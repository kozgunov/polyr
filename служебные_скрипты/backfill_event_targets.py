"""Заполняет исторический Price to Beat и пересобирает target-based признаки."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import api_config as api
import app_config as settings
import httpx

from polybot.collectors.pipeline import now
from polybot.labeling.resolution import features_for_snapshot

SCHEMA = """
CREATE TABLE IF NOT EXISTS event_targets (
  event_slug TEXT PRIMARY KEY, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
  target_price REAL NOT NULL, latest_reference_price REAL,
  source_timestamp TEXT, completed INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL, fetched_at TEXT NOT NULL, raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reference_price_snapshots (
  id INTEGER PRIMARY KEY, collected_at TEXT NOT NULL, event_slug TEXT NOT NULL,
  target_price REAL NOT NULL, reference_price REAL,
  source_timestamp TEXT, completed INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL, raw_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reference_event_time
  ON reference_price_snapshots(event_slug,collected_at);
CREATE INDEX IF NOT EXISTS idx_prices_source_time
  ON external_prices(source,collected_at);
"""


def _event_times(connection: sqlite3.Connection, slug: str, raw: str) -> tuple[str, str]:
    data = json.loads(raw)
    market_row = connection.execute(
        """SELECT m.raw_json FROM markets m JOIN events e ON e.event_id=m.event_id
           WHERE e.slug=? AND m.enable_order_book=1 LIMIT 1""", (slug,),
    ).fetchone()
    market = json.loads(market_row[0]) if market_row else {}
    start = market.get("eventStartTime") or data.get("startTime")
    end = market.get("endDate") or market.get("endDateIso") or data.get("endDate")
    if start and end:
        return str(start), str(end)
    response = httpx.get(f"{api.POLYMARKET_GAMMA_URL}/events/slug/{slug}", timeout=15)
    response.raise_for_status()
    event = response.json()
    return str(event["startTime"]), str(event["endDate"])


def _fetch(client: httpx.Client, start: str, end: str) -> dict[str, Any]:
    for attempt in range(5):
        response = client.get(
            api.POLYMARKET_CRYPTO_PRICE_URL,
            params={"symbol": "BTC", "eventStartTime": start, "variant": "fiveminute", "endDate": end},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://polymarket.com/"},
        )
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response.json()
        time.sleep(min(8.0, 1.0 * (2 ** attempt)))
    response.raise_for_status()
    raise RuntimeError("unreachable")


def backfill(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    connection.executescript(SCHEMA)
    # Старый experiment variant=five возвращал ненулевой closePrice у незавершённых окон.
    # Сохраняем raw для аудита, но исключаем только эти заведомо неверные live-снимки.
    connection.execute(
        """UPDATE reference_price_snapshots SET source='invalid_wrong_variant_five'
           WHERE source='polymarket_crypto_price' AND completed=0 AND reference_price IS NOT NULL"""
    )
    connection.commit()
    events = connection.execute(
        """SELECT slug,raw_json FROM events WHERE slug LIKE 'btc-updown-5m-%'
           AND slug NOT IN (SELECT event_slug FROM event_targets WHERE source='polymarket_crypto_price') ORDER BY slug"""
    ).fetchall()
    fetched = failed = 0
    failures: list[dict[str, str]] = []
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for row in events:
            slug = str(row["slug"])
            try:
                start, end = _event_times(connection, slug, str(row["raw_json"]))
                data = _fetch(client, start, end)
                target = float(data["openPrice"])
                reference = float(data["closePrice"]) if data.get("closePrice") is not None else None
                raw = json.dumps(data, ensure_ascii=False)
                completed = int(bool(data.get("completed")))
                connection.execute(
                    """INSERT INTO event_targets(event_slug,start_time,end_time,target_price,
                       latest_reference_price,source_timestamp,completed,source,fetched_at,raw_json)
                       VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(event_slug) DO UPDATE SET
                       target_price=excluded.target_price,latest_reference_price=excluded.latest_reference_price,
                       source_timestamp=excluded.source_timestamp,completed=excluded.completed,
                       source=excluded.source,fetched_at=excluded.fetched_at,raw_json=excluded.raw_json""",
                    (slug, start, end, target, reference, str(data.get("timestamp") or ""), completed,
                     "polymarket_crypto_price", now(), raw),
                )
                if reference is not None and completed:
                    connection.execute(
                        """INSERT INTO reference_price_snapshots(collected_at,event_slug,target_price,
                           reference_price,source_timestamp,completed,source,raw_json)
                           SELECT ?,?,?,?,?,1,'historical_close_only',?
                           WHERE NOT EXISTS (SELECT 1 FROM reference_price_snapshots
                           WHERE event_slug=? AND source='historical_close_only')""",
                        (end, slug, target, reference, str(data.get("timestamp") or ""), raw, slug),
                    )
                connection.commit()
                fetched += 1
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failed += 1
                detail = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    detail = f"HTTP {exc.response.status_code}: {exc.response.text[:160]}"
                failures.append({"slug": slug, "error": type(exc).__name__, "detail": detail})
            time.sleep(0.25)

    examples = connection.execute(
        """SELECT t.snapshot_id,s.* FROM training_examples t
           JOIN market_snapshots s ON s.id=t.snapshot_id ORDER BY t.snapshot_id"""
    ).fetchall()
    rebuilt = with_target = 0
    for row in examples:
        features = features_for_snapshot(connection, row)
        connection.execute(
            "UPDATE training_examples SET features_json=?,labeled_at=? WHERE snapshot_id=?",
            (json.dumps(features, ensure_ascii=False), now(), row["snapshot_id"]),
        )
        rebuilt += 1
        with_target += int(features.get("target_price") is not None)
        if rebuilt % 250 == 0:
            connection.commit()
    connection.commit()
    connection.close()
    return {
        "events_seen": len(events), "targets_fetched": fetched, "targets_failed": failed,
        "examples_rebuilt": rebuilt, "examples_with_target": with_target,
        "failures": failures[:20],
    }


def main() -> None:
    report = backfill(settings.DATABASE_PATH)
    output = settings.MODEL_DIR / "target_backfill_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
