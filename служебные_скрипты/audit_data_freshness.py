"""Read-only проверка фактического возраста данных каждого источника."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as settings


def _age(value: str | None) -> float | None:
    if not value:
        return None
    return max(0.0, (datetime.now(UTC) - datetime.fromisoformat(value)).total_seconds())


def main() -> None:
    db = sqlite3.connect(settings.DATABASE_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    prices = [dict(row) for row in db.execute(
        """SELECT source,MAX(collected_at) latest FROM external_prices
           GROUP BY source ORDER BY source"""
    )]
    for row in prices:
        row["age_seconds"] = _age(row["latest"])
    result = {
        "checked_at": datetime.now(UTC).isoformat(),
        "external_prices": prices,
        "polymarket_book": None,
        "polymarket_reference": None,
        "latest_event": None,
        "collector_run": None,
        "wal_bytes": settings.DATABASE_PATH.with_name(settings.DATABASE_PATH.name + "-wal").stat().st_size
            if settings.DATABASE_PATH.with_name(settings.DATABASE_PATH.name + "-wal").exists() else 0,
    }
    for key, query in {
        "polymarket_book": "SELECT MAX(collected_at) latest FROM market_snapshots",
        "polymarket_reference": "SELECT MAX(collected_at) latest FROM reference_price_snapshots",
        "latest_event": "SELECT MAX(fetched_at) latest FROM events",
    }.items():
        latest = db.execute(query).fetchone()["latest"]
        result[key] = {"latest": latest, "age_seconds": _age(latest)}
    run = db.execute(
        "SELECT event_slug,status,started_at,ended_at FROM collector_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    result["collector_run"] = dict(run) if run else None
    db.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
