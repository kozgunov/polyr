"""Label historical snapshots with resolved Polymarket outcomes; never creates trades."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median, pstdev
from typing import Any

import api_config as api
import app_config as settings
import httpx

from polybot.collectors.pipeline import ROOT, now, parse_json


def features_for_snapshot(connection: sqlite3.Connection, snapshot: sqlite3.Row) -> dict[str, Any]:
    values: dict[str, Any] = {
        "best_bid": snapshot["best_bid"], "best_ask": snapshot["best_ask"],
        "midpoint": snapshot["midpoint"], "spread": snapshot["spread"],
        "best_bid_size": snapshot["best_bid_size"], "best_ask_size": snapshot["best_ask_size"],
    }
    source_prices: dict[str, float] = {}
    for source in ("bybit", "okx", "pyth"):
        row = connection.execute(
            """SELECT price, confidence, source_timestamp FROM external_prices
               WHERE source=? AND collected_at<=? ORDER BY collected_at DESC LIMIT 1""",
            (source, snapshot["collected_at"]),
        ).fetchone()
        if row:
            values[f"{source}_price"] = row[0]
            values[f"{source}_confidence"] = row[1]
            values[f"{source}_timestamp"] = row[2]
            source_prices[source] = float(row[0])

    target = connection.execute(
        "SELECT target_price,source FROM event_targets WHERE event_slug=? AND source='polymarket_crypto_price'",
        (snapshot["event_slug"],),
    ).fetchone()
    target_price = float(target[0]) if target else None
    official = connection.execute(
        """SELECT reference_price FROM reference_price_snapshots
           WHERE event_slug=? AND collected_at<=? AND reference_price IS NOT NULL
           AND source IN ('polymarket_crypto_price','historical_close_only')
           ORDER BY collected_at DESC LIMIT 1""",
        (snapshot["event_slug"], snapshot["collected_at"]),
    ).fetchone()
    reference_price = float(official[0]) if official else median(source_prices.values()) if source_prices else None
    reference_source = "polymarket_crypto_price" if official else "external_median_proxy"
    start_timestamp = int(snapshot["event_slug"].rsplit("-", 1)[-1])
    observed = datetime.fromisoformat(snapshot["collected_at"])
    elapsed_seconds = min(300.0, max(0.0, observed.timestamp() - start_timestamp))
    remaining_seconds = max(0.0, 300.0 - elapsed_seconds)
    values.update({
        "target_price": target_price,
        "target_source": str(target[1]) if target else None,
        "reference_price": reference_price,
        "reference_source": reference_source,
        "elapsed_seconds": elapsed_seconds,
        "remaining_seconds": remaining_seconds,
    })
    if target_price and reference_price:
        values["distance_to_target_usd"] = reference_price - target_price
        values["distance_to_target_pct"] = (reference_price / target_price - 1.0) * 100.0
        for source, price in source_prices.items():
            values[f"{source}_to_target_pct"] = (price / target_price - 1.0) * 100.0

    cutoff = (observed - timedelta(seconds=60)).isoformat()
    rows = connection.execute(
        """SELECT price FROM external_prices WHERE source='bybit' AND collected_at BETWEEN ? AND ?
           ORDER BY collected_at""",
        (cutoff, snapshot["collected_at"]),
    ).fetchall()
    returns = [
        math.log(float(rows[index][0]) / float(rows[index - 1][0])) * 100.0
        for index in range(1, len(rows)) if float(rows[index - 1][0]) > 0
    ]
    volatility = pstdev(returns) if len(returns) >= 2 else 0.0
    values["realized_volatility_60s_pct"] = volatility
    distance_pct = float(values.get("distance_to_target_pct") or 0.0)
    time_scale = math.sqrt(max(remaining_seconds, 1.0) / 60.0)
    values["distance_time_score"] = distance_pct / max(volatility * time_scale, 0.01)
    if target_price:
        current_bybit_distance = values.get("bybit_to_target_pct")
        for lag_seconds in (15, 30, 60):
            lag_time = (observed - timedelta(seconds=lag_seconds)).isoformat()
            lag_row = connection.execute(
                """SELECT price FROM external_prices WHERE source='bybit' AND collected_at<=?
                   ORDER BY collected_at DESC LIMIT 1""", (lag_time,),
            ).fetchone()
            if lag_row:
                lag_distance = (float(lag_row[0]) / target_price - 1.0) * 100.0
                values[f"distance_lag_{lag_seconds}s_pct"] = lag_distance
                if current_bybit_distance is not None:
                    values[f"target_momentum_{lag_seconds}s_pct"] = float(current_bybit_distance) - lag_distance
    return values


async def resolved_event(client: httpx.AsyncClient, slug: str) -> dict[str, Any] | None:
    response = await client.get(f"{api.POLYMARKET_GAMMA_URL}/events/slug/{slug}")
    if response.status_code == 404: return None
    response.raise_for_status()
    data = response.json()
    return data if data.get("closed") else None


async def label_database(path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS training_examples (
          snapshot_id INTEGER PRIMARY KEY, event_slug TEXT NOT NULL, market_id TEXT NOT NULL,
          token_id TEXT NOT NULL, outcome TEXT NOT NULL, observed_at TEXT NOT NULL,
          label INTEGER NOT NULL CHECK(label IN (0,1)), label_kind TEXT NOT NULL,
          features_json TEXT NOT NULL, labeled_at TEXT NOT NULL
        )"""
    )
    connection.commit()
    slugs = [row[0] for row in connection.execute("""SELECT DISTINCT event_slug FROM market_snapshots
        WHERE event_slug NOT IN (SELECT DISTINCT event_slug FROM training_examples)""")]
    labeled, pending = 0, 0
    async with httpx.AsyncClient(timeout=15) as client:
        for slug in slugs:
            event = await resolved_event(client, slug)
            if not event:
                pending += 1; continue
            result_by_token: dict[str, int] = {}
            for market in event.get("markets", []):
                outcomes, prices, token_ids = parse_json(market.get("outcomes")), parse_json(market.get("outcomePrices")), parse_json(market.get("clobTokenIds"))
                if not (len(outcomes) == len(prices) == len(token_ids)): continue
                for token_id, price in zip(token_ids, prices, strict=True):
                    try: value = float(price)
                    except (TypeError, ValueError): continue
                    if value in (0.0, 1.0): result_by_token[str(token_id)] = int(value)
            snapshots = connection.execute("""SELECT * FROM market_snapshots WHERE event_slug=?
                AND id NOT IN (SELECT snapshot_id FROM training_examples)""", (slug,)).fetchall()
            for snapshot in snapshots:
                label = result_by_token.get(snapshot["token_id"])
                if label is None: continue
                connection.execute("""INSERT OR IGNORE INTO training_examples
                    (snapshot_id,event_slug,market_id,token_id,outcome,observed_at,label,label_kind,features_json,labeled_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""", (snapshot["id"], slug, snapshot["market_id"], snapshot["token_id"], snapshot["outcome"], snapshot["collected_at"], label, "resolved_outcome", json.dumps(features_for_snapshot(connection, snapshot)), now()))
                labeled += 1
            for attempt in range(settings.SQLITE_WRITE_RETRY_ATTEMPTS):
                try:
                    connection.commit()
                    break
                except sqlite3.OperationalError as error:
                    connection.rollback()
                    if "locked" not in str(error).lower() or attempt + 1 >= settings.SQLITE_WRITE_RETRY_ATTEMPTS:
                        raise
                    time.sleep(settings.SQLITE_WRITE_RETRY_DELAY_SECONDS * (attempt + 1))
    connection.close()
    return labeled, pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(settings.DATABASE_PATH))
    args = parser.parse_args()
    path = ROOT / args.db
    if not path.exists(): raise SystemExit(f"LABEL_ERROR database not found: {path}")
    labeled, pending = asyncio.run(label_database(path))
    print(f"LABEL_OK new_examples={labeled} pending_events={pending}")


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
