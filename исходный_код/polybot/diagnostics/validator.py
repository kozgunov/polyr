"""Integrity and freshness checks for the local read-only collector database."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime, timedelta

import app_config as settings

ROOT = settings.PROJECT_ROOT


def count(connection: sqlite3.Connection, query: str, values: tuple = ()) -> int:
    return int(connection.execute(query, values).fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(settings.DATABASE_PATH))
    parser.add_argument("--max-age-seconds", type=int, default=90)
    args = parser.parse_args()
    path = ROOT / args.db
    if not path.exists(): raise SystemExit(f"VALIDATION_ERROR database not found: {path}")
    connection = sqlite3.connect(path)
    cutoff = (datetime.now(UTC) - timedelta(seconds=args.max_age_seconds)).isoformat()
    checks = {
        "successful_runs": count(connection, "SELECT COUNT(*) FROM collector_runs WHERE status='ok'"),
        "events": count(connection, "SELECT COUNT(*) FROM events"),
        "closed_events": count(connection, "SELECT COUNT(*) FROM events WHERE closed=1"),
        "markets": count(connection, "SELECT COUNT(*) FROM markets"),
        "book_snapshots": count(connection, "SELECT COUNT(*) FROM market_snapshots"),
        "external_prices": count(connection, "SELECT COUNT(*) FROM external_prices"),
        "recent_books": count(connection, "SELECT COUNT(*) FROM market_snapshots WHERE collected_at >= ?", (cutoff,)),
        "recent_prices": count(connection, "SELECT COUNT(*) FROM external_prices WHERE collected_at >= ?", (cutoff,)),
        "crossed_books": count(connection, "SELECT COUNT(*) FROM market_snapshots WHERE best_bid > best_ask"),
        "invalid_probability": count(connection, "SELECT COUNT(*) FROM market_snapshots WHERE best_bid NOT BETWEEN 0 AND 1 OR best_ask NOT BETWEEN 0 AND 1"),
    }
    connection.close()
    for name, value in checks.items(): print(f"{name}={value}")
    required = ["successful_runs", "events", "markets", "external_prices", "recent_prices"]
    if checks["closed_events"] == 0:
        required += ["book_snapshots", "recent_books"]
    failed = [name for name in required if checks[name] == 0]
    failed += [name for name in ("crossed_books", "invalid_probability") if checks[name] != 0]
    if failed: raise SystemExit("VALIDATION_ERROR " + ", ".join(failed))
    if checks["closed_events"] and checks["book_snapshots"] == 0:
        print("VALIDATION_WARNING closed event has no active CLOB orderbook")
    print("VALIDATION_OK")


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
