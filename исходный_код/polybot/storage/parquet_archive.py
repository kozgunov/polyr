"""Hourly compact Parquet rollover for completed collector intervals."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import app_config as settings
import pyarrow as pa
import pyarrow.parquet as pq


TABLES: dict[str, tuple[str, str]] = {
    "market_snapshots": (
        "collected_at",
        "SELECT id,collected_at,event_slug,market_id,token_id,outcome,best_bid,best_bid_size,best_ask,best_ask_size,midpoint,spread,book_timestamp,book_hash FROM market_snapshots WHERE collected_at>=? AND collected_at<? ORDER BY id",
    ),
    "external_prices": (
        "collected_at",
        "SELECT id,collected_at,source,symbol,price,confidence,source_timestamp FROM external_prices WHERE collected_at>=? AND collected_at<? ORDER BY id",
    ),
    "reference_price_snapshots": (
        "collected_at",
        "SELECT id,collected_at,event_slug,target_price,reference_price,source_timestamp,completed,source FROM reference_price_snapshots WHERE collected_at>=? AND collected_at<? ORDER BY id",
    ),
    "source_metrics": (
        "measured_at",
        "SELECT id,measured_at,source,operation,status,latency_ms,detail FROM source_metrics WHERE measured_at>=? AND measured_at<? ORDER BY id",
    ),
}


def _hour_floor(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    temporary.replace(path)


def archive_previous_hour(connection: sqlite3.Connection, current: datetime | None = None) -> dict[str, Any]:
    """Archives only the fully completed UTC hour; repeated calls are idempotent."""
    end = _hour_floor(current or datetime.now(UTC))
    start = end - timedelta(hours=1)
    hour_key = start.isoformat()
    connection.execute(
        """CREATE TABLE IF NOT EXISTS parquet_archive_manifest(
             hour_start TEXT PRIMARY KEY,archived_at TEXT NOT NULL,rows_json TEXT NOT NULL)"""
    )
    if connection.execute("SELECT 1 FROM parquet_archive_manifest WHERE hour_start=?", (hour_key,)).fetchone():
        return {"hour_start": hour_key, "status": "already_archived", "rows": {}}

    import json

    partition = (
        settings.PARQUET_ARCHIVE_DIR
        / f"year={start:%Y}" / f"month={start:%m}" / f"day={start:%d}" / f"hour={start:%H}"
    )
    counts: dict[str, int] = {}
    previous_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        for table, (_column, query) in TABLES.items():
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
            ).fetchone()
            if not exists:
                continue
            rows = [dict(row) for row in connection.execute(query, (start.isoformat(), end.isoformat())).fetchall()]
            counts[table] = len(rows)
            if rows:
                _write_atomic(partition / f"{table}.parquet", rows)
        connection.execute(
            "INSERT INTO parquet_archive_manifest(hour_start,archived_at,rows_json) VALUES(?,?,?)",
            (hour_key, datetime.now(UTC).isoformat(), json.dumps(counts, ensure_ascii=False)),
        )
        connection.commit()
    finally:
        connection.row_factory = previous_factory
    return {"hour_start": hour_key, "status": "archived", "rows": counts, "path": str(partition)}

