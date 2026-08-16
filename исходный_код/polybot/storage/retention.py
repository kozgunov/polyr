"""Bounded SQLite retention; preserves labeled/model/trading records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import app_config as settings


def prune(connection: sqlite3.Connection) -> dict[str, int]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    deleted: dict[str, int] = {}
    policies = {
        "raw_messages": ("received_at", settings.RAW_DATA_RETENTION_DAYS),
        "source_metrics": ("measured_at", settings.NORMALIZED_DATA_RETENTION_DAYS),
        # После разметки графиковые признаки уже находятся в training_examples и
        # action_value_examples, поэтому бесконечно хранить каждый сырой тик не нужно.
        "external_prices": ("collected_at", settings.NORMALIZED_DATA_RETENTION_DAYS),
        "market_snapshots": ("collected_at", settings.NORMALIZED_DATA_RETENTION_DAYS),
        "reference_price_snapshots": ("collected_at", settings.NORMALIZED_DATA_RETENTION_DAYS),
        "paper_equity_snapshots": ("observed_at", settings.EQUITY_SNAPSHOT_RETENTION_DAYS),
        "model_decisions": ("observed_at", settings.DECISION_RETENTION_DAYS),
    }
    for table, (column, days) in policies.items():
        if table not in tables:
            continue
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        cursor = connection.execute(f'DELETE FROM "{table}" WHERE "{column}"<?', (cutoff,))
        deleted[table] = max(0, cursor.rowcount)
    # Raw payloads are not training data. With storage disabled they can be safely discarded.
    if not settings.STORE_RAW_MESSAGES and "raw_messages" in tables:
        cursor = connection.execute("DELETE FROM raw_messages")
        deleted["raw_messages"] = deleted.get("raw_messages", 0) + max(0, cursor.rowcount)
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
    connection.execute("PRAGMA optimize")
    return deleted
