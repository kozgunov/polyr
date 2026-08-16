"""Показывает безопасный снимок состояния демо-движка без изменения базы."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as settings


def rows(connection: sqlite3.Connection, query: str) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(query)]


def main() -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    report = {
        "controls": rows(connection, "SELECT * FROM runtime_controls WHERE control_key IN ('engine_state','selected_model','trading_mode')"),
        "sessions": rows(connection, "SELECT session_id,status,strategy_version,run_label,cash_balance_usdc,realized_pnl_usdc FROM paper_sessions ORDER BY started_at DESC LIMIT 3"),
        "open_positions": rows(connection, "SELECT id,session_id,event_slug,outcome,status,cost_usdc,current_price FROM paper_positions WHERE status='open'"),
        "event_targets": connection.execute("SELECT COUNT(*) FROM event_targets").fetchone()[0],
    }
    connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
        print(f"all_done{Path(__file__).name}")
    except Exception:
        print(f"error_in_{Path(__file__).name}")
        raise
