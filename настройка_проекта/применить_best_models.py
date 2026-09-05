"""Применяет воспроизводимую конфигурацию best-models, оставляя режим PAPER."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app_config as settings


def main() -> None:
    config = json.loads((Path(__file__).with_name("best_models_runtime.json")).read_text(encoding="utf-8"))
    with sqlite3.connect(settings.DATABASE_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS runtime_controls(
            control_key TEXT PRIMARY KEY,control_value TEXT NOT NULL,updated_at TEXT NOT NULL,reason TEXT)""")
        stamp = datetime.now(UTC).isoformat()
        values = {
            "selected_entry_model": "custom",
            "selected_exit_model": "catboost",
            "selected_model": "custom",
            "trade_size_multiplier": str(config["recommended_live_size_multiplier"]),
            "trading_mode": "paper",
            "requested_trading_mode": "paper",
        }
        for key, value in values.items():
            db.execute("INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
                       (key, value, stamp, f"applied {config['release']}"))
    print(f"all_done_{__file__}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"error_in_{__file__}")
        raise
