"""Однократно архивирует прибыльные исторические версии моделей по demo-сессиям."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for item in (PROJECT_ROOT, SOURCE_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import app_config as settings
from polybot.models.version_archive import archive_profitable_session


def main() -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    created = []
    try:
        sessions = connection.execute("SELECT session_id FROM paper_sessions WHERE status IN ('archived','paused')").fetchall()
        for row in sessions:
            created.extend(archive_profitable_session(connection, str(row[0])))
    finally:
        connection.close()
    print(f"Архивировано версий: {len(created)}")


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
