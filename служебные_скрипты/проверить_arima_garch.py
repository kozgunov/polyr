"""Проводит причинный ARIMA/GARCH shadow-бэктест без отправки заявок."""

from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for directory in (PROJECT_ROOT, SOURCE_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import app_config as settings
from polybot.analytics.timeseries_walk_forward import run


def main() -> None:
    warnings.filterwarnings("ignore", module="statsmodels")
    with sqlite3.connect(f"file:{settings.DATABASE_PATH.as_posix()}?mode=ro", uri=True, timeout=60) as connection:
        # 60 последних независимых событий × 3 причинных checkpoint — достаточно
        # для первого challenger-screening и не отбирает CPU у торгового движка надолго.
        payload = run(connection, max_events=60)
    settings.TIMESERIES_CHALLENGER_REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
        print("all_doneпроверить_arima_garch.py")
    except Exception:
        print("error_in_проверить_arima_garch.py")
        raise
