"""Сохраняет модели v13 вместе с PAPER/LIVE событиями и результатами."""

from __future__ import annotations

import json

from polybot.models.experiment_snapshot import create_snapshot


if __name__ == "__main__":
    try:
        print(json.dumps(create_snapshot(), ensure_ascii=False, indent=2))
        print("all_doneсохранить_эксперимент_v13.py")
    except Exception:
        print("error_in_сохранить_эксперимент_v13.py")
        raise

