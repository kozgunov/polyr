"""Пересобирает переносимый Parquet-датасет для временных GPU-моделей."""

from __future__ import annotations

import json

from polybot.models.sequence_dataset import build


if __name__ == "__main__":
    try:
        print(json.dumps(build(), ensure_ascii=False, indent=2))
        print("all_doneсобрать_sequence_dataset.py")
    except Exception:
        print("error_in_собрать_sequence_dataset.py")
        raise

