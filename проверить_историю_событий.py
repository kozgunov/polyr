"""Запускает сравнение контекста 0/3/12 событий и проверяет гипотезу после убытка."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "исходный_код"))

from polybot.analytics.history_context_experiment import evaluate as evaluate_history
from polybot.analytics.loss_reversal_hypothesis import evaluate as evaluate_reversal


if __name__ == "__main__":
    try:
        print(json.dumps({"history_context": evaluate_history(), "loss_reversal": evaluate_reversal()}, ensure_ascii=False, indent=2))
        print("all_doneпроверить_историю_событий.py")
    except Exception:
        print("error_in_проверить_историю_событий.py")
        raise
