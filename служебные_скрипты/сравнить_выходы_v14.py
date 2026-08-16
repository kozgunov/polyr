"""Сравнивает фактический partial, полный выход в тот же момент и HOLD."""

from __future__ import annotations

import json
from polybot.analytics.exit_policy_comparison import compare

if __name__ == "__main__":
    try:
        print(json.dumps(compare(), ensure_ascii=False, indent=2))
        print("all_doneсравнить_выходы_v14.py")
    except Exception:
        print("error_in_сравнить_выходы_v14.py")
        raise

