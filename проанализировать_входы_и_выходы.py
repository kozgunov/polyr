"""Запускает статистический эксперимент цен входа и раннего выхода.

Результат сохраняется в ``модели/experiments`` как JSON и два Parquet-датасета.
Рабочие модели и режим торговли этот файл не переключает.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from polybot.analytics.entry_exit_price_study import study


if __name__ == "__main__":
    try:
        result = study()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"all_done{Path(__file__).name}")
    except Exception:
        print(f"error_in_{Path(__file__).name}")
        raise
