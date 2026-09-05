"""Подготовка всех датасетов одной командой без остановки сборщика и торговли.

Запуск: ``python подготовить_дообучение.py``.
Рабочие модели не заменяются: создаётся отдельный неизменяемый training bundle.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from polybot.models.training_bundle import enrich_direction_dataset, prepare


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--enrich-bundle", type=Path)
        args = parser.parse_args()
        result = enrich_direction_dataset(args.enrich_bundle) if args.enrich_bundle else prepare()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("all_doneподготовить_дообучение.py")
    except Exception:
        print("error_in_подготовить_дообучение.py")
        raise
