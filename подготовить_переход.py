"""Проверяет базу, модели и файлы перед переносом проекта на новый компьютер."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "исходный_код"))

from polybot.tools.migration_audit import build_manifest


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", action="store_true", help="checkpoint WAL после остановки всех процессов")
        args = parser.parse_args()
        report = build_manifest(checkpoint=args.checkpoint)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("all_doneподготовить_переход.py")
    except Exception:
        print("error_in_подготовить_переход.py")
        raise
