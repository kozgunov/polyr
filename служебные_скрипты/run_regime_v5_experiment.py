"""Запускает воспроизводимый walk-forward эксперимент режимной entry-модели v5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

import app_config as settings
from polybot.models.train_regime_entry import run


def main() -> None:
    snapshot = (
        settings.MODEL_CANDIDATE_DIR
        / "entry_value_v4_tail_risk"
        / "training_snapshot.sqlite3"
    )
    if not snapshot.exists():
        raise FileNotFoundError(
            "Не найден фиксированный training_snapshot.sqlite3 из эксперимента v4."
        )
    report = run(snapshot)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
