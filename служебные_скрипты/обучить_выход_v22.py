"""Создаёт и обучает exit v22 как отдельного кандидата, не меняя рабочую модель."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for directory in (PROJECT_ROOT, SOURCE_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import app_config as settings
from polybot.models.exit_sequence_v22 import build_dataset, train


def main() -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = settings.MODEL_CANDIDATE_DIR / f"exit_sequence_v22_{stamp}"
    dataset = output_dir / "exit_sequence_v22.parquet"
    manifest = output_dir / "exit_sequence_v22_manifest.json"
    dataset_report = build_dataset(settings.DATABASE_PATH, dataset, manifest)
    training_report = train(dataset, output_dir)
    print(json.dumps({"dataset": dataset_report, "training": training_report,
                      "activation": "candidate_not_activated"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
