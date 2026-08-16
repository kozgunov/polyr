"""Проверяет готовность и формирует воспроизводимый план GPU-обучения."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import app_config as settings
from polybot.models.temporal_validation import purged_expanding_folds


def inspect_training(model: str) -> dict:
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(settings.SEQUENCE_DATASET_PATH)
    table = pq.read_table(settings.SEQUENCE_DATASET_PATH, columns=["event_slug", "observed_at"])
    frame = table.to_pandas().sort_values("observed_at")
    events = frame["event_slug"].drop_duplicates().tolist()
    plan = json.loads(Path(settings.GPU_TRAINING_PLAN_PATH).read_text(encoding="utf-8"))
    validation = plan["validation"]
    split = purged_expanding_folds(
        events, folds=validation["minimum_folds"], purge=validation["purge_events"], embargo=validation["embargo_events"]
    )
    return {
        "model": model, "dataset_rows": metadata.num_rows, "events": len(events),
        "folds": len(split), "features": len(pq.read_schema(settings.SEQUENCE_DATASET_PATH).names),
        "status": "ready" if len(split) >= validation["minimum_folds"] else "not_enough_events",
        "next_command": f"python -m polybot.models.train_gpu_model --model {model}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("tcn", "gru", "tiny_transformer"), default="tcn")
    args = parser.parse_args()
    if not settings.SEQUENCE_DATASET_PATH.exists():
        raise SystemExit("Сначала соберите sequence dataset.")
    print(json.dumps(inspect_training(args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

