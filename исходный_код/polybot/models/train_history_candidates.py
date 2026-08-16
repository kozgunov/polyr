"""Обучает отдельные candidate-only артефакты для контекста 0/3/12 событий."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import app_config as settings

from polybot.models.train_direction_model import train


def run() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "candidate_only": True, "variants": {},
    }
    for raw_window in settings.EVENT_HISTORY_WINDOWS:
        windows = () if int(raw_window) == 0 else (int(raw_window),)
        directory = settings.HISTORY_MODEL_CANDIDATE_DIR / f"history_{raw_window}"
        artifact = directory / "direction_model.joblib"
        metrics = train(settings.DATABASE_PATH, artifact_path=artifact, history_windows=windows)
        report["variants"][f"history_{raw_window}"] = {
            "artifact": str(artifact), "history_windows": list(windows), "metrics": metrics,
        }
    output = settings.HISTORY_MODEL_CANDIDATE_DIR / "candidate_training_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
