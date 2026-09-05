"""Train the CatBoost candidate with an event-level chronological split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import app_config as settings
import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.train_direction_model import load_dataset
from polybot.models.artifact_versions import save_version_bundle


def train(
    path: Path,
    allow_small_sample: bool = False,
    output: Path | None = None,
    history_windows: tuple[int, ...] = (),
) -> dict[str, float | int | bool]:
    x, y, groups = load_dataset(path, history_windows)
    events = list(dict.fromkeys(groups.tolist()))
    if len(events) < settings.TRAINING_MIN_INDEPENDENT_EVENTS and not allow_small_sample:
        raise RuntimeError(f"TRAINING_BLOCKED: only {len(events)} independent events")
    train_end = max(1, int(len(events) * (1 - settings.TRAINING_TEST_SIZE - settings.TRAINING_CALIBRATION_SIZE)))
    calibration_end = max(train_end + 1, int(len(events) * (1 - settings.TRAINING_TEST_SIZE)))
    train_events = set(events[:train_end])
    calibration_events = set(events[train_end:calibration_end])
    test_events = set(events[calibration_end:])
    train_mask = np.array([group in train_events for group in groups])
    calibration_mask = np.array([group in calibration_events for group in groups])
    test_mask = np.array([group in test_events for group in groups])
    model = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.04, loss_function="Logloss",
        random_seed=settings.TRAINING_RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    model.fit(x[train_mask], y[train_mask])
    raw_calibration = np.clip(model.predict_proba(x[calibration_mask])[:, 1], 1e-6, 1 - 1e-6)
    calibration_logit = np.log(raw_calibration / (1.0 - raw_calibration)).reshape(-1, 1)
    calibration_groups = groups[calibration_mask]
    group_counts = {group: int(np.sum(calibration_groups == group)) for group in calibration_events}
    weights = np.asarray([1.0 / group_counts[group] for group in calibration_groups])
    calibrator = LogisticRegression(C=0.25, random_state=settings.TRAINING_RANDOM_STATE)
    calibrator.fit(calibration_logit, y[calibration_mask], sample_weight=weights)
    raw_probability = np.clip(model.predict_proba(x[test_mask])[:, 1], 1e-6, 1 - 1e-6)
    test_logit = np.log(raw_probability / (1.0 - raw_probability)).reshape(-1, 1)
    probability = calibrator.predict_proba(test_logit)[:, 1]
    prediction = (probability >= 0.5).astype(np.int8)
    metrics: dict[str, float | int | bool] = {
        "independent_events": len(events), "train_events": len(train_events),
        "calibration_events": len(calibration_events), "test_events": len(test_events),
        "rows": len(y), "accuracy": float(accuracy_score(y[test_mask], prediction)),
        "brier": float(brier_score_loss(y[test_mask], probability)),
        "log_loss": float(log_loss(y[test_mask], probability, labels=[0, 1])),
        "raw_brier": float(brier_score_loss(y[test_mask], raw_probability)),
        "raw_log_loss": float(log_loss(y[test_mask], raw_probability, labels=[0, 1])),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_mask], prediction)),
        "roc_auc": float(roc_auc_score(y[test_mask], probability)) if len(set(y[test_mask])) == 2 else 0.0,
        "pr_auc": float(average_precision_score(y[test_mask], probability)) if len(set(y[test_mask])) == 2 else 0.0,
        "production_ready": len(events) >= settings.TRAINING_MIN_INDEPENDENT_EVENTS,
    }
    output = output or settings.MODEL_DIR / "catboost"
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "catboost_btc_5m.cbm"
    bundle_path = output / "catboost_bundle.joblib"
    metrics_path = output / "metrics.json"
    model.save_model(model_path)
    joblib.dump({
        "calibrator": calibrator,
        "metrics": metrics,
        "history_windows": list(history_windows),
        "version": "v5_target_time_path_history_event_platt_calibrated",
        "splits": {
            "train_events": sorted(train_events),
            "calibration_events": sorted(calibration_events),
            "test_events": sorted(test_events),
        },
    }, bundle_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    version = save_version_bundle("catboost", "entry", [model_path, bundle_path, metrics_path], metrics)
    metrics["artifact_version"] = str(version["version"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--allow-small-sample", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.db, args.allow_small_sample, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
