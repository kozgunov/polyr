"""Event-level evaluation for a compact numeric baseline; prevents snapshot leakage."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import app_config as settings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.artifact_versions import save_version_bundle
from polybot.models.event_history import build_summaries, context as history_context, feature_names as history_feature_names

FEATURE_NAMES = [
    "is_up", "best_bid", "best_ask", "midpoint", "spread", "log_bid_size", "log_ask_size",
    "source_dispersion_pct", "bybit_vs_median_pct", "okx_vs_median_pct", "pyth_vs_median_pct",
    "distance_to_target_pct", "absolute_distance_to_target_pct", "bybit_to_target_pct",
    "okx_to_target_pct", "pyth_to_target_pct", "realized_volatility_60s_pct",
    "distance_time_score", "remaining_fraction", "official_reference_available",
    "distance_lag_15s_pct", "distance_lag_30s_pct", "distance_lag_60s_pct",
    "target_momentum_15s_pct", "target_momentum_30s_pct", "target_momentum_60s_pct",
]


def vector(
    event_slug: str, outcome: str, observed_at: str, features: dict,
    history_windows: tuple[int, ...] | list[int] = (),
) -> list[float]:
    prices = [float(features[key]) for key in ("bybit_price", "okx_price", "pyth_price") if features.get(key)]
    centre = float(np.median(prices)) if prices else 0.0
    dispersion = (max(prices) - min(prices)) / centre * 100 if len(prices) >= 2 and centre else 999.0
    start = int(event_slug.rsplit("-", 1)[-1])
    observed = __import__("datetime").datetime.fromisoformat(observed_at).timestamp()
    elapsed = min(1.0, max(0.0, (observed - start) / 300.0))
    distance = float(features.get("distance_to_target_pct") or 0.0)
    def deviation(key: str) -> float:
        return (float(features[key]) / centre - 1) * 100 if centre and features.get(key) else 0.0
    values = [
        float(outcome == "Up"), float(features.get("best_bid") or 0), float(features.get("best_ask") or 0),
        float(features.get("midpoint") or 0.5), float(features.get("spread") or 1),
        float(np.log1p(features.get("best_bid_size") or 0)), float(np.log1p(features.get("best_ask_size") or 0)),
        dispersion, deviation("bybit_price"), deviation("okx_price"), deviation("pyth_price"),
        distance, abs(distance), float(features.get("bybit_to_target_pct") or 0.0),
        float(features.get("okx_to_target_pct") or 0.0), float(features.get("pyth_to_target_pct") or 0.0),
        float(features.get("realized_volatility_60s_pct") or 0.0),
        float(features.get("distance_time_score") or 0.0), 1.0 - elapsed,
        float(features.get("reference_source") == "polymarket_crypto_price"),
        float(features.get("distance_lag_15s_pct") or 0.0),
        float(features.get("distance_lag_30s_pct") or 0.0),
        float(features.get("distance_lag_60s_pct") or 0.0),
        float(features.get("target_momentum_15s_pct") or 0.0),
        float(features.get("target_momentum_30s_pct") or 0.0),
        float(features.get("target_momentum_60s_pct") or 0.0),
    ]
    values.extend(float(features.get(name) or 0.0) for name in history_feature_names(history_windows))
    return values


def load_dataset(path: Path, history_windows: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        "SELECT event_slug,outcome,observed_at,label,features_json FROM training_examples ORDER BY observed_at"
    ).fetchall()
    summaries = build_summaries(rows)
    connection.close()
    x, y, groups = [], [], []
    context_cache: dict[str, dict[str, float]] = {}
    for slug, outcome, observed_at, label, raw_features in rows:
        features = json.loads(raw_features)
        if features.get("target_price") is None or features.get("reference_price") is None:
            continue
        if str(slug) not in context_cache:
            context_cache[str(slug)] = history_context(summaries, slug, observed_at, history_windows)
        features.update(context_cache[str(slug)])
        x.append(vector(slug, outcome, observed_at, features, history_windows))
        y.append(int(label))
        groups.append(slug)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int8), np.asarray(groups)


def train(
    path: Path, allow_small_sample: bool = False, artifact_path: Path | None = None,
    history_windows: tuple[int, ...] = (),
) -> dict[str, float | int | bool]:
    x, y, groups = load_dataset(path, history_windows)
    events = list(dict.fromkeys(groups.tolist()))
    if len(events) < settings.TRAINING_MIN_INDEPENDENT_EVENTS and not allow_small_sample:
        raise RuntimeError(
            f"TRAINING_BLOCKED: only {len(events)} independent events; need {settings.TRAINING_MIN_INDEPENDENT_EVENTS}"
        )
    train_end = max(1, int(len(events) * (1 - settings.TRAINING_TEST_SIZE - settings.TRAINING_CALIBRATION_SIZE)))
    calibration_end = max(train_end + 1, int(len(events) * (1 - settings.TRAINING_TEST_SIZE)))
    train_events = set(events[:train_end])
    calibration_events = set(events[train_end:calibration_end])
    test_events = set(events[calibration_end:])
    train_mask = np.array([group in train_events for group in groups])
    calibration_mask = np.array([group in calibration_events for group in groups])
    test_mask = np.array([group in test_events for group in groups])
    if not test_mask.any() or not calibration_mask.any() or len(set(y[train_mask])) < 2:
        raise RuntimeError("TRAINING_BLOCKED: event-level split is not usable")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
        random_state=settings.TRAINING_RANDOM_STATE,
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
    artifact_path = artifact_path or settings.TRAINING_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "calibrator": calibrator,
        "features": [*FEATURE_NAMES, *history_feature_names(history_windows)], "metrics": metrics,
        "history_windows": list(history_windows),
        "version": "v5_target_time_neighbor_history_event_platt_calibrated",
        "splits": {
            "train_events": sorted(train_events), "calibration_events": sorted(calibration_events),
            "test_events": sorted(test_events),
        },
    }, artifact_path)
    version = save_version_bundle("custom", "entry", [artifact_path], metrics)
    metrics["artifact_version"] = str(version["version"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--allow-small-sample", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.db, args.allow_small_sample, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
