"""Обучение последовательной exit-модели на всей траектории открытой позиции."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import app_config as settings
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, precision_score, roc_auc_score

from polybot.models.artifact_versions import save_version_bundle


FEATURES = [
    "seconds_in_position", "remaining_seconds", "current_bid", "average_price", "marked_return",
    "oriented_distance_to_target_pct", "momentum_bid_3ticks", "momentum_distance_3ticks",
    "peak_bid_since_entry", "trough_bid_since_entry", "drawdown_from_peak", "recovery_from_trough",
    "spread", "log_bid_size", "log_ask_size", "shares", "original_cost_usdc",
]


def train() -> dict:
    rows = pq.read_table(settings.EXIT_SEQUENCE_DATASET_PATH).to_pylist()
    # Не даём событиям с большим числом тиков доминировать и оставляем примерно один кадр в 5 секунд.
    sampled = [row for row in rows if int(float(row["seconds_in_position"])) % 5 == 0]
    events = list(dict.fromkeys(str(row["event_slug"]) for row in sampled))
    train_end, validation_end = max(1, int(len(events) * .60)), max(2, int(len(events) * .80))
    split = {event: ("train" if i < train_end else "validation" if i < validation_end else "test")
             for i, event in enumerate(events)}
    x = np.asarray([[float(row.get(name) or 0) for name in FEATURES] for row in sampled], dtype=float)
    y = np.asarray([float(row["close_advantage_usdc"]) for row in sampled], dtype=float)
    slugs = np.asarray([str(row["event_slug"]) for row in sampled])
    outcomes = np.asarray([str(row["outcome"]) for row in sampled])
    names = np.asarray([split[slug] for slug in slugs])
    masks = {name: names == name for name in ("train", "validation", "test")}
    margin = float(settings.EXIT_VALUE_MARGIN_USDC)
    label = (y >= margin).astype(int)
    event_sizes = {event: max(1, int((slugs[masks["train"]] == event).sum())) for event in set(slugs[masks["train"]])}
    weights = np.asarray([1 / event_sizes[event] for event in slugs[masks["train"]]], dtype=float)
    classifier = HistGradientBoostingClassifier(
        max_iter=350, max_leaf_nodes=15, learning_rate=.035, l2_regularization=8,
        min_samples_leaf=30, random_state=43,
    ).fit(x[masks["train"]], label[masks["train"]], sample_weight=weights)
    regressor = HistGradientBoostingRegressor(
        max_iter=350, max_leaf_nodes=15, learning_rate=.04, l2_regularization=8,
        min_samples_leaf=30, random_state=43,
    ).fit(x[masks["train"]], y[masks["train"]], sample_weight=weights)

    validation_probability = classifier.predict_proba(x[masks["validation"]])[:, 1]
    validation_advantage = regressor.predict(x[masks["validation"]])
    validation_truth = y[masks["validation"]]
    policies = []
    for probability_threshold in np.arange(.55, .96, .05):
        for advantage_threshold in np.arange(margin, 1.51, .10):
            selected = (validation_probability >= probability_threshold) & (validation_advantage >= advantage_threshold)
            truth = validation_truth[selected]
            policies.append({
                "probability_threshold": float(probability_threshold),
                "advantage_threshold": float(advantage_threshold), "closes": int(selected.sum()),
                "precision": float((truth >= margin).mean()) if len(truth) else 0.0,
                "advantage_sum_usdc": float(truth.sum()) if len(truth) else 0.0,
            })
    eligible = [p for p in policies if p["closes"] >= 25 and p["precision"] >= .60 and p["advantage_sum_usdc"] > 0]
    shadow_policy = max(policies, key=lambda p: (p["advantage_sum_usdc"], p["precision"]))
    policy = max(eligible or policies, key=lambda p: (p["advantage_sum_usdc"] if p["closes"] >= 25 else -1e9, p["precision"]))
    if not eligible:
        policy = {**policy, "probability_threshold": 1.01, "advantage_threshold": float("inf")}

    def metrics(name: str) -> dict:
        mask = masks[name]
        probability = classifier.predict_proba(x[mask])[:, 1]
        advantage = regressor.predict(x[mask])
        selected = ((probability >= policy["probability_threshold"])
                    & (advantage >= policy["advantage_threshold"]))
        truth, binary = y[mask], label[mask]
        return {
            "rows": int(mask.sum()), "events": len(set(slugs[mask])), "closes": int(selected.sum()),
            "close_precision": float(precision_score(binary, selected, zero_division=0)),
            "selected_advantage_sum_usdc": float(truth[selected].sum()) if selected.any() else 0.0,
            "selected_advantage_mean_usdc": float(truth[selected].mean()) if selected.any() else 0.0,
            "roc_auc": float(roc_auc_score(binary, probability)) if len(set(binary)) == 2 else None,
            "brier": float(brier_score_loss(binary, probability)),
            "up_rows": int((outcomes[mask] == "Up").sum()), "down_rows": int((outcomes[mask] == "Down").sum()),
        }

    report = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "target": "close_now_net_pnl_minus_hold_to_resolution_net_pnl",
        "features": FEATURES, "rows": len(sampled), "events": len(events), "selected_policy": policy,
        "shadow_policy": shadow_policy,
        "train": metrics("train"), "validation": metrics("validation"), "test": metrics("test"),
    }
    test = report["test"]
    report["promotion_gate"] = {"passed": bool(test["events"] >= 30 and test["closes"] >= 20
        and test["close_precision"] >= .60 and test["selected_advantage_sum_usdc"] > 0),
        "candidate_only": True}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = settings.MODEL_DIR / "candidates" / f"exit_sequence_v14_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "exit_sequence_v14.joblib"
    report_path = directory / "exit_sequence_report.json"
    joblib.dump({"close_classifier": classifier, "advantage_model": regressor, "features": FEATURES,
                 "probability_threshold": policy["probability_threshold"],
                 "advantage_threshold": policy["advantage_threshold"],
                 "shadow_probability_threshold": shadow_policy["probability_threshold"],
                 "shadow_advantage_threshold": shadow_policy["advantage_threshold"], "report": report}, artifact)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    version = save_version_bundle("exit_sequence", "exit", [artifact, report_path], report)
    report.update({"artifact_version": version["version"], "artifact_path": str(artifact), "report_path": str(report_path)})
    return report


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, lambda: print(json.dumps(train(), ensure_ascii=False, indent=2)))
