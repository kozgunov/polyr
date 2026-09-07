"""Обучение последовательной exit-модели на всей траектории открытой позиции."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import app_config as settings
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, roc_auc_score

from polybot.models.artifact_versions import save_version_bundle


FEATURES = [
    "seconds_in_position", "remaining_seconds", "current_bid", "average_price", "marked_return",
    "oriented_distance_to_target_pct", "momentum_bid_3ticks", "momentum_distance_3ticks",
    "momentum_bid_15s", "momentum_bid_30s", "momentum_bid_60s",
    "bid_slope_15s", "bid_slope_30s", "target_distance_available",
    "peak_bid_since_entry", "trough_bid_since_entry", "drawdown_from_peak", "recovery_from_trough",
    "seconds_since_peak", "maximum_favorable_excursion", "maximum_adverse_excursion",
    "spread", "log_bid_size", "log_ask_size", "shares", "original_cost_usdc",
]


def train(dataset_path: Path | None = None, output_dir: Path | None = None) -> dict:
    dataset_path = dataset_path or settings.EXIT_SEQUENCE_DATASET_PATH
    rows = pq.read_table(dataset_path).to_pylist()
    # Не даём событиям с большим числом тиков доминировать и оставляем примерно один кадр в 5 секунд.
    sampled = [row for row in rows if int(float(row["seconds_in_position"])) % 5 == 0]
    events = list(dict.fromkeys(str(row["event_slug"]) for row in sampled))
    train_end, validation_end = max(1, int(len(events) * .60)), max(2, int(len(events) * .80))
    split = {event: ("train" if i < train_end else "validation" if i < validation_end else "test")
             for i, event in enumerate(events)}
    x = np.asarray([[float(row.get(name) or 0) for name in FEATURES] for row in sampled], dtype=float)
    # Строгая контрфактуальная цель: закрыться сейчас стоит лишь тогда, когда это
    # лучше не только HOLD до resolution, но и доступного более позднего выхода.
    # Будущее используется только как supervised label, но не попадает в features.
    target_name = "close_advantage_vs_best_wait_usdc"
    y = np.asarray([float(row[target_name]) for row in sampled], dtype=float)
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

    def policy_metrics(name: str, probability_threshold: float, advantage_threshold: float) -> dict:
        mask_indices = np.flatnonzero(masks[name])
        probability = classifier.predict_proba(x[mask_indices])[:, 1]
        predicted_advantage = regressor.predict(x[mask_indices])
        grouped: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
        for local_index, global_index in enumerate(mask_indices.tolist()):
            row = sampled[global_index]
            key = (str(row.get("source", "paper")), int(row["position_id"]))
            grouped.setdefault(key, []).append((global_index, float(probability[local_index]), float(predicted_advantage[local_index])))
        policy_pnl: list[float] = []
        hold_pnl: list[float] = []
        realised_advantage: list[float] = []
        selected_outcomes: list[str] = []
        for candidates in grouped.values():
            candidates.sort(key=lambda item: str(sampled[item[0]]["observed_at"]))
            last = sampled[candidates[-1][0]]
            baseline = float(last["hold_pnl_usdc"])
            selected = next((item for item in candidates if item[1] >= probability_threshold and item[2] >= advantage_threshold), None)
            if selected is None:
                realised = baseline
            else:
                selected_row = sampled[selected[0]]
                realised = float(selected_row["close_now_pnl_usdc"])
                realised_advantage.append(realised - baseline)
                selected_outcomes.append(str(selected_row["outcome"]))
            policy_pnl.append(realised); hold_pnl.append(baseline)
        advantages = np.asarray(policy_pnl) - np.asarray(hold_pnl)
        return {
            "positions": len(grouped), "closes": len(realised_advantage),
            "close_precision": float(np.mean(np.asarray(realised_advantage) >= margin)) if realised_advantage else 0.0,
            "policy_pnl_usdc": float(np.sum(policy_pnl)), "hold_pnl_usdc": float(np.sum(hold_pnl)),
            "advantage_vs_hold_usdc": float(np.sum(advantages)),
            "selected_advantage_mean_usdc": float(np.mean(realised_advantage)) if realised_advantage else 0.0,
            "up_closes": selected_outcomes.count("Up"), "down_closes": selected_outcomes.count("Down"),
        }

    validation_probability = classifier.predict_proba(x[masks["validation"]])[:, 1]
    validation_advantage = regressor.predict(x[masks["validation"]])
    policies = []
    for probability_threshold in np.arange(.55, .96, .05):
        for advantage_threshold in np.arange(-.25, 1.51, .10):
            result = policy_metrics("validation", float(probability_threshold), float(advantage_threshold))
            policies.append({
                "probability_threshold": float(probability_threshold),
                "advantage_threshold": float(advantage_threshold), **result,
            })
    eligible = [p for p in policies if p["closes"] >= 15 and p["close_precision"] >= .70 and p["advantage_vs_hold_usdc"] > 0]
    shadow_policy = max(policies, key=lambda p: (p["advantage_vs_hold_usdc"], p["close_precision"]))
    policy = max(eligible or policies, key=lambda p: (p["advantage_vs_hold_usdc"] if p["closes"] >= 20 else -1e9, p["close_precision"]))
    if not eligible:
        policy = {**policy, "probability_threshold": 1.01, "advantage_threshold": float("inf")}

    def metrics(name: str) -> dict:
        mask = masks[name]
        probability = classifier.predict_proba(x[mask])[:, 1]
        advantage = regressor.predict(x[mask])
        truth, binary = y[mask], label[mask]
        result = policy_metrics(name, float(policy["probability_threshold"]), float(policy["advantage_threshold"]))
        return {**result,
            "rows": int(mask.sum()), "events": len(set(slugs[mask])),
            "roc_auc": float(roc_auc_score(binary, probability)) if len(set(binary)) == 2 else None,
            "pr_auc": float(average_precision_score(binary, probability)) if len(set(binary)) == 2 else None,
            "brier": float(brier_score_loss(binary, probability)),
            "up_rows": int((outcomes[mask] == "Up").sum()), "down_rows": int((outcomes[mask] == "Down").sum()),
        }

    report = {
        "schema_version": 6, "created_at": datetime.now(UTC).isoformat(),
        "target": target_name,
        "decision_semantics": "first causal CLOSE_NOW trigger versus best of later exit and HOLD, evaluated after fees",
        "features": FEATURES, "rows": len(sampled), "events": len(events), "selected_policy": policy,
        "shadow_policy": shadow_policy,
        "train": metrics("train"), "validation": metrics("validation"), "test": metrics("test"),
    }
    test = report["test"]
    report["promotion_gate"] = {"passed": bool(test["events"] >= 30 and test["closes"] >= 20
        and test["close_precision"] >= .60 and test["advantage_vs_hold_usdc"] > 0
        and test["up_closes"] >= 5 and test["down_closes"] >= 5),
        "candidate_only": True}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = output_dir or settings.MODEL_DIR / "candidates" / f"exit_sequence_v21_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "exit_sequence_v21.joblib"
    report_path = directory / "exit_sequence_report.json"
    joblib.dump({"close_classifier": classifier, "advantage_model": regressor, "features": FEATURES,
                 "probability_threshold": policy["probability_threshold"],
                 "advantage_threshold": policy["advantage_threshold"],
                 "shadow_probability_threshold": shadow_policy["probability_threshold"],
                 "shadow_advantage_threshold": shadow_policy["advantage_threshold"], "report": report}, artifact)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_dir is None:
        version = save_version_bundle("exit_sequence_v21", "exit", [artifact, report_path], report)
        report["artifact_version"] = version["version"]
    report.update({"artifact_path": str(artifact), "report_path": str(report_path)})
    return report


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, lambda: print(json.dumps(train(), ensure_ascii=False, indent=2)))
