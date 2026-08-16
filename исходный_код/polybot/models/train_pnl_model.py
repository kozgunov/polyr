"""Двухступенчатая value-модель: P(fill) × E[net PnL | fill]."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)

from polybot.models.counterfactual_actions import ACTION_FEATURE_NAMES, action_vector
from polybot.trading.fees import total_fee_usdc


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA query_only=ON")
    rows = connection.execute(
        """SELECT c.event_slug,c.outcome,c.observed_at,c.action,c.target_net_pnl_usdc,
                  c.candidate_price,c.candidate_notional_usdc,c.limit_level,c.filled,
                  c.resolution_label,t.features_json
           FROM counterfactual_action_examples c
           JOIN training_examples t ON t.snapshot_id=c.snapshot_id
           WHERE c.phase='entry' AND c.action LIKE 'BUY_%'
           ORDER BY c.event_slug,c.observed_at,c.action,c.candidate_price,c.candidate_notional_usdc"""
    ).fetchall()
    connection.close()
    x, pnl, filled, groups, metadata = [], [], [], [], []
    labels = []
    for slug, outcome, observed_at, action, target_pnl, price, notional, level, did_fill, label, raw in rows:
        features = json.loads(raw)
        if features.get("target_price") is None:
            continue
        x.append(action_vector(slug, outcome or "Up", observed_at, features, action, price, notional))
        pnl.append(float(target_pnl)); filled.append(int(did_fill)); labels.append(int(label)); groups.append(str(slug))
        metadata.append({
            "event_slug": str(slug), "outcome": str(outcome or ""), "action": str(action),
            "observed_at": str(observed_at), "entry_price": float(price or 0),
            "notional": float(notional), "limit_level": str(level),
            "filled": int(did_fill), "actual_pnl": float(target_pnl),
        })
    return (np.asarray(x, dtype=float), np.asarray(pnl, dtype=float), np.asarray(filled, dtype=int),
            np.asarray(labels, dtype=int), np.asarray(groups), metadata)


def _bootstrap_mean_lower(values: list[float], seed: int = 42) -> float | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(sample, size=(2000, len(sample)), replace=True), axis=1)
    return float(np.quantile(means, 0.025))


def policy_metrics(indices: np.ndarray, predictions: np.ndarray, metadata: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_event: dict[str, dict[str, list[tuple[float, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    for index, prediction in zip(indices.tolist(), predictions.tolist(), strict=True):
        row = metadata[index]
        by_event[row["event_slug"]][row["observed_at"]].append((float(prediction), row))
    pnls, selected_actions, nonfills = [], [], 0
    for snapshots in by_event.values():
        for observed_at, candidates in sorted(snapshots.items()):
            prediction, row = max(candidates, key=lambda item: item[0])
            start = int(row["event_slug"].rsplit("-", 1)[-1])
            elapsed = datetime.fromisoformat(observed_at).timestamp() - start
            if not settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN <= elapsed <= 300 - settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:
                continue
            if prediction >= threshold:
                pnls.append(float(row["actual_pnl"])); selected_actions.append(row["action"])
                nonfills += int(not row["filled"])
                break
    wins = sum(value > 0 for value in pnls)
    losses = abs(sum(value for value in pnls if value < 0))
    profit = sum(value for value in pnls if value > 0)
    return {
        "threshold": threshold, "trades": len(pnls), "net_pnl_usdc": float(sum(pnls)),
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "profit_factor": profit / losses if losses else None,
        "expectancy_usdc": sum(pnls) / len(pnls) if pnls else 0.0,
        "expectancy_ci95_lower_usdc": _bootstrap_mean_lower(pnls),
        "nonfills": nonfills, "fill_rate": 1.0 - nonfills / len(pnls) if pnls else 0.0,
        "up": selected_actions.count("BUY_UP"), "down": selected_actions.count("BUY_DOWN"),
    }


def _select_threshold(indices: np.ndarray, predictions: np.ndarray, metadata: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    grid = np.round(np.arange(0.00, 1.51, 0.02), 2)
    policies = [policy_metrics(indices, predictions, metadata, float(value)) for value in grid]
    eligible = [item for item in policies if item["trades"] >= 30 and min(item["up"], item["down"]) >= 5]
    selected = max(eligible or policies, key=lambda item: (item["net_pnl_usdc"], item["profit_factor"] or 0.0))
    return selected, policies


def train(path: Path, artifact_path: Path | None = None, report_path: Path | None = None) -> dict[str, Any]:
    x, pnl, filled, resolution, groups, metadata = load(path)
    events = list(dict.fromkeys(groups.tolist()))
    if len(events) < 100 or len(np.unique(filled)) < 2:
        raise RuntimeError(f"TWO_STAGE_TRAINING_BLOCKED: events={len(events)}, fill_classes={np.unique(filled).tolist()}")
    train_end = int(len(events) * 0.55)
    calibration_end = int(len(events) * 0.70)
    validation_end = int(len(events) * 0.85)
    train_events = set(events[:train_end])
    calibration_events = set(events[train_end:calibration_end])
    validation_events = set(events[calibration_end:validation_end])
    test_events = set(events[validation_end:])
    train_mask = np.asarray([group in train_events for group in groups])
    calibration_mask = np.asarray([group in calibration_events for group in groups])
    validation_mask = np.asarray([group in validation_events for group in groups])
    test_mask = np.asarray([group in test_events for group in groups])

    fill_model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, l2_regularization=3.0,
        min_samples_leaf=30, random_state=settings.TRAINING_RANDOM_STATE,
    ).fit(x[train_mask], filled[train_mask])
    raw_calibration = np.clip(fill_model.predict_proba(x[calibration_mask])[:, 1], 1e-6, 1 - 1e-6)
    fill_calibrator = LogisticRegression(C=0.25, random_state=settings.TRAINING_RANDOM_STATE)
    fill_calibrator.fit(np.log(raw_calibration / (1.0 - raw_calibration)).reshape(-1, 1), filled[calibration_mask])
    conditional_train = train_mask & (filled == 1)
    outcome_model = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=300, max_leaf_nodes=15, l2_regularization=3.0,
        min_samples_leaf=30, random_state=settings.TRAINING_RANDOM_STATE,
    ).fit(x[conditional_train], resolution[conditional_train])
    conditional_calibration = calibration_mask & (filled == 1)
    raw_outcome_calibration = np.clip(outcome_model.predict_proba(x[conditional_calibration])[:, 1], 1e-6, 1 - 1e-6)
    outcome_calibrator = LogisticRegression(C=0.25, random_state=settings.TRAINING_RANDOM_STATE)
    outcome_calibrator.fit(
        np.log(raw_outcome_calibration / (1.0 - raw_outcome_calibration)).reshape(-1, 1),
        resolution[conditional_calibration],
    )

    # Направления калибруются раздельно: одинаковый raw-score не обязан означать
    # одинаковую вероятность для Up и Down при временном дрейфе выборки.
    directional_outcome_calibrators: dict[str, LogisticRegression] = {}
    calibration_indices = np.flatnonzero(conditional_calibration)
    for outcome in ("Up", "Down"):
        local = np.asarray([metadata[index]["outcome"] == outcome for index in calibration_indices])
        local_labels = resolution[calibration_indices][local]
        if local.sum() >= 30 and len(np.unique(local_labels)) == 2:
            local_raw = raw_outcome_calibration[local]
            calibrator = LogisticRegression(C=0.25, random_state=settings.TRAINING_RANDOM_STATE)
            calibrator.fit(np.log(local_raw / (1.0 - local_raw)).reshape(-1, 1), local_labels)
            directional_outcome_calibrators[outcome] = calibrator

    severe_loss = pnl <= -np.maximum(0.01, np.asarray([row["notional"] for row in metadata])
                                    * settings.ACTION_TAIL_LOSS_FRACTION)
    tail_classifier = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=250, max_leaf_nodes=15, l2_regularization=5.0,
        min_samples_leaf=30, random_state=settings.TRAINING_RANDOM_STATE,
    ).fit(x[conditional_train], severe_loss[conditional_train].astype(int))
    tail_train = conditional_train & severe_loss
    tail_loss_model = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=250, max_leaf_nodes=15, l2_regularization=5.0,
        min_samples_leaf=20, random_state=settings.TRAINING_RANDOM_STATE,
    ).fit(x[tail_train], pnl[tail_train])

    # Честный baseline: прежняя единая регрессия обучается на том же train split.
    baseline_model = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=250, max_leaf_nodes=15, l2_regularization=2.0,
        random_state=settings.TRAINING_RANDOM_STATE,
    ).fit(x[train_mask], pnl[train_mask])

    def predict(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_fill = np.clip(fill_model.predict_proba(x[mask])[:, 1], 1e-6, 1 - 1e-6)
        p_fill = fill_calibrator.predict_proba(np.log(raw_fill / (1.0 - raw_fill)).reshape(-1, 1))[:, 1]
        raw_outcome = np.clip(outcome_model.predict_proba(x[mask])[:, 1], 1e-6, 1 - 1e-6)
        selected_indices = np.flatnonzero(mask)
        selected_rows = [metadata[index] for index in selected_indices]
        p_win = outcome_calibrator.predict_proba(np.log(raw_outcome / (1.0 - raw_outcome)).reshape(-1, 1))[:, 1]
        for outcome, calibrator in directional_outcome_calibrators.items():
            local = np.asarray([row["outcome"] == outcome for row in selected_rows])
            if local.any():
                p_win[local] = calibrator.predict_proba(
                    np.log(raw_outcome[local] / (1.0 - raw_outcome[local])).reshape(-1, 1)
                )[:, 1]
        pnl_if_filled = np.asarray([
            (row["notional"] / row["entry_price"]) * probability - row["notional"]
            - total_fee_usdc(row["notional"] / row["entry_price"], row["entry_price"], taker=row["limit_level"] == "ask")
            for row, probability in zip(selected_rows, p_win, strict=True)
        ])
        p_tail = tail_classifier.predict_proba(x[mask])[:, 1]
        tail_pnl = tail_loss_model.predict(x[mask])
        expected_tail_loss = p_fill * p_tail * np.maximum(0.0, -tail_pnl)
        risk_adjusted = p_fill * pnl_if_filled - settings.ACTION_TAIL_RISK_PENALTY * expected_tail_loss
        return risk_adjusted, p_fill, pnl_if_filled, p_tail, expected_tail_loss

    validation_prediction, _, _, validation_p_tail, validation_tail_loss = predict(validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    validation_base_ev = validation_prediction + settings.ACTION_TAIL_RISK_PENALTY * validation_tail_loss
    risk_policies = []
    for risk_penalty in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
        for max_tail_probability in (0.25, 0.35, 0.50, 0.75, 1.0):
            candidate_prediction = validation_base_ev - risk_penalty * validation_tail_loss
            candidate_prediction = np.where(validation_p_tail <= max_tail_probability,
                                            candidate_prediction, float("-inf"))
            candidate, _ = _select_threshold(validation_indices, candidate_prediction, metadata)
            risk_policies.append((candidate, risk_penalty, max_tail_probability))
    eligible_risk = [item for item in risk_policies if item[0]["trades"] >= 30
                     and min(item[0]["up"], item[0]["down"]) >= 5]
    selected, selected_risk_penalty, selected_max_tail_probability = max(
        eligible_risk or risk_policies,
        key=lambda item: (item[0]["net_pnl_usdc"], item[0]["profit_factor"] or 0.0),
    )
    test_prediction, test_p_fill, test_conditional, test_p_tail, test_tail_loss = predict(test_mask)
    test_indices = np.flatnonzero(test_mask)
    test_base_ev = test_prediction + settings.ACTION_TAIL_RISK_PENALTY * test_tail_loss
    test_risk_prediction = test_base_ev - selected_risk_penalty * test_tail_loss
    test_risk_prediction = np.where(test_p_tail <= selected_max_tail_probability,
                                    test_risk_prediction, float("-inf"))
    test_policy = policy_metrics(test_indices, test_risk_prediction, metadata, float(selected["threshold"]))

    baseline_validation = baseline_model.predict(x[validation_mask])
    baseline_selected, _ = _select_threshold(validation_indices, baseline_validation, metadata)
    baseline_test = policy_metrics(test_indices, baseline_model.predict(x[test_mask]), metadata, float(baseline_selected["threshold"]))
    filled_test = test_mask & (filled == 1)
    test_labels = filled[test_mask]
    raw_test_outcome = np.clip(outcome_model.predict_proba(x[filled_test])[:, 1], 1e-6, 1 - 1e-6)
    test_p_win = outcome_calibrator.predict_proba(
        np.log(raw_test_outcome / (1.0 - raw_test_outcome)).reshape(-1, 1)
    )[:, 1]
    report = {
        "version": "entry_value_v4_tail_risk",
        "created_at": datetime.now(UTC).isoformat(),
        "formula": "P(fill) * E(net PnL | filled) - learned tail-risk penalty",
        "rows": len(pnl), "independent_events": len(events),
        "train_events": len(train_events), "calibration_events": len(calibration_events),
        "validation_events": len(validation_events), "test_events": len(test_events),
        "train_fill_rate": float(filled[train_mask].mean()), "test_fill_rate": float(test_labels.mean()),
        "fill_model": {
            "roc_auc": float(roc_auc_score(test_labels, test_p_fill)),
            "brier": float(brier_score_loss(test_labels, test_p_fill)),
        },
        "conditional_pnl_model": {
            "test_filled_rows": int(filled_test.sum()),
            "mae": float(mean_absolute_error(pnl[filled_test], test_conditional[filled[test_mask] == 1])),
            "rmse": float(mean_squared_error(pnl[filled_test], test_conditional[filled[test_mask] == 1]) ** 0.5),
            "outcome_roc_auc": float(roc_auc_score(resolution[filled_test], test_p_win)),
            "outcome_brier": float(brier_score_loss(resolution[filled_test], test_p_win)),
            "target": "calibrated_P(contract_wins|filled), converted to net PnL by payout and fee formula",
        },
        "tail_risk_model": {
            "enabled": True,
            "loss_definition": f"net_pnl <= -{settings.ACTION_TAIL_LOSS_FRACTION:.2f} * notional",
            "test_tail_rate": float(severe_loss[test_mask].mean()),
            "predicted_tail_probability_mean": float(test_p_tail.mean()),
            "expected_tail_loss_mean_usdc": float(test_tail_loss.mean()),
            "directional_calibrators": sorted(directional_outcome_calibrators),
            "selected_risk_penalty": selected_risk_penalty,
            "selected_max_tail_probability": selected_max_tail_probability,
            "score": "P(fill)*E(PnL|fill) - tail_risk_penalty*expected_tail_loss",
        },
        "two_stage": {"selected_validation_policy": selected, "test_policy": test_policy},
        "unified_baseline_same_split": {"selected_validation_policy": baseline_selected, "test_policy": baseline_test},
    }
    report["promotion_gate"] = {
        "passed": bool(
            selected["net_pnl_usdc"] > 0 and test_policy["net_pnl_usdc"] > 0
            and (test_policy["profit_factor"] or 0.0) > 1.0
            and (test_policy["expectancy_ci95_lower_usdc"] or -1.0) > 0
            and min(test_policy["up"], test_policy["down"]) >= 10
            and test_policy["trades"] >= 50
            and test_policy["net_pnl_usdc"] > baseline_test["net_pnl_usdc"]
        ),
        "candidate_only": True,
        "requirements": [
            "positive validation/test net PnL", "test PF > 1", "95% expectancy lower CI > 0",
            ">=50 test trades", ">=10 Up and >=10 Down", "beats unified baseline on same split",
        ],
    }
    artifact_path = artifact_path or settings.MODEL_DIR / "candidates" / "two_stage_value_v3_2_structural_pnl" / "pnl_model.joblib"
    report_path = report_path or artifact_path.parent / "pnl_model_report.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True); report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "fill_model": fill_model, "fill_calibrator": fill_calibrator,
        "conditional_outcome_model": outcome_model, "outcome_calibrator": outcome_calibrator,
        "directional_outcome_calibrators": directional_outcome_calibrators,
        "tail_classifier": tail_classifier, "tail_loss_model": tail_loss_model,
        "tail_loss_fraction": settings.ACTION_TAIL_LOSS_FRACTION,
        "tail_risk_penalty": selected_risk_penalty,
        "max_tail_probability": selected_max_tail_probability,
        "conditional_target": "contract_win_probability_to_fee_adjusted_pnl",
        "features": ACTION_FEATURE_NAMES, "threshold": selected["threshold"], "report": report,
    }, artifact_path)
    report["artifact_path"] = str(artifact_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--artifact", type=Path); parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.db, args.artifact, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
