"""Обучает кандидата: CLOSE сейчас против HOLD до официального resolution."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import app_config as settings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    brier_score_loss, f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)

from polybot.models.artifact_versions import save_version_bundle
from polybot.models.exit_features import FEATURES, feature_map, vector


def _load_pairs(path: Path) -> list[tuple]:
    """Загружает пары без тяжёлого self-JOIN по рабочей SQLite-базе."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA query_only=ON")
    holds = connection.execute(
        """SELECT decision_id,event_slug,observed_at,outcome,features_json,current_bid,
                  shares,cost_usdc,net_pnl_usdc
           FROM action_counterfactuals
           WHERE action='HOLD' AND horizon_seconds>=300 AND status='resolved'
             AND net_pnl_usdc IS NOT NULL ORDER BY observed_at,id"""
    ).fetchall()
    decision_ids = {int(row[0]) for row in holds if row[0] is not None}
    closes = {}
    for decision_id, pnl in connection.execute(
        """SELECT decision_id,net_pnl_usdc FROM action_counterfactuals
           WHERE action='CLOSE' AND status='evaluated' AND net_pnl_usdc IS NOT NULL"""
    ):
        if decision_id is not None and int(decision_id) in decision_ids:
            closes[int(decision_id)] = float(pnl)
    connection.close()
    return [(*row, closes[int(row[0])]) for row in holds if row[0] is not None and int(row[0]) in closes]


def _classification(y_true: np.ndarray, prediction: np.ndarray, margin: float) -> dict:
    actual = y_true >= margin
    selected = prediction >= margin
    selected_advantages = y_true[selected]
    return {
        "close_candidates": int(selected.sum()),
        "close_rate": float(selected.mean()),
        "close_precision": float(precision_score(actual, selected, zero_division=0)),
        "close_recall": float(recall_score(actual, selected, zero_division=0)),
        "close_f1": float(f1_score(actual, selected, zero_division=0)),
        "selected_realized_advantage_sum_usdc": float(selected_advantages.sum()) if selected.any() else 0.0,
        "selected_realized_advantage_mean_usdc": float(selected_advantages.mean()) if selected.any() else 0.0,
    }


def train(path: Path, artifact_path: Path | None = None, report_path: Path | None = None) -> dict:
    rows = _load_pairs(path)
    if len(rows) < settings.EXIT_MODEL_MIN_ROWS:
        raise RuntimeError(f"EXIT_TRAINING_BLOCKED: {len(rows)} resolution pairs; need {settings.EXIT_MODEL_MIN_ROWS}")

    x, y, groups, outcomes = [], [], [], []
    for _, slug, _, outcome, raw, bid, shares, cost, hold_pnl, close_pnl in rows:
        state = json.loads(raw)
        values = feature_map(state, str(outcome), bid, shares, cost)
        x.append(vector(values)); y.append(float(close_pnl) - float(hold_pnl))
        groups.append(str(slug)); outcomes.append(str(outcome))

    events = list(dict.fromkeys(groups))
    train_end = max(1, int(len(events) * 0.60))
    validation_end = max(train_end + 1, int(len(events) * 0.80))
    train_events = set(events[:train_end])
    validation_events = set(events[train_end:validation_end])
    test_events = set(events[validation_end:])
    split_names = np.asarray([
        "train" if group in train_events else "validation" if group in validation_events else "test"
        for group in groups
    ])
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    train_mask, validation_mask, test_mask = (split_names == "train"), (split_names == "validation"), (split_names == "test")

    margin = float(settings.EXIT_VALUE_MARGIN_USDC)
    close_label = (y_array >= margin).astype(int)
    train_groups = np.asarray(groups)[train_mask]
    train_outcomes = np.asarray(outcomes)[train_mask]
    class_counts = np.bincount(close_label[train_mask], minlength=2)
    event_counts = Counter(train_groups.tolist())
    outcome_counts = Counter(train_outcomes.tolist())
    sample_weight = np.asarray([
        (len(close_label[train_mask]) / max(1, 2 * class_counts[label]))
        * (len(train_groups) / max(1, len(outcome_counts) * outcome_counts[outcome]))
        / max(1, event_counts[group])
        for label, group, outcome in zip(
            close_label[train_mask], train_groups, train_outcomes, strict=True
        )
    ])
    close_classifier = HistGradientBoostingClassifier(
        max_iter=300, max_leaf_nodes=15, learning_rate=0.04,
        l2_regularization=5.0, min_samples_leaf=25, random_state=42,
    ).fit(x_array[train_mask], close_label[train_mask], sample_weight=sample_weight)
    advantage_model = HistGradientBoostingRegressor(
        max_iter=300, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=4.0, min_samples_leaf=25, random_state=42,
    ).fit(x_array[train_mask], y_array[train_mask], sample_weight=sample_weight)

    validation_probability = close_classifier.predict_proba(x_array[validation_mask])[:, 1]
    validation_advantage = advantage_model.predict(x_array[validation_mask])
    validation_truth = y_array[validation_mask]
    policies = []
    for threshold in np.arange(0.50, 0.96, 0.02):
        for advantage_threshold in np.arange(margin, 2.01, 0.10):
            selected = ((validation_probability >= threshold)
                        & (validation_advantage >= advantage_threshold))
            selected_truth = validation_truth[selected]
            policies.append({
                "threshold": float(threshold), "advantage_threshold": float(advantage_threshold),
                "closes": int(selected.sum()),
                "precision": float((selected_truth >= margin).mean()) if selected.any() else 0.0,
                "advantage_sum_usdc": float(selected_truth.sum()) if selected.any() else 0.0,
                "advantage_mean_usdc": float(selected_truth.mean()) if selected.any() else 0.0,
            })
    eligible = [item for item in policies if item["closes"] >= 20 and item["precision"] >= 0.55
                and item["advantage_sum_usdc"] > 0]
    selected_policy = max(
        eligible or policies,
        key=lambda item: (item["advantage_sum_usdc"] if item["closes"] >= 20 else float("-inf"), item["precision"]),
    )
    probability_threshold = float(selected_policy["threshold"] if eligible else 1.01)
    advantage_threshold = float(selected_policy.get("advantage_threshold", margin) if eligible else float("inf"))

    def metrics(mask: np.ndarray) -> dict:
        probability = close_classifier.predict_proba(x_array[mask])[:, 1]
        prediction = advantage_model.predict(x_array[mask])
        truth = y_array[mask]
        actual = truth >= margin
        selected = (probability >= probability_threshold) & (prediction >= advantage_threshold)
        result = {
            "rows": int(mask.sum()),
            "events": len(set(np.asarray(groups)[mask])),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
            "r2": float(r2_score(truth, prediction)),
            "close_roc_auc": float(roc_auc_score(actual, probability)) if len(set(actual)) == 2 else None,
            "close_brier": float(brier_score_loss(actual, probability)),
        }
        result.update(_classification(truth, np.where(selected, margin, -1.0), margin))
        return result

    test_outcomes = np.asarray(outcomes)[test_mask]
    test_metrics = metrics(test_mask)
    report = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "target": "close_now_net_pnl_minus_hold_to_resolution_net_pnl",
        "label_policy": "two_stage: P(CLOSE beats HOLD) and E(close advantage); horizon_300 resolved pairs",
        "rows": len(rows), "events": len(events), "features": FEATURES,
        "selected_validation_policy": selected_policy,
        "probability_threshold": probability_threshold,
        "advantage_threshold": advantage_threshold,
        "train": metrics(train_mask), "validation": metrics(validation_mask), "test": test_metrics,
        "test_outcome_rows": {"Up": int((test_outcomes == "Up").sum()), "Down": int((test_outcomes == "Down").sum())},
    }
    report["promotion_gate"] = {
        "passed": bool(
            test_metrics["events"] >= 30
            and test_metrics["close_candidates"] >= 20
            and test_metrics["close_precision"] >= 0.60
            and test_metrics["selected_realized_advantage_sum_usdc"] > 0
            and report["test_outcome_rows"]["Up"] >= 20
            and report["test_outcome_rows"]["Down"] >= 20
        ),
        "candidate_only": True,
        "requirements": "honest temporal test >=30 events; 20 exits; precision>=0.60; positive net advantage; >=20 rows each Up/Down",
    }

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate_dir = settings.MODEL_DIR / "candidates" / f"exit_value_v2_{stamp}"
    artifact_path = artifact_path or candidate_dir / "btc_5m_exit_value.joblib"
    report_path = report_path or candidate_dir / "exit_model_report.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "close_classifier": close_classifier, "advantage_model": advantage_model,
        "probability_threshold": probability_threshold, "advantage_threshold": advantage_threshold,
        "features": FEATURES, "report": report,
    }, artifact_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    version = save_version_bundle("exit_value", "exit", [artifact_path, report_path], report)
    report["artifact_version"] = str(version["version"])
    report["artifact_path"] = str(artifact_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.db, args.artifact, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
