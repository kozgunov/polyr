"""Честное сравнение entry champion/challenger на одном последнем holdout."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import app_config as settings
import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.event_history import build_summaries, context as history_context
from polybot.models.train_direction_model import vector
from polybot.trading.fees import total_fee_usdc


def _calibrate(raw: float, calibrator: Any | None) -> float:
    raw = max(1e-6, min(1 - 1e-6, float(raw)))
    if calibrator is None:
        return raw
    return float(calibrator.predict_proba([[math.log(raw / (1 - raw))]])[0, 1])


def _bootstrap_lower(values: list[float], seed: int = 42) -> float | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    sample = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(sample, size=(2000, len(sample)), replace=True), axis=1)
    return float(np.quantile(means, 0.025))


def _load_points(database: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT event_slug,outcome,observed_at,label,features_json FROM training_examples ORDER BY event_slug,observed_at"
    ).fetchall()
    connection.close()
    summaries = build_summaries(rows)
    grouped: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row[0])][str(row[1])].append(row)
    points = []
    for slug, outcomes in grouped.items():
        if not outcomes.get("Up") or not outcomes.get("Down"):
            continue
        start = int(slug.rsplit("-", 1)[-1])
        selected = {}
        for outcome in ("Up", "Down"):
            selected[outcome] = min(
                outcomes[outcome],
                key=lambda row: abs((__import__("datetime").datetime.fromisoformat(str(row[2])).timestamp() - start) - 180),
            )
        up, down = selected["Up"], selected["Down"]
        up_features, down_features = json.loads(up[4]), json.loads(down[4])
        if not (0 < float(up_features.get("best_ask") or 0) < 1 and 0 < float(down_features.get("best_ask") or 0) < 1):
            continue
        points.append({
            "event_slug": slug, "observed_at": str(up[2]), "label": int(up[3]),
            "Up": up_features, "Down": down_features,
        })
    return sorted(points, key=lambda item: item["event_slug"]), summaries


def _predictor(kind: str, model_path: Path, metadata_path: Path | None = None):
    if kind == "custom":
        artifact = joblib.load(model_path)
        model, calibrator = artifact["model"], artifact.get("calibrator")
        windows = tuple(int(value) for value in artifact.get("history_windows", ()))
    else:
        model = CatBoostClassifier(); model.load_model(str(model_path))
        artifact = joblib.load(metadata_path) if metadata_path and metadata_path.exists() else {}
        calibrator = artifact.get("calibrator")
        windows = tuple(int(value) for value in artifact.get("history_windows", ()))

    def predict(point: dict[str, Any], summaries: list[Any]) -> float:
        history = history_context(summaries, point["event_slug"], point["observed_at"], windows)
        probabilities = {}
        for outcome in ("Up", "Down"):
            features = {**point[outcome], **history}
            row = vector(point["event_slug"], outcome, point["observed_at"], features, windows)
            probabilities[outcome] = _calibrate(float(model.predict_proba([row])[0, 1]), calibrator)
        return max(0.001, min(0.999, (probabilities["Up"] + 1 - probabilities["Down"]) / 2))
    return predict


def _policy(rows: list[dict[str, Any]], probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    pnls, directions = [], []
    for row, p_up in zip(rows, probabilities, strict=True):
        confidence = max(float(p_up), 1 - float(p_up))
        if confidence < threshold:
            continue
        direction = "Up" if p_up >= 0.5 else "Down"
        price = float(row[direction]["best_ask"])
        notional = 3.0
        shares = notional / price
        won = row["label"] if direction == "Up" else 1 - row["label"]
        pnls.append(shares * won - notional - total_fee_usdc(shares, price, taker=True))
        directions.append(direction)
    profit, loss = sum(max(0, x) for x in pnls), abs(sum(min(0, x) for x in pnls))
    return {
        "threshold": threshold, "trades": len(pnls), "net_pnl_usdc": float(sum(pnls)),
        "expectancy_usdc": float(np.mean(pnls)) if pnls else 0.0,
        "expectancy_ci95_lower_usdc": _bootstrap_lower(pnls),
        "profit_factor": profit / loss if loss else None,
        "up": directions.count("Up"), "down": directions.count("Down"),
    }


def evaluate(database: Path, candidates: dict[str, tuple[str, Path, Path | None]], output: Path) -> dict[str, Any]:
    points, summaries = _load_points(database)
    validation_start, test_start = int(len(points) * 0.60), int(len(points) * 0.80)
    validation, test = points[validation_start:test_start], points[test_start:]
    report: dict[str, Any] = {"events": len(points), "validation_events": len(validation), "test_events": len(test), "models": {}}
    for name, (kind, model_path, metadata_path) in candidates.items():
        predictor = _predictor(kind, model_path, metadata_path)
        validation_probability = np.asarray([predictor(row, summaries) for row in validation])
        test_probability = np.asarray([predictor(row, summaries) for row in test])
        policies = [_policy(validation, validation_probability, float(value)) for value in np.arange(.50, .91, .02)]
        eligible = [row for row in policies if row["trades"] >= 50 and min(row["up"], row["down"]) >= 10]
        selected = max(eligible or policies, key=lambda row: (row["net_pnl_usdc"], row["profit_factor"] or 0))
        test_policy = _policy(test, test_probability, float(selected["threshold"]))
        labels = np.asarray([row["label"] for row in test])
        report["models"][name] = {
            "roc_auc": float(roc_auc_score(labels, test_probability)),
            "pr_auc": float(average_precision_score(labels, test_probability)),
            "brier": float(brier_score_loss(labels, test_probability)),
            "log_loss": float(log_loss(labels, test_probability, labels=[0, 1])),
            "validation_policy": selected, "test_policy": test_policy,
        }
    ranked = sorted(report["models"], key=lambda name: (
        report["models"][name]["test_policy"]["net_pnl_usdc"],
        report["models"][name]["roc_auc"],
    ), reverse=True)
    report["ranking"] = ranked
    report["winner"] = ranked[0] if ranked else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
