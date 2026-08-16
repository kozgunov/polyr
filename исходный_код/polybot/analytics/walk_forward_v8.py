"""Walk-forward v8: обучение, калибровка и выбор действия без утечки будущего."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import app_config as settings
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from polybot.models.train_direction_model import vector
from polybot.trading.fees import total_fee_usdc


def load(path: Path) -> list[dict]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        "SELECT event_slug,outcome,observed_at,label,features_json "
        "FROM training_examples ORDER BY observed_at"
    ).fetchall()
    connection.close()
    result: list[dict] = []
    for slug, outcome, observed, label, raw in rows:
        features = json.loads(raw)
        if features.get("target_price") is None or features.get("reference_price") is None:
            continue
        result.append({
            "event": str(slug), "outcome": str(outcome), "observed": str(observed),
            "label": int(label), "features": features,
            "x": vector(str(slug), str(outcome), str(observed), features),
        })
    return result


def _event_weights(rows: list[dict]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["event"]] += 1
    return np.asarray([1.0 / counts[row["event"]] for row in rows])


def _fit_model(rows: list[dict]) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=260, depth=5, learning_rate=0.04, loss_function="Logloss",
        verbose=False, allow_writing_files=False, random_seed=settings.TRAINING_RANDOM_STATE,
    )
    model.fit(
        np.asarray([row["x"] for row in rows]), np.asarray([row["label"] for row in rows]),
        sample_weight=_event_weights(rows),
    )
    return model


def _calibrator(model: CatBoostClassifier, rows: list[dict]) -> LogisticRegression:
    logits = model.predict(np.asarray([row["x"] for row in rows]), prediction_type="RawFormulaVal")
    calibration = LogisticRegression(C=1.0, solver="lbfgs")
    calibration.fit(
        np.asarray(logits).reshape(-1, 1), np.asarray([row["label"] for row in rows]),
        sample_weight=_event_weights(rows),
    )
    return calibration


def _probabilities(model: CatBoostClassifier, calibration: LogisticRegression, rows: list[dict]) -> np.ndarray:
    logits = model.predict(np.asarray([row["x"] for row in rows]), prediction_type="RawFormulaVal")
    return calibration.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]


def _paired_snapshots(rows: list[dict], probabilities: np.ndarray):
    by_event: dict[str, dict[str, list[tuple[dict, float]]]] = defaultdict(lambda: defaultdict(list))
    for row, probability in zip(rows, probabilities, strict=True):
        by_event[row["event"]][row["outcome"].lower()].append((row, float(probability)))
    for event, outcomes in by_event.items():
        up = sorted(outcomes.get("up", []), key=lambda item: item[0]["observed"])
        down = sorted(outcomes.get("down", []), key=lambda item: item[0]["observed"])
        yield event, zip(up, down)


def _expected_pnl(probability: float, price: float, notional: float) -> float:
    shares = notional / price
    return shares * probability - notional - total_fee_usdc(shares, price)


def _bootstrap_lower(values: list[float]) -> float | None:
    if len(values) < 10:
        return None
    rng = np.random.default_rng(settings.TRAINING_RANDOM_STATE)
    samples = rng.choice(np.asarray(values), size=(2000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025))


def _policy(
    rows: list[dict], probabilities: np.ndarray, model_weight: float, min_expected: float,
    allow_low_probability: bool = False,
) -> dict:
    pnls: list[float] = []
    directions: dict[str, int] = defaultdict(int)
    for event, pairs in _paired_snapshots(rows, probabilities):
        event_start = int(event.rsplit("-", 1)[-1])
        for (up_row, up_raw), (down_row, down_raw) in pairs:
            elapsed = min(
                datetime.fromisoformat(up_row["observed"]).timestamp(),
                datetime.fromisoformat(down_row["observed"]).timestamp(),
            ) - event_start
            if not settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN <= elapsed <= 300 - settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:
                continue
            candidates = []
            symmetric_model_up = (up_raw + (1.0 - down_raw)) / 2.0
            action_probabilities: dict[str, float] = {}
            raw_mid_up = up_row["features"].get("midpoint") or up_row["features"].get("last_trade_price")
            raw_mid_down = down_row["features"].get("midpoint") or down_row["features"].get("last_trade_price")
            if raw_mid_up is None or raw_mid_down is None or float(raw_mid_up) + float(raw_mid_down) <= 0:
                continue
            market_up = float(raw_mid_up) / (float(raw_mid_up) + float(raw_mid_down))
            for direction, row, raw_probability in (
                ("Up", up_row, symmetric_model_up), ("Down", down_row, 1.0 - symmetric_model_up)
            ):
                price = row["features"].get("best_ask")
                if price is None or not 0.01 <= float(price) <= 0.99:
                    continue
                market_probability = market_up if direction == "Up" else 1.0 - market_up
                probability = model_weight * raw_probability + (1.0 - model_weight) * market_probability
                action_probabilities[direction] = probability
                notional = float(settings.PAPER_ENTRY_NOTIONAL_USDC)
                expected = _expected_pnl(probability, float(price), notional)
                candidates.append((expected, direction, row, probability, float(price), notional))
            if not candidates:
                continue
            expected, direction, row, probability, price, notional = max(candidates, key=lambda item: item[0])
            signal_confidence = max(action_probabilities.values())
            if (
                signal_confidence < settings.MIN_ENTRY_CONFIDENCE
                or (not allow_low_probability and probability < settings.MIN_ENTRY_CONFIDENCE)
                or expected < min_expected
            ):
                continue
            shares = notional / price
            pnls.append(float(shares * row["label"] - notional - total_fee_usdc(shares, price)))
            directions[direction] += 1
            break
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_loss = -sum(losses)
    return {
        "trades": len(pnls), "net_pnl": float(sum(pnls)),
        "expectancy": float(sum(pnls) / len(pnls)) if pnls else 0.0,
        "expectancy_ci95_lower": _bootstrap_lower(pnls),
        "profit_factor": float(sum(wins) / gross_loss) if gross_loss else (999.0 if wins else 0.0),
        "up": directions["Up"], "down": directions["Down"],
    }


def _select_policy(
    rows: list[dict], probabilities: np.ndarray, allow_low_probability: bool = False,
) -> tuple[float, float, dict]:
    candidates = []
    for model_weight in (0.25, 0.50, 0.75, 1.0):
        for min_expected in (0.08, 0.15, 0.25, 0.40, 0.60, 1.00):
            metrics = _policy(rows, probabilities, model_weight, min_expected, allow_low_probability)
            if metrics["trades"] >= 10:
                candidates.append((metrics["net_pnl"], metrics["profit_factor"], model_weight, min_expected, metrics))
    if not candidates:
        metrics = _policy(rows, probabilities, 0.5, 0.4, allow_low_probability)
        return 0.5, 0.4, metrics
    _, _, model_weight, min_expected, metrics = max(candidates, key=lambda item: (item[0], item[1]))
    return model_weight, min_expected, metrics


def run(path: Path, minimum_train_events: int = 200, fold_events: int = 50, calibration_events: int = 50) -> dict:
    rows = load(path)
    events = list(dict.fromkeys(row["event"] for row in rows))
    folds = []
    for start in range(minimum_train_events + calibration_events, len(events), fold_events):
        fit_events = set(events[:start - calibration_events])
        validation_events = set(events[start - calibration_events:start])
        test_list = events[start:start + fold_events]
        if not test_list:
            break
        train = [row for row in rows if row["event"] in fit_events]
        validation = [row for row in rows if row["event"] in validation_events]
        test = [row for row in rows if row["event"] in set(test_list)]
        model = _fit_model(train)
        calibration = _calibrator(model, validation)
        validation_probabilities = _probabilities(model, calibration, validation)
        test_probabilities = _probabilities(model, calibration, test)
        model_weight, min_expected, validation_metrics = _select_policy(validation, validation_probabilities)
        low_weight, low_min_expected, low_validation = _select_policy(
            validation, validation_probabilities, allow_low_probability=True,
        )
        labels = np.asarray([row["label"] for row in test])
        weights = _event_weights(test)
        metrics = _policy(test, test_probabilities, model_weight, min_expected)
        low_metrics = _policy(
            test, test_probabilities, low_weight, low_min_expected, allow_low_probability=True,
        )
        folds.append({
            "fit_events": len(fit_events), "calibration_events": len(validation_events),
            "test_events": len(test_list),
            "roc_auc": float(roc_auc_score(labels, test_probabilities, sample_weight=weights)),
            "brier": float(brier_score_loss(labels, test_probabilities, sample_weight=weights)),
            "selected_model_weight": model_weight, "selected_min_expected_pnl": min_expected,
            "validation": validation_metrics, **metrics,
            "low_probability_experiment": {
                "selected_model_weight": low_weight,
                "selected_min_expected_pnl": low_min_expected,
                "validation": low_validation,
                **low_metrics,
            },
        })
    positive_folds = sum(fold["net_pnl"] > 0 for fold in folds)
    ci_positive_folds = sum(
        fold["expectancy_ci95_lower"] is not None and fold["expectancy_ci95_lower"] > 0
        for fold in folds
    )
    report = {
        "protocol": "v8 expanding walk-forward; fit -> Platt calibration/policy selection -> untouched test",
        "events": len(events), "folds": folds,
        "total_test_events": sum(fold["test_events"] for fold in folds),
        "total_trades": sum(fold["trades"] for fold in folds),
        "total_net_pnl": sum(fold["net_pnl"] for fold in folds),
        "mean_roc_auc": sum(fold["roc_auc"] for fold in folds) / len(folds) if folds else None,
        "mean_brier": sum(fold["brier"] for fold in folds) / len(folds) if folds else None,
        "production_gate": {
            "passed": bool(
                sum(fold["test_events"] for fold in folds) >= 500
                and positive_folds / max(1, len(folds)) >= 0.80
                and ci_positive_folds == len(folds)
            ),
            "positive_folds": positive_folds,
            "total_folds": len(folds),
            "ci_positive_folds": ci_positive_folds,
            "minimum_oos_events": 500,
            "reason": "Production requires >=500 OOS events, >=80% positive folds and positive lower CI in every fold.",
        },
        "low_probability_experiment": {
            "status": "shadow_only",
            "reason": "Requires a separately validated early-exit model; hold-to-resolution is not equivalent.",
            "total_trades": sum(fold["low_probability_experiment"]["trades"] for fold in folds),
            "total_net_pnl": sum(fold["low_probability_experiment"]["net_pnl"] for fold in folds),
        },
    }
    settings.WALK_FORWARD_V8_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.WALK_FORWARD_V8_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--fold", type=int, default=50)
    parser.add_argument("--calibration", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run(args.db, args.min_train, args.fold, args.calibration), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
