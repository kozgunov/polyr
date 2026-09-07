"""Симметричная entry-модель: одинаковая семантика признаков для Up и Down."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import app_config as settings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.artifact_versions import save_version_bundle
from polybot.models.event_history import build_summaries, context as history_context
from polybot.trading.fees import total_fee_usdc


BASE_FEATURES = (
    "candidate_bid", "candidate_ask", "candidate_midpoint", "candidate_spread",
    "candidate_log_bid_size", "candidate_log_ask_size",
    "opposite_bid", "opposite_ask", "opposite_midpoint", "opposite_spread",
    "opposite_log_bid_size", "opposite_log_ask_size", "ask_pair_overround",
    "oriented_distance_to_target_pct", "absolute_distance_to_target_pct",
    "oriented_bybit_to_target_pct", "oriented_okx_to_target_pct", "oriented_pyth_to_target_pct",
    "realized_volatility_60s_pct", "oriented_distance_time_score", "remaining_fraction",
    "oriented_distance_lag_15s_pct", "oriented_distance_lag_30s_pct", "oriented_distance_lag_60s_pct",
    "oriented_target_momentum_15s_pct", "oriented_target_momentum_30s_pct",
    "oriented_target_momentum_60s_pct", "source_dispersion_pct",
)

HISTORY_SUFFIXES = (
    "available_fraction", "candidate_win_rate", "flip_rate", "last_candidate_win",
    "oriented_signed_streak", "oriented_mean_final_distance_pct",
    "mean_abs_final_distance_pct", "mean_distance_range_pct", "mean_volatility_pct",
    "oriented_mean_reference_return_pct", "mean_target_crossings",
    "candidate_favorable_extreme_pct", "candidate_adverse_extreme_pct",
    "oriented_mean_distance_at_30s_pct", "oriented_mean_distance_at_60s_pct",
    "oriented_mean_distance_at_120s_pct", "oriented_mean_distance_at_240s_pct", "gap_seconds",
)


def feature_names(history_windows: tuple[int, ...]) -> list[str]:
    return [*BASE_FEATURES, *(f"history_{window}_{suffix}" for window in history_windows for suffix in HISTORY_SUFFIXES)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def canonical_features(
    event_slug: str,
    outcome: str,
    observed_at: str,
    candidate: dict[str, Any],
    opposite: dict[str, Any],
    history: dict[str, Any],
    history_windows: tuple[int, ...] = (3, 12),
) -> dict[str, float]:
    """Ориентирует все направленные величины «в пользу кандидата»."""
    sign = 1.0 if outcome == "Up" else -1.0
    start = int(str(event_slug).rsplit("-", 1)[-1])
    elapsed = min(300.0, max(0.0, datetime.fromisoformat(str(observed_at)).timestamp() - start))
    prices = [_number(candidate.get(key), float("nan")) for key in ("bybit_price", "okx_price", "pyth_price")]
    prices = [value for value in prices if math.isfinite(value) and value > 0]
    centre = float(np.median(prices)) if prices else 0.0
    dispersion = (max(prices) - min(prices)) / centre * 100.0 if len(prices) >= 2 and centre else 0.0
    candidate_ask = _number(candidate.get("best_ask"), 1.0)
    opposite_ask = _number(opposite.get("best_ask"), 1.0)
    result = {
        "candidate_bid": _number(candidate.get("best_bid")),
        "candidate_ask": candidate_ask,
        "candidate_midpoint": _number(candidate.get("midpoint"), 0.5),
        "candidate_spread": _number(candidate.get("spread"), 1.0),
        "candidate_log_bid_size": math.log1p(max(0.0, _number(candidate.get("best_bid_size")))),
        "candidate_log_ask_size": math.log1p(max(0.0, _number(candidate.get("best_ask_size")))),
        "opposite_bid": _number(opposite.get("best_bid")),
        "opposite_ask": opposite_ask,
        "opposite_midpoint": _number(opposite.get("midpoint"), 0.5),
        "opposite_spread": _number(opposite.get("spread"), 1.0),
        "opposite_log_bid_size": math.log1p(max(0.0, _number(opposite.get("best_bid_size")))),
        "opposite_log_ask_size": math.log1p(max(0.0, _number(opposite.get("best_ask_size")))),
        "ask_pair_overround": candidate_ask + opposite_ask - 1.0,
        "oriented_distance_to_target_pct": sign * _number(candidate.get("distance_to_target_pct")),
        "absolute_distance_to_target_pct": abs(_number(candidate.get("distance_to_target_pct"))),
        "oriented_bybit_to_target_pct": sign * _number(candidate.get("bybit_to_target_pct")),
        "oriented_okx_to_target_pct": sign * _number(candidate.get("okx_to_target_pct")),
        "oriented_pyth_to_target_pct": sign * _number(candidate.get("pyth_to_target_pct")),
        "realized_volatility_60s_pct": _number(candidate.get("realized_volatility_60s_pct")),
        "oriented_distance_time_score": sign * _number(candidate.get("distance_time_score")),
        "remaining_fraction": 1.0 - elapsed / 300.0,
        "oriented_distance_lag_15s_pct": sign * _number(candidate.get("distance_lag_15s_pct")),
        "oriented_distance_lag_30s_pct": sign * _number(candidate.get("distance_lag_30s_pct")),
        "oriented_distance_lag_60s_pct": sign * _number(candidate.get("distance_lag_60s_pct")),
        "oriented_target_momentum_15s_pct": sign * _number(candidate.get("target_momentum_15s_pct")),
        "oriented_target_momentum_30s_pct": sign * _number(candidate.get("target_momentum_30s_pct")),
        "oriented_target_momentum_60s_pct": sign * _number(candidate.get("target_momentum_60s_pct")),
        "source_dispersion_pct": dispersion,
    }
    for window in history_windows:
        source = f"history_{window}_"
        target = f"history_{window}_"
        up_rate = _number(history.get(source + "up_rate"), 0.5)
        last_up = _number(history.get(source + "last_up"), 0.5)
        max_positive = _number(history.get(source + "mean_max_positive_distance_pct"))
        max_negative = _number(history.get(source + "mean_max_negative_distance_pct"))
        values = {
            "available_fraction": _number(history.get(source + "available_fraction")),
            "candidate_win_rate": up_rate if outcome == "Up" else 1.0 - up_rate,
            "flip_rate": _number(history.get(source + "flip_rate")),
            "last_candidate_win": last_up if outcome == "Up" else 1.0 - last_up,
            "oriented_signed_streak": sign * _number(history.get(source + "signed_streak")),
            "oriented_mean_final_distance_pct": sign * _number(history.get(source + "mean_final_distance_pct")),
            "mean_abs_final_distance_pct": _number(history.get(source + "mean_abs_final_distance_pct")),
            "mean_distance_range_pct": _number(history.get(source + "mean_distance_range_pct")),
            "mean_volatility_pct": _number(history.get(source + "mean_volatility_pct")),
            "oriented_mean_reference_return_pct": sign * _number(history.get(source + "mean_reference_return_pct")),
            "mean_target_crossings": _number(history.get(source + "mean_target_crossings")),
            "candidate_favorable_extreme_pct": max_positive if outcome == "Up" else -max_negative,
            "candidate_adverse_extreme_pct": max_negative if outcome == "Up" else -max_positive,
            "gap_seconds": _number(history.get(source + "gap_seconds"), 86_400.0),
        }
        for second in (30, 60, 120, 240):
            values[f"oriented_mean_distance_at_{second}s_pct"] = sign * _number(
                history.get(source + f"mean_distance_at_{second}s_pct")
            )
        result.update({target + key: value for key, value in values.items()})
    return result


def vector(*args: Any, history_windows: tuple[int, ...] = (3, 12), **kwargs: Any) -> list[float]:
    values = canonical_features(*args, history_windows=history_windows, **kwargs)
    return [values[name] for name in feature_names(history_windows)]


def paired_dataset(
    database: Path, history_windows: tuple[int, ...] = (3, 12), sample_seconds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT event_slug,outcome,observed_at,label,features_json FROM training_examples ORDER BY observed_at"
    ).fetchall()
    connection.close()
    summaries = build_summaries(rows)
    buckets: dict[tuple[str, int], dict[str, tuple[Any, ...]]] = defaultdict(dict)
    for row in rows:
        slug, outcome, observed_at = str(row[0]), str(row[1]), str(row[2])
        features = json.loads(str(row[4]))
        if features.get("target_price") is None or features.get("reference_price") is None:
            continue
        bucket = int(datetime.fromisoformat(observed_at).timestamp()) // max(1, int(sample_seconds))
        buckets[(slug, bucket)][outcome] = (slug, outcome, observed_at, int(row[3]), features)
    context_cache: dict[str, dict[str, float]] = {}
    x: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    outcomes: list[str] = []
    paired: list[dict[str, Any]] = []
    for (slug, bucket), by_outcome in sorted(buckets.items(), key=lambda item: (int(item[0][0].rsplit("-", 1)[-1]), item[0][1])):
        if "Up" not in by_outcome or "Down" not in by_outcome:
            continue
        up, down = by_outcome["Up"], by_outcome["Down"]
        if not (0 < _number(up[4].get("best_ask")) < 1 and 0 < _number(down[4].get("best_ask")) < 1):
            continue
        if slug not in context_cache:
            context_cache[slug] = history_context(summaries, slug, up[2], history_windows)
        history = context_cache[slug]
        pair = {"event_slug": slug, "bucket": bucket, "observed_at": up[2], "label_up": up[3], "Up": up[4], "Down": down[4]}
        paired.append(pair)
        for outcome, row, opposite in (("Up", up, down), ("Down", down, up)):
            x.append(vector(slug, outcome, row[2], row[4], opposite[4], history, history_windows=history_windows))
            y.append(int(row[3]))
            groups.append(slug)
            outcomes.append(outcome)
    return np.asarray(x), np.asarray(y, dtype=np.int8), np.asarray(groups), np.asarray(outcomes), paired


def _calibrated_candidate_scores(artifact: dict[str, Any], rows: np.ndarray) -> np.ndarray:
    raw = np.clip(artifact["model"].predict_proba(rows)[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    return artifact["calibrator"].predict_proba(logits)[:, 1]


def _metrics(labels: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probability, sample_weight=weights)),
        "pr_auc": float(average_precision_score(labels, probability, sample_weight=weights)),
        "brier": float(np.average((probability - labels) ** 2, weights=weights)),
        "log_loss": float(log_loss(labels, probability, sample_weight=weights, labels=[0, 1])),
    }


def _policy(
    pairs: list[dict[str, Any]], probability_by_key: dict[tuple[str, int, str], float],
    minimum_edge: float | dict[str, float], minimum_confidence: float | dict[str, float],
    allowed_outcomes: tuple[str, ...] = ("Up", "Down"),
) -> dict[str, Any]:
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_event[pair["event_slug"]].append(pair)
    pnls: list[float] = []
    directions: list[str] = []
    for slug, event_rows in by_event.items():
        for pair in sorted(event_rows, key=lambda row: row["bucket"]):
            scores = {outcome: probability_by_key[(slug, pair["bucket"], outcome)] for outcome in ("Up", "Down")}
            total = max(1e-9, scores["Up"] + scores["Down"])
            probabilities = {outcome: scores[outcome] / total for outcome in ("Up", "Down")}
            candidates = []
            for outcome in allowed_outcomes:
                price = _number(pair[outcome].get("best_ask"))
                if not settings.PAPER_MIN_ENTRY_PRICE <= price <= settings.PAPER_MAX_ENTRY_PRICE:
                    continue
                fee_per_share = total_fee_usdc(1.0, price, taker=True)
                edge = probabilities[outcome] - price - fee_per_share
                edge_gate = float(minimum_edge[outcome]) if isinstance(minimum_edge, dict) else float(minimum_edge)
                confidence_gate = (
                    float(minimum_confidence[outcome])
                    if isinstance(minimum_confidence, dict) else float(minimum_confidence)
                )
                if edge >= edge_gate and probabilities[outcome] >= confidence_gate:
                    # В тандеме сравниваем запас над порогом, а не
                    # только сырой edge: у Up/Down могут быть разные execution gates.
                    candidates.append((edge - edge_gate, probabilities[outcome] - confidence_gate,
                                       edge, probabilities[outcome], outcome, price))
            if not candidates:
                continue
            _, _, edge, confidence, outcome, price = max(candidates)
            shares = max(5.0, settings.LIVE_MIN_POSITION_USDC / price)
            won = int(pair["label_up"] == int(outcome == "Up"))
            pnl = shares * won - shares * price - total_fee_usdc(shares, price, taker=True)
            pnls.append(float(pnl))
            directions.append(outcome)
            break
    profit = sum(max(0.0, value) for value in pnls)
    loss = abs(sum(min(0.0, value) for value in pnls))
    equity = np.cumsum(pnls) if pnls else np.asarray([])
    peaks = np.maximum.accumulate(np.r_[0.0, equity]) if pnls else np.asarray([])
    drawdown = peaks[1:] - equity if pnls else np.asarray([])
    return {
        "minimum_edge": minimum_edge, "minimum_confidence": minimum_confidence,
        "trades": len(pnls), "up": directions.count("Up"), "down": directions.count("Down"),
        "up_share": directions.count("Up") / len(directions) if directions else 0.0,
        "net_pnl_usdc": float(sum(pnls)), "expectancy_usdc": float(np.mean(pnls)) if pnls else 0.0,
        "win_rate": float(np.mean(np.asarray(pnls) > 0)) if pnls else 0.0,
        "profit_factor": profit / loss if loss else None,
        "max_drawdown_usdc": float(max(drawdown)) if len(drawdown) else 0.0,
        "up_pnl_usdc": float(sum(value for value, side in zip(pnls, directions, strict=True) if side == "Up")),
        "down_pnl_usdc": float(sum(value for value, side in zip(pnls, directions, strict=True) if side == "Down")),
    }


def train(
    database: Path = settings.DATABASE_PATH,
    artifact_path: Path = settings.BIDIRECTIONAL_ENTRY_CANDIDATE_PATH,
    report_path: Path = settings.BIDIRECTIONAL_ENTRY_REPORT_PATH,
    history_windows: tuple[int, ...] = (3, 12),
) -> dict[str, Any]:
    x, y, groups, outcomes, pairs = paired_dataset(database, history_windows)
    events = sorted(set(groups.tolist()), key=lambda slug: int(slug.rsplit("-", 1)[-1]))
    train_end, calibration_end = int(len(events) * .60), int(len(events) * .80)
    event_split = {slug: ("train" if i < train_end else "calibration" if i < calibration_end else "test") for i, slug in enumerate(events)}
    masks = {name: np.asarray([event_split[group] == name for group in groups]) for name in ("train", "calibration", "test")}
    event_counts = {slug: int(np.sum(groups == slug)) for slug in events}
    weights = np.asarray([1.0 / event_counts[group] for group in groups])
    model = HistGradientBoostingClassifier(
        learning_rate=.04, max_iter=260, max_leaf_nodes=31, min_samples_leaf=60,
        l2_regularization=2.0, random_state=settings.TRAINING_RANDOM_STATE,
    )
    model.fit(x[masks["train"]], y[masks["train"]], sample_weight=weights[masks["train"]])
    raw_cal = np.clip(model.predict_proba(x[masks["calibration"]])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=.20, random_state=settings.TRAINING_RANDOM_STATE)
    calibrator.fit(
        np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), y[masks["calibration"]],
        sample_weight=weights[masks["calibration"]],
    )
    artifact = {
        "model": model, "calibrator": calibrator, "features": feature_names(history_windows),
        "history_windows": list(history_windows), "direction_semantics": "candidate_side_probability_v1",
        "version": "bidirectional_entry_v1",
    }
    probabilities = _calibrated_candidate_scores(artifact, x)
    score_map: dict[tuple[str, int, str], float] = {}
    pair_keys = [(pair["event_slug"], pair["bucket"], outcome) for pair in pairs for outcome in ("Up", "Down")]
    for key, value in zip(pair_keys, probabilities, strict=True):
        score_map[key] = float(value)
    validation_pairs = [pair for pair in pairs if event_split[pair["event_slug"]] == "calibration"]
    test_pairs = [pair for pair in pairs if event_split[pair["event_slug"]] == "test"]
    grids = [(float(edge), float(confidence))
             for edge in np.arange(.00, .121, .01) for confidence in np.arange(.50, .76, .05)]
    direction_selection: dict[str, dict[str, Any]] = {}
    for outcome in ("Up", "Down"):
        side_policies = [
            _policy(validation_pairs, score_map, edge, confidence, (outcome,))
            for edge, confidence in grids
        ]
        eligible = [row for row in side_policies if row["trades"] >= 20 and row["net_pnl_usdc"] > 0]
        # Сначала максимизируем expectancy, затем общий PnL и минимизируем DD.
        direction_selection[outcome] = max(
            eligible or side_policies,
            key=lambda row: (row["expectancy_usdc"], row["net_pnl_usdc"], -row["max_drawdown_usdc"]),
        )
    edge_gates = {outcome: float(direction_selection[outcome]["minimum_edge"]) for outcome in ("Up", "Down")}
    confidence_gates = {
        outcome: float(direction_selection[outcome]["minimum_confidence"]) for outcome in ("Up", "Down")
    }
    selected = _policy(validation_pairs, score_map, edge_gates, confidence_gates)
    test_policy = _policy(test_pairs, score_map, edge_gates, confidence_gates)
    split_metrics: dict[str, Any] = {}
    for name, mask in masks.items():
        split_metrics[name] = {"all": _metrics(y[mask], probabilities[mask], weights[mask])}
        for outcome in ("Up", "Down"):
            side = mask & (outcomes == outcome)
            split_metrics[name][outcome.lower()] = _metrics(y[side], probabilities[side], weights[side])
    test_pair_raw_sums = []
    for pair in test_pairs:
        test_pair_raw_sums.append(score_map[(pair["event_slug"], pair["bucket"], "Up")] + score_map[(pair["event_slug"], pair["bucket"], "Down")])
    gate = {
        "passed": bool(
            selected["trades"] >= 50 and min(selected["up"], selected["down"]) >= 15
            and selected["up_pnl_usdc"] > 0 and selected["down_pnl_usdc"] > 0
            and test_policy["trades"] >= 50 and min(test_policy["up"], test_policy["down"]) >= 15
            and test_policy["up_pnl_usdc"] > 0 and test_policy["down_pnl_usdc"] > 0
            and test_policy["net_pnl_usdc"] > 0 and test_policy["max_drawdown_usdc"] <= 20
        ),
        "requirements": "validation and test trades>=50; Up/Down>=15; both side PnL>0; test net PnL>0; max DD<=20",
    }
    report = {
        "version": "bidirectional_entry_v1", "created_at": datetime.now().astimezone().isoformat(),
        "rows": int(len(y)), "paired_points": len(pairs), "independent_events": len(events),
        "events": {name: sum(value == name for value in event_split.values()) for name in masks},
        "features": feature_names(history_windows), "metrics": split_metrics,
        "mean_raw_pair_probability_sum_test": float(np.mean(test_pair_raw_sums)),
        "direction_validation_selection": direction_selection,
        "validation_policy": selected, "test_policy": test_policy, "promotion_gate": gate,
        "deployment_state": "shadow_candidate" if gate["passed"] else "candidate_not_activated",
    }
    artifact["report"] = report
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    version = save_version_bundle("custom_bidir", "entry", [artifact_path, report_path], report)
    report["artifact_version"] = version["version"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
