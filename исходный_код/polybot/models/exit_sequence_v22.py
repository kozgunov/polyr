"""Честный датасет и обучение exit-модели v22 без активации в торговле.

Модель оценивает полный выход позиции сейчас против реалистичного ожидания
15/30/60 секунд и HOLD до исхода. Будущие значения используются только как
supervised-разметка; признаки строятся исключительно из прошлого и настоящего.
"""

from __future__ import annotations

import json
import math
import sqlite3
from bisect import bisect_left, bisect_right
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from statistics import median

import app_config as settings
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from polybot.trading.fees import total_fee_usdc


WAIT_HORIZONS = (15, 30, 60)
FUTURE_QUOTE_TOLERANCE_SECONDS = 5.0
SAMPLE_SECONDS = 5
MIN_ABSOLUTE_ADVANTAGE_USDC = 0.05
MIN_RELATIVE_ADVANTAGE = 0.02
MINIMUM_TRAINING_EVENTS = 100

FEATURES = [
    "held_is_up", "seconds_in_position", "remaining_seconds", "remaining_fraction",
    "current_bid", "average_price", "marked_return", "oriented_distance_to_target_pct",
    "distance_momentum_15s", "distance_momentum_30s", "distance_momentum_60s",
    "bid_momentum_15s", "bid_momentum_30s", "bid_momentum_60s",
    "bid_slope_15s", "bid_slope_30s", "peak_bid_since_entry", "trough_bid_since_entry",
    "drawdown_from_peak", "recovery_from_trough", "seconds_since_peak",
    "maximum_favorable_excursion", "maximum_adverse_excursion", "spread",
    "log_bid_size", "log_ask_size", "depth_coverage", "shares", "original_cost_usdc",
]


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _executable_close_pnl(shares: float, cost: float, bid: float, bid_size: float) -> float | None:
    """Консервативно разрешает полный выход только при доступной глубине best bid."""
    if not 0 < bid < 1 or bid_size + 1e-9 < shares:
        return None
    return shares * bid - total_fee_usdc(shares, bid) - cost


def _book_close_pnl(shares: float, cost: float, snapshot: dict[str, Any]) -> tuple[float | None, float | None]:
    """Считает полный FAK-выход по сохранённым уровням стакана и price cap."""
    best_bid = float(snapshot["best_bid"] if snapshot.get("best_bid") is not None else snapshot.get("midpoint") or 0)
    try:
        raw = json.loads(str(snapshot.get("raw_json") or "{}"))
        levels = raw.get("bids") or []
    except (TypeError, ValueError, json.JSONDecodeError):
        levels = []
    parsed: list[tuple[float, float]] = []
    for level in levels:
        try:
            price = float(level.get("price") if isinstance(level, dict) else level[0])
            size = float(level.get("size") if isinstance(level, dict) else level[1])
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        if 0 < price < 1 and size > 0:
            parsed.append((price, size))
    if not parsed:
        fallback = _executable_close_pnl(shares, cost, best_bid, float(snapshot.get("best_bid_size") or 0))
        return fallback, best_bid if fallback is not None else None
    cap = max(0.01, best_bid * (1 - float(settings.FAK_PRICE_CAP_SLIPPAGE_BPS) / 10_000))
    remaining = shares; proceeds = 0.0
    for price, size in sorted(parsed, reverse=True):
        if price + 1e-12 < cap:
            continue
        filled = min(remaining, size)
        proceeds += filled * price; remaining -= filled
        if remaining <= 1e-9:
            average = proceeds / shares
            return proceeds - total_fee_usdc(shares, average) - cost, average
    return None, None


def _resolved_labels(connection: sqlite3.Connection) -> dict[tuple[str, str], int]:
    return {(str(row[0]), str(row[1])): int(row[2]) for row in connection.execute(
        "SELECT event_slug,outcome,MAX(label) FROM training_examples GROUP BY event_slug,outcome"
    )}


def _external_reference_at(
    source_rows: dict[str, list[tuple[float, float]]], observed_ts: float,
    maximum_age_seconds: float,
) -> float | None:
    """Медиана последних причинно доступных BTC-цен; будущие тики исключены."""
    values: list[float] = []
    for rows in source_rows.values():
        timestamps = [item[0] for item in rows]
        index = bisect_right(timestamps, observed_ts) - 1
        if index < 0:
            continue
        timestamp, price = rows[index]
        if 0 <= observed_ts - timestamp <= maximum_age_seconds and price > 0:
            values.append(price)
    return float(median(values)) if values else None


def _positions(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source, table in (("paper", "paper_positions"), ("live", "live_positions")):
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        valid = "AND COALESCE(p.execution_valid,1)=1" if "execution_valid" in columns else ""
        query = f"""SELECT p.*,COALESCE(d.model_name,'unknown') model
                    FROM {table} p LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                    WHERE p.status IN ('closed','resolved','provisionally_resolved') {valid}
                    ORDER BY p.opened_at,p.id"""
        result.extend({**dict(row), "source": source} for row in connection.execute(query))
    return result


def _fills(connection: sqlite3.Connection, position: dict[str, Any]) -> tuple[float, float]:
    slug, outcome = str(position["event_slug"]), str(position["outcome"])
    if position["source"] == "paper":
        rows = connection.execute(
            """SELECT shares,filled_price,fee_usdc FROM paper_orders
               WHERE session_id=? AND event_slug=? AND action=?
                 AND status IN ('filled','partially_filled') AND COALESCE(execution_valid,1)=1
                 AND shares>0 AND filled_price>0""",
            (position["session_id"], slug, f"BUY_{outcome.upper()}"),
        ).fetchall()
        shares = sum(float(row[0]) for row in rows)
        cost = sum(float(row[0]) * float(row[1]) + float(row[2] or 0) for row in rows)
    else:
        rows = connection.execute(
            """SELECT matched_size,COALESCE(average_fill_price,requested_price),fee_usdc
               FROM live_orders WHERE event_slug=? AND outcome=? AND side='BUY'
                 AND matched_size>0 AND COALESCE(execution_valid,1)=1""",
            (slug, outcome),
        ).fetchall()
        shares = sum(float(row[0]) for row in rows)
        cost = sum(float(row[0]) * float(row[1]) + float(row[2] or 0) for row in rows)
    # Восстановленные старые LIVE-позиции могут не иметь полной локальной истории
    # ордеров, но их агрегаты уже сверены и помечены execution_valid=1.
    if shares <= 0 or cost <= 0:
        shares = float(position.get("shares") or 0)
        cost = float(position.get("cost_usdc") or 0)
    return shares, cost


def build_dataset(database: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")  # стабильный WAL snapshot без остановки collector-а
    labels = _resolved_labels(connection)
    records: list[dict[str, Any]] = []
    skipped = {"no_label": 0, "no_fill": 0, "no_market_path": 0, "no_executable_bid": 0}

    for position in _positions(connection):
        slug, outcome = str(position["event_slug"]), str(position["outcome"])
        label = labels.get((slug, outcome))
        if label is None:
            skipped["no_label"] += 1
            continue
        shares, cost = _fills(connection, position)
        if shares <= 0 or cost <= 0:
            skipped["no_fill"] += 1
            continue
        average_price = cost / shares
        opened_at = str(position["opened_at"])
        opened_ts = _timestamp(opened_at)
        event_end = int(slug.rsplit("-", 1)[-1]) + 300
        snapshots = [dict(row) for row in connection.execute(
            """SELECT collected_at,best_bid,midpoint,best_bid_size,best_ask_size,spread,raw_json
               FROM market_snapshots WHERE event_slug=? AND outcome=? AND collected_at>=?
               ORDER BY collected_at""", (slug, outcome, opened_at))]
        target_row = connection.execute(
            "SELECT target_price FROM event_targets WHERE event_slug=?", (slug,),
        ).fetchone()
        external_rows: dict[str, list[tuple[float, float]]] = {}
        for row in connection.execute(
            """SELECT source,collected_at,price FROM external_prices
               WHERE source IN ('bybit','okx','pyth') AND collected_at>=? AND collected_at<=?
               ORDER BY source,collected_at""",
            (opened_at, datetime.fromtimestamp(event_end, UTC).isoformat()),
        ):
            external_rows.setdefault(str(row[0]), []).append((_timestamp(str(row[1])), float(row[2])))
        if not snapshots or not target_row or not external_rows:
            skipped["no_market_path"] += 1
            continue
        snapshot_times = [_timestamp(str(row["collected_at"])) for row in snapshots]
        target_price = float(target_row[0])
        history_times: list[float] = []
        bids: list[float] = []
        distances: list[float] = []
        hold_pnl = shares * label - cost
        last_sample_bucket = -1

        for index, snapshot in enumerate(snapshots):
            observed_at = str(snapshot["collected_at"])
            observed_ts = snapshot_times[index]
            if observed_ts > event_end:
                break
            bucket = int(max(0, observed_ts - opened_ts) // SAMPLE_SECONDS)
            if bucket == last_sample_bucket:
                continue
            last_sample_bucket = bucket
            bid = float(snapshot["best_bid"] if snapshot["best_bid"] is not None else snapshot["midpoint"] or 0)
            bid_size = max(0.0, float(snapshot["best_bid_size"] or 0))
            close_pnl, close_fill_price = _book_close_pnl(shares, cost, snapshot)
            if close_pnl is None:
                skipped["no_executable_bid"] += 1
                continue
            reference_price = _external_reference_at(
                external_rows, observed_ts, float(settings.MAX_EXCHANGE_AGE_SECONDS),
            )
            if reference_price is None or target_price <= 0:
                continue
            distance = (reference_price / target_price - 1) * 100
            oriented_distance = distance if outcome == "Up" else -distance
            history_times.append(observed_ts); bids.append(bid); distances.append(oriented_distance)

            def lag(values: list[float], seconds: int) -> float:
                lag_index = max(0, bisect_right(history_times, observed_ts - seconds) - 1)
                return values[lag_index]

            delayed: dict[int, float | None] = {}
            for horizon in WAIT_HORIZONS:
                target_ts = observed_ts + horizon
                future_index = bisect_left(snapshot_times, target_ts, lo=index + 1)
                if future_index >= len(snapshots) or snapshot_times[future_index] > event_end:
                    delayed[horizon] = None
                    continue
                if snapshot_times[future_index] - target_ts > FUTURE_QUOTE_TOLERANCE_SECONDS:
                    delayed[horizon] = None
                    continue
                future = snapshots[future_index]
                delayed[horizon], _ = _book_close_pnl(shares, cost, future)
            wait_candidates = {"HOLD": hold_pnl, **{
                f"WAIT_{h}": value for h, value in delayed.items() if value is not None
            }}
            best_wait_action, best_wait_pnl = max(wait_candidates.items(), key=lambda item: item[1])
            advantage = close_pnl - best_wait_pnl
            relative_advantage = advantage / max(cost, 1.0)
            peak, trough = max(bids), min(bids)
            peak_index = max(range(len(bids)), key=bids.__getitem__)
            remaining = max(0.0, event_end - observed_ts)
            records.append({
                "source": position["source"], "position_id": int(position["id"]),
                "event_slug": slug, "outcome": outcome, "model": position["model"],
                "observed_at": observed_at, "held_is_up": int(outcome == "Up"),
                "seconds_in_position": max(0.0, observed_ts - opened_ts),
                "remaining_seconds": remaining, "remaining_fraction": remaining / 300,
                "current_bid": bid, "executable_close_price": close_fill_price,
                "average_price": average_price,
                "marked_return": bid / average_price - 1,
                "oriented_distance_to_target_pct": oriented_distance,
                "distance_momentum_15s": oriented_distance - lag(distances, 15),
                "distance_momentum_30s": oriented_distance - lag(distances, 30),
                "distance_momentum_60s": oriented_distance - lag(distances, 60),
                "bid_momentum_15s": bid - lag(bids, 15), "bid_momentum_30s": bid - lag(bids, 30),
                "bid_momentum_60s": bid - lag(bids, 60),
                "bid_slope_15s": (bid - lag(bids, 15)) / 15,
                "bid_slope_30s": (bid - lag(bids, 30)) / 30,
                "peak_bid_since_entry": peak, "trough_bid_since_entry": trough,
                "drawdown_from_peak": bid - peak, "recovery_from_trough": bid - trough,
                "seconds_since_peak": observed_ts - history_times[peak_index],
                "maximum_favorable_excursion": peak - average_price,
                "maximum_adverse_excursion": trough - average_price,
                "spread": float(snapshot["spread"] or 0),
                "log_bid_size": math.log1p(bid_size),
                "log_ask_size": math.log1p(max(0.0, float(snapshot["best_ask_size"] or 0))),
                "depth_coverage": min(5.0, bid_size / max(shares, 1e-9)),
                "shares": shares, "original_cost_usdc": cost,
                "resolved_label": label, "close_now_pnl_usdc": close_pnl,
                "hold_pnl_usdc": hold_pnl, "best_wait_action": best_wait_action,
                "best_wait_pnl_usdc": best_wait_pnl,
                "close_advantage_vs_best_wait_usdc": advantage,
                "close_advantage_vs_best_wait_fraction": relative_advantage,
                "close_is_materially_better": int(
                    advantage >= MIN_ABSOLUTE_ADVANTAGE_USDC and relative_advantage >= MIN_RELATIVE_ADVANTAGE
                ),
                **{f"pnl_wait_{h:03d}s_usdc": delayed[h] for h in WAIT_HORIZONS},
            })
    connection.rollback(); connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd", use_dictionary=True)
    temporary.replace(output)
    manifest = {
        "schema_version": 7, "created_at": datetime.now(UTC).isoformat(), "output": str(output),
        "rows": len(records), "events": len({row["event_slug"] for row in records}),
        "positions": len({(row["source"], row["position_id"]) for row in records}),
        "sources": {source: sum(row["source"] == source for row in records) for source in ("paper", "live")},
        "labels": {"close": sum(row["close_is_materially_better"] for row in records),
                   "hold": sum(not row["close_is_materially_better"] for row in records)},
        "skipped": skipped, "wait_horizons_seconds": WAIT_HORIZONS,
        "future_quote_tolerance_seconds": FUTURE_QUOTE_TOLERANCE_SECONDS,
        "execution_rule": "valid actual fills; full size available across bid levels; exact fees; no invalid positions",
        "reference_rule": "official Polymarket openPrice target + causal fresh median(Bybit,OKX,Pyth); closePrice only labels resolution",
        "leakage_rule": "features <= observed_at; future quotes and resolution appear only in labels",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _cluster_bootstrap_lower(values: list[float], seed: int = 43, samples: int = 2000) -> float | None:
    if len(values) < 20:
        return None
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    totals = [float(generator.choice(array, size=len(array), replace=True).sum()) for _ in range(samples)]
    return float(np.quantile(totals, 0.025))


def train(dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = pq.read_table(dataset_path).to_pylist()
    events = sorted({str(row["event_slug"]) for row in rows}, key=lambda slug: int(slug.rsplit("-", 1)[-1]))
    if len(events) < MINIMUM_TRAINING_EVENTS:
        report = {
            "schema_version": 7, "created_at": datetime.now(UTC).isoformat(),
            "training_status": "insufficient_execution_quality_data",
            "rows": len(rows), "events": len(events), "minimum_events": MINIMUM_TRAINING_EVENTS,
            "promotion_gate": {"passed": False, "candidate_only": True},
            "reason": "Недостаточно независимых событий с проверяемым полным выходом по стакану.",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "exit_sequence_v22_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report
    train_end, validation_end = max(1, int(len(events) * .60)), max(2, int(len(events) * .80))
    split = {event: ("train" if i < train_end else "validation" if i < validation_end else "test")
             for i, event in enumerate(events)}
    x = np.asarray([[float(row.get(feature) or 0) for feature in FEATURES] for row in rows], dtype=float)
    y_class = np.asarray([int(row["close_is_materially_better"]) for row in rows])
    y_value = np.asarray([float(row["close_advantage_vs_best_wait_fraction"]) for row in rows])
    names = np.asarray([split[str(row["event_slug"])] for row in rows])
    masks = {name: names == name for name in ("train", "validation", "test")}
    train_positions = [(str(row["source"]), int(row["position_id"])) for row in np.asarray(rows, dtype=object)[masks["train"]]]
    position_counts = {key: train_positions.count(key) for key in set(train_positions)}
    weights = np.asarray([1 / position_counts[key] for key in train_positions], dtype=float)
    classifier = HistGradientBoostingClassifier(
        max_iter=450, max_leaf_nodes=15, learning_rate=.025, l2_regularization=12,
        min_samples_leaf=35, random_state=43,
    ).fit(x[masks["train"]], y_class[masks["train"]], sample_weight=weights)
    regressor = HistGradientBoostingRegressor(
        max_iter=450, max_leaf_nodes=15, learning_rate=.025, l2_regularization=12,
        min_samples_leaf=35, loss="absolute_error", random_state=43,
    ).fit(x[masks["train"]], y_value[masks["train"]], sample_weight=weights)

    def evaluate(name: str, probability_threshold: float, value_threshold: float) -> dict[str, Any]:
        indices = np.flatnonzero(masks[name])
        probabilities = classifier.predict_proba(x[indices])[:, 1]
        values = regressor.predict(x[indices])
        grouped: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
        for local, global_index in enumerate(indices):
            row = rows[int(global_index)]
            grouped.setdefault((str(row["source"]), int(row["position_id"])), []).append(
                (int(global_index), float(probabilities[local]), float(values[local])))
        advantages: list[float] = []; outcomes: list[str] = []; policy_pnl = 0.0; hold_pnl = 0.0
        false_winner_exits = 0; rescued_losers = 0
        for candidates in grouped.values():
            candidates.sort(key=lambda item: str(rows[item[0]]["observed_at"]))
            baseline = float(rows[candidates[-1][0]]["hold_pnl_usdc"])
            hold_pnl += baseline
            selected = next((item for item in candidates if item[1] >= probability_threshold and item[2] >= value_threshold), None)
            if selected is None:
                policy_pnl += baseline
                continue
            row = rows[selected[0]]; realised = float(row["close_now_pnl_usdc"])
            advantage = realised - baseline
            policy_pnl += realised; advantages.append(advantage); outcomes.append(str(row["outcome"]))
            false_winner_exits += int(baseline > 0 and advantage < 0)
            rescued_losers += int(baseline < 0 and advantage > 0)
        selected_truth = y_class[indices]
        return {
            "rows": len(indices), "events": len({str(rows[i]["event_slug"]) for i in indices}),
            "positions": len(grouped), "closes": len(advantages),
            "roc_auc": float(roc_auc_score(selected_truth, probabilities)) if len(set(selected_truth)) == 2 else None,
            "pr_auc": float(average_precision_score(selected_truth, probabilities)) if len(set(selected_truth)) == 2 else None,
            "brier": float(brier_score_loss(selected_truth, probabilities)),
            "policy_pnl_usdc": policy_pnl, "hold_pnl_usdc": hold_pnl,
            "advantage_vs_hold_usdc": policy_pnl - hold_pnl,
            "advantage_ci95_lower_usdc": _cluster_bootstrap_lower(advantages),
            "close_precision": float(np.mean(np.asarray(advantages) >= MIN_ABSOLUTE_ADVANTAGE_USDC)) if advantages else 0.0,
            "average_selected_advantage_usdc": float(np.mean(advantages)) if advantages else 0.0,
            "up_closes": outcomes.count("Up"), "down_closes": outcomes.count("Down"),
            "false_winner_exits": false_winner_exits, "rescued_losers": rescued_losers,
        }

    candidates = []
    for probability_threshold in np.arange(.60, .96, .05):
        for value_threshold in np.arange(.00, .101, .01):
            metrics = evaluate("validation", float(probability_threshold), float(value_threshold))
            candidates.append({"probability_threshold": float(probability_threshold),
                               "value_threshold": float(value_threshold), **metrics})
    eligible = [row for row in candidates if row["closes"] >= 20 and row["up_closes"] >= 5
                and row["down_closes"] >= 5 and row["close_precision"] >= .65
                and row["advantage_vs_hold_usdc"] > 0]
    selected = max(eligible or candidates, key=lambda row: (
        row["advantage_ci95_lower_usdc"] if row["advantage_ci95_lower_usdc"] is not None else -1e9,
        row["advantage_vs_hold_usdc"], row["close_precision"],
    ))
    if not eligible:
        production_probability, production_value = 1.01, float("inf")
    else:
        production_probability = float(selected["probability_threshold"])
        production_value = float(selected["value_threshold"])
    report = {
        "schema_version": 7, "created_at": datetime.now(UTC).isoformat(),
        "target": "P(material CLOSE advantage) + E(relative CLOSE advantage)",
        "features": FEATURES, "rows": len(rows), "events": len(events),
        "selected_validation_policy": selected,
        "production_policy": {"probability_threshold": production_probability, "value_threshold": production_value},
        "train": evaluate("train", production_probability, production_value),
        "validation": evaluate("validation", production_probability, production_value),
        "test": evaluate("test", production_probability, production_value),
    }
    test = report["test"]
    report["promotion_gate"] = {
        "passed": bool(eligible and test["closes"] >= 20 and test["up_closes"] >= 5
                       and test["down_closes"] >= 5 and test["close_precision"] >= .60
                       and test["advantage_vs_hold_usdc"] > 0
                       and (test["advantage_ci95_lower_usdc"] or -1) >= 0),
        "candidate_only": True,
        "reason": "Никогда не активируется автоматически; требуется PAPER shadow на новых событиях.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "exit_sequence_v22.joblib"
    report_path = output_dir / "exit_sequence_v22_report.json"
    joblib.dump({"close_classifier": classifier, "advantage_model": regressor, "features": FEATURES,
                 "probability_threshold": production_probability, "value_threshold": production_value,
                 "report": report}, artifact)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report.update({"artifact_path": str(artifact), "report_path": str(report_path)})
    return report
