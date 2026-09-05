"""Event-level исследование цен входа и каузальных сигналов раннего выхода."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import app_config as settings
import pyarrow as pa
import pyarrow.parquet as pq


def _price_bucket(value: float) -> str:
    low = max(0.0, math.floor(value * 10) / 10)
    return f"{low:.1f}–{min(1.0, low + .1):.1f}"


def _range_bucket(value: float, edges: tuple[float, ...], suffix: str = "") -> str:
    absolute = abs(float(value))
    previous = 0.0
    for edge in edges:
        if absolute < edge:
            return f"{previous:g}–{edge:g}{suffix}"
        previous = edge
    return f">={previous:g}{suffix}"


def _aggregate(rows: list[dict[str, Any]], pnl_key: str) -> dict[str, Any]:
    pnls = [float(row[pnl_key]) for row in rows]
    n = len(pnls)
    average = mean(pnls) if pnls else 0.0
    standard_error = stdev(pnls) / math.sqrt(n) if n > 1 else None
    return {
        "events": len({str(row["event_slug"]) for row in rows}),
        "positions": n,
        "total_pnl_usdc": sum(pnls),
        "average_pnl_usdc": average,
        "mean_ci95_lower_usdc": average - 1.96 * standard_error if standard_error is not None else None,
        "mean_ci95_upper_usdc": average + 1.96 * standard_error if standard_error is not None else None,
        "profitable_fraction": sum(value > 0 for value in pnls) / n if n else None,
        "useful_profit_fraction": sum(value >= settings.MIN_ACCEPTABLE_NET_PNL_USDC for value in pnls) / n if n else None,
    }


def _group(rows: list[dict[str, Any]], key: str, pnl_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: _aggregate(values, pnl_key) for name, values in sorted(grouped.items())}


def _entry_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    labels = {(str(row[0]), str(row[1])): int(row[2]) for row in connection.execute(
        "SELECT event_slug,outcome,MAX(label) FROM training_examples GROUP BY event_slug,outcome"
    )}
    decisions = {int(row[0]): str(row[1] or "{}") for row in connection.execute(
        "SELECT id,market_state_json FROM model_decisions"
    )}
    rows: list[dict[str, Any]] = []
    for source, table in (("paper", "paper_positions"), ("live", "live_positions")):
        for raw in connection.execute(
            f"""SELECT * FROM {table} WHERE status IN ('closed','resolved')
                 AND COALESCE(execution_valid,1)=1 ORDER BY opened_at"""
        ):
            position = dict(raw)
            label = labels.get((str(position["event_slug"]), str(position["outcome"])))
            if label is None:
                continue
            try:
                state = json.loads(decisions.get(int(position.get("entry_decision_id") or -1), "{}"))
            except json.JSONDecodeError:
                state = {}
            price = float(position.get("average_price") or 0.0)
            shares = float(position.get("shares") or 0.0)
            cost = float(position.get("cost_usdc") or 0.0)
            remaining = float(state.get("remaining_seconds") or 0.0)
            distance = float(state.get("distance_to_target_pct") or 0.0)
            row = {
                "source": source, "position_id": int(position["id"]),
                "event_slug": str(position["event_slug"]), "outcome": str(position["outcome"]),
                "entry_price": price, "entry_price_bucket": _price_bucket(price),
                "remaining_seconds": remaining,
                "remaining_bucket": _range_bucket(remaining, (30, 60, 120, 180, 240, 300), "s"),
                "distance_to_target_pct": distance,
                "target_distance_bucket": _range_bucket(distance, (.02, .05, .10, .20), "%"),
                "realized_volatility_60s_pct": float(state.get("realized_volatility_60s_pct") or 0.0),
                "source_disagreement_pct": float(state.get("source_disagreement_pct") or 0.0),
                "consensus_return_pct": float(state.get("consensus_return_pct") or 0.0),
                "sharp_move_pct": float(state.get("sharp_move_pct") or 0.0),
                "official_label": label,
                "actual_pnl_usdc": float(position.get("realized_pnl_usdc") or 0.0),
                "hold_pnl_usdc": shares * label - cost,
                "failed_high_confidence_price": int(price >= .90 and label == 0),
            }
            rows.append(row)
    return rows


def _exit_analysis(exit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sampled = [row for row in exit_rows if int(float(row["seconds_in_position"])) % 5 == 0]
    by_position: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        by_position[(str(row["source"]), int(row["position_id"]))].append(row)
    oracle: list[dict[str, Any]] = []
    for key, sequence in by_position.items():
        sequence.sort(key=lambda row: str(row["observed_at"]))
        hold = float(sequence[0]["hold_pnl_usdc"])
        best = max(sequence, key=lambda row: float(row["close_now_pnl_usdc"]))
        best_close = float(best["close_now_pnl_usdc"])
        oracle.append({
            "source": key[0], "position_id": key[1], "event_slug": str(best["event_slug"]),
            "outcome": str(best["outcome"]), "entry_price": float(best["average_price"]),
            "entry_price_bucket": _price_bucket(float(best["average_price"])),
            "oracle_action": "CLOSE" if best_close >= hold + settings.MIN_ACCEPTABLE_NET_PNL_USDC else "HOLD",
            "oracle_exit_price": float(best["current_bid"]),
            "oracle_exit_remaining_seconds": float(best["remaining_seconds"]),
            "oracle_close_pnl_usdc": best_close, "hold_pnl_usdc": hold,
            "oracle_advantage_usdc": max(0.0, best_close - hold),
        })

    price_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        price_groups[_price_bucket(float(row["current_bid"]))].append(row)
    exit_price_surface = {}
    for name, rows in sorted(price_groups.items()):
        advantages = [float(row["close_advantage_vs_best_wait_usdc"]) for row in rows]
        exit_price_surface[name] = {
            "rows": len(rows), "events": len({str(row["event_slug"]) for row in rows}),
            "average_close_advantage_vs_best_wait_usdc": mean(advantages),
            "profitable_close_fraction": sum(value >= settings.MIN_ACCEPTABLE_NET_PNL_USDC for value in advantages) / len(advantages),
        }

    signal_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled:
        drawdown = float(row["drawdown_from_peak"])
        bid_momentum = float(row["momentum_bid_3ticks"])
        target_momentum = float(row["momentum_distance_3ticks"])
        signal_groups["all_ticks"].append(row)
        if drawdown <= -.05:
            signal_groups["drawdown_5c"].append(row)
        if bid_momentum < 0 and target_momentum < 0:
            signal_groups["bid_and_target_momentum_against"].append(row)
        if bid_momentum < 0:
            signal_groups["bid_momentum_against"].append(row)
        if drawdown <= -.05 and bid_momentum < 0:
            signal_groups["drawdown_and_bid_momentum_against"].append(row)
        if drawdown <= -.05 and bid_momentum < 0 and target_momentum < 0:
            signal_groups["combined_reversal"].append(row)
    reversal_signals = {}
    for name, rows in signal_groups.items():
        values = [float(row["close_advantage_vs_best_wait_usdc"]) for row in rows]
        reversal_signals[name] = {
            "rows": len(rows), "events": len({str(row["event_slug"]) for row in rows}),
            "average_advantage_usdc": mean(values),
            "close_better_fraction": sum(value >= settings.MIN_ACCEPTABLE_NET_PNL_USDC for value in values) / len(values),
        }

    policies = []
    for drawdown_threshold in (-.02, -.05, -.10, -.15):
        for momentum_threshold in (0.0, -.005, -.01, -.02):
            selected_total = hold_total = 0.0
            exits = winners_saved = winners_cut = 0
            for sequence in by_position.values():
                sequence.sort(key=lambda row: str(row["observed_at"]))
                hold = float(sequence[0]["hold_pnl_usdc"])
                trigger = next((row for row in sequence if float(row["seconds_in_position"]) >= 5
                    and float(row["drawdown_from_peak"]) <= drawdown_threshold
                    and float(row["momentum_bid_3ticks"]) <= momentum_threshold), None)
                selected = float(trigger["close_now_pnl_usdc"]) if trigger else hold
                hold_total += hold; selected_total += selected
                exits += int(trigger is not None)
                winners_saved += int(trigger is not None and hold < 0 and selected > hold)
                winners_cut += int(trigger is not None and hold > 0 and selected < hold)
            policies.append({
                "drawdown_threshold": drawdown_threshold, "bid_momentum_threshold": momentum_threshold,
                "positions": len(by_position), "early_exits": exits,
                "policy_pnl_usdc": selected_total, "hold_pnl_usdc": hold_total,
                "advantage_vs_hold_usdc": selected_total - hold_total,
                "losing_positions_improved": winners_saved, "winning_positions_cut": winners_cut,
                "validation": "in_sample_exploratory_only",
            })
    policies.sort(key=lambda row: float(row["advantage_vs_hold_usdc"]), reverse=True)
    return {
        "sampled_rows": len(sampled), "positions": len(by_position),
        "oracle": {
            "positions": len(oracle),
            "early_exit_preferred": sum(row["oracle_action"] == "CLOSE" for row in oracle),
            "total_oracle_advantage_usdc": sum(float(row["oracle_advantage_usdc"]) for row in oracle),
            "by_entry_price": _group(oracle, "entry_price_bucket", "oracle_advantage_usdc") if oracle else {},
        },
        "oracle_rows": oracle,
        "exit_price_surface": exit_price_surface,
        "reversal_signals": reversal_signals,
        "exploratory_threshold_policies": policies[:10],
    }


def study(database: Path = settings.DATABASE_PATH, output_dir: Path | None = None) -> dict[str, Any]:
    connection = sqlite3.connect(database); connection.row_factory = sqlite3.Row
    try:
        entries = _entry_rows(connection)
    finally:
        connection.close()
    exit_rows = pq.read_table(settings.EXIT_SEQUENCE_DATASET_PATH).to_pylist()
    exit_report = _exit_analysis(exit_rows)
    high_price_failures = [row for row in entries if row["failed_high_confidence_price"]]
    report = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "entry": "actual valid positions; official labels; actual and HOLD net PnL after stored fees",
            "exit": "causal features at each tick; future outcome used only as offline label",
            "warning": "descriptive/in-sample results are hypotheses; promote only after chronological OOS validation",
        },
        "entry": {
            "positions": len(entries), "events": len({row["event_slug"] for row in entries}),
            "by_price": _group(entries, "entry_price_bucket", "actual_pnl_usdc"),
            "hold_by_price": _group(entries, "entry_price_bucket", "hold_pnl_usdc"),
            "by_remaining_time": _group(entries, "remaining_bucket", "actual_pnl_usdc"),
            "by_target_distance": _group(entries, "target_distance_bucket", "actual_pnl_usdc"),
            "by_direction": _group(entries, "outcome", "actual_pnl_usdc"),
            "price_090_plus_failures": _aggregate(high_price_failures, "actual_pnl_usdc"),
        },
        "exit": {key: value for key, value in exit_report.items() if key != "oracle_rows"},
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = output_dir or settings.MODEL_DIR / "experiments" / f"entry_exit_price_study_v1_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    entry_path, oracle_path = root / "entry_positions.parquet", root / "oracle_exit_opportunities.parquet"
    pq.write_table(pa.Table.from_pylist(entries), entry_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(exit_report["oracle_rows"]), oracle_path, compression="zstd")
    report.update({"output_dir": str(root), "entry_dataset": str(entry_path), "oracle_exit_dataset": str(oracle_path)})
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2))
