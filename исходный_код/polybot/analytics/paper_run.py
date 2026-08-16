"""Diagnose a paper-trading session without changing it."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import app_config as settings

from polybot.models.model_policy import probability_up
from polybot.trading.fees import net_buy_edge
from polybot.trading.policy import MarketState


def _duration_seconds(opened_at: str, closed_at: str | None) -> float | None:
    if not closed_at:
        return None
    return (datetime.fromisoformat(closed_at) - datetime.fromisoformat(opened_at)).total_seconds()


def analyze(path: Path, session_id: str | None = None) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    if session_id is None:
        row = connection.execute("SELECT session_id FROM paper_sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        if not row:
            raise RuntimeError("No paper sessions")
        session_id = str(row[0])
    session_row = connection.execute("SELECT * FROM paper_sessions WHERE session_id=?", (session_id,)).fetchone()
    if not session_row:
        raise RuntimeError(f"Unknown session: {session_id}")
    session = dict(session_row)
    positions = [dict(row) for row in connection.execute(
        "SELECT * FROM paper_positions WHERE session_id=? ORDER BY id", (session_id,),
    )]
    orders = [dict(row) for row in connection.execute(
        "SELECT * FROM paper_orders WHERE session_id=? ORDER BY id", (session_id,),
    )]
    decisions = {int(row["id"]): dict(row) for row in connection.execute(
        "SELECT * FROM model_decisions WHERE session_id=?", (session_id,),
    )}
    closed = [row for row in positions if row["status"] in {"closed", "resolved"}]
    pnls = [float(row["realized_pnl_usdc"] or 0) for row in closed]
    durations = [value for row in closed if (value := _duration_seconds(row["opened_at"], row["closed_at"])) is not None]
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))

    by_reason: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for row in closed:
        reason = str(row["close_reason"] or row["status"])
        by_reason[reason]["count"] += 1
        by_reason[reason]["pnl"] += float(row["realized_pnl_usdc"] or 0)

    action_counts: dict[str, int] = defaultdict(int)
    for order in orders:
        action_counts[str(order["action"])] += 1

    hold_comparisons: list[dict[str, Any]] = []
    for position in closed:
        entry = next((order for order in orders if order["decision_id"] == position["entry_decision_id"] and order["action"] in {"BUY_UP", "BUY_DOWN", "FLIP"}), None)
        label_row = connection.execute(
            "SELECT label FROM training_examples WHERE event_slug=? AND token_id=? LIMIT 1",
            (position["event_slug"], position["token_id"]),
        ).fetchone()
        if not entry or not label_row:
            continue
        entry_cost = float(entry["notional_usdc"] or 0) + float(entry["fee_usdc"] or 0)
        hold_pnl = float(entry["shares"] or 0) * int(label_row[0]) - entry_cost
        actual_pnl = float(position["realized_pnl_usdc"] or 0)
        hold_comparisons.append({
            "position_id": position["id"], "event_slug": position["event_slug"],
            "outcome": position["outcome"], "close_reason": position["close_reason"],
            "actual_pnl": actual_pnl, "hold_to_resolution_pnl": hold_pnl,
            "hold_minus_actual": hold_pnl - actual_pnl,
        })

    signal_exits = 0
    exits_without_opposite_signal = 0
    for position in closed:
        decision = decisions.get(int(position["exit_decision_id"] or 0))
        if not decision:
            continue
        p_up = decision.get("predicted_up_probability")
        if p_up is None:
            continue
        predicted = "Up" if float(p_up) >= 0.5 else "Down"
        if predicted != position["outcome"]:
            signal_exits += 1
        elif position["close_reason"] != "market_resolution":
            exits_without_opposite_signal += 1

    entry_signals: list[dict[str, float | int | str]] = []
    replay_v2: list[dict[str, float | int | str | bool]] = []
    for position in positions:
        decision = decisions.get(int(position["entry_decision_id"] or 0))
        label_row = connection.execute(
            "SELECT label FROM training_examples WHERE event_slug=? AND token_id=? LIMIT 1",
            (position["event_slug"], position["token_id"]),
        ).fetchone()
        if not decision or not label_row or decision.get("predicted_up_probability") is None:
            continue
        p_up = float(decision["predicted_up_probability"])
        probability = p_up if position["outcome"] == "Up" else 1.0 - p_up
        entry_signals.append({
            "outcome": position["outcome"], "probability": probability,
            "confidence": float(decision["confidence"]), "correct": int(label_row[0]),
            "entry_price": float(position["average_price"]),
        })
        raw_state = json.loads(decision["market_state_json"])
        state = MarketState(**{key: raw_state[key] for key in MarketState.__dataclass_fields__})
        new_p_up = probability_up(state)
        new_probability = new_p_up if position["outcome"] == "Up" else 1.0 - new_p_up
        new_confidence = abs(new_p_up - 0.5) * 2.0
        ask = state.up_ask if position["outcome"] == "Up" else state.down_ask
        spread = state.book_json.get(position["outcome"], {}).get("spread")
        new_edge = net_buy_edge(new_probability, ask) if ask else -1.0
        accepted = bool(
            ask and spread is not None
            and settings.PAPER_MIN_ENTRY_PRICE <= ask <= settings.PAPER_MAX_ENTRY_PRICE
            and float(spread) <= settings.MAX_ALLOWED_SPREAD
            and new_confidence >= settings.MIN_ENTRY_CONFIDENCE
            and new_edge >= settings.PAPER_MIN_ENTRY_NET_EDGE
            and state.elapsed_seconds >= settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN
            and state.remaining_seconds >= settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE
        )
        replay_v2.append({
            "accepted": accepted, "probability": new_probability, "confidence": new_confidence,
            "edge": new_edge, "correct": int(label_row[0]), "ask": float(ask or 0),
        })

    connection.close()
    wins = sum(value > 0 for value in pnls)
    comparison_delta = [float(row["hold_minus_actual"]) for row in hold_comparisons]
    replay_grid = {}
    for name, min_confidence, min_edge, max_ask in (
        ("strict", 0.65, 0.06, 0.72),
        ("balanced", 0.55, 0.04, 0.75),
        ("exploratory", 0.50, 0.03, 0.78),
    ):
        selected = [
            row for row in replay_v2
            if float(row["confidence"]) >= min_confidence
            and float(row["edge"]) >= min_edge
            and settings.PAPER_MIN_ENTRY_PRICE <= float(row["ask"]) <= max_ask
        ]
        replay_grid[name] = {
            "selected": len(selected),
            "win_rate": mean(int(row["correct"]) for row in selected) if selected else 0.0,
            "mean_probability": mean(float(row["probability"]) for row in selected) if selected else 0.0,
            "mean_ask": mean(float(row["ask"]) for row in selected) if selected else 0.0,
        }
    return {
        "session": session,
        "summary": {
            "positions": len(positions), "closed_positions": len(closed),
            "independent_events": len({row["event_slug"] for row in positions}),
            "wins": wins, "losses": sum(value < 0 for value in pnls),
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "net_pnl_usdc": sum(pnls), "fees_usdc": sum(float(row["fee_usdc"] or 0) for row in orders),
            "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
            "average_pnl_usdc": mean(pnls) if pnls else 0.0,
            "median_pnl_usdc": median(pnls) if pnls else 0.0,
            "average_holding_seconds": mean(durations) if durations else 0.0,
            "median_holding_seconds": median(durations) if durations else 0.0,
            "signal_exits": signal_exits,
            "exits_without_opposite_signal": exits_without_opposite_signal,
        },
        "orders": dict(action_counts),
        "by_close_reason": dict(by_reason),
        "hold_to_resolution": {
            "comparable_positions": len(hold_comparisons),
            "actual_pnl_usdc": sum(float(row["actual_pnl"]) for row in hold_comparisons),
            "hold_pnl_usdc": sum(float(row["hold_to_resolution_pnl"]) for row in hold_comparisons),
            "hold_minus_actual_usdc": sum(comparison_delta),
            "hold_better_count": sum(value > 0 for value in comparison_delta),
            "exit_better_count": sum(value < 0 for value in comparison_delta),
        },
        "entry_calibration": {
            "signals": len(entry_signals),
            "mean_predicted_probability": mean(float(row["probability"]) for row in entry_signals) if entry_signals else 0.0,
            "actual_win_rate": mean(int(row["correct"]) for row in entry_signals) if entry_signals else 0.0,
            "brier": mean((float(row["probability"]) - int(row["correct"])) ** 2 for row in entry_signals) if entry_signals else 0.0,
            "log_loss": mean(
                -(int(row["correct"]) * math.log(max(1e-9, float(row["probability"])))
                  + (1 - int(row["correct"])) * math.log(max(1e-9, 1 - float(row["probability"]))))
                for row in entry_signals
            ) if entry_signals else 0.0,
            "mean_entry_price": mean(float(row["entry_price"]) for row in entry_signals) if entry_signals else 0.0,
        },
        "v2_replay_on_old_entries": {
            "diagnostic_only_in_sample": True,
            "accepted": sum(bool(row["accepted"]) for row in replay_v2),
            "rejected": sum(not bool(row["accepted"]) for row in replay_v2),
            "accepted_win_rate": mean(int(row["correct"]) for row in replay_v2 if row["accepted"]) if any(row["accepted"] for row in replay_v2) else 0.0,
            "accepted_mean_probability": mean(float(row["probability"]) for row in replay_v2 if row["accepted"]) if any(row["accepted"] for row in replay_v2) else 0.0,
            "accepted_mean_ask": mean(float(row["ask"]) for row in replay_v2 if row["accepted"]) if any(row["accepted"] for row in replay_v2) else 0.0,
            "threshold_grid": replay_grid,
        },
        "largest_losses": sorted(hold_comparisons, key=lambda row: row["actual_pnl"])[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--session-id")
    args = parser.parse_args()
    print(json.dumps(analyze(args.db, args.session_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
