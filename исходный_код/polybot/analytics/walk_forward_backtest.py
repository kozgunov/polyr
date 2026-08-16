"""Sequential out-of-sample replay of the current BTC 5m policy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import app_config as settings
import joblib

from polybot.models.model_policy import decide_with_model
from polybot.trading.fees import total_fee_usdc
from polybot.trading.policy import Decision, MarketState, PositionState


def _load(path: Path, allowed_events: set[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        "SELECT event_slug,outcome,observed_at,label,features_json FROM training_examples ORDER BY observed_at"
    ).fetchall()
    connection.close()
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for slug, outcome, observed_at, label, raw in rows:
        if slug in allowed_events:
            result[str(slug)][str(outcome)].append({
                "observed_at": str(observed_at), "label": int(label), "features": json.loads(raw),
            })
    return result


def _state(slug: str, up: dict[str, Any], down: dict[str, Any], starts: dict[str, float]) -> MarketState:
    timestamp = max(datetime.fromisoformat(up["observed_at"]), datetime.fromisoformat(down["observed_at"]))
    started = datetime.fromtimestamp(int(slug.rsplit("-", 1)[-1]), UTC)
    prices = {
        source: float(up["features"][f"{source}_price"])
        for source in ("bybit", "okx", "pyth") if up["features"].get(f"{source}_price")
    }
    returns = {
        source: (price / starts[source] - 1.0) * 100.0
        for source, price in prices.items() if source in starts and starts[source] > 0
    }
    values = list(prices.values())
    disagreement = (max(values) - min(values)) / median(values) * 100 if len(values) >= 2 else 999.0
    books = {
        "Up": {key: up["features"].get(key) for key in ("best_bid", "best_ask", "best_bid_size", "best_ask_size", "midpoint", "spread")},
        "Down": {key: down["features"].get(key) for key in ("best_bid", "best_ask", "best_bid_size", "best_ask_size", "midpoint", "spread")},
    }
    elapsed = (timestamp - started).total_seconds()
    return MarketState(
        slug, timestamp.isoformat(), elapsed, 300.0 - elapsed, returns, prices, disagreement, 0.0,
        books["Up"]["best_bid"], books["Up"]["best_ask"], books["Down"]["best_bid"],
        books["Down"]["best_ask"], books,
        up["features"].get("target_price"), up["features"].get("reference_price"),
        up["features"].get("target_source"), timestamp.isoformat(),
        float(up["features"].get("realized_volatility_60s_pct") or 0.0),
        {str(lag): float(up["features"].get(f"distance_lag_{lag}s_pct") or 0.0) for lag in (15, 30, 60)},
    )


def _required(decision: Decision) -> int:
    if decision.action.startswith("BUY_"):
        return settings.ENTRY_SIGNAL_CONFIRMATIONS
    if decision.action == "CLOSE" and "reversal_exit" in decision.tags:
        return settings.REVERSAL_SIGNAL_CONFIRMATIONS
    return 0


def run(path: Path) -> dict[str, Any]:
    artifact = joblib.load(settings.TRAINING_ARTIFACT_PATH)
    test_events = set(artifact.get("splits", {}).get("test_events", []))
    if not test_events:
        raise RuntimeError("Artifact has no out-of-sample event split")
    data = _load(path, test_events)
    event_pnls: list[float] = []
    trade_log: list[dict[str, Any]] = []
    fees_total = 0.0
    trades = entries = exits = 0
    for slug in sorted(data):
        up_rows, down_rows = data[slug].get("Up", []), data[slug].get("Down", [])
        if not up_rows or not down_rows:
            continue
        starts = {
            source: float(up_rows[0]["features"][f"{source}_price"])
            for source in ("bybit", "okx", "pyth") if up_rows[0]["features"].get(f"{source}_price")
        }
        position: dict[str, Any] | None = None
        event_pnl = 0.0
        fresh_entries = 0
        confirmation_key = None
        confirmation_count = 0
        confirmation_started = 0.0
        last_action_at = -999.0
        event_trace: dict[str, Any] = {"event_slug": slug, "label_up": int(up_rows[0]["label"])}
        for up, down in zip(up_rows, down_rows, strict=False):
            state = _state(slug, up, down, starts)
            position_state = None
            if position:
                bid = state.up_bid if position["outcome"] == "Up" else state.down_bid
                position_state = PositionState(
                    1, slug, position["outcome"], "test", position["shares"], position["cost"],
                    position["average_price"], bid,
                )
            decision = decide_with_model(state, position_state)
            if position is None and decision.action.startswith("BUY_") and fresh_entries >= settings.MAX_FRESH_ENTRIES_PER_EVENT:
                decision = Decision("WAIT", decision.confidence, "entry limit", ["event_entry_limit"])
            required = _required(decision)
            if required:
                key = f"{decision.action}:{decision.direction or ''}:{position['outcome'] if position else ''}"
                if key != confirmation_key:
                    confirmation_key, confirmation_count, confirmation_started = key, 1, state.elapsed_seconds
                else:
                    confirmation_count += 1
                if confirmation_count < required or state.elapsed_seconds - confirmation_started < settings.SIGNAL_CONFIRMATION_MIN_SECONDS:
                    continue
            else:
                confirmation_key, confirmation_count = None, 0
            if state.elapsed_seconds - last_action_at < settings.PAPER_MIN_SECONDS_BETWEEN_ACTIONS:
                continue

            def close_current(current_state: MarketState) -> None:
                nonlocal position, event_pnl, fees_total, exits
                if not position:
                    return
                bid = current_state.up_bid if position["outcome"] == "Up" else current_state.down_bid
                if bid is None:
                    return
                fill = max(0.01, float(bid) * (1 - settings.ESTIMATED_SLIPPAGE_BPS / 10_000))
                fee = total_fee_usdc(position["shares"], fill)
                event_pnl += position["shares"] * fill - fee - position["cost"]
                event_trace["exit"] = {
                    "elapsed_seconds": current_state.elapsed_seconds,
                    "reference_price": current_state.reference_price,
                    "target_price": current_state.target_price,
                    "distance_to_target_pct": current_state.distance_to_target_pct,
                    "bid": bid,
                    "fill": fill,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "tags": decision.tags,
                }
                fees_total += fee
                exits += 1
                position = None

            if decision.action == "CLOSE" and position:
                close_current(state)
                last_action_at = state.elapsed_seconds
            if decision.action.startswith("BUY_") and decision.direction and decision.limit_price:
                fill = min(0.99, decision.limit_price * (1 + settings.ESTIMATED_SLIPPAGE_BPS / 10_000))
                shares = settings.PAPER_ENTRY_NOTIONAL_USDC / fill
                fee = total_fee_usdc(shares, fill)
                position = {"outcome": decision.direction, "shares": shares, "cost": settings.PAPER_ENTRY_NOTIONAL_USDC + fee, "average_price": fill}
                event_trace["entry"] = {
                    "outcome": decision.direction,
                    "elapsed_seconds": state.elapsed_seconds,
                    "remaining_seconds": state.remaining_seconds,
                    "reference_price": state.reference_price,
                    "target_price": state.target_price,
                    "distance_to_target_pct": state.distance_to_target_pct,
                    "ask": decision.limit_price,
                    "fill": fill,
                    "confidence": decision.confidence,
                    "tags": decision.tags,
                }
                fees_total += fee
                trades += 1
                if decision.action.startswith("BUY_"):
                    fresh_entries += 1
                    entries += 1
                last_action_at = state.elapsed_seconds
                confirmation_key, confirmation_count = None, 0
        if position:
            label = up_rows[0]["label"] if position["outcome"] == "Up" else down_rows[0]["label"]
            event_pnl += position["shares"] * label - position["cost"]
            event_trace["resolution"] = {"outcome": position["outcome"], "label": int(label)}
        if "entry" in event_trace:
            event_trace["net_pnl_usdc"] = event_pnl
            trade_log.append(event_trace)
        event_pnls.append(event_pnl)
    traded_pnls = [value for value in event_pnls if value != 0]
    wins = sum(value > 0 for value in traded_pnls)
    gross_profit = sum(value for value in event_pnls if value > 0)
    gross_loss = abs(sum(value for value in event_pnls if value < 0))
    return {
        "out_of_sample_events": len(event_pnls), "events_with_trades": sum(value != 0 for value in event_pnls),
        "entries": entries, "flips": 0, "exits": exits, "trades": trades,
        "net_pnl_usdc": sum(event_pnls), "fees_usdc": fees_total,
        "win_rate_on_traded_events": wins / len(traded_pnls) if traded_pnls else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "average_event_pnl": mean(event_pnls) if event_pnls else 0.0,
        "event_pnls": event_pnls, "trade_log": trade_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    args = parser.parse_args()
    report = run(args.db)
    settings.WALK_FORWARD_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
