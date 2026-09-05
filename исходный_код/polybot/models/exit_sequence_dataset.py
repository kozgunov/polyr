"""Траектории открытых позиций: на каждом тике сравнивает CLOSE с HOLD."""

from __future__ import annotations

import json
import math
import sqlite3
from bisect import bisect_left, bisect_right
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import pyarrow as pa
import pyarrow.parquet as pq

from polybot.trading.fees import total_fee_usdc


EXIT_FRACTIONS = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
WAIT_HORIZONS = tuple(int(value) for value in settings.EXIT_WAIT_HORIZONS_SECONDS)


def _ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _reward(pnl: float) -> float:
    """Net PnL ниже $0.10 после комиссии является отрицательной учебной наградой."""
    threshold = float(settings.MIN_ACCEPTABLE_NET_PNL_USDC)
    return pnl if pnl >= threshold else -max(0.001, threshold - pnl)


def build(
    database: Path = settings.DATABASE_PATH,
    output: Path = settings.EXIT_SEQUENCE_DATASET_PATH,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    connection = sqlite3.connect(database); connection.row_factory = sqlite3.Row
    labels = {(str(row[0]), str(row[1])): int(row[2]) for row in connection.execute(
        "SELECT event_slug,outcome,MAX(label) FROM training_examples GROUP BY event_slug,outcome"
    )}
    positions = []
    for source, table in (("paper", "paper_positions"), ("live", "live_positions")):
        query = f"""SELECT p.*,COALESCE(d.model_name,'unknown') model FROM {table} p
                    LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                    WHERE p.status IN ('closed','resolved') ORDER BY p.opened_at"""
        positions.extend([{**dict(row), "source": source} for row in connection.execute(query)])
    records = []
    for position in positions:
        slug, outcome = str(position["event_slug"]), str(position["outcome"])
        label = labels.get((slug, outcome))
        if label is None:
            continue
        if position["source"] == "paper":
            buys = connection.execute(
                """SELECT shares,filled_price,fee_usdc FROM paper_orders
                   WHERE session_id=? AND event_slug=? AND action LIKE 'BUY_%' AND status IN ('filled','partially_filled')""",
                (position["session_id"], slug),
            ).fetchall()
        else:
            buys = connection.execute(
                "SELECT matched_size,requested_price,0.0 FROM live_orders WHERE event_slug=? AND side='BUY' AND matched_size>0",
                (slug,),
            ).fetchall()
        shares = sum(float(row[0] or 0) for row in buys)
        cost = sum(float(row[0] or 0) * float(row[1] or 0) + float(row[2] or 0) for row in buys)
        if shares <= 0:
            continue
        average_price = cost / shares
        opened_at = str(position["opened_at"]); opened_ts = _ts(opened_at)
        event_end = int(slug.rsplit("-", 1)[-1]) + 300
        snapshots = connection.execute(
            """SELECT collected_at,best_bid,midpoint,best_bid_size,best_ask_size,spread
               FROM market_snapshots WHERE event_slug=? AND outcome=? AND collected_at>=?
               ORDER BY collected_at""", (slug, outcome, opened_at),
        ).fetchall()
        refs = [dict(row) for row in connection.execute(
            """SELECT collected_at,reference_price,target_price FROM reference_price_snapshots
               WHERE event_slug=? AND collected_at>=? ORDER BY collected_at""", (slug, opened_at),
        )]
        ref_index = 0; bids: list[float] = []; bid_times: list[float] = []; distances: list[float] = []
        hold_pnl = shares * label - cost
        snapshot_rows = list(snapshots)
        snapshot_times = [_ts(str(candidate[0])) for candidate in snapshot_rows]
        for snapshot_index, snap in enumerate(snapshot_rows):
            observed = str(snap[0]); observed_ts = _ts(observed)
            if observed_ts > event_end + 10:
                break
            while ref_index + 1 < len(refs) and _ts(refs[ref_index + 1]["collected_at"]) <= observed_ts:
                ref_index += 1
            reference = refs[ref_index] if refs else {}
            bid = float(snap[1] if snap[1] is not None else snap[2] or 0)
            if not 0 <= bid <= 1:
                continue
            ref_price, target = reference.get("reference_price"), reference.get("target_price")
            distance = ((float(ref_price) / float(target) - 1) * 100) if ref_price and target else 0.0
            oriented_distance = distance if outcome == "Up" else -distance
            bids.append(bid); bid_times.append(observed_ts); distances.append(oriented_distance)
            close_pnl = shares * bid - total_fee_usdc(shares, bid) - cost
            advantage = close_pnl - hold_pnl
            delayed_pnls: dict[int, float | None] = {}
            for horizon in WAIT_HORIZONS:
                target_ts = observed_ts + horizon
                future_index = bisect_left(snapshot_times, target_ts, lo=snapshot_index + 1)
                future = (
                    snapshot_rows[future_index]
                    if future_index < len(snapshot_rows) and snapshot_times[future_index] <= event_end
                    else None
                )
                if future is None:
                    delayed_pnls[horizon] = None
                else:
                    future_bid = float(future[1] if future[1] is not None else future[2] or 0)
                    delayed_pnls[horizon] = (
                        shares * future_bid - total_fee_usdc(shares, future_bid) - cost
                        if 0 <= future_bid <= 1 else None
                    )
            wait_candidates = {"HOLD": hold_pnl}
            wait_candidates.update({f"WAIT_{h}": value for h, value in delayed_pnls.items() if value is not None})
            best_wait_action, best_wait_pnl = max(wait_candidates.items(), key=lambda item: item[1])
            close_advantage_vs_best_wait = close_pnl - best_wait_pnl
            timing_candidates = {"CLOSE_NOW": close_pnl, **wait_candidates}
            optimal_exit_timing, optimal_timing_pnl = max(timing_candidates.items(), key=lambda item: item[1])
            fraction_pnls: dict[float, float] = {}
            for exit_fraction in EXIT_FRACTIONS:
                sold_shares = shares * exit_fraction
                remaining_shares = shares - sold_shares
                allocated_cost = cost * exit_fraction
                remaining_cost = cost - allocated_cost
                realised_now = (
                    sold_shares * bid
                    - total_fee_usdc(sold_shares, bid)
                    - allocated_cost
                )
                resolved_remainder = remaining_shares * label - remaining_cost
                fraction_pnls[exit_fraction] = realised_now + resolved_remainder
            fraction_rewards = {fraction: _reward(pnl) for fraction, pnl in fraction_pnls.items()}
            optimal_fraction = max(EXIT_FRACTIONS, key=lambda value: fraction_rewards[value])
            if optimal_fraction <= 0:
                optimal_exit_action = "HOLD"
            elif optimal_fraction >= 1:
                optimal_exit_action = "CLOSE"
            else:
                optimal_exit_action = "PARTIAL_CLOSE"
            seconds_in_position = max(0.0, observed_ts - opened_ts)
            remaining = max(0.0, event_end - observed_ts)
            peak = max(bids); trough = min(bids)
            def lag_bid(seconds: int) -> float:
                cutoff = observed_ts - seconds
                index = max(0, bisect_right(bid_times, cutoff) - 1)
                return bids[index]

            momentum_15s = bid - lag_bid(15)
            momentum_30s = bid - lag_bid(30)
            momentum_60s = bid - lag_bid(60)
            peak_index = max(range(len(bids)), key=bids.__getitem__)
            seconds_since_peak = max(0.0, observed_ts - bid_times[peak_index])
            records.append({
                "source": position["source"], "position_id": int(position["id"]), "event_slug": slug,
                "outcome": outcome, "model": position["model"], "observed_at": observed,
                "seconds_in_position": seconds_in_position, "remaining_seconds": remaining,
                "current_bid": bid, "average_price": average_price, "marked_return": bid / average_price - 1,
                "oriented_distance_to_target_pct": oriented_distance,
                "momentum_bid_3ticks": bid - bids[max(0, len(bids)-4)],
                "momentum_bid_15s": momentum_15s,
                "momentum_bid_30s": momentum_30s,
                "momentum_bid_60s": momentum_60s,
                "bid_slope_15s": momentum_15s / 15.0,
                "bid_slope_30s": momentum_30s / 30.0,
                "momentum_distance_3ticks": oriented_distance - distances[max(0, len(distances)-4)],
                "target_distance_available": int(bool(ref_price and target)),
                "peak_bid_since_entry": peak, "trough_bid_since_entry": trough,
                "drawdown_from_peak": bid - peak, "recovery_from_trough": bid - trough,
                "seconds_since_peak": seconds_since_peak,
                "maximum_favorable_excursion": peak - average_price,
                "maximum_adverse_excursion": trough - average_price,
                "spread": float(snap[5] or 0),
                "log_bid_size": math.log1p(max(0.0, float(snap[3] or 0))),
                "log_ask_size": math.log1p(max(0.0, float(snap[4] or 0))),
                "shares": shares, "original_cost_usdc": cost, "resolved_label": label,
                "close_now_pnl_usdc": close_pnl, "hold_pnl_usdc": hold_pnl,
                "close_advantage_usdc": advantage, "optimal_action": "CLOSE" if advantage > settings.EXIT_VALUE_MARGIN_USDC else "HOLD",
                "best_wait_action": best_wait_action, "best_wait_pnl_usdc": best_wait_pnl,
                "close_advantage_vs_best_wait_usdc": close_advantage_vs_best_wait,
                "optimal_exit_timing": optimal_exit_timing,
                "optimal_timing_pnl_usdc": optimal_timing_pnl,
                **{f"pnl_wait_{h:03d}s_usdc": delayed_pnls[h] for h in WAIT_HORIZONS},
                "optimal_exit_action": optimal_exit_action,
                "optimal_exit_fraction": optimal_fraction,
                "optimal_fraction_pnl_usdc": fraction_pnls[optimal_fraction],
                "optimal_fraction_reward": fraction_rewards[optimal_fraction],
                **{
                    f"pnl_exit_{int(exit_fraction * 100):03d}_pct_usdc": pnl
                    for exit_fraction, pnl in fraction_pnls.items()
                },
                **{
                    f"reward_exit_{int(exit_fraction * 100):03d}_pct": reward
                    for exit_fraction, reward in fraction_rewards.items()
                },
                "actual_exit_timing": position.get("exit_timing"), "actual_realized_pnl_usdc": float(position.get("realized_pnl_usdc") or 0),
            })
    connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd", use_dictionary=True)
    temporary.replace(output)
    manifest = {
        "schema_version": 4, "created_at": datetime.now(UTC).isoformat(), "output": str(output),
        "rows": len(records), "events": len({row["event_slug"] for row in records}),
        "positions": len({(row["source"], row["position_id"]) for row in records}),
        "sources": {source: sum(row["source"] == source for row in records) for source in ("paper", "live")},
        "target": "argmax quality-adjusted reward among HOLD/20%/40%/60%/80%/100%; net PnL below $0.10 is penalized",
        "exit_fractions": EXIT_FRACTIONS,
        "wait_horizons_seconds": WAIT_HORIZONS,
        "timing_target": "CLOSE_NOW versus WAIT_15/WAIT_30/WAIT_60/HOLD, net of exit fees",
        "leakage_rule": "all features use only observations at or before observed_at; labels use later resolution",
    }
    manifest_path = manifest_path or settings.EXIT_SEQUENCE_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
