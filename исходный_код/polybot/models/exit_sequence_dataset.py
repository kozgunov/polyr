"""Траектории открытых позиций: на каждом тике сравнивает CLOSE с HOLD."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import pyarrow as pa
import pyarrow.parquet as pq

from polybot.trading.fees import total_fee_usdc


def _ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def build(database: Path = settings.DATABASE_PATH, output: Path = settings.EXIT_SEQUENCE_DATASET_PATH) -> dict[str, Any]:
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
        ref_index = 0; bids: list[float] = []; distances: list[float] = []
        hold_pnl = shares * label - cost
        for snap in snapshots:
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
            bids.append(bid); distances.append(oriented_distance)
            close_pnl = shares * bid - total_fee_usdc(shares, bid) - cost
            advantage = close_pnl - hold_pnl
            seconds_in_position = max(0.0, observed_ts - opened_ts)
            remaining = max(0.0, event_end - observed_ts)
            peak = max(bids); trough = min(bids)
            records.append({
                "source": position["source"], "position_id": int(position["id"]), "event_slug": slug,
                "outcome": outcome, "model": position["model"], "observed_at": observed,
                "seconds_in_position": seconds_in_position, "remaining_seconds": remaining,
                "current_bid": bid, "average_price": average_price, "marked_return": bid / average_price - 1,
                "oriented_distance_to_target_pct": oriented_distance,
                "momentum_bid_3ticks": bid - bids[max(0, len(bids)-4)],
                "momentum_distance_3ticks": oriented_distance - distances[max(0, len(distances)-4)],
                "peak_bid_since_entry": peak, "trough_bid_since_entry": trough,
                "drawdown_from_peak": bid - peak, "recovery_from_trough": bid - trough,
                "spread": float(snap[5] or 0),
                "log_bid_size": math.log1p(max(0.0, float(snap[3] or 0))),
                "log_ask_size": math.log1p(max(0.0, float(snap[4] or 0))),
                "shares": shares, "original_cost_usdc": cost, "resolved_label": label,
                "close_now_pnl_usdc": close_pnl, "hold_pnl_usdc": hold_pnl,
                "close_advantage_usdc": advantage, "optimal_action": "CLOSE" if advantage > settings.EXIT_VALUE_MARGIN_USDC else "HOLD",
                "actual_exit_timing": position.get("exit_timing"), "actual_realized_pnl_usdc": float(position.get("realized_pnl_usdc") or 0),
            })
    connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd", use_dictionary=True)
    temporary.replace(output)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(), "output": str(output),
        "rows": len(records), "events": len({row["event_slug"] for row in records}),
        "positions": len({(row["source"], row["position_id"]) for row in records}),
        "sources": {source: sum(row["source"] == source for row in records) for source in ("paper", "live")},
        "target": "close_now_net_pnl_minus_hold_to_official_resolution_net_pnl",
        "leakage_rule": "all features use only observations at or before observed_at; labels use later resolution",
    }
    settings.EXIT_SEQUENCE_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))

