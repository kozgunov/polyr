"""Честное сравнение ранних partial/full выходов с HOLD на тех же событиях."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
from polybot.trading.fees import total_fee_usdc


def _labels(connection: sqlite3.Connection) -> dict[tuple[str, str], int]:
    return {(str(row[0]), str(row[1])): int(row[2]) for row in connection.execute(
        "SELECT event_slug,outcome,MAX(label) FROM training_examples GROUP BY event_slug,outcome"
    )}


def _position_rows(connection: sqlite3.Connection, source: str) -> list[dict[str, Any]]:
    if source == "paper":
        query = """SELECT p.*,COALESCE(d.model_name,s.model_name,'unknown') model
                   FROM paper_positions p JOIN paper_sessions s ON s.session_id=p.session_id
                   LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                   WHERE p.had_early_exit=1 ORDER BY p.opened_at"""
    else:
        query = """SELECT p.*,COALESCE(d.model_name,'unknown') model
                   FROM live_positions p LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                   WHERE p.had_early_exit=1 ORDER BY p.opened_at"""
    return [dict(row) for row in connection.execute(query)]


def _orders(connection: sqlite3.Connection, position: dict[str, Any], source: str) -> tuple[list[dict], list[dict]]:
    if source == "paper":
        rows = [dict(row) for row in connection.execute(
            """SELECT action,filled_price price,shares,fee_usdc,status,created_at
               FROM paper_orders WHERE session_id=? AND event_slug=? AND status IN ('filled','partially_filled')
               ORDER BY id""", (position["session_id"], position["event_slug"]),
        )]
        buys = [row for row in rows if str(row["action"]).startswith("BUY")]
        exits = [row for row in rows if row["action"] in {"CLOSE", "PARTIAL_CLOSE"}]
    else:
        rows = [dict(row) for row in connection.execute(
            """SELECT side action,requested_price price,matched_size shares,0.0 fee_usdc,status,created_at
               FROM live_orders WHERE event_slug=? AND matched_size>0 ORDER BY id""", (position["event_slug"],),
        )]
        buys = [row for row in rows if row["action"] == "BUY"]
        exits = [row for row in rows if row["action"] == "SELL"]
    return buys, exits


def compare(database: Path = settings.DATABASE_PATH) -> dict[str, Any]:
    connection = sqlite3.connect(database); connection.row_factory = sqlite3.Row
    labels = _labels(connection)
    events = []
    for source in ("paper", "live"):
        for position in _position_rows(connection, source):
            buys, exits = _orders(connection, position, source)
            if not buys or not exits:
                continue
            original_shares = sum(float(row.get("shares") or 0) for row in buys)
            original_cost = sum(float(row.get("shares") or 0) * float(row.get("price") or 0) + float(row.get("fee_usdc") or 0) for row in buys)
            first_exit_price = float(exits[0].get("price") or 0)
            full_exit_pnl = original_shares * first_exit_price - total_fee_usdc(original_shares, first_exit_price) - original_cost
            label = labels.get((str(position["event_slug"]), str(position["outcome"])))
            if label is None:
                continue
            hold_pnl = original_shares * label - original_cost
            actual_pnl = float(position.get("realized_pnl_usdc") or 0)
            exit_shares = [float(row.get("shares") or 0) for row in exits]
            technical_partial_fill = source == "live" and any(
                0 < shares < original_shares - 1e-6 for shares in exit_shares
            )
            intended_partial = source == "paper" and any(row.get("action") == "PARTIAL_CLOSE" for row in exits)
            events.append({
                "source": source, "event_slug": position["event_slug"], "model": position["model"],
                "outcome": position["outcome"], "exit_timing": position["exit_timing"],
                "first_exit_at": exits[0].get("created_at"), "first_exit_price": first_exit_price,
                "original_shares": original_shares, "original_cost_usdc": original_cost,
                "actual_policy_pnl_usdc": actual_pnl, "full_at_first_exit_pnl_usdc": full_exit_pnl,
                "hold_to_resolution_pnl_usdc": hold_pnl,
                "actual_vs_hold_usdc": actual_pnl - hold_pnl,
                "full_vs_hold_usdc": full_exit_pnl - hold_pnl,
                "actual_vs_full_usdc": actual_pnl - full_exit_pnl,
                "intended_partial": intended_partial, "technical_partial_fill": technical_partial_fill,
                "exit_fill_count": len(exits), "resolved_label": label,
            })
    connection.close()

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "events": len(rows),
            "actual_pnl_usdc": sum(row["actual_policy_pnl_usdc"] for row in rows),
            "full_at_first_exit_pnl_usdc": sum(row["full_at_first_exit_pnl_usdc"] for row in rows),
            "hold_pnl_usdc": sum(row["hold_to_resolution_pnl_usdc"] for row in rows),
            "actual_vs_hold_usdc": sum(row["actual_vs_hold_usdc"] for row in rows),
            "full_vs_hold_usdc": sum(row["full_vs_hold_usdc"] for row in rows),
            "actual_vs_full_usdc": sum(row["actual_vs_full_usdc"] for row in rows),
        }
    report = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "interpretation": "full_at_first_exit uses the same first exit time/price; HOLD uses official resolution",
        "all_early_exits": aggregate(events),
        "paper_intended_partial": aggregate([row for row in events if row["intended_partial"]]),
        "live_technical_partial_fill": aggregate([row for row in events if row["technical_partial_fill"]]),
        "paper": aggregate([row for row in events if row["source"] == "paper"]),
        "live": aggregate([row for row in events if row["source"] == "live"]),
        "events": events,
    }
    output = settings.MODEL_DIR / "exit_partial_full_hold_comparison_v14.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output"] = str(output)
    return report


if __name__ == "__main__":
    print(json.dumps(compare(), ensure_ascii=False, indent=2))

