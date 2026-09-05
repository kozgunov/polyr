"""Сверяет LIVE PnL и ранние выходы исключительно по CLOB trade fills."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "исходный_код")]
import app_config as settings
from polybot.trading.live_executor import build_client


def main() -> None:
    with sqlite3.connect(settings.DATABASE_PATH) as db:
        db.row_factory = sqlite3.Row
        orders = {str(r["order_id"]): dict(r) for r in db.execute("SELECT * FROM live_orders")}
        labels = {(str(a), str(b)): int(c) for a, b, c in db.execute(
            "SELECT event_slug,outcome,MAX(label) FROM training_examples GROUP BY event_slug,outcome")}
    fills: list[dict] = []
    for trade in build_client().get_trades():
        raw = trade if isinstance(trade, dict) else vars(trade)
        taker = str(raw.get("taker_order_id") or "")
        if taker in orders:
            fills.append({"order_id": taker, "size": float(raw["size"]), "price": float(raw["price"]),
                          "fee_bps": float(raw.get("fee_rate_bps") or 0)})
        for maker in raw.get("maker_orders") or []:
            oid = str(maker.get("order_id") or "")
            if oid in orders:
                fills.append({"order_id": oid, "size": float(maker["matched_amount"]),
                              "price": float(maker["price"]), "fee_bps": float(maker.get("fee_rate_bps") or 0)})
    events: dict[str, dict] = defaultdict(lambda: {"buy_cost": 0.0, "buy_shares": 0.0, "sell_proceeds": 0.0,
                                                   "sell_shares": 0.0, "fees": 0.0, "outcome": None})
    for fill in fills:
        order = orders[fill["order_id"]]
        event = events[str(order["event_slug"])]
        notional = fill["size"] * fill["price"]
        fee = notional * fill["fee_bps"] / 10_000
        event["fees"] += fee; event["outcome"] = str(order["outcome"])
        if str(order["side"]).upper() == "BUY":
            event["buy_cost"] += notional; event["buy_shares"] += fill["size"]
        else:
            event["sell_proceeds"] += notional; event["sell_shares"] += fill["size"]
    rows = []
    for slug, event in sorted(events.items()):
        label = labels.get((slug, str(event["outcome"])))
        remaining = max(0.0, event["buy_shares"] - event["sell_shares"])
        pnl = event["sell_proceeds"] + (remaining * label if label is not None else 0) - event["buy_cost"] - event["fees"]
        exit_vs_hold = event["sell_proceeds"] - event["sell_shares"] * label - event["fees"] if label is not None else None
        rows.append({"event_slug": slug, **event, "resolved_label": label, "remaining_shares": remaining,
                     "net_pnl_usdc": pnl if label is not None else None, "early_exit_advantage_vs_hold_usdc": exit_vs_hold})
    exited = [r for r in rows if r["sell_shares"] > 0 and r["early_exit_advantage_vs_hold_usdc"] is not None]
    report = {"created_at": datetime.now(UTC).isoformat(), "source": "authenticated_clob_trade_history",
              "matched_fills": len(fills), "events": rows,
              "early_exit": {"events": len(exited),
                 "advantage_vs_hold_usdc": sum(r["early_exit_advantage_vs_hold_usdc"] for r in exited),
                 "helped_events": sum(r["early_exit_advantage_vs_hold_usdc"] > 0 for r in exited),
                 "hurt_events": sum(r["early_exit_advantage_vs_hold_usdc"] < 0 for r in exited)}}
    output = ROOT / "документация" / "live_fill_audit_20260905.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["early_exit"], ensure_ascii=False))
    print(f"all_done_{__file__}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"error_in_{__file__}")
        raise
