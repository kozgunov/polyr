"""Одноразовая сверка трёх повреждённых LIVE-позиций с фактическими CLOB fills."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import app_config as settings
from polybot.trading.live_executor import build_client


CORRUPT_IDS = (19, 24, 28)


def main() -> None:
    db = sqlite3.connect(settings.DATABASE_PATH)
    db.row_factory = sqlite3.Row
    for table, columns in {
        "live_positions": {"ledger_validated": "INTEGER NOT NULL DEFAULT 0"},
        "live_orders": {"average_fill_price": "REAL", "fill_notional_usdc": "REAL NOT NULL DEFAULT 0",
                        "fee_usdc": "REAL NOT NULL DEFAULT 0", "fill_source": "TEXT"},
    }.items():
        existing = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
        for name, definition in columns.items():
            if name not in existing:
                db.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}')
    existing_backups = sorted(settings.DATA_DIR.glob("market_data_before_live_reconcile_*.sqlite3"))
    complete_backup = next((p for p in reversed(existing_backups)
                            if p.stat().st_size >= settings.DATABASE_PATH.stat().st_size * 0.95), None)
    if complete_backup is None:
        backup = settings.DATA_DIR / f"market_data_before_live_reconcile_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.sqlite3"
        with sqlite3.connect(backup) as target:
            db.backup(target)

    db.execute("""CREATE TABLE IF NOT EXISTS live_ledger_audit (
        id INTEGER PRIMARY KEY, audited_at TEXT NOT NULL, position_id INTEGER NOT NULL,
        original_json TEXT NOT NULL, clob_evidence_json TEXT NOT NULL,
        action TEXT NOT NULL, UNIQUE(position_id, action)
    )""")
    trades = build_client().get_trades()
    for position_id in CORRUPT_IDS:
        position = db.execute("SELECT * FROM live_positions WHERE id=?", (position_id,)).fetchone()
        if not position:
            continue
        orders = db.execute("SELECT * FROM live_orders WHERE event_slug=? ORDER BY id", (position["event_slug"],)).fetchall()
        order_ids = {str(row["order_id"]): row for row in orders}
        fills: list[dict] = []
        for trade in trades:
            raw = trade if isinstance(trade, dict) else vars(trade)
            if str(raw.get("asset_id")) != str(position["token_id"]):
                continue
            if str(raw.get("taker_order_id")) in order_ids:
                fills.append({"order_id": raw["taker_order_id"], "side": raw["side"], "price": raw["price"],
                              "size": raw["size"], "fee_rate_bps": raw.get("fee_rate_bps", "0"), "trade_id": raw["id"]})
            for maker in raw.get("maker_orders") or []:
                if str(maker.get("order_id")) in order_ids:
                    fills.append({"order_id": maker["order_id"], "side": maker["side"], "price": maker["price"],
                                  "size": maker["matched_amount"], "fee_rate_bps": maker.get("fee_rate_bps") or "0",
                                  "trade_id": raw["id"]})
        original = dict(position)
        evidence = {"orders": [dict(row) for row in orders], "fills": fills}
        db.execute("UPDATE live_positions SET execution_valid=0,ledger_validated=0,invalid_reason=? WHERE id=?",
                   ("duplicate_cumulative_fill_reconciliation_bug; superseded_by_clob_audit", position_id))
        for row in orders:
            matching = [f for f in fills if f["order_id"] == row["order_id"]]
            size = sum(float(f["size"]) for f in matching)
            notional = sum(float(f["size"]) * float(f["price"]) for f in matching)
            fee = sum(float(f["size"]) * float(f["price"]) * float(f["fee_rate_bps"] or 0) / 10_000 for f in matching)
            db.execute("""UPDATE live_orders SET matched_size=?,average_fill_price=?,fill_notional_usdc=?,fee_usdc=?,
                          fill_source='clob_trade_history',execution_valid=?,invalid_reason=? WHERE id=?""",
                       (size, notional / size if size else None, notional, fee, 1 if size else 0,
                        None if size else "no_matching_clob_trade", row["id"]))
        db.execute("INSERT OR IGNORE INTO live_ledger_audit(audited_at,position_id,original_json,clob_evidence_json,action) VALUES(?,?,?,?,?)",
                   (datetime.now(UTC).isoformat(), position_id, json.dumps(original), json.dumps(evidence), "invalidate_and_reconcile_orders"))
    db.commit()
    print(f"all_done_{__file__}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"error_in_{__file__}")
        raise
