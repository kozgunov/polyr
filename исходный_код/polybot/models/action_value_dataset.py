"""Постоянно обновляемый датасет: состояние рынка -> чистый PnL BUY или WAIT."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import app_config as settings

from polybot.collectors.pipeline import now
from polybot.trading.fees import total_fee_usdc


SCHEMA = """
CREATE TABLE IF NOT EXISTS action_value_examples (
  snapshot_id INTEGER PRIMARY KEY, event_slug TEXT NOT NULL, observed_at TEXT NOT NULL,
  outcome TEXT NOT NULL, resolution_label INTEGER NOT NULL,
  entry_price REAL NOT NULL, notional_usdc REAL NOT NULL,
  net_pnl_if_buy_resolution REAL NOT NULL, optimal_action TEXT NOT NULL,
  model_action TEXT, model_confidence REAL, model_predicted_up_probability REAL,
  model_expected_net_edge REAL, model_executed INTEGER NOT NULL DEFAULT 0,
  actual_realized_pnl_usdc REAL, features_json TEXT NOT NULL, built_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_action_value_event_time
  ON action_value_examples(event_slug,observed_at);
"""


def build(path: Path = settings.DATABASE_PATH, output: Path | None = None) -> dict[str, int]:
    connection = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    connection.executescript(SCHEMA)
    realized = {
        int(row[0]): float(row[1])
        for row in connection.execute(
            """SELECT entry_decision_id,realized_pnl_usdc FROM paper_positions
               WHERE entry_decision_id IS NOT NULL AND realized_pnl_usdc IS NOT NULL"""
        )
    }
    rows = connection.execute(
        """SELECT t.snapshot_id,t.event_slug,t.outcome,t.observed_at,t.label,t.features_json,
                  d.id decision_id,d.action,d.confidence,d.predicted_up_probability,
                  d.expected_net_edge,d.executed
           FROM training_examples t
           LEFT JOIN model_decisions d ON d.id=(
             SELECT d2.id FROM model_decisions d2
             WHERE d2.event_slug=t.event_slug AND d2.observed_at<=t.observed_at
             ORDER BY d2.observed_at DESC LIMIT 1
           )
           ORDER BY t.event_slug,t.observed_at"""
    ).fetchall()
    written = skipped = 0
    export_lines: list[str] = []
    for row in rows:
        features = json.loads(row["features_json"])
        ask = features.get("best_ask")
        if ask is None or not settings.PNL_DATASET_MIN_ENTRY_PRICE <= float(ask) <= settings.PNL_DATASET_MAX_ENTRY_PRICE:
            skipped += 1
            continue
        price = float(ask)
        notional = float(settings.PAPER_ENTRY_NOTIONAL_USDC)
        shares = notional / price
        fee = total_fee_usdc(shares, price, taker=True)
        net_pnl = shares * int(row["label"]) - notional - fee
        decision_id = int(row["decision_id"]) if row["decision_id"] is not None else None
        values = (
            int(row["snapshot_id"]), str(row["event_slug"]), str(row["observed_at"]), str(row["outcome"]),
            int(row["label"]), price, notional, net_pnl, "BUY" if net_pnl > 0 else "WAIT",
            str(row["action"]) if row["action"] is not None else None,
            float(row["confidence"]) if row["confidence"] is not None else None,
            float(row["predicted_up_probability"]) if row["predicted_up_probability"] is not None else None,
            float(row["expected_net_edge"]) if row["expected_net_edge"] is not None else None,
            int(row["executed"] or 0), realized.get(decision_id) if decision_id is not None else None,
            json.dumps(features, ensure_ascii=False), now(),
        )
        connection.execute("INSERT OR REPLACE INTO action_value_examples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
        export_lines.append(json.dumps({
            "snapshot_id": values[0], "event_slug": values[1], "observed_at": values[2],
            "outcome": values[3], "resolution_label": values[4], "entry_price": price,
            "notional_usdc": notional, "entry_fee_usdc": fee,
            "net_pnl_if_buy_resolution": net_pnl, "optimal_action": values[8],
            "model_action": values[9], "actual_realized_pnl_usdc": values[14], "features": features,
        }, ensure_ascii=False))
        written += 1
        if written % 500 == 0:
            connection.commit()
    connection.commit()
    events = int(connection.execute("SELECT COUNT(DISTINCT event_slug) FROM action_value_examples").fetchone()[0])
    connection.execute("PRAGMA optimize")
    connection.close()
    output = output or settings.ACTION_VALUE_DATASET_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(export_lines) + ("\n" if export_lines else ""), encoding="utf-8")
    return {"rows": written, "events": events, "skipped": skipped}
