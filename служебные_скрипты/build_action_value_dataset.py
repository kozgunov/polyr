"""Строит action-value датасет: признаки на момент решения -> чистый PnL действия."""

from __future__ import annotations

import bisect
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

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


def legacy_main() -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    decisions: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        """SELECT id,event_slug,observed_at,action,confidence,predicted_up_probability,
                  expected_net_edge,executed FROM model_decisions ORDER BY event_slug,observed_at"""
    ):
        decisions[str(row["event_slug"])].append(row)
    decision_times = {
        slug: [datetime.fromisoformat(str(row["observed_at"])).timestamp() for row in rows]
        for slug, rows in decisions.items()
    }
    realized = {
        int(row[0]): float(row[1]) for row in connection.execute(
            """SELECT entry_decision_id,realized_pnl_usdc FROM paper_positions
               WHERE entry_decision_id IS NOT NULL AND realized_pnl_usdc IS NOT NULL"""
        )
    }
    examples = connection.execute(
        """SELECT snapshot_id,event_slug,outcome,observed_at,label,features_json
           FROM training_examples ORDER BY event_slug,observed_at"""
    ).fetchall()
    written = skipped = 0
    settings.ACTION_VALUE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_lines: list[str] = []
    for row in examples:
        features = json.loads(row["features_json"])
        ask = features.get("best_ask")
        if ask is None or not settings.PNL_DATASET_MIN_ENTRY_PRICE <= float(ask) <= settings.PNL_DATASET_MAX_ENTRY_PRICE:
            skipped += 1
            continue
        price = float(ask)
        notional = float(settings.PAPER_ENTRY_NOTIONAL_USDC)
        shares = notional / price
        fee = total_fee_usdc(shares, price)
        net_pnl = shares * int(row["label"]) - notional - fee
        slug = str(row["event_slug"])
        observed_ts = datetime.fromisoformat(str(row["observed_at"])).timestamp()
        nearest = None
        if slug in decisions:
            index = bisect.bisect_right(decision_times[slug], observed_ts) - 1
            if index >= 0 and observed_ts - decision_times[slug][index] <= 15:
                nearest = decisions[slug][index]
        payload = {
            "snapshot_id": int(row["snapshot_id"]), "event_slug": slug,
            "observed_at": str(row["observed_at"]), "outcome": str(row["outcome"]),
            "resolution_label": int(row["label"]), "entry_price": price,
            "notional_usdc": notional, "net_pnl_if_buy_resolution": net_pnl,
            "optimal_action": "BUY" if net_pnl > 0 else "WAIT",
            "model_action": str(nearest["action"]) if nearest else None,
            "model_confidence": float(nearest["confidence"]) if nearest else None,
            "model_predicted_up_probability": float(nearest["predicted_up_probability"]) if nearest and nearest["predicted_up_probability"] is not None else None,
            "model_expected_net_edge": float(nearest["expected_net_edge"]) if nearest and nearest["expected_net_edge"] is not None else None,
            "model_executed": int(nearest["executed"]) if nearest else 0,
            "actual_realized_pnl_usdc": realized.get(int(nearest["id"])) if nearest else None,
            "features": features,
        }
        connection.execute(
            """INSERT OR REPLACE INTO action_value_examples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (payload["snapshot_id"], slug, payload["observed_at"], payload["outcome"], payload["resolution_label"],
             price, notional, net_pnl, payload["optimal_action"], payload["model_action"], payload["model_confidence"],
             payload["model_predicted_up_probability"], payload["model_expected_net_edge"], payload["model_executed"],
             payload["actual_realized_pnl_usdc"], json.dumps(features, ensure_ascii=False), now()),
        )
        export_lines.append(json.dumps(payload, ensure_ascii=False))
        written += 1
        if written % 500 == 0:
            connection.commit()
    connection.commit()
    settings.ACTION_VALUE_DATASET_PATH.write_text("\n".join(export_lines) + "\n", encoding="utf-8")
    events = connection.execute("SELECT COUNT(DISTINCT event_slug) FROM action_value_examples").fetchone()[0]
    connection.close()
    print(json.dumps({"rows": written, "events": events, "skipped": skipped, "path": str(settings.ACTION_VALUE_DATASET_PATH)}, ensure_ascii=False, indent=2))


def main() -> None:
    # Единая реализация с тем же fee/net-PnL контрактом, что использует
    # автоматический цикл переобучения.
    from polybot.models.action_value_dataset import build
    print(json.dumps(build(settings.DATABASE_PATH), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
        print(f"all_done{Path(__file__).name}")
    except Exception:
        print(f"error_in_{Path(__file__).name}")
        raise
