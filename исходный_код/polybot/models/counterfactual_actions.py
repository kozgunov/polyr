"""Контрфактические Q-метки действий входа: BUY_UP, BUY_DOWN и WAIT."""

from __future__ import annotations

import bisect
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import app_config as settings

from polybot.collectors.pipeline import now
from polybot.models.train_direction_model import FEATURE_NAMES, vector
from polybot.trading.fees import total_fee_usdc


SCHEMA = """
CREATE TABLE IF NOT EXISTS counterfactual_action_examples(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_id INTEGER NOT NULL,event_slug TEXT NOT NULL,observed_at TEXT NOT NULL,
  phase TEXT NOT NULL,action TEXT NOT NULL,outcome TEXT,
  candidate_price REAL,candidate_notional_usdc REAL NOT NULL,
  limit_level TEXT,filled INTEGER NOT NULL,fill_price REAL,fee_usdc REAL NOT NULL,
  target_net_pnl_usdc REAL NOT NULL,resolution_label INTEGER,
  label_source TEXT NOT NULL,features_json TEXT NOT NULL,built_at TEXT NOT NULL,
  UNIQUE(snapshot_id,phase,action,candidate_price,candidate_notional_usdc,limit_level)
);
CREATE INDEX IF NOT EXISTS idx_counterfactual_action_event_time
ON counterfactual_action_examples(event_slug,observed_at,phase,action);
"""

ACTION_FEATURE_NAMES = [
    *FEATURE_NAMES, "action_buy_up", "action_buy_down", "action_wait",
    "candidate_price", "candidate_notional_usdc", "limit_offset_from_ask",
]


def action_vector(slug: str, outcome: str, observed_at: str, features: dict[str, Any],
                  action: str, price: float | None, notional: float) -> list[float]:
    base = vector(slug, outcome, observed_at, features)
    ask = float(features.get("best_ask") or price or 0.0)
    candidate = float(price or 0.0)
    return [*base, float(action == "BUY_UP"), float(action == "BUY_DOWN"),
            float(action == "WAIT"), candidate, float(notional), candidate - ask]


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _fill(rows: list[dict[str, Any]], index: int, price: float) -> tuple[bool, float | None, bool]:
    current = rows[index]
    ask = current["features"].get("best_ask")
    if ask is not None and price >= float(ask):
        return True, float(ask), True
    deadline = _timestamp(current["observed_at"]) + settings.GTD_EFFECTIVE_LIFETIME_SECONDS
    for future in rows[index + 1:]:
        if _timestamp(future["observed_at"]) > deadline:
            break
        future_ask = future["features"].get("best_ask")
        if future_ask is not None and float(future_ask) <= price:
            return True, price, False
    return False, None, False


def build(path: Path = settings.DATABASE_PATH) -> dict[str, Any]:
    connection = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    connection.executescript(SCHEMA)
    source = [dict(row) for row in connection.execute(
        """SELECT snapshot_id,event_slug,outcome,observed_at,label,features_json
           FROM training_examples ORDER BY event_slug,outcome,observed_at"""
    )]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        row["features"] = json.loads(row.pop("features_json"))
        grouped[(str(row["event_slug"]), str(row["outcome"]))].append(row)
    # Один причинный снимок на интервал: сохраняем динамику, но не размножаем
    # почти одинаковые тики в сотни гигабайт контрфактических комбинаций.
    for key, rows in list(grouped.items()):
        sampled: dict[int, dict[str, Any]] = {}
        event_start = int(key[0].rsplit("-", 1)[-1])
        for row in rows:
            elapsed = max(0.0, _timestamp(str(row["observed_at"])) - event_start)
            bucket = int(elapsed // settings.COUNTERFACTUAL_SAMPLE_SECONDS)
            sampled.setdefault(bucket, row)
        grouped[key] = list(sampled.values())

    examples: list[dict[str, Any]] = []
    buy_by_event_time: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, outcome), rows in grouped.items():
        action = f"BUY_{outcome.upper()}"
        for index, row in enumerate(rows):
            features = row["features"]
            prices = {
                "ask": features.get("best_ask"),
                "midpoint": features.get("midpoint"),
                "bid": features.get("best_bid"),
            }
            for level in settings.COUNTERFACTUAL_LIMIT_LEVELS:
                raw_price = prices.get(level)
                if raw_price is None:
                    continue
                price = round(float(raw_price), 3)
                if not settings.PNL_DATASET_MIN_ENTRY_PRICE <= price <= settings.PNL_DATASET_MAX_ENTRY_PRICE:
                    continue
                filled, fill_price, taker = _fill(rows, index, price)
                for notional in settings.COUNTERFACTUAL_ENTRY_NOTIONALS_USDC:
                    fee = 0.0
                    pnl = 0.0
                    if filled and fill_price:
                        shares = float(notional) / fill_price
                        fee = total_fee_usdc(shares, fill_price, taker=taker)
                        pnl = shares * int(row["label"]) - float(notional) - fee
                    item = {
                        "snapshot_id": int(row["snapshot_id"]), "event_slug": str(row["event_slug"]),
                        "observed_at": str(row["observed_at"]), "phase": "entry", "action": action,
                        "outcome": outcome, "candidate_price": price,
                        "candidate_notional_usdc": float(notional), "limit_level": level,
                        "filled": int(filled), "fill_price": fill_price, "fee_usdc": fee,
                        "target_net_pnl_usdc": pnl, "resolution_label": int(row["label"]),
                        "label_source": "future_gtd_fill_then_resolution", "features": features,
                    }
                    examples.append(item)
                    buy_by_event_time[str(row["event_slug"])].append(item)

    # WAIT получает ценность лучшего доступного действия в ближайшем будущем,
    # а не искусственный ноль. Это Q-метка ожидания, рассчитанная только из будущего label.
    for (slug, outcome), rows in grouped.items():
        if outcome != "Up":
            continue
        candidates = sorted(buy_by_event_time.get(slug, []), key=lambda x: x["observed_at"])
        candidate_times = [_timestamp(item["observed_at"]) for item in candidates]
        for row in rows:
            t0 = _timestamp(str(row["observed_at"]))
            start = bisect.bisect_right(candidate_times, t0)
            end = bisect.bisect_right(candidate_times, t0 + settings.COUNTERFACTUAL_WAIT_LOOKAHEAD_SECONDS)
            future_value = max([0.0, *(float(item["target_net_pnl_usdc"]) for item in candidates[start:end])])
            examples.append({
                "snapshot_id": int(row["snapshot_id"]), "event_slug": slug,
                "observed_at": str(row["observed_at"]), "phase": "entry", "action": "WAIT",
                "outcome": "Up", "candidate_price": None, "candidate_notional_usdc": 0.0,
                "limit_level": "wait", "filled": 1, "fill_price": None, "fee_usdc": 0.0,
                "target_net_pnl_usdc": future_value, "resolution_label": int(row["label"]),
                "label_source": f"best_future_action_within_{settings.COUNTERFACTUAL_WAIT_LOOKAHEAD_SECONDS}s",
                "features": row["features"],
            })

    connection.execute("DELETE FROM counterfactual_action_examples")
    built_at = now()
    for index, item in enumerate(examples, start=1):
        connection.execute(
            """INSERT INTO counterfactual_action_examples(
               snapshot_id,event_slug,observed_at,phase,action,outcome,candidate_price,
               candidate_notional_usdc,limit_level,filled,fill_price,fee_usdc,
               target_net_pnl_usdc,resolution_label,label_source,features_json,built_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (item["snapshot_id"], item["event_slug"], item["observed_at"], item["phase"],
             item["action"], item["outcome"], item["candidate_price"], item["candidate_notional_usdc"],
             item["limit_level"], item["filled"], item["fill_price"], item["fee_usdc"],
             item["target_net_pnl_usdc"], item["resolution_label"], item["label_source"],
             "{}", built_at),
        )
        if index % 1000 == 0:
            connection.commit()
    connection.commit()
    connection.close()

    settings.COUNTERFACTUAL_ACTION_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_examples = [{key: value for key, value in item.items() if key != "features"} for item in examples]
    lines = [json.dumps(item, ensure_ascii=False) for item in export_examples]
    settings.COUNTERFACTUAL_ACTION_DATASET_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    parquet_status = "not_available"
    try:
        import pandas as pd
        frame = pd.DataFrame(export_examples)
        frame.to_parquet(settings.COUNTERFACTUAL_ACTION_PARQUET_PATH, index=False)
        parquet_status = "ok"
    except (ImportError, ModuleNotFoundError):
        pass
    return {"rows": len(examples), "events": len(buy_by_event_time),
            "buy_rows": sum(item["action"].startswith("BUY") for item in examples),
            "wait_rows": sum(item["action"] == "WAIT" for item in examples),
            "parquet": parquet_status}
