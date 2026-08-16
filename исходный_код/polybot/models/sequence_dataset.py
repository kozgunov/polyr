"""Строит переносимый Parquet sequence-датасет без зависимости от PyTorch/CUDA."""

from __future__ import annotations

import json
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import pyarrow as pa
import pyarrow.parquet as pq
from polybot.models.event_history import build_summaries, context as history_context, feature_names as history_feature_names


BASE_FEATURES = (
    "best_bid", "best_ask", "midpoint", "spread", "best_bid_size", "best_ask_size",
    "reference_price", "target_price", "distance_to_target_pct", "remaining_seconds",
    "realized_volatility_60s_pct", "distance_time_score", "target_momentum_15s_pct",
    "target_momentum_30s_pct", "target_momentum_60s_pct", "bybit_to_target_pct",
    "okx_to_target_pct", "pyth_to_target_pct",
)
FEATURES = (*BASE_FEATURES, *history_feature_names(settings.EVENT_HISTORY_LIVE_WINDOWS))


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if result == result else 0.0
    except (TypeError, ValueError):
        return 0.0


def build(database: Path = settings.DATABASE_PATH, output: Path = settings.SEQUENCE_DATASET_PATH) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT snapshot_id,event_slug,outcome,observed_at,label,features_json
           FROM training_examples ORDER BY event_slug,outcome,observed_at"""
    ).fetchall()
    summaries = build_summaries(rows)
    action_labels: dict[int, dict[str, Any]] = {}
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='action_value_examples'").fetchone():
        for row in connection.execute(
            """SELECT snapshot_id,outcome,entry_price,notional_usdc,net_pnl_if_buy_resolution,
                      optimal_action,model_executed
               FROM action_value_examples ORDER BY snapshot_id"""
        ):
            snapshot_id, outcome = int(row[0]), str(row[1]).lower()
            utility_key = "utility_down" if outcome == "down" else "utility_up"
            payload = action_labels.setdefault(snapshot_id, {"utility_hold": 0.0})
            payload.update({
                utility_key: _number(row[4]), "best_action": str(row[5]),
                "best_action_price": _number(row[2]), "best_action_notional": _number(row[3]),
                "best_limit_level": "resolution_value", "best_action_net_pnl": max(0.0, _number(row[4])),
            })
    # Честные метки исполнения: PAPER использует реалистичный стаканный симулятор,
    # LIVE — фактический matched_size. Метка относится к ближайшему снимку решения.
    snapshot_index: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        snapshot_index[(str(row["event_slug"]), str(row["outcome"]))].append((
            datetime.fromisoformat(str(row["observed_at"])).timestamp(), int(row["snapshot_id"]),
        ))
    for values in snapshot_index.values():
        values.sort()

    def attach_fill(event_slug: str, outcome: str, observed_at: str, filled: bool) -> None:
        candidates = snapshot_index.get((event_slug, outcome), [])
        if not candidates:
            return
        timestamp = datetime.fromisoformat(observed_at).timestamp()
        position = bisect_left(candidates, (timestamp, -1))
        nearest = min(candidates[max(0, position - 1):position + 1], key=lambda item: abs(item[0] - timestamp))
        if abs(nearest[0] - timestamp) <= 15:
            action_labels.setdefault(nearest[1], {})["fill_down" if outcome == "Down" else "fill_up"] = int(filled)

    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "paper_orders" in tables and "model_decisions" in tables:
        for row in connection.execute(
            """SELECT o.event_slug,o.action,d.observed_at,o.status
               FROM paper_orders o JOIN model_decisions d ON d.id=o.decision_id
               WHERE o.action IN ('BUY_UP','BUY_DOWN') AND COALESCE(o.execution_valid,1)=1"""
        ):
            attach_fill(str(row[0]), "Down" if str(row[1]).endswith("DOWN") else "Up", str(row[2]),
                        str(row[3]) in {"filled", "partial", "partially_filled", "partial_cancelled"})
    if "live_orders" in tables:
        for row in connection.execute(
            """SELECT event_slug,outcome,created_at,status,matched_size FROM live_orders
               WHERE side='BUY' AND COALESCE(execution_valid,1)=1"""
        ):
            attach_fill(str(row[0]), str(row[1]), str(row[2]), float(row[4] or 0) > 0)
    connection.close()

    grouped: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["event_slug"]), str(row["outcome"]))].append(row)
    records: list[dict[str, Any]] = []
    history_cache: dict[str, dict[str, float]] = {}
    sequence_steps = max(1, settings.SEQUENCE_PRIMARY_LENGTH_SECONDS // settings.SEQUENCE_SAMPLE_SECONDS)
    for (event_slug, outcome), event_rows in grouped.items():
        sampled: dict[int, sqlite3.Row] = {}
        start = int(event_slug.rsplit("-", 1)[-1])
        for row in event_rows:
            bucket = max(0, int((datetime.fromisoformat(str(row["observed_at"])).timestamp() - start) // settings.SEQUENCE_SAMPLE_SECONDS))
            sampled.setdefault(bucket, row)
        ordered = list(sampled.values())
        feature_rows = []
        for row in ordered:
            values = json.loads(str(row["features_json"]))
            if event_slug not in history_cache:
                history_cache[event_slug] = history_context(
                    summaries, event_slug, str(row["observed_at"]), settings.EVENT_HISTORY_LIVE_WINDOWS,
                )
            values.update(history_cache[event_slug])
            feature_rows.append(values)
        for end_index, row in enumerate(ordered):
            begin = max(0, end_index + 1 - sequence_steps)
            window = feature_rows[begin:end_index + 1]
            padding = sequence_steps - len(window)
            action = {
                "best_action": None, "best_action_price": None, "best_action_notional": None,
                "best_limit_level": None, "best_action_net_pnl": None, "best_action_filled": None,
                "utility_hold": None, "utility_up": None, "utility_down": None,
                "fill_hold": None, "fill_up": None, "fill_down": None,
                **action_labels.get(int(row["snapshot_id"]), {}),
            }
            record: dict[str, Any] = {
                "sequence_id": f"{event_slug}:{outcome}:{row['snapshot_id']}",
                "snapshot_id": int(row["snapshot_id"]), "event_slug": event_slug,
                "outcome": outcome, "observed_at": str(row["observed_at"]),
                "resolution_label": int(row["label"]), "sequence_steps": sequence_steps,
                "valid_steps": len(window), "padding_steps": padding, **action,
            }
            for feature in FEATURES:
                record[feature] = [0.0] * padding + [_number(item.get(feature)) for item in window]
            records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd", use_dictionary=True)
    temporary.replace(output)
    manifest = {
        "schema_version": 2, "built_at": datetime.now(UTC).isoformat(), "source_database": str(database),
        "output": str(output), "rows": len(records), "events": len({r["event_slug"] for r in records}),
        "features": list(FEATURES), "sample_seconds": settings.SEQUENCE_SAMPLE_SECONDS,
        "sequence_steps": sequence_steps, "context_seconds": settings.SEQUENCE_PRIMARY_LENGTH_SECONDS,
        "neighbor_event_windows": list(settings.EVENT_HISTORY_LIVE_WINDOWS),
        "leakage_policy": "only events with end_timestamp <= decision observed_at",
        "targets": ["resolution_label", "utility_hold", "utility_up", "utility_down", "fill_up", "fill_down"],
        "target_non_null_rows": {
            name: sum(record.get(name) is not None for record in records)
            for name in ("resolution_label", "utility_hold", "utility_up", "utility_down", "fill_up", "fill_down")
        },
        "warnings": [
            "fill_up/fill_down зарезервированы до накопления честных limit-order fill labels"
        ] if not any(record.get("fill_up") is not None or record.get("fill_down") is not None for record in records) else [],
    }
    settings.SEQUENCE_DATASET_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
