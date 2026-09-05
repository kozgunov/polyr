"""Создаёт неизменяемый пакет данных для безопасного дообучения всех моделей."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import pyarrow.parquet as pq

from polybot.models.action_value_dataset import build as build_action_values
from polybot.models.counterfactual_actions import build as build_counterfactuals
from polybot.models.event_history import context as history_context, summaries_from_connection
from polybot.models.exit_sequence_dataset import build as build_exit_sequences
from polybot.models.sequence_dataset import build as build_sequences


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(source, timeout=60)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection, pages=4096)
    finally:
        destination_connection.close()
        source_connection.close()


def _event_splits(connection: sqlite3.Connection) -> tuple[dict[str, str], dict[str, Any]]:
    events = [str(row[0]) for row in connection.execute(
        "SELECT DISTINCT event_slug FROM training_examples ORDER BY event_slug"
    )]
    train_end = max(1, int(len(events) * 0.70))
    validation_start = min(len(events), train_end + 2)
    validation_end = max(validation_start, int(len(events) * 0.85))
    test_start = min(len(events), validation_end + 2)
    groups = {
        "train": events[:train_end],
        "validation": events[validation_start:validation_end],
        "test": events[test_start:],
        "purged": [*events[train_end:validation_start], *events[validation_end:test_start]],
    }
    mapping = {event: split for split, values in groups.items() for event in values}
    return mapping, {
        "method": "chronological event split with two-event purge at both boundaries",
        "counts": {name: len(values) for name, values in groups.items()},
        "events": groups,
        "overlap": 0,
    }


def _compact_features(features: dict[str, Any], history: dict[str, float]) -> dict[str, Any]:
    names = (
        "target_price", "reference_price", "distance_to_target_pct", "remaining_seconds",
        "best_bid", "best_ask", "midpoint", "spread", "best_bid_size", "best_ask_size",
        "bybit_to_target_pct", "okx_to_target_pct", "pyth_to_target_pct",
        "realized_volatility_60s_pct", "distance_time_score",
        "target_momentum_15s_pct", "target_momentum_30s_pct", "target_momentum_60s_pct",
    )
    return {**{name: features.get(name) for name in names}, **history}


def _write_qwen_entry(connection: sqlite3.Connection, root: Path, splits: dict[str, str]) -> dict[str, Any]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    query = connection.execute(
        """SELECT event_slug,observed_at,snapshot_id,action,outcome,candidate_price,
                  candidate_notional_usdc,limit_level,filled,target_net_pnl_usdc
           FROM counterfactual_action_examples WHERE phase='entry'
           ORDER BY event_slug,observed_at"""
    )
    for row in query:
        slug, observed = str(row[0]), str(row[1])
        elapsed = max(0, int(datetime.fromisoformat(observed).timestamp()) - int(slug.rsplit("-", 1)[-1]))
        key = (slug, elapsed // settings.COUNTERFACTUAL_SAMPLE_SECONDS)
        item = {
            "event_slug": slug, "observed_at": observed, "snapshot_id": int(row[2]),
            "action": str(row[3]), "outcome": row[4], "limit_price": row[5],
            "notional_usdc": float(row[6]), "limit_level": row[7], "filled": int(row[8]),
            "net_pnl_target": float(row[9]),
        }
        current = best.get(key)
        if current is None or item["net_pnl_target"] > current["net_pnl_target"]:
            best[key] = item
    summaries = summaries_from_connection(connection)
    history_cache: dict[str, dict[str, float]] = {}
    handles = {name: (root / f"qwen_entry_{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    counts = {name: Counter() for name in handles}
    try:
        for item in sorted(best.values(), key=lambda value: (value["event_slug"], value["observed_at"])):
            split = splits.get(item["event_slug"])
            if split not in handles:
                continue
            raw = connection.execute(
                "SELECT features_json FROM training_examples WHERE snapshot_id=?", (item["snapshot_id"],),
            ).fetchone()
            if not raw:
                continue
            features = json.loads(str(raw[0]))
            if item["event_slug"] not in history_cache:
                history_cache[item["event_slug"]] = history_context(
                    summaries, item["event_slug"], item["observed_at"], settings.EVENT_HISTORY_LIVE_WINDOWS,
                )
            target = {
                "action": item["action"], "direction": item["outcome"],
                "limit_level": item["limit_level"], "limit_price": item["limit_price"],
                "notional_usdc": item["notional_usdc"], "filled": item["filled"],
                "net_pnl_target": item["net_pnl_target"],
                "reason": "best counterfactual net PnL after fee and realistic GTD fill",
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "BTC Up/Down 5m. Choose WAIT or one limit BUY, its side and size $1..$10. Maximize long-run net PnL after fees; return JSON."},
                    {"role": "user", "content": json.dumps({"event": item["event_slug"], "observed_at": item["observed_at"], "features": _compact_features(features, history_cache[item["event_slug"]])}, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
                "event_slug": item["event_slug"], "split": split,
                "label_source": "counterfactual_limit_fill_then_resolution",
            }
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split][item["action"]] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {name: dict(values) for name, values in counts.items()}


def _write_qwen_exit(exit_path: Path, root: Path, splits: dict[str, str]) -> dict[str, Any]:
    rows = pq.read_table(exit_path).to_pylist()
    handles = {name: (root / f"qwen_exit_{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    counts = {name: Counter() for name in handles}
    try:
        for row in rows:
            split = splits.get(str(row["event_slug"]))
            if split not in handles or int(float(row["seconds_in_position"])) % 5:
                continue
            features = {name: row.get(name) for name in (
                "outcome", "seconds_in_position", "remaining_seconds", "current_bid", "average_price",
                "marked_return", "oriented_distance_to_target_pct", "momentum_bid_3ticks",
                "momentum_bid_15s", "momentum_bid_30s", "momentum_bid_60s",
                "bid_slope_15s", "bid_slope_30s", "target_distance_available",
                "momentum_distance_3ticks", "peak_bid_since_entry", "trough_bid_since_entry",
                "drawdown_from_peak", "recovery_from_trough", "seconds_since_peak",
                "maximum_favorable_excursion", "maximum_adverse_excursion",
                "spread", "shares", "original_cost_usdc",
            )}
            pnl_surface = {
                key: row.get(key) for key in row if key.startswith("pnl_exit_")
            }
            reward_surface = {
                key: row.get(key) for key in row if key.startswith("reward_exit_")
            }
            target = {
                "action": row["optimal_exit_action"],
                "exit_fraction": row["optimal_exit_fraction"],
                "pnl_by_exit_fraction": pnl_surface,
                "reward_by_exit_fraction": reward_surface,
                "minimum_acceptable_net_pnl_usdc": settings.MIN_ACCEPTABLE_NET_PNL_USDC,
                "hold_pnl_usdc": row["hold_pnl_usdc"],
                "reason": "counterfactual net PnL versus holding remainder to official resolution",
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "For an open BTC 5m position choose HOLD, PARTIAL_CLOSE, or CLOSE. Preserve upside but reduce downside; return JSON."},
                    {"role": "user", "content": json.dumps(features, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
                "event_slug": row["event_slug"], "split": split,
                "label_source": "exit_fraction_counterfactual_to_resolution",
            }
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split][str(row["optimal_exit_action"])] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {name: dict(values) for name, values in counts.items()}


def _write_qwen_direction(connection: sqlite3.Connection, root: Path, splits: dict[str, str]) -> dict[str, Any]:
    """Балансирует обучение направления по событиям, не смешивая его с value/sizing."""
    summaries = summaries_from_connection(connection)
    history_cache: dict[str, dict[str, float]] = {}
    handles = {name: (root / f"qwen_direction_{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    counts = {name: Counter() for name in handles}
    selected_buckets: set[tuple[str, int]] = set()
    try:
        rows = connection.execute(
            """SELECT event_slug,observed_at,label,features_json FROM training_examples
               WHERE outcome='Up' ORDER BY event_slug,observed_at"""
        )
        for slug, observed, label, raw_features in rows:
            slug, observed = str(slug), str(observed)
            split = splits.get(slug)
            if split not in handles:
                continue
            elapsed = max(0, int(datetime.fromisoformat(observed).timestamp()) - int(slug.rsplit("-", 1)[-1]))
            key = (slug, min(9, elapsed // 30))
            if key in selected_buckets:
                continue
            selected_buckets.add(key)
            features = json.loads(str(raw_features))
            if slug not in history_cache:
                history_cache[slug] = history_context(
                    summaries, slug, observed, settings.EVENT_HISTORY_LIVE_WINDOWS,
                )
            direction = "Up" if int(label) else "Down"
            payload = {
                "messages": [
                    {"role": "system", "content": "Estimate the official BTC Up/Down 5m resolution direction from causal market data. Do not choose order price or size. Return JSON direction and a short evidence-based explanation; confidence is calibrated outside the LLM."},
                    {"role": "user", "content": json.dumps({"event": slug, "observed_at": observed, "features": _compact_features(features, history_cache[slug])}, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps({"direction": direction, "label_source": "official_resolution", "confidence_policy": "external_calibrator"}, ensure_ascii=False)},
                ],
                "event_slug": slug, "split": split, "resolution_up": int(label),
                "label_source": "official_resolution_direction_only",
            }
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split][direction] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {name: dict(values) for name, values in counts.items()}


def enrich_direction_dataset(root: Path) -> dict[str, Any]:
    """Добавляет безопасный direction-only QLoRA набор в уже готовый bundle."""
    connection = sqlite3.connect(root / "training_snapshot.sqlite3")
    try:
        split_map, _ = _event_splits(connection)
        report = _write_qwen_direction(connection, root, split_map)
    finally:
        connection.close()
    preparation_path = root / "preparation_report.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    action_counts = preparation.get("qwen_entry_actions", {})
    total_buys = sum(
        int(counts.get("BUY_UP", 0)) + int(counts.get("BUY_DOWN", 0))
        for counts in action_counts.values()
    )
    down_buys = sum(int(counts.get("BUY_DOWN", 0)) for counts in action_counts.values())
    preparation["qwen_direction"] = report
    preparation["qwen_action_dataset_gate"] = {
        "passed": bool(total_buys >= 1000 and min(down_buys, total_buys - down_buys) >= 500),
        "reason": "direct action SFT is blocked when economically optimal BUY labels collapse to one side",
        "buy_up": total_buys - down_buys, "buy_down": down_buys,
        "recommended_task": "direction",
    }
    for split in ("train", "validation", "test"):
        path = root / f"qwen_direction_{split}.jsonl"
        preparation.setdefault("sha256", {})[path.name] = _sha256(path)
    preparation["next_commands"][1] = (
        f'python служебные_скрипты/train_qwen_qlora.py --bundle "{root}" --task direction'
    )
    preparation_path.write_text(json.dumps(preparation, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"qwen_direction": report, "action_gate": preparation["qwen_action_dataset_gate"]}


def prepare(output_root: Path | None = None) -> dict[str, Any]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = output_root or settings.PREPARED_TRAINING_DIR / f"training_bundle_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "training_snapshot.sqlite3"
    _snapshot(settings.DATABASE_PATH, snapshot)
    action_report = build_action_values(snapshot, root / "action_value.jsonl")
    counterfactual_report = build_counterfactuals(
        snapshot, root / "counterfactual_entry.jsonl", root / "counterfactual_entry.parquet",
    )
    sequence_path = root / "entry_sequences.parquet"
    exit_path = root / "exit_sequences.parquet"
    sequence_report = build_sequences(snapshot, sequence_path, root / "entry_sequence_manifest.json")
    exit_report = build_exit_sequences(snapshot, exit_path, root / "exit_sequence_manifest.json")
    connection = sqlite3.connect(snapshot)
    try:
        split_map, split_report = _event_splits(connection)
        entry_llm = _write_qwen_entry(connection, root, split_map)
        direction_llm = _write_qwen_direction(connection, root, split_map)
        exit_llm = _write_qwen_exit(exit_path, root, split_map)
        labels = dict(connection.execute(
            "SELECT outcome||':'||label,COUNT(*) FROM training_examples GROUP BY outcome,label"
        ).fetchall())
    finally:
        connection.close()
    config = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "event_budget_usdc": settings.PAPER_MAX_EVENT_EXPOSURE_USDC,
        "entry_sizes_usdc": list(settings.COUNTERFACTUAL_ENTRY_NOTIONALS_USDC),
        "history_windows": list(settings.EVENT_HISTORY_LIVE_WINDOWS),
        "tasks": {
            "direction": "calibrated P(Up/Down at official resolution)",
            "entry_value": "P(fill) * E(net PnL | fill) including market fee",
            "sizing": "argmax net PnL over limit price and $1..$10",
            "exit": "value surface for HOLD/20/40/60/80/100 percent close",
        },
        "validation": split_report,
        "promotion": {
            "candidate_only": True, "automatic_production_replacement": False,
            "minimum_oos_events": 500, "minimum_trades_each_direction": 50,
            "pnl_ci95_lower_must_be_positive": True, "shadow_events_before_paper": 100,
        },
    }
    (root / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [
        root / "action_value.jsonl", root / "counterfactual_entry.parquet", sequence_path, exit_path,
        root / "qwen_entry_train.jsonl", root / "qwen_entry_validation.jsonl", root / "qwen_entry_test.jsonl",
        root / "qwen_exit_train.jsonl", root / "qwen_exit_validation.jsonl", root / "qwen_exit_test.jsonl",
    ]
    report = {
        "status": "ready", "root": str(root), "snapshot": str(snapshot),
        "labels": labels, "action_value": action_report, "counterfactual_entry": counterfactual_report,
        "entry_sequences": sequence_report, "exit_sequences": exit_report,
        "qwen_entry_actions": entry_llm, "qwen_direction": direction_llm, "qwen_exit_actions": exit_llm,
        "splits": split_report["counts"],
        "sha256": {path.name: _sha256(path) for path in artifacts if path.exists()},
        "next_commands": [
            f"python дообучить_модели.py --bundle \"{root}\"",
            f"python служебные_скрипты/train_qwen_qlora.py --bundle \"{root}\" --task direction",
        ],
    }
    total_buys = sum(values.get("BUY_UP", 0) + values.get("BUY_DOWN", 0) for values in entry_llm.values())
    down_buys = sum(values.get("BUY_DOWN", 0) for values in entry_llm.values())
    report["qwen_action_dataset_gate"] = {
        "passed": bool(total_buys >= 1000 and min(down_buys, total_buys - down_buys) >= 500),
        "reason": "direct action SFT is blocked when economically optimal BUY labels collapse to one side",
        "buy_up": total_buys - down_buys, "buy_down": down_buys,
        "recommended_task": "direction",
    }
    (root / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (settings.PREPARED_TRAINING_DIR / "latest.txt").write_text(str(root), encoding="utf-8")
    return report
