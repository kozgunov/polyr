"""Честное event-level сравнение моделей на одном отложенном наборе BTC 5m."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.llm_runtime import infer as infer_llm
from polybot.models.llm_runtime import release_model
from polybot.models.model_policy import _llm_context, probability_up
from polybot.models.model_registry import MODEL_SPECS, model_is_ready
from polybot.trading.fees import net_buy_edge, total_fee_usdc
from polybot.trading.execution_simulator import limit_buy
from polybot.trading.policy import MarketState


def _events(path: Path, allowed: set[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    connection = sqlite3.connect(path)
    rows = connection.execute(
        "SELECT event_slug,outcome,observed_at,label,features_json FROM training_examples ORDER BY observed_at"
    ).fetchall()
    connection.close()
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for slug, outcome, observed_at, label, raw in rows:
        if slug in allowed:
            result[str(slug)][str(outcome)].append({
                "observed_at": str(observed_at), "label": int(label), "features": json.loads(raw),
            })
    return result


def _state(slug: str, up: dict[str, Any], down: dict[str, Any], starts: dict[str, float]) -> MarketState:
    timestamp = max(datetime.fromisoformat(up["observed_at"]), datetime.fromisoformat(down["observed_at"]))
    started = datetime.fromtimestamp(int(slug.rsplit("-", 1)[-1]), UTC)
    prices = {
        source: float(up["features"][f"{source}_price"])
        for source in ("bybit", "okx", "pyth") if up["features"].get(f"{source}_price")
    }
    returns = {source: (price / starts[source] - 1.0) * 100 for source, price in prices.items() if source in starts}
    values = list(prices.values())
    disagreement = (max(values) - min(values)) / float(np.median(values)) * 100 if len(values) >= 2 else 999.0
    books = {
        outcome: {key: row["features"].get(key) for key in ("best_bid", "best_ask", "best_bid_size", "best_ask_size", "midpoint", "spread")}
        for outcome, row in (("Up", up), ("Down", down))
    }
    elapsed = (timestamp - started).total_seconds()
    return MarketState(
        slug, timestamp.isoformat(), elapsed, 300.0 - elapsed, returns, prices, disagreement, 0.0,
        books["Up"]["best_bid"], books["Up"]["best_ask"], books["Down"]["best_bid"], books["Down"]["best_ask"], books,
        up["features"].get("target_price"), up["features"].get("reference_price"),
        up["features"].get("target_source"), timestamp.isoformat(),
        float(up["features"].get("realized_volatility_60s_pct") or 0.0),
        {str(lag): float(up["features"].get(f"distance_lag_{lag}s_pct") or 0.0) for lag in (15, 30, 60)},
    )


def _representative_states(path: Path) -> list[tuple[MarketState, int]]:
    artifact = joblib.load(settings.TRAINING_ARTIFACT_PATH)
    allowed = set(artifact.get("splits", {}).get("test_events", []))
    data = _events(path, allowed)
    result: list[tuple[MarketState, int]] = []
    for slug in sorted(data):
        up_rows, down_rows = data[slug].get("Up", []), data[slug].get("Down", [])
        if not up_rows or not down_rows:
            continue
        starts = {
            source: float(up_rows[0]["features"][f"{source}_price"])
            for source in ("bybit", "okx", "pyth") if up_rows[0]["features"].get(f"{source}_price")
        }
        pairs = list(zip(up_rows, down_rows, strict=False))
        eligible = []
        for up, down in pairs:
            state = _state(slug, up, down, starts)
            if settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN <= state.elapsed_seconds <= 300 - settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:
                eligible.append((abs(state.elapsed_seconds - 180.0), state, int(up["label"])))
        if eligible:
            _, state, label = min(eligible, key=lambda item: item[0])
            result.append((state, label))
    return result


def _export_labeled_dataset(states: list[tuple[MarketState, int]]) -> dict[str, Any]:
    records = []
    for state, label in states:
        records.append({
            "event_slug": state.event_slug, "observed_at": state.observed_at,
            "elapsed_seconds": state.elapsed_seconds, "remaining_seconds": state.remaining_seconds,
            "label_up": int(label), "winning_outcome": "Up" if label else "Down",
            "target_price": state.target_price, "reference_price": state.reference_price,
            "distance_to_target_pct": state.distance_to_target_pct,
            "up_bid": state.up_bid, "up_ask": state.up_ask,
            "down_bid": state.down_bid, "down_ask": state.down_ask,
            "source_disagreement_pct": state.source_disagreement_pct,
            "realized_volatility_60s_pct": state.realized_volatility_60s_pct,
            "state_json": json.dumps(state.as_dict(), ensure_ascii=False, sort_keys=True),
        })
    settings.OFFLINE_TOURNAMENT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    settings.OFFLINE_TOURNAMENT_DATASET_PATH.write_bytes(payload)
    parquet_status = "not_written"
    try:
        import pandas as pd
        pd.DataFrame(records).to_parquet(settings.OFFLINE_TOURNAMENT_PARQUET_PATH, index=False)
        parquet_status = "ok"
    except (ImportError, OSError, ValueError) as error:
        parquet_status = f"{type(error).__name__}: {error}"
    labels = [int(record["label_up"]) for record in records]
    return {
        "rows": len(records), "events": len({record["event_slug"] for record in records}),
        "up": sum(labels), "down": len(labels) - sum(labels),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "jsonl": str(settings.OFFLINE_TOURNAMENT_DATASET_PATH),
        "parquet": str(settings.OFFLINE_TOURNAMENT_PARQUET_PATH), "parquet_status": parquet_status,
        "first_observed_at": records[0]["observed_at"] if records else None,
        "last_observed_at": records[-1]["observed_at"] if records else None,
    }


def _financial_metrics(states: list[tuple[MarketState, int]], probabilities: list[float]) -> dict[str, Any]:
    pnls: list[float] = []
    total_fees = 0.0
    up_trades = down_trades = attempted = non_fills = 0
    for (state, label), p_up in zip(states, probabilities, strict=True):
        direction = "Up" if p_up >= 0.5 else "Down"
        confidence = max(p_up, 1.0 - p_up)
        ask = state.up_ask if direction == "Up" else state.down_ask
        probability = p_up if direction == "Up" else 1.0 - p_up
        if ask is None or confidence < settings.MIN_ENTRY_CONFIDENCE:
            continue
        if not settings.PAPER_MIN_ENTRY_PRICE <= ask <= settings.PAPER_MAX_ENTRY_PRICE:
            continue
        spread = float(state.book_json.get(direction, {}).get("spread") or 1.0)
        edge = net_buy_edge(probability, ask)
        execution_buffer = spread + ask * settings.ESTIMATED_SLIPPAGE_BPS / 10_000
        required_edge = max(settings.PAPER_MIN_ENTRY_NET_EDGE, settings.ENTRY_VALUE_SAFETY_MARGIN + execution_buffer)
        if edge < required_edge:
            continue
        shares = settings.PAPER_ENTRY_NOTIONAL_USDC / ask
        attempted += 1
        book = state.book_json.get(direction, {})
        fill = limit_buy(
            f"offline:{state.event_slug}:{direction}", ask, book.get("best_ask"),
            book.get("best_ask_size"), shares, spread,
        )
        if fill.filled_shares <= 0 or fill.filled_price is None:
            non_fills += 1
            continue
        shares = float(fill.filled_shares)
        fill_price = float(fill.filled_price)
        fee = total_fee_usdc(shares, fill_price, taker=True)
        won = label == int(direction == "Up")
        pnls.append(shares * int(won) - shares * fill_price - fee)
        total_fees += fee
        up_trades += int(direction == "Up")
        down_trades += int(direction == "Down")
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    equity = peak = max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl; peak = max(peak, equity); max_drawdown = max(max_drawdown, peak - equity)
    return {
        "trades": len(pnls),
        "coverage": len(pnls) / len(states) if states else 0.0,
        "net_pnl_usdc": sum(pnls),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "trade_win_rate": sum(value > 0 for value in pnls) / len(pnls) if pnls else 0.0,
        "attempted_orders": attempted, "non_fills": non_fills,
        "fill_rate": len(pnls) / attempted if attempted else 0.0,
        "fees_usdc": total_fees, "max_drawdown_usdc": max_drawdown,
        "expectancy_usdc": sum(pnls) / len(pnls) if pnls else 0.0,
        "up_trades": up_trades, "down_trades": down_trades,
    }


def _metrics(name: str, states: list[tuple[MarketState, int]], probabilities: list[float]) -> dict[str, Any]:
    labels = np.asarray([label for _, label in states], dtype=np.int8)
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    active = ~np.isclose(values, 0.5)
    predictions = (values[active] > 0.5).astype(np.int8)
    active_labels = labels[active]
    return {
        "name": name, "status": "ok", "events": len(labels),
        "accuracy": float(accuracy_score(active_labels, predictions)) if active.any() else None,
        "balanced_accuracy": float(balanced_accuracy_score(active_labels, predictions)) if active.any() else None,
        "directional_coverage": float(active.mean()),
        "abstention_rate": float(1.0 - active.mean()),
        "brier": float(brier_score_loss(labels, values)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, values)) if len(set(labels.tolist())) == 2 else None,
        **_financial_metrics(states, values.tolist()),
    }


def _llm_probabilities(key: str, states: list[tuple[MarketState, int]]) -> list[float]:
    probabilities = []
    limit = settings.GEMMA_OFFLINE_MAX_EVENTS if key == "gemma" else settings.QWEN_OFFLINE_MAX_EVENTS
    for state, _ in states[:limit]:
        signal = infer_llm(key, _llm_context(state, None), use_cache=False)
        direction, confidence = signal["direction"], float(signal["confidence"])
        probabilities.append(confidence if direction == "Up" else 1.0 - confidence if direction == "Down" else 0.5)
    release_model()
    return probabilities


def run(path: Path, skip_llm: bool = False, llm_models: tuple[str, ...] = ("qwen", "gemma"),
        preserve_existing: bool = False) -> dict[str, Any]:
    previous = {}
    if preserve_existing and settings.MODEL_COMPARISON_REPORT_PATH.exists():
        try:
            previous = json.loads(settings.MODEL_COMPARISON_REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    states = _representative_states(path)
    result: dict[str, Any] = {
        "protocol": "v5 locked labeled holdout; same event snapshot near t=180s; target-aware; fee/value gate; deterministic L2 execution simulation; no snapshot leakage",
        "events": len(states), "models": {},
        "dataset": _export_labeled_dataset(states),
    }
    numeric: dict[str, list[float]] = {}
    for key in ("catboost", "custom"):
        if not model_is_ready(key):
            result["models"][key] = {"name": MODEL_SPECS[key].name, "status": "not_installed"}
            continue
        values = [probability_up(state, key) for state, _ in states]
        numeric[key] = values
        result["models"][key] = _metrics(MODEL_SPECS[key].name, states, values)

    llm_values: dict[str, list[float]] = {}
    # Быструю Qwen считаем первой и сразу сохраняем, затем медленную Gemma.
    for key in llm_models:
        if skip_llm or not model_is_ready(key):
            reason = "LLM пропущена параметром" if skip_llm else "локальные веса не установлены"
            result["models"][key] = {"name": MODEL_SPECS[key].name, "status": "not_evaluated", "reason": reason}
            continue
        started = time.perf_counter()
        try:
            values = _llm_probabilities(key, states)
        except Exception as error:  # noqa: BLE001 - одна LLM не отменяет весь турнир
            release_model()
            result["models"][key] = {
                "name": MODEL_SPECS[key].name, "status": "error",
                "reason": f"{type(error).__name__}: {error}"[-1000:],
                "evaluation_seconds": time.perf_counter() - started,
            }
            settings.MODEL_COMPARISON_REPORT_PATH.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            continue
        elapsed = time.perf_counter() - started
        llm_states = states[: len(values)]
        llm_values[key] = values
        result["models"][key] = _metrics(MODEL_SPECS[key].name, llm_states, values)
        result["models"][key]["evaluation_seconds"] = elapsed
        result["models"][key]["seconds_per_event"] = elapsed / len(values) if values else None
        result["models"][key]["small_sample_warning"] = (
            f"LLM latency audit is limited to {len(values)} events and is not statistically significant"
        )
        settings.MODEL_COMPARISON_REPORT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    if preserve_existing:
        for key in ("qwen", "gemma"):
            if key not in result["models"] and key in previous.get("models", {}):
                result["models"][key] = previous["models"][key]

    # Честная общая таблица: каждая реально оценённая модель получает строго
    # те же события. Полный numeric holdout сохраняется отдельно выше.
    evaluated_probabilities = {**numeric, **llm_values}
    common_count = min((len(values) for values in evaluated_probabilities.values()), default=0)
    result["common_subset"] = {
        "events": common_count,
        "event_slugs": [state.event_slug for state, _ in states[:common_count]],
        "models": {
            key: _metrics(MODEL_SPECS[key].name, states[:common_count], values[:common_count])
            for key, values in evaluated_probabilities.items()
        },
        "statistically_sufficient": common_count >= 30,
        "warning": None if common_count >= 30 else "Общая LLM-выборка меньше 30 событий; это smoke comparison, не основание для LIVE.",
    }

    if preserve_existing and not llm_values and previous.get("common_subset"):
        result["common_subset"] = previous["common_subset"]

    if model_is_ready("qwen_lora"):
        result["models"]["qwen_lora"] = {
            "name": MODEL_SPECS["qwen_lora"].name,
            "status": "not_evaluated",
            "reason": "LoRA adapter требует отдельного безопасного runtime-профиля; не смешан с базовой Qwen.",
        }
    else:
        result["models"]["qwen_lora"] = {
            "name": MODEL_SPECS["qwen_lora"].name,
            "status": "not_installed",
            "reason": "adapter_config.json отсутствует",
        }

    if "custom" in numeric:
        for llm_key, values in llm_values.items():
            custom_values = numeric["custom"][: len(values)]
            consensus = []
            for numeric_p, llm_p in zip(custom_values, values, strict=True):
                numeric_direction, llm_direction = numeric_p >= 0.5, llm_p >= 0.5
                numeric_confidence = max(numeric_p, 1.0 - numeric_p)
                llm_confidence = max(llm_p, 1.0 - llm_p)
                if numeric_direction != llm_direction or min(numeric_confidence, llm_confidence) < settings.CONSENSUS_ENTRY_CONFIDENCE:
                    consensus.append(0.5)
                elif numeric_direction:
                    consensus.append(min(numeric_p, llm_p))
                else:
                    consensus.append(1.0 - min(1.0 - numeric_p, 1.0 - llm_p))
            key = f"consensus_{llm_key}_custom"
            result["models"][key] = _metrics(MODEL_SPECS[key].name, states[: len(consensus)], consensus)

    ranked = [
        (key, value) for key, value in result["models"].items()
        if value.get("status") == "ok" and value.get("events") == len(states)
        and value.get("trades", 0) >= 3
    ]
    ranked.sort(key=lambda item: (item[1]["net_pnl_usdc"], -item[1]["brier"]), reverse=True)
    result["ranking"] = [key for key, _ in ranked]
    result["llm_smoke_models"] = [
        key for key in ("qwen", "gemma")
        if result["models"].get(key, {}).get("status") == "ok"
        and result["models"][key].get("events", 0) < len(states)
    ]
    result["warning"] = "Малый holdout не доказывает будущую прибыльность; выбор делается по net PnL, Brier, drawdown и стабильности на новых событиях."
    settings.MODEL_COMPARISON_REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-models", nargs="*", choices=("qwen", "gemma"), default=("qwen", "gemma"))
    parser.add_argument("--preserve-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.db, args.skip_llm, tuple(args.llm_models), args.preserve_existing), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
