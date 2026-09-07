"""Причинный shadow-бэктест ARIMA/GARCH на завершённых BTC 5m событиях."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from polybot.models.timeseries_challenger import forecast


def terminal_up_probability(arima_price: float, target_price: float, sigma_per_step: float,
                            horizon_steps: int) -> float:
    """Преобразует прогноз цены и GARCH-риск в P(close >= Price to Beat)."""
    sigma = max(float(sigma_per_step) * math.sqrt(max(1, int(horizon_steps))), 1e-7)
    z_score = math.log(max(arima_price, 1e-9) / max(target_price, 1e-9)) / sigma
    return float(min(0.999, max(0.001, NormalDist().cdf(z_score))))


def _metric_block(rows: list[dict[str, Any]], probability_key: str) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    truth = np.asarray([int(row["label_up"]) for row in rows])
    probability = np.asarray([float(row[probability_key]) for row in rows])
    prediction = probability >= 0.5
    return {
        "events": len(rows),
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "roc_auc": float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
        "pr_auc": float(average_precision_score(truth, probability)) if len(set(truth)) == 2 else None,
        "brier": float(brier_score_loss(truth, probability)),
        "predicted_up": int(prediction.sum()), "predicted_down": int((~prediction).sum()),
        "actual_up": int(truth.sum()), "actual_down": int((1 - truth).sum()),
    }


def run(connection: sqlite3.Connection, max_events: int = 180,
        checkpoints: tuple[int, ...] = (60, 150, 240)) -> dict[str, Any]:
    """Оценивает модели только по данным, доступным к каждому checkpoint."""
    connection.row_factory = sqlite3.Row
    events = connection.execute(
        """SELECT e.event_slug,e.target_price,MAX(t.label) label_up
           FROM event_targets e JOIN training_examples t ON t.event_slug=e.event_slug AND t.outcome='Up'
           WHERE e.target_price>0 GROUP BY e.event_slug,e.target_price
           ORDER BY CAST(substr(e.event_slug,length('btc-updown-5m-')+1) AS INTEGER) DESC LIMIT ?""",
        (max_events,),
    ).fetchall()
    records: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    for event in reversed(events):
        slug = str(event["event_slug"])
        try:
            start = int(slug.rsplit("-", 1)[-1])
        except ValueError:
            continue
        target = float(event["target_price"])
        for checkpoint in checkpoints:
            cutoff = start + checkpoint
            rows = connection.execute(
                """SELECT source,collected_at,price FROM external_prices
                   WHERE source IN ('bybit','okx') AND collected_at>=? AND collected_at<=?
                   ORDER BY collected_at""",
                (datetime.fromtimestamp(start - 1200, UTC).isoformat(),
                 datetime.fromtimestamp(cutoff, UTC).isoformat()),
            ).fetchall()
            # Пятиисекундные causal close-точки из медианы доступных источников.
            buckets: dict[int, list[float]] = {}
            for row in rows:
                timestamp = int(datetime.fromisoformat(str(row["collected_at"])).timestamp())
                buckets.setdefault(timestamp // 5, []).append(float(row["price"]))
            prices = [float(np.median(buckets[key])) for key in sorted(buckets)]
            if len(prices) < 40:
                failures["insufficient_prices"] = failures.get("insufficient_prices", 0) + 1
                continue
            horizon = max(1, math.ceil((300 - checkpoint) / 5))
            try:
                prediction = forecast(prices[-360:], horizon_steps=horizon)
            except Exception as error:  # isolated candidate failure must not affect trading
                key = type(error).__name__
                failures[key] = failures.get(key, 0) + 1
                continue
            p_up = terminal_up_probability(
                prediction.arima_price, target, prediction.garch_sigma_per_step, horizon,
            )
            current_probability = terminal_up_probability(
                prices[-1], target, prediction.garch_sigma_per_step, horizon,
            )
            market = connection.execute(
                """SELECT midpoint,best_bid,best_ask FROM market_snapshots
                   WHERE event_slug=? AND outcome='Up' AND collected_at<=?
                   ORDER BY collected_at DESC LIMIT 1""",
                (slug, datetime.fromtimestamp(cutoff, UTC).isoformat()),
            ).fetchone()
            market_probability = None
            if market:
                market_probability = market["midpoint"]
                if market_probability is None and market["best_bid"] is not None and market["best_ask"] is not None:
                    market_probability = (float(market["best_bid"]) + float(market["best_ask"])) / 2
            records.append({
                "event_slug": slug, "checkpoint_seconds": checkpoint,
                "label_up": int(event["label_up"]), "target_price": target,
                "current_price": prices[-1], "arima_price": prediction.arima_price,
                "garch_sigma_per_step": prediction.garch_sigma_per_step,
                "arima_garch_p_up": p_up, "current_price_p_up": current_probability,
                "market_p_up": float(market_probability) if market_probability is not None else None,
            })
    by_checkpoint: dict[str, Any] = {}
    for checkpoint in checkpoints:
        subset = [row for row in records if row["checkpoint_seconds"] == checkpoint]
        market_subset = [row for row in subset if row["market_p_up"] is not None]
        by_checkpoint[str(checkpoint)] = {
            "arima_garch": _metric_block(subset, "arima_garch_p_up"),
            "current_price_garch": _metric_block(subset, "current_price_p_up"),
            "polymarket_midpoint": _metric_block(market_subset, "market_p_up"),
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(), "status": "SHADOW_ONLY",
        "method": "causal per-event checkpoints; official openPrice target; future resolution only as label",
        "checkpoints_seconds_after_open": list(checkpoints), "events_requested": len(events),
        "predictions": len(records), "failures": failures, "metrics": by_checkpoint,
        "promotion_gate": "ARIMA/GARCH may become an entry feature only after positive incremental temporal OOS value",
    }
