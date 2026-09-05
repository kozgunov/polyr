"""Единое построение признаков для обучения и применения exit-value модели."""

from __future__ import annotations

import math
from typing import Any, Mapping


FEATURES = [
    "oriented_distance_to_target_pct",
    "remaining_seconds",
    "remaining_fraction",
    "realized_volatility_60s_pct",
    "current_bid",
    "average_price",
    "marked_return",
    "cost_usdc",
    "source_disagreement_pct",
    "sharp_move_pct",
    "agreement",
    "held_spread",
    "log_held_bid_size",
    "log_held_ask_size",
    "oriented_momentum_15s_pct",
    "oriented_momentum_30s_pct",
    "oriented_momentum_60s_pct",
    "held_side_currently_winning",
]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def feature_map(
    state: Mapping[str, Any], outcome: str, current_bid: float | None,
    shares: float, cost_usdc: float, runtime_features: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Возвращает признаки, доступные строго в момент принятия решения."""
    direction = 1.0 if outcome == "Up" else -1.0
    distance = _number(state.get("distance_to_target_pct"))
    oriented_distance = direction * distance
    remaining = max(0.0, _number(state.get("remaining_seconds")))
    bid = max(0.0, _number(current_bid))
    shares = max(0.0, _number(shares))
    cost = max(0.0, _number(cost_usdc))
    average_price = cost / shares if shares > 0 else 0.0
    marked_return = bid / average_price - 1.0 if average_price > 0 else 0.0
    book = (state.get("book_json") or {}).get(outcome, {}) or {}
    lags = state.get("target_distance_lags_pct") or {}

    values = {
        "oriented_distance_to_target_pct": oriented_distance,
        "remaining_seconds": remaining,
        "remaining_fraction": min(1.0, remaining / 300.0),
        "realized_volatility_60s_pct": _number(state.get("realized_volatility_60s_pct")),
        "current_bid": bid,
        "average_price": average_price,
        "marked_return": marked_return,
        "cost_usdc": cost,
        "source_disagreement_pct": _number(state.get("source_disagreement_pct")),
        "sharp_move_pct": _number(state.get("sharp_move_pct")),
        "agreement": _number(state.get("agreement")),
        "held_spread": _number(book.get("spread")),
        "log_held_bid_size": math.log1p(max(0.0, _number(book.get("best_bid_size")))),
        "log_held_ask_size": math.log1p(max(0.0, _number(book.get("best_ask_size")))),
        "held_side_currently_winning": float(oriented_distance >= 0.0),
    }
    for seconds in (15, 30, 60):
        lag = _number(lags.get(str(seconds), lags.get(seconds, distance)), distance)
        values[f"oriented_momentum_{seconds}s_pct"] = direction * (distance - lag)
    # Последовательная exit-модель обучается на истории позиции. Эти значения
    # рассчитываются движком только из снимков, существовавших к моменту решения.
    for name, value in (runtime_features or {}).items():
        values[str(name)] = _number(value)
    return values


def vector(values: Mapping[str, float], features: list[str] | None = None) -> list[float]:
    return [_number(values.get(name)) for name in (features or FEATURES)]
