"""Рыночные режимы BTC 5m, вычисляемые только из доступных на текущем тике данных."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegimeThresholds:
    volatility_low: float
    volatility_high: float
    momentum_flat: float
    distance_near: float
    distance_far: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_thresholds(features: list[dict[str, Any]]) -> RegimeThresholds:
    volatility = np.asarray([abs(float(row.get("realized_volatility_60s_pct") or 0)) for row in features])
    momentum = np.asarray([abs(float(row.get("target_momentum_30s_pct") or 0)) for row in features])
    distance = np.asarray([abs(float(row.get("distance_to_target_pct") or 0)) for row in features])
    return RegimeThresholds(
        volatility_low=float(np.quantile(volatility, 0.33)),
        volatility_high=float(np.quantile(volatility, 0.67)),
        momentum_flat=float(np.quantile(momentum, 0.40)),
        distance_near=float(np.quantile(distance, 0.33)),
        distance_far=float(np.quantile(distance, 0.67)),
    )


def classify(features: dict[str, Any], thresholds: RegimeThresholds) -> dict[str, str]:
    volatility = abs(float(features.get("realized_volatility_60s_pct") or 0))
    momentum = float(features.get("target_momentum_30s_pct") or 0)
    distance = abs(float(features.get("distance_to_target_pct") or 0))
    remaining = max(0.0, min(300.0, float(features.get("remaining_seconds") or 0)))
    if abs(momentum) <= thresholds.momentum_flat:
        trend = "flat"
    else:
        trend = "uptrend" if momentum > 0 else "downtrend"
    volatility_band = "low" if volatility <= thresholds.volatility_low else "high" if volatility >= thresholds.volatility_high else "mid"
    distance_band = "near" if distance <= thresholds.distance_near else "far" if distance >= thresholds.distance_far else "mid"
    elapsed = 300.0 - remaining
    phase = "early" if elapsed < 100 else "late" if elapsed >= 200 else "middle"
    return {
        "trend": trend, "volatility": volatility_band,
        "phase": phase, "distance": distance_band,
        "expert": f"{trend}__{volatility_band}",
    }


def one_hot(regime: dict[str, str]) -> list[float]:
    values: list[float] = []
    for key, categories in (
        ("trend", ("downtrend", "flat", "uptrend")),
        ("volatility", ("low", "mid", "high")),
        ("phase", ("early", "middle", "late")),
        ("distance", ("near", "mid", "far")),
    ):
        values.extend(float(regime[key] == category) for category in categories)
    return values
