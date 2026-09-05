"""Сравнение CLOSE сейчас с HOLD до resolution для удерживаемой стороны."""

from __future__ import annotations

import app_config as settings
import joblib
from dataclasses import asdict
from functools import lru_cache

from polybot.models.exit_features import feature_map, vector
from polybot.trading.fees import state_fee_usdc
from polybot.trading.policy import MarketState, PositionState


def _calibrated_held_probability(artifact, model_input) -> float:
    """Калиброванная вероятность победы уже удерживаемого контракта."""
    import math
    import numpy as np

    raw = float(np.clip(artifact["held_probability_model"].predict_proba([model_input])[0, 1], 1e-6, 1 - 1e-6))
    calibrator = artifact.get("held_probability_calibrator")
    if calibrator is None:
        return raw
    return float(calibrator.predict_proba([[math.log(raw / (1.0 - raw))]])[0, 1])


@lru_cache(maxsize=1)
def _artifact():
    if not settings.EXIT_MODEL_ARTIFACT_PATH.exists():
        return None
    artifact = joblib.load(settings.EXIT_MODEL_ARTIFACT_PATH)
    gate = artifact.get("report", {}).get("promotion_gate") if isinstance(artifact, dict) else None
    return artifact if not gate or bool(gate.get("passed")) else None


def compare(state: MarketState, position: PositionState, held_probability: float) -> dict[str, float | bool]:
    bid = state.up_bid if position.outcome == "Up" else state.down_bid
    if bid is None or position.shares <= 0:
        return {"close_pnl": float("-inf"), "hold_pnl": 0.0, "close_advantage": float("-inf"), "exit": False}
    close_pnl = position.shares * bid - state_fee_usdc(state, position.shares, bid) - position.cost_usdc
    hold_pnl = position.shares * held_probability - position.cost_usdc
    advantage = close_pnl - hold_pnl
    learned_advantage = None
    artifact = _artifact() if settings.EXIT_VALUE_ENABLED else None
    if artifact:
        features = artifact.get("features", [])
        if "oriented_distance_to_target_pct" in features:
            values = feature_map(
                asdict(state), position.outcome, bid, position.shares, position.cost_usdc,
                position.exit_features,
            )
            model_input = vector(values, features)
        else:
            # Совместимость с сохранённой моделью v1 до безопасного promotion v2.
            model_input = [
                float(state.distance_to_target_pct or 0), float(state.remaining_seconds),
                float(state.realized_volatility_60s_pct), float(bid), float(position.shares),
                float(position.cost_usdc),
            ]
        if "held_probability_model" in artifact:
            learned_probability = _calibrated_held_probability(artifact, model_input)
            learned_hold_pnl = position.shares * learned_probability - position.cost_usdc
            learned_advantage = close_pnl - learned_hold_pnl
            advantage_threshold = float(artifact.get("advantage_threshold", settings.EXIT_VALUE_MARGIN_USDC))
            probability_threshold = None
            model_exit = learned_advantage >= advantage_threshold
            advantage = learned_advantage
        elif "close_classifier" in artifact:
            learned_probability = float(artifact["close_classifier"].predict_proba([model_input])[0, 1])
            learned_advantage = float(artifact["advantage_model"].predict([model_input])[0])
            probability_threshold = float(artifact.get("probability_threshold", 1.01))
            advantage_threshold = float(artifact.get("advantage_threshold", settings.EXIT_VALUE_MARGIN_USDC))
            model_exit = learned_probability >= probability_threshold and learned_advantage >= advantage_threshold
            advantage = learned_advantage
        else:
            learned_probability = None
            probability_threshold = None
            advantage_threshold = settings.EXIT_VALUE_MARGIN_USDC
            learned_advantage = float(artifact["model"].predict([model_input])[0])
            advantage = 0.5 * advantage + 0.5 * learned_advantage
            model_exit = advantage >= settings.EXIT_VALUE_MARGIN_USDC
    else:
        learned_probability = None
        probability_threshold = None
        advantage_threshold = min(
            settings.EXIT_FALLBACK_MIN_ADVANTAGE_USDC,
            max(settings.EXIT_VALUE_MARGIN_USDC,
                position.cost_usdc * settings.EXIT_FALLBACK_MIN_ADVANTAGE_FRACTION),
        )
        model_exit = advantage >= advantage_threshold
    return {
        "close_pnl": close_pnl, "hold_pnl": hold_pnl, "close_advantage": advantage,
        "learned_close_advantage": learned_advantage,
        "learned_close_probability": learned_probability,
        "close_probability_threshold": probability_threshold,
        "close_advantage_threshold": advantage_threshold,
        "exit": model_exit,
    }
