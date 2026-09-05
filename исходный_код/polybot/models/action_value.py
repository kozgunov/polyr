"""Отделяет вероятность исхода от денежной ценности BUY_UP/BUY_DOWN/WAIT."""

from __future__ import annotations

from functools import lru_cache

import app_config as settings
import joblib

from polybot.models.counterfactual_actions import action_vector
from polybot.models.train_direction_model import vector
from polybot.trading.fees import state_fee_usdc, total_fee_usdc
from polybot.trading.policy import MarketState


@lru_cache(maxsize=1)
def _artifact():
    if not settings.PNL_MODEL_ARTIFACT_PATH.exists():
        return None
    artifact = joblib.load(settings.PNL_MODEL_ARTIFACT_PATH)
    report = artifact.get("report", {})
    is_two_stage = "fill_model" in artifact and (
        "conditional_pnl_model" in artifact or "conditional_outcome_model" in artifact
    )
    # Непрошедший promotion gate артефакт остаётся кандидатом/shadow и никогда не управляет капиталом.
    if is_two_stage and not bool(report.get("promotion_gate", {}).get("passed")):
        return None
    if not is_two_stage and float(report.get("r2", float("-inf"))) < settings.ACTION_VALUE_MIN_R2:
        return None
    return artifact


@lru_cache(maxsize=1)
def _fill_artifact():
    """Загружает только валидированный P(fill), не активируя провалившую gate value-модель."""
    if not settings.FILL_PROBABILITY_MODEL_ENABLED or not settings.PNL_MODEL_ARTIFACT_PATH.exists():
        return None
    artifact = joblib.load(settings.PNL_MODEL_ARTIFACT_PATH)
    metrics = artifact.get("report", {}).get("fill_model", {}) if isinstance(artifact, dict) else {}
    if (
        not isinstance(artifact, dict)
        or "fill_model" not in artifact
        or float(metrics.get("roc_auc", 0.0)) < settings.FILL_PROBABILITY_MIN_ROC_AUC
        or float(metrics.get("brier", 1.0)) > settings.FILL_PROBABILITY_MAX_BRIER
    ):
        return None
    return artifact


def _predict_fill(artifact, row) -> float:
    import numpy as np

    probability = float(artifact["fill_model"].predict_proba([row])[0, 1])
    calibrator = artifact.get("fill_calibrator")
    if calibrator is not None:
        raw = float(np.clip(probability, 1e-6, 1 - 1e-6))
        logit = np.log(raw / (1.0 - raw))
        probability = float(calibrator.predict_proba([[logit]])[0, 1])
    return max(0.0, min(1.0, probability))


def expected_pnl(state: MarketState, outcome: str, probability: float, price: float, notional: float) -> tuple[float, dict]:
    shares = notional / price
    analytical = shares * probability - notional - state_fee_usdc(state, shares, price)
    artifact = _artifact() if settings.ACTION_VALUE_ENABLED else None
    learned = None
    learned_fill_probability = None
    learned_pnl_if_filled = None
    fill_only_artifact = _fill_artifact()
    if fill_only_artifact is not None:
        from polybot.models.model_policy import _features

        fill_row = action_vector(
            state.event_slug, outcome, state.observed_at, _features(state, outcome),
            f"BUY_{outcome.upper()}", price, notional,
        )
        learned_fill_probability = _predict_fill(fill_only_artifact, fill_row)
    if artifact:
        from polybot.models.model_policy import _features
        features = _features(state, outcome)
        if "fill_model" in artifact and ("conditional_pnl_model" in artifact or "conditional_outcome_model" in artifact):
            row = action_vector(state.event_slug, outcome, state.observed_at, features,
                                f"BUY_{outcome.upper()}", price, notional)
            learned_fill_probability = _predict_fill(artifact, row)
            if "conditional_outcome_model" in artifact:
                raw_win = float(artifact["conditional_outcome_model"].predict_proba([row])[0, 1])
                if artifact.get("outcome_calibrator") is not None:
                    raw_win = float(np.clip(raw_win, 1e-6, 1 - 1e-6))
                    win_logit = np.log(raw_win / (1.0 - raw_win))
                    win_probability = float(artifact["outcome_calibrator"].predict_proba([[win_logit]])[0, 1])
                else:
                    win_probability = raw_win
                directional = artifact.get("directional_outcome_calibrators", {}).get(outcome)
                if directional is not None:
                    win_probability = float(directional.predict_proba([[win_logit]])[0, 1])
                shares = notional / price
                learned_pnl_if_filled = shares * win_probability - notional - state_fee_usdc(state, shares, price)
            else:
                learned_pnl_if_filled = float(artifact["conditional_pnl_model"].predict([row])[0])
                if artifact.get("conditional_target") == "net_return_per_usdc":
                    learned_pnl_if_filled *= notional
            learned = learned_fill_probability * learned_pnl_if_filled
            tail_probability = None
            expected_tail_loss = None
            if settings.ACTION_TAIL_RISK_ENABLED and artifact.get("tail_classifier") is not None:
                tail_probability = float(artifact["tail_classifier"].predict_proba([row])[0, 1])
                tail_pnl = float(artifact["tail_loss_model"].predict([row])[0])
                expected_tail_loss = learned_fill_probability * tail_probability * max(0.0, -tail_pnl)
                learned -= float(artifact.get("tail_risk_penalty", settings.ACTION_TAIL_RISK_PENALTY)) * expected_tail_loss
        elif str(artifact.get("report", {}).get("version", "")).startswith("counterfactual_"):
            row = action_vector(state.event_slug, outcome, state.observed_at, features,
                                f"BUY_{outcome.upper()}", price, notional)
            learned = float(artifact["model"].predict([row])[0])
        else:
            learned = float(artifact["model"].predict([vector(
                state.event_slug, outcome, state.observed_at, features,
            )])[0])
            learned *= notional / max(settings.PAPER_ENTRY_NOTIONAL_USDC, 0.01)
    weight = settings.ACTION_VALUE_MODEL_WEIGHT if learned is not None else 0.0
    if learned is not None:
        combined = analytical * (1.0 - weight) + float(learned) * weight
    elif learned_fill_probability is not None:
        combined = learned_fill_probability * analytical
    else:
        combined = analytical
    return combined, {
        "analytical": analytical, "learned": learned, "combined": combined,
        "fill_probability": learned_fill_probability,
        "pnl_if_filled": learned_pnl_if_filled,
        "tail_probability": locals().get("tail_probability"),
        "expected_tail_loss": locals().get("expected_tail_loss"),
        "max_tail_probability": artifact.get("max_tail_probability") if artifact else None,
    }


def position_notional(
    net_edge: float,
    expected_pnl_usdc: float,
    *,
    win_probability: float | None = None,
    entry_price: float | None = None,
    fill_probability: float | None = None,
) -> float:
    if not settings.ADAPTIVE_POSITION_SIZING_ENABLED:
        return settings.PAPER_ENTRY_NOTIONAL_USDC
    if (
        settings.POSITION_SIZING_MODE == "calibrated_fractional_kelly_v1"
        and win_probability is not None
        and entry_price is not None
        and 0 < entry_price < 1
    ):
        probability = max(0.0, min(1.0, float(win_probability)))
        fill = max(0.0, min(1.0, float(fill_probability if fill_probability is not None else 1.0)))
        # Для бинарного контракта Kelly = (p-price)/(1-price). Используем только
        # малую долю Kelly и дополнительно уменьшаем риск вероятностью fill.
        full_kelly = max(0.0, (probability - float(entry_price)) / max(1e-6, 1.0 - float(entry_price)))
        risk_fraction = min(
            1.0,
            full_kelly ** float(settings.POSITION_SIZE_KELLY_POWER)
            * float(settings.POSITION_SIZE_KELLY_FRACTION),
        )
        risk_fraction *= fill ** float(settings.POSITION_SIZE_FILL_EXPONENT)
        selected = float(settings.PAPER_MAX_EVENT_EXPOSURE_USDC) * risk_fraction
        # Явно ограничиваем ожидаемый полный проигрыш, а не цену контракта как таковую.
        expected_loss_rate = max(1e-6, 1.0 - probability)
        selected = min(selected, float(settings.POSITION_SIZE_MAX_EXPECTED_LOSS_USDC) / expected_loss_rate)
        if selected > 0:
            selected = max(float(settings.POSITION_SIZE_MIN_USDC), selected)
        if expected_pnl_usdc < settings.ACTION_VALUE_MIN_EXPECTED_PNL_USDC:
            return 0.0
        return min(selected, settings.PAPER_MAX_EVENT_EXPOSURE_USDC, settings.MAX_POSITION_USDC)
    selected = 0.0
    for threshold, size in settings.POSITION_SIZE_EDGE_TIERS:
        if net_edge >= threshold:
            selected = float(size)
    if expected_pnl_usdc < settings.ACTION_VALUE_MIN_EXPECTED_PNL_USDC:
        return 0.0
    return min(selected, settings.PAPER_MAX_EVENT_EXPOSURE_USDC, settings.MAX_POSITION_USDC)
