"""Автономная ML-policy: модель выбирает действие, цену и размер через argmax utility."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import app_config as settings

from polybot.models.action_value import expected_pnl
from polybot.models.exit_value import compare as compare_exit_value
from polybot.trading.policy import Decision, MarketState, PositionState


def policy_confidence(best: float, runner_up: float) -> float:
    """Диагностический score разрыва utility; это не вероятность правильного направления."""
    gap = max(0.0, float(best) - float(runner_up))
    return float(1.0 - math.exp(-gap / max(settings.PAPER_ENTRY_NOTIONAL_USDC, 0.01)))


def _technical_guard(state: MarketState, position: PositionState | None) -> Decision | None:
    passive = "HOLD" if position else "WAIT"
    if state.target_price is None or state.reference_price is None:
        return Decision(passive, 0.0, "Нет Price to Beat или актуальной reference-цены", ["ml_policy", "invalid_market_data"])
    if settings.ML_POLICY_REQUIRE_FRESH_DATA:
        try:
            observed = datetime.fromisoformat(state.observed_at)
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            age = max(0.0, (datetime.now(UTC) - observed.astimezone(UTC)).total_seconds())
        except (TypeError, ValueError):
            age = float("inf")
        if age > settings.MAX_POLYMARKET_AGE_SECONDS:
            return Decision(passive, 0.0, f"Снимок рынка устарел на {age:.1f} сек", ["ml_policy", "stale_market_data"])
    if position is not None and position.event_slug != state.event_slug:
        return Decision("HOLD", 0.0, "Позиция ожидает официального расчёта", ["ml_policy", "awaiting_resolution"])
    if position is None and state.remaining_seconds <= settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:
        return Decision("WAIT", 0.0, "Окно входа в событие уже закрыто", ["ml_policy", "entry_window_closed"])
    if position is None and state.remaining_seconds > 300 - settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN:
        return Decision("WAIT", 0.0, "Ожидаем открытия разрешённого окна входа", ["ml_policy", "entry_window_not_open"])
    return None


def decide(
    state: MarketState,
    position: PositionState | None,
    probabilities: dict[str, float],
    model_key: str,
    base_tags: list[str] | None = None,
) -> Decision:
    guard = _technical_guard(state, position)
    if guard is not None:
        return guard
    tags = [*(base_tags or []), "ml_autonomous_policy", "manual_trade_gates_disabled", f"model={model_key}"]

    if position is not None:
        held_probability = float(probabilities[position.outcome])
        result = compare_exit_value(state, position, held_probability)
        learned = result.get("learned_close_advantage")
        close_utility = float(learned if learned is not None else result["close_advantage"])
        hold_utility = 0.0
        confidence = policy_confidence(max(close_utility, hold_utility), min(close_utility, hold_utility))
        tags.extend([
            "ml_exit_argmax", f"hold_utility={hold_utility:.6f}", f"close_utility={close_utility:.6f}",
            f"learned_close_probability={result.get('learned_close_probability')}",
            f"close_pnl_now={float(result['close_pnl']):.6f}", f"hold_expected_pnl={float(result['hold_pnl']):.6f}",
            f"policy_confidence={confidence:.6f}",
        ])
        if bool(result.get("exit")):
            return Decision("CLOSE", confidence, f"Exit-модель выбрала CLOSE: преимущество над HOLD {close_utility:+.3f}", tags,
                            direction=position.outcome, exit_fraction=1.0)
        return Decision("HOLD", confidence, f"Exit-модель выбрала HOLD: оценка CLOSE против HOLD {close_utility:+.3f}", tags)

    direction = max(("Up", "Down"), key=lambda outcome: float(probabilities.get(outcome, 0.0)))
    direction_probability = float(probabilities.get(direction, 0.0))
    if not math.isfinite(direction_probability) or direction_probability < settings.ML_POLICY_MIN_DIRECTION_CONFIDENCE:
        tags.extend([
            "direction_preserved_from_entry_model",
            "entry_model_low_confidence_wait",
            f"selected_direction={direction}",
            f"selected_outcome_probability={direction_probability:.6f}",
        ])
        return Decision(
            "WAIT", max(0.0, min(1.0, direction_probability)),
            f"Entry-модель выбрала {direction}, но уверенность {direction_probability:.1%} ниже порога",
            tags,
        )

    candidates: list[dict[str, Any]] = [{
        "action": "WAIT", "utility": float(settings.ML_POLICY_WAIT_UTILITY_USDC),
        "direction": None, "price": None, "notional": 0.0, "level": "wait", "parts": {},
    }]
    # Направление события выбирает только калиброванная entry-модель. Value-слой
    # оптимизирует лимит и размер либо выбирает WAIT, но не может купить обратную сторону.
    outcomes = (direction,) if settings.ML_POLICY_DIRECTION_PRESERVING else ("Up", "Down")
    for outcome in outcomes:
        book = state.book_json.get(outcome, {}) or {}
        for level in settings.ML_POLICY_LIMIT_LEVELS:
            raw_price = book.get("midpoint" if level == "midpoint" else f"best_{level}")
            if raw_price is None:
                continue
            price = float(raw_price)
            if not 0.0 < price < 1.0:
                continue
            for raw_notional in settings.ML_POLICY_NOTIONALS_USDC:
                notional = min(float(raw_notional), settings.PAPER_MAX_EVENT_EXPOSURE_USDC, settings.MAX_POSITION_USDC)
                utility, parts = expected_pnl(state, outcome, probabilities[outcome], price, notional)
                if notional >= 5.0 and (
                    float(probabilities[outcome]) < settings.ML_POLICY_FIVE_DOLLAR_MIN_OUTCOME_PROBABILITY
                    or float(utility) < settings.ML_POLICY_FIVE_DOLLAR_MIN_UTILITY_USDC
                ):
                    continue
                candidates.append({
                    "action": f"BUY_{outcome.upper()}", "utility": float(utility), "direction": outcome,
                    "price": price, "notional": notional, "level": level, "parts": parts,
                })
    ranked = sorted(candidates, key=lambda item: item["utility"], reverse=True)
    best = ranked[0]
    runner_up = next(
        (item for item in ranked[1:] if item["action"] != best["action"]),
        ranked[1] if len(ranked) > 1 else candidates[0],
    )
    utility_margin_score = policy_confidence(best["utility"], runner_up["utility"])
    if best["direction"] in {"Up", "Down"}:
        selected_probability = float(probabilities[best["direction"]])
        confidence = max(0.0, min(1.0, selected_probability))
    else:
        selected_probability = None
        confidence = utility_margin_score
    compact_scores = ";".join(
        f"{item['action']}@{item['level']}x{item['notional']:.0f}={item['utility']:.4f}" for item in ranked[:6]
    )
    tags.extend([
        "ml_entry_argmax", f"best_utility={best['utility']:.6f}", f"runner_up_utility={runner_up['utility']:.6f}",
        f"utility_gap={best['utility']-runner_up['utility']:.6f}",
        f"utility_margin_score={utility_margin_score:.6f}",
        f"selected_outcome_probability={selected_probability}",
        f"direction_confidence={confidence:.6f}", f"policy_confidence={confidence:.6f}",
        f"candidate_count={len(candidates)}", f"top_action_scores={compact_scores}",
        "direction_preserved_from_entry_model",
        f"selected_direction={direction}",
        "explicit_analytical_value_policy" if not settings.ACTION_VALUE_ENABLED else "learned_value_policy",
    ])
    if best["action"] == "WAIT":
        return Decision("WAIT", confidence, "ML-policy выбрала WAIT как действие с максимальной полезностью", tags)
    parts = best["parts"]
    tags.extend([
        f"selected_limit_level={best['level']}", f"learned_value={parts.get('learned')}",
        f"fill_probability={parts.get('fill_probability')}", f"pnl_if_filled={parts.get('pnl_if_filled')}",
        f"tail_probability={parts.get('tail_probability')}",
    ])
    return Decision(
        best["action"], confidence,
        f"ML-policy выбрала {best['action']} по {best['level']} за ${best['notional']:.2f}; utility={best['utility']:+.3f}",
        tags, direction=best["direction"], limit_price=best["price"], notional_usdc=best["notional"],
    )
