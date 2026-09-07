"""Общая консервативная политика для числовых, LLM и consensus-моделей."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import app_config as settings
import joblib
from catboost import CatBoostClassifier

from polybot.models.model_registry import get_model, model_is_ready
from polybot.models.action_value import expected_pnl, position_notional
from polybot.models.exit_value import compare as compare_exit_value
from polybot.models.train_direction_model import vector
from polybot.models.bidirectional_entry import vector as bidirectional_vector
from polybot.trading.fees import net_buy_edge
from polybot.trading.policy import Decision, MarketState, PositionState


@lru_cache(maxsize=1)
def _custom_artifact() -> dict[str, Any]:
    return joblib.load(settings.TRAINING_ARTIFACT_PATH)


@lru_cache(maxsize=1)
def _catboost_artifact() -> tuple[CatBoostClassifier, dict[str, Any]]:
    model = CatBoostClassifier()
    model.load_model(str(settings.CATBOOST_ARTIFACT_PATH))
    metadata = joblib.load(settings.CATBOOST_METADATA_PATH) if settings.CATBOOST_METADATA_PATH.exists() else {}
    return model, metadata


@lru_cache(maxsize=1)
def _bidirectional_artifact() -> dict[str, Any]:
    return joblib.load(settings.BIDIRECTIONAL_ENTRY_CANDIDATE_PATH)


def _features(state: MarketState, outcome: str) -> dict[str, Any]:
    book = state.book_json.get(outcome, {})
    result = {
        "best_bid": book.get("best_bid"), "best_ask": book.get("best_ask"),
        "midpoint": book.get("midpoint"), "spread": book.get("spread"),
        "best_bid_size": book.get("best_bid_size"), "best_ask_size": book.get("best_ask_size"),
    }
    for source, price in state.source_prices.items():
        if source in {"bybit", "okx", "pyth"}:
            result[f"{source}_price"] = price
            if state.target_price and state.target_price > 0:
                result[f"{source}_to_target_pct"] = (price / state.target_price - 1.0) * 100.0
    distance = state.distance_to_target_pct
    volatility = state.realized_volatility_60s_pct
    time_scale = math.sqrt(max(state.remaining_seconds, 1.0) / 60.0)
    result.update({
        "target_price": state.target_price,
        "target_source": state.target_source,
        "reference_price": state.reference_price,
        "reference_source": "polymarket_crypto_price" if state.reference_price is not None else None,
        "elapsed_seconds": state.elapsed_seconds,
        "remaining_seconds": state.remaining_seconds,
        "distance_to_target_usd": state.distance_to_target_usd,
        "distance_to_target_pct": distance,
        "realized_volatility_60s_pct": volatility,
        "distance_time_score": (float(distance or 0.0) / max(volatility * time_scale, 0.01)),
    })
    current_bybit_distance = result.get("bybit_to_target_pct")
    for lag_seconds in (15, 30, 60):
        lag_distance = state.target_distance_lags_pct.get(str(lag_seconds))
        result[f"distance_lag_{lag_seconds}s_pct"] = lag_distance
        result[f"target_momentum_{lag_seconds}s_pct"] = (
            float(current_bybit_distance) - lag_distance
            if current_bybit_distance is not None and lag_distance is not None else None
        )
    result.update(state.history_features)
    return result


def _calibrate(raw: float, calibrator: Any | None) -> float:
    if calibrator is None:
        return raw
    raw = max(1e-6, min(1 - 1e-6, raw))
    return float(calibrator.predict_proba([[math.log(raw / (1.0 - raw))]])[0][1])


def probability_up(state: MarketState, model_key: str = "custom") -> float:
    """Вероятность Up от выбранной числовой модели с симметрией Up/Down."""
    if model_key == "custom_bidir":
        artifact = _bidirectional_artifact()
        model = artifact["model"]
        calibrator = artifact.get("calibrator")
        windows = tuple(int(value) for value in artifact.get("history_windows", (3, 12)))
        scores = {}
        for outcome in ("Up", "Down"):
            opposite = "Down" if outcome == "Up" else "Up"
            row = bidirectional_vector(
                state.event_slug, outcome, state.observed_at, _features(state, outcome),
                _features(state, opposite), state.history_features, history_windows=windows,
            )
            scores[outcome] = _calibrate(float(model.predict_proba([row])[0][1]), calibrator)
        total = max(1e-9, scores["Up"] + scores["Down"])
        return float(max(0.001, min(0.999, scores["Up"] / total)))
    if model_key == "custom":
        artifact = _custom_artifact()
        model = artifact["model"]
        calibrator = artifact.get("calibrator")
    elif model_key == "catboost":
        model, artifact = _catboost_artifact()
        calibrator = artifact.get("calibrator")
    else:
        raise ValueError(f"{model_key} не является числовой моделью")

    def calibrated(outcome: str) -> float:
        history_windows = tuple(int(value) for value in artifact.get("history_windows", ()))
        row = vector(state.event_slug, outcome, state.observed_at, _features(state, outcome), history_windows)
        raw = float(model.predict_proba([row])[0][1])
        return _calibrate(raw, calibrator)

    up = calibrated("Up")
    down = calibrated("Down")
    return float(max(0.001, min(0.999, (up + (1.0 - down)) / 2.0)))


def _llm_context(state: MarketState, position: PositionState | None) -> dict[str, Any]:
    return {
        "event_slug": state.event_slug,
        "prediction_contract": "P(final_reference_price >= target_price)",
        "target_price": state.target_price,
        "target_source": state.target_source,
        "reference_price": state.reference_price,
        "distance_to_target_usd": state.distance_to_target_usd,
        "distance_to_target_pct": state.distance_to_target_pct,
        "external_median_price": state.external_median_price,
        "target_side_validated": state.target_side_validated,
        "reference_external_deviation_pct": state.reference_external_deviation_pct,
        "elapsed_seconds": round(state.elapsed_seconds, 1),
        "remaining_seconds": round(state.remaining_seconds, 1),
        "realized_volatility_60s_pct": round(state.realized_volatility_60s_pct, 6),
        "target_distance_lags_pct": state.target_distance_lags_pct,
        "completed_event_history": state.history_features,
        "external_prices": {key: round(value, 4) for key, value in state.source_prices.items()},
        "external_returns_pct": {key: round(value, 5) for key, value in state.source_returns_pct.items()},
        "source_disagreement_pct": round(state.source_disagreement_pct, 5),
        "sharp_move_pct": round(state.sharp_move_pct, 5),
        "contracts": {
            "Up": {key: state.book_json.get("Up", {}).get(key) for key in ("best_bid", "best_ask", "spread", "best_bid_size", "best_ask_size")},
            "Down": {key: state.book_json.get("Down", {}).get(key) for key in ("best_bid", "best_ask", "spread", "best_bid_size", "best_ask_size")},
        },
        "position": None if position is None else {
            "outcome": position.outcome,
            "average_price": position.average_price,
            "current_bid": position.current_bid,
        },
    }


def _signal_for(state: MarketState, position: PositionState | None, model_key: str) -> tuple[str, float, float, list[str]]:
    spec = get_model(model_key)
    if spec.kind == "numeric":
        p_up = probability_up(state, model_key)
        direction = "Up" if p_up >= 0.5 else "Down"
        confidence = max(p_up, 1.0 - p_up)
        return direction, confidence, p_up, [f"p_up={p_up:.4f}", f"p_down={1.0-p_up:.4f}", "target_outcome_probability"]
    if spec.kind == "llm":
        # Torch/Transformers импортируются только при фактическом выборе LLM.
        from polybot.models.llm_runtime import infer as infer_llm

        signal = infer_llm(model_key, _llm_context(state, position))
        direction = str(signal["direction"])
        confidence = float(signal["confidence"])
        p_up = confidence if direction == "Up" else 1.0 - confidence if direction == "Down" else 0.5
        return direction, confidence, p_up, [f"p_up={p_up:.4f}", f"p_down={1.0-p_up:.4f}", f"llm_reason={signal['reason']}", "target_outcome_probability"]
    raise ValueError(f"Неподдерживаемый тип модели: {spec.kind}")


def _preflight(state: MarketState, position: PositionState | None) -> Decision | None:
    wait = "HOLD" if position else "WAIT"
    if settings.REQUIRE_OFFICIAL_EVENT_TARGET and (
        state.target_price is None or state.target_source != "polymarket_crypto_price"
    ):
        return Decision(wait, 0.0, "Нет официальной Price to Beat", ["target_unavailable"])
    if state.reference_price is None:
        return Decision(wait, 0.0, "Нет валидированной live reference-цены", ["reference_unavailable"])
    if state.target_side_validated is not True:
        return Decision(
            wait, 0.0,
            "Polymarket reference и медиана Bybit/OKX/Pyth находятся по разные стороны Price to Beat",
            ["target_side_conflict", "trading_blocked"],
        )
    if (
        state.reference_external_deviation_pct is None
        or state.reference_external_deviation_pct > settings.PAPER_MAX_SOURCE_DISAGREEMENT_PCT
    ):
        return Decision(
            wait, 0.0, "Polymarket reference недостаточно согласован с внешней медианой",
            ["target_reference_deviation", "trading_blocked"],
        )
    required = {"bybit", "okx", "pyth"}
    if len(required.intersection(state.source_returns_pct)) < settings.MIN_REQUIRED_PRICE_SOURCES:
        return Decision(wait, 0.0, "Недостаточно независимых ценовых источников", ["insufficient_sources"])
    if state.source_disagreement_pct > settings.PAPER_MAX_SOURCE_DISAGREEMENT_PCT:
        return Decision(wait, 0.0, "Источники цены расходятся", ["oracle_conflict"])
    if position is None and state.sharp_move_pct >= settings.PAPER_SHARP_MOVE_BLOCK_PCT:
        return Decision(wait, 0.0, "Резкий скачок: новые действия заблокированы", ["sharp_move_block"])
    return None


def _from_signal(
    state: MarketState,
    position: PositionState | None,
    direction: str,
    confidence: float,
    p_up: float,
    model_key: str,
    tags: list[str],
) -> Decision:
    llm_reason = next((tag.split("=", 1)[1] for tag in tags if tag.startswith("llm_reason=")), "")
    llm_reason = " ".join(llm_reason.split())[:180]
    distance_pct = (
        (state.reference_price / state.target_price - 1.0) * 100.0
        if state.reference_price is not None and state.target_price else 0.0
    )
    market_context = f"до цели {distance_pct:+.4f}%, осталось {state.remaining_seconds:.0f}с"
    tags = [
        *tags, f"model={model_key}", "model_registry", "target=price_to_beat",
        f"target_price={state.target_price}", f"reference_price={state.reference_price}",
        f"remaining_seconds={state.remaining_seconds:.1f}",
    ]
    raw_probabilities = {"Up": p_up, "Down": 1.0 - p_up}
    probabilities = dict(raw_probabilities)
    asks = {"Up": state.up_ask, "Down": state.down_ask}
    if get_model(model_key).kind == "numeric":
        mid_up = state.book_json.get("Up", {}).get("midpoint")
        mid_down = state.book_json.get("Down", {}).get("midpoint")
        if mid_up is not None and mid_down is not None and float(mid_up) + float(mid_down) > 0:
            market_up = float(mid_up) / (float(mid_up) + float(mid_down))
            weight = float(settings.ACTION_PROBABILITY_MODEL_WEIGHT)
            action_p_up = weight * p_up + (1.0 - weight) * market_up
            probabilities = {"Up": action_p_up, "Down": 1.0 - action_p_up}
            confidence = max(probabilities.values())
            tags.extend([
                f"action_p_up={action_p_up:.4f}", f"action_p_down={1.0-action_p_up:.4f}",
                f"market_p_up={market_up:.4f}", f"probability_model_weight={weight:.2f}",
            ])

    if position:
        if position.event_slug != state.event_slug:
            return Decision("HOLD", confidence, "Позиция ожидает официального расчёта", [*tags, "awaiting_resolution"])
        held_probability = probabilities[position.outcome]
        tags.append(f"held_win_probability={held_probability:.4f}")
        held_bid = state.up_bid if position.outcome == "Up" else state.down_bid
        marked_return = (
            (held_bid / position.average_price - 1.0)
            if held_bid is not None and position.average_price > 0 else 0.0
        )
        next_stage = min(5, max(1, position.exit_stage + 1))
        profit_trigger = settings.EXIT_STAGE_PROFIT_RETURN_PCT[next_stage - 1]
        risk_trigger = settings.EXIT_STAGE_MAX_HELD_PROBABILITY[next_stage - 1]
        reference_above_target = bool(
            state.reference_price is not None and state.target_price is not None
            and state.reference_price >= state.target_price
        )
        held_side_currently_winning = (
            reference_above_target if position.outcome == "Up" else not reference_above_target
        )
        adverse_side_confirmed = (
            not held_side_currently_winning
            or not settings.EXIT_RISK_REQUIRES_ADVERSE_TARGET_SIDE
        )
        stage_fraction = float(settings.EXIT_STAGE_REMAINING_FRACTIONS[next_stage - 1])
        exit_value = compare_exit_value(state, position, held_probability)
        tags.extend([
            f"exit_stage={next_stage}", f"marked_return={marked_return:.4f}",
            f"stage_profit_trigger={profit_trigger:.4f}",
            f"stage_risk_probability={risk_trigger:.4f}",
            f"close_pnl_now={float(exit_value['close_pnl']):.5f}",
            f"hold_expected_pnl={float(exit_value['hold_pnl']):.5f}",
            f"close_advantage={float(exit_value['close_advantage']):.5f}",
        ])
        if (
            settings.FIVE_STAGE_EXIT_ENABLED
            and settings.EXIT_ON_PROFIT_ALONE
            and marked_return >= profit_trigger
        ):
            action = "CLOSE" if next_stage == 5 else "PARTIAL_CLOSE"
            return Decision(
                action, held_probability,
                f"Ступень {next_stage}/5: доходность {marked_return:+.1%} достигла лимита; "
                f"P({position.outcome})={held_probability:.1%}, {market_context}"
                + (f"; модель: {llm_reason}" if llm_reason else ""),
                [*tags, "five_stage_exit", "staged_profit_exit", "limit_exit"],
                exit_fraction=stage_fraction,
            )
        if (
            settings.FIVE_STAGE_EXIT_ENABLED
            and held_probability <= risk_trigger
            and adverse_side_confirmed
            and (not settings.EXIT_VALUE_ENABLED or bool(exit_value["exit"]))
        ):
            action = "CLOSE" if next_stage == 5 else "PARTIAL_CLOSE"
            return Decision(
                action, 1.0 - held_probability,
                f"Ступень {next_stage}/5: P({position.outcome}) упала до {held_probability:.1%} и цена на "
                f"противоположной стороне Price to Beat; {market_context}"
                + (f"; модель: {llm_reason}" if llm_reason else ""),
                [*tags, "five_stage_exit", "staged_risk_exit", "held_direction_exit", "reversal_exit", "flip_disabled", "limit_exit"],
                exit_fraction=stage_fraction,
            )
        return Decision(
            "HOLD", held_probability,
            f"Удерживаем {position.outcome}: P={held_probability:.1%}, доходность {marked_return:+.1%}; "
            f"условия ступени {next_stage}/5 не выполнены, {market_context}",
            [*tags, "hold_held_direction", "five_stage_hold"],
        )

    if direction not in {"Up", "Down"}:
        return Decision("WAIT", confidence, "Модель не выбрала направление", [*tags, "model_hold"])
    if state.remaining_seconds <= 0:
        return Decision("WAIT", confidence, "Событие уже завершено", [*tags, "event_ended"])
    if settings.MODEL_TIME_GATES_ENABLED:
        if state.elapsed_seconds < settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN:
            return Decision("WAIT", confidence, "Недостаточно истории внутри события", [*tags, "early_noise_window"])
        if state.remaining_seconds < settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:
            return Decision("WAIT", confidence, "Поздно для нового входа", [*tags, "late_entry_block"])
    # Модель входа единолично определяет сторону Price to Beat. Денежная модель
    # может отклонить вход или изменить размер, но не имеет права незаметно
    # заменить Down на Up (и наоборот) из-за иной ask-цены контракта.
    tags.extend([
        "direction_preserved_from_entry_model",
        "action_value_is_gate_not_direction_selector",
        f"selected_contract_probability={probabilities[direction]:.4f}",
        f"signal_confidence={confidence:.4f}",
    ])
    ask = asks[direction]
    if ask is None:
        return Decision("WAIT", confidence, "Нет исполнимой ask-цены", [*tags, "book_unavailable"])
    spread = state.book_json.get(direction, {}).get("spread")
    if spread is None or float(spread) > settings.MAX_ALLOWED_SPREAD:
        return Decision("WAIT", confidence, "Spread слишком велик", [*tags, "spread_guard"])
    if not settings.PAPER_MIN_ENTRY_PRICE <= ask <= settings.PAPER_MAX_ENTRY_PRICE:
        return Decision("WAIT", confidence, "Цена контракта вне допустимого диапазона", [*tags, "entry_price_guard"])
    edge = net_buy_edge(probabilities[direction], ask)
    execution_buffer = float(spread) + ask * settings.ESTIMATED_SLIPPAGE_BPS / 10_000
    required_edge = max(
        settings.PAPER_MIN_ENTRY_NET_EDGE,
        settings.ENTRY_VALUE_SAFETY_MARGIN + execution_buffer,
    )
    tags.extend([
        f"net_edge={edge:.5f}",
        f"required_net_edge={required_edge:.5f}",
        f"execution_buffer={execution_buffer:.5f}",
        "calibrated_probability_value_gate",
    ])
    selected_probability_too_low = (
        not settings.LOW_PROBABILITY_TRADING_ENABLED
        and probabilities[direction] < settings.MIN_ENTRY_CONFIDENCE
    )
    if confidence < settings.MIN_ENTRY_CONFIDENCE or selected_probability_too_low or edge < required_edge:
        return Decision(
            "WAIT", confidence,
            "Калиброванная вероятность не покрывает цену, комиссию, исполнение и запас ошибки",
            [*tags, "no_positive_edge", "value_gate_reject"],
        )
    base_expected_pnl, value_parts = expected_pnl(
        state, direction, probabilities[direction], ask, settings.PAPER_ENTRY_NOTIONAL_USDC,
    )
    notional = position_notional(
        edge, base_expected_pnl,
        win_probability=probabilities[direction], entry_price=ask,
        fill_probability=value_parts.get("fill_probability"),
    )
    minimum_shares = max(
        float(state.minimum_order_size or 0.0),
        float(settings.DEFAULT_CLOB_MIN_ORDER_SIZE_SHARES),
    )
    minimum_notional = max(float(settings.POSITION_SIZE_MIN_USDC), minimum_shares * ask)
    if notional > 0:
        notional = min(
            max(float(notional), minimum_notional),
            float(settings.PAPER_MAX_EVENT_EXPOSURE_USDC),
            float(settings.MAX_POSITION_USDC),
        )
    if notional + 1e-9 < minimum_notional:
        notional = 0.0
    tags.extend([
        f"action_value={base_expected_pnl:.5f}",
        f"analytical_value={float(value_parts['analytical']):.5f}",
        f"learned_value={value_parts['learned']}",
        f"learned_fill_probability={value_parts.get('fill_probability')}",
        f"learned_pnl_if_filled={value_parts.get('pnl_if_filled')}",
        f"tail_probability={value_parts.get('tail_probability')}",
        f"expected_tail_loss={value_parts.get('expected_tail_loss')}",
        f"position_notional={notional:.2f}",
    ])
    tail_probability = value_parts.get("tail_probability")
    expected_tail_loss = value_parts.get("expected_tail_loss")
    if (
        settings.ACTION_TAIL_RISK_ENABLED
        and tail_probability is not None
        and expected_tail_loss is not None
        and (
            float(tail_probability) > float(value_parts.get("max_tail_probability") or settings.ACTION_TAIL_MAX_PROBABILITY)
            or float(expected_tail_loss) > settings.ACTION_TAIL_MAX_EXPECTED_LOSS_USDC
        )
    ):
        return Decision(
            "WAIT", confidence,
            f"Хвостовой риск слишком велик: P(крупный убыток)={float(tail_probability):.1%}, "
            f"expected tail loss=${float(expected_tail_loss):.2f}",
            [*tags, "tail_risk_reject"],
        )
    if notional <= 0:
        return Decision(
            "WAIT", confidence,
            f"Направление {direction} вероятно, но ожидаемый PnL {base_expected_pnl:+.3f} недостаточен",
            [*tags, "action_value_reject"],
        )
    return Decision(
        f"BUY_{direction.upper()}", confidence,
        f"Вход {direction}: P={probabilities[direction]:.1%}, ask={ask:.3f}, net-edge={edge:+.3f}, "
        f"EV={base_expected_pnl:+.3f}, размер=${notional:.2f}; "
        f"{market_context}" + (f"; модель: {llm_reason}" if llm_reason else ""),
        tags, direction, ask, notional,
    )


def _consensus(state: MarketState, position: PositionState | None, model_key: str) -> Decision:
    llm_key = "qwen" if model_key == "consensus_qwen_custom" else "gemma"
    numeric = _signal_for(state, position, settings.CONSENSUS_NUMERIC_MODEL)
    llm = _signal_for(state, position, llm_key)
    n_direction, n_confidence, n_p_up, n_tags = numeric
    l_direction, l_confidence, l_p_up, l_tags = llm
    tags = [
        *n_tags, *l_tags, "consensus",
        f"numeric_confidence={n_confidence:.4f}", f"llm_confidence={l_confidence:.4f}",
    ]
    if position:
        numeric_held = n_p_up if position.outcome == "Up" else 1.0 - n_p_up
        llm_held = l_p_up if position.outcome == "Up" else 1.0 - l_p_up
        tags.extend([
            f"numeric_held_win_probability={numeric_held:.4f}",
            f"llm_held_win_probability={llm_held:.4f}",
        ])
        conservative_held = max(numeric_held, llm_held)
        consensus_p_up = conservative_held if position.outcome == "Up" else 1.0 - conservative_held
        return _from_signal(
            state, position,
            position.outcome if conservative_held >= 0.5 else ("Down" if position.outcome == "Up" else "Up"),
            max(conservative_held, 1.0 - conservative_held), consensus_p_up, model_key,
            [*tags, "consensus_held_probability_conservative"],
        )
    threshold = settings.CONSENSUS_ENTRY_CONFIDENCE
    if n_direction != l_direction or n_direction not in {"Up", "Down"}:
        action = "HOLD" if position else "WAIT"
        return Decision(action, min(n_confidence, l_confidence), "Модели не согласны по направлению", [*tags, "consensus_disagreement"])
    selected_probabilities = (
        (n_p_up, l_p_up) if n_direction == "Up" else (1.0 - n_p_up, 1.0 - l_p_up)
    )
    if min(selected_probabilities) < threshold:
        action = "HOLD" if position else "WAIT"
        return Decision(action, min(n_confidence, l_confidence), "Недостаточная общая уверенность моделей", [*tags, "consensus_low_confidence"])
    if n_direction == "Up":
        p_up = min(n_p_up, l_p_up)
    else:
        p_up = 1.0 - min(1.0 - n_p_up, 1.0 - l_p_up)
    return _from_signal(state, position, n_direction, min(selected_probabilities), p_up, model_key, tags)


def decide_with_model(
    state: MarketState,
    position: PositionState | None = None,
    model_key: str = "custom",
    active_collection: bool = False,
) -> Decision:
    if not model_is_ready(model_key):
        return Decision("HOLD" if position else "WAIT", 0.0, "Выбранная модель не установлена", ["model_not_ready", f"model={model_key}"])
    if settings.ML_AUTONOMOUS_POLICY_ENABLED and get_model(model_key).kind == "numeric":
        from polybot.models.autonomous_policy import decide as decide_autonomously

        direction, _, p_up, tags = _signal_for(state, position, model_key)
        return decide_autonomously(
            state, position, {"Up": p_up, "Down": 1.0 - p_up}, model_key,
            [*tags, f"direction_hint={direction}"],
            active_collection=active_collection,
        )
    guard = _preflight(state, position)
    if guard:
        return guard
    if get_model(model_key).kind == "consensus":
        return _consensus(state, position, model_key)
    direction, confidence, p_up, tags = _signal_for(state, position, model_key)
    return _from_signal(state, position, direction, confidence, p_up, model_key, tags)
