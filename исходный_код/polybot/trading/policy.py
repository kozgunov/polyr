"""Conservative BTC 5-minute decision policy used as a tested baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

import app_config as settings


@dataclass(slots=True)
class MarketState:
    event_slug: str
    observed_at: str
    elapsed_seconds: float
    remaining_seconds: float
    source_returns_pct: dict[str, float]
    source_prices: dict[str, float]
    source_disagreement_pct: float
    sharp_move_pct: float
    up_bid: float | None
    up_ask: float | None
    down_bid: float | None
    down_ask: float | None
    book_json: dict[str, Any] = field(default_factory=dict)
    target_price: float | None = None
    reference_price: float | None = None
    target_source: str | None = None
    reference_observed_at: str | None = None
    realized_volatility_60s_pct: float = 0.0
    target_distance_lags_pct: dict[str, float] = field(default_factory=dict)
    history_features: dict[str, float] = field(default_factory=dict)
    fees_enabled: bool = True
    fee_rate: float = settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE
    fee_exponent: float = 1.0
    fee_taker_only: bool = True
    minimum_order_size: float = 0.0
    tick_size: float = 0.01

    @property
    def distance_to_target_usd(self) -> float | None:
        if self.target_price is None or self.reference_price is None:
            return None
        return self.reference_price - self.target_price

    @property
    def distance_to_target_pct(self) -> float | None:
        if self.target_price is None or self.reference_price is None or self.target_price <= 0:
            return None
        return (self.reference_price / self.target_price - 1.0) * 100.0

    @property
    def external_median_price(self) -> float | None:
        values = [self.source_prices[key] for key in ("bybit", "okx", "pyth") if key in self.source_prices]
        return median(values) if values else None

    @property
    def reference_external_deviation_pct(self) -> float | None:
        external = self.external_median_price
        if self.reference_price is None or external is None or external <= 0:
            return None
        return abs(self.reference_price / external - 1.0) * 100.0

    @property
    def target_side_validated(self) -> bool | None:
        external = self.external_median_price
        if self.target_price is None or self.reference_price is None or external is None:
            return None
        return (self.reference_price >= self.target_price) == (external >= self.target_price)

    @property
    def consensus_return_pct(self) -> float:
        return median(self.source_returns_pct.values()) if self.source_returns_pct else 0.0

    @property
    def agreement(self) -> float:
        if not self.source_returns_pct:
            return 0.0
        direction = 1 if self.consensus_return_pct >= 0 else -1
        agreed = sum(1 for value in self.source_returns_pct.values() if (value >= 0) == (direction > 0))
        return agreed / len(self.source_returns_pct)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["consensus_return_pct"] = self.consensus_return_pct
        data["agreement"] = self.agreement
        data["distance_to_target_usd"] = self.distance_to_target_usd
        data["distance_to_target_pct"] = self.distance_to_target_pct
        data["external_median_price"] = self.external_median_price
        data["reference_external_deviation_pct"] = self.reference_external_deviation_pct
        data["target_side_validated"] = self.target_side_validated
        return data


@dataclass(slots=True)
class PositionState:
    position_id: int
    event_slug: str
    outcome: str
    token_id: str
    shares: float
    cost_usdc: float
    average_price: float
    current_bid: float | None
    exit_stage: int = 0
    opened_at: str | None = None
    original_shares: float | None = None
    original_cost_usdc: float | None = None
    exit_features: dict[str, float] = field(default_factory=dict)

    @property
    def unrealized_pnl(self) -> float:
        return self.shares * self.current_bid - self.cost_usdc if self.current_bid is not None else 0.0


@dataclass(slots=True)
class Decision:
    action: str
    confidence: float
    reason: str
    tags: list[str] = field(default_factory=list)
    direction: str | None = None
    limit_price: float | None = None
    notional_usdc: float = 0.0
    exit_fraction: float = 0.0


def confidence_for(state: MarketState) -> float:
    magnitude = min(abs(state.consensus_return_pct) / 0.35, 1.0)
    disagreement_penalty = min(
        state.source_disagreement_pct / max(settings.PAPER_MAX_SOURCE_DISAGREEMENT_PCT, 0.001), 1.0
    )
    return max(0.0, min(0.98, 0.55 + 0.27 * magnitude + 0.18 * state.agreement - 0.12 * disagreement_penalty))


def decide(state: MarketState, position: PositionState | None = None) -> Decision:
    tags: list[str] = []
    if len(state.source_returns_pct) < settings.MIN_REQUIRED_PRICE_SOURCES:
        return Decision("WAIT", 0.0, "Недостаточно независимых ценовых источников", ["insufficient_sources"])
    if state.source_disagreement_pct > settings.PAPER_MAX_SOURCE_DISAGREEMENT_PCT:
        return Decision("WAIT", 0.0, "Источники цены расходятся сильнее допустимого", ["oracle_conflict"])
    if state.sharp_move_pct >= settings.PAPER_SHARP_MOVE_BLOCK_PCT:
        return Decision("WAIT", 0.0, "Обнаружен резкий скачок; вход временно заблокирован", ["sharp_move_block"])

    move = state.consensus_return_pct
    direction = "Up" if move > 0 else "Down"
    confidence = confidence_for(state)
    direction_price = state.up_ask if direction == "Up" else state.down_ask
    direction_bid = state.up_bid if direction == "Up" else state.down_bid
    all_same_direction = state.agreement == 1.0
    implied = direction_price
    if all_same_direction and implied is not None and implied < 0.5 - settings.PAPER_CONSENSUS_MISPRICING_PROB:
        tags.append("consensus_misalignment")
        confidence = min(0.98, confidence + 0.08)

    if position is not None:
        if position.event_slug != state.event_slug:
            return Decision("HOLD", 1.0, "Предыдущая позиция ожидает официального расчёта", ["awaiting_resolution"])
        position_direction_bid = state.up_bid if position.outcome == "Up" else state.down_bid
        marked_return = (
            (position_direction_bid - position.average_price) / position.average_price
            if position_direction_bid is not None and position.average_price > 0
            else 0.0
        )
        if state.remaining_seconds <= settings.PAPER_FORCE_EXIT_SECONDS_BEFORE_CLOSE:
            return Decision("CLOSE", 1.0, "Принудительное закрытие перед расчётом рынка", ["time_exit"])
        if marked_return >= settings.TAKE_PROFIT_PCT:
            return Decision("CLOSE", 0.95, "Достигнут тестовый take-profit", ["take_profit"])
        if marked_return <= -settings.STOP_LOSS_PCT:
            return Decision("CLOSE", 0.95, "Достигнут тестовый stop-loss", ["stop_loss"])
        if direction != position.outcome and confidence >= 1.0 - settings.MAX_HELD_WIN_PROBABILITY_FOR_EXIT:
            return Decision(
                "CLOSE", confidence,
                "Сильный подтверждённый сигнал против позиции; закрываем без переворота",
                [*tags, "reversal_exit", "flip_disabled"],
            )
        if (
            direction == position.outcome
            and confidence >= 0.92
            and position.cost_usdc + settings.PAPER_ADD_NOTIONAL_USDC <= settings.PAPER_MAX_EVENT_EXPOSURE_USDC
            and direction_price is not None
        ):
            return Decision(
                "ADD", confidence, "Сигнал усилился; разрешено небольшое добавление", tags,
                direction, direction_price, settings.PAPER_ADD_NOTIONAL_USDC,
            )
        return Decision("HOLD", confidence, "Позиция сохраняется; условий выхода или добавления нет", tags)

    if state.remaining_seconds <= 0:
        return Decision("WAIT", confidence, "Событие уже завершено", ["event_ended"])
    if settings.MODEL_TIME_GATES_ENABLED:
        if state.elapsed_seconds < settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN:
            return Decision("WAIT", confidence, "Слишком рано: ждём формирование устойчивого сигнала", ["early_noise_window"])
        if state.remaining_seconds < settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE:
            return Decision("WAIT", confidence, "Для нового входа осталось слишком мало времени", ["late_entry_block"])
    if abs(move) < settings.PAPER_MIN_BTC_MOVE_PCT:
        return Decision("WAIT", confidence, "Движение BTC недостаточно сильное", ["weak_signal"])
    if confidence < settings.MIN_ENTRY_CONFIDENCE:
        return Decision("WAIT", confidence, "Уверенность ниже порога входа", ["low_confidence"])
    if direction_price is None or direction_bid is None:
        return Decision("WAIT", confidence, "В стакане нет исполнимой цены", ["book_unavailable"])
    if not settings.PAPER_MIN_ENTRY_PRICE <= direction_price <= settings.PAPER_MAX_ENTRY_PRICE:
        return Decision("WAIT", confidence, "Цена контракта вне консервативного диапазона", ["price_guard"])
    return Decision(
        f"BUY_{direction.upper()}", confidence, "Согласованный сигнал цены и допустимый риск", tags,
        direction, direction_price, settings.PAPER_ENTRY_NOTIONAL_USDC,
    )
