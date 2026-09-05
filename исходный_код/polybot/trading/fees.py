"""Polymarket fee helpers used by paper/backtest economics."""

from __future__ import annotations

import app_config as settings


def platform_fee_usdc(
    shares: float,
    price: float,
    *,
    taker: bool = True,
    fee_rate: float | None = None,
    fee_exponent: float = 1.0,
    taker_only: bool = True,
    fees_enabled: bool = True,
) -> float:
    """Комиссия CLOB V2 с параметрами конкретного рынка.

    Для BTC 5m API сейчас возвращает rate=0.07, exponent=1 и takerOnly=true.
    Параметры аргументами позволяют не зашивать категорию рынка в симуляцию.
    """
    if not fees_enabled or shares <= 0 or not 0 < price < 1:
        return 0.0
    if taker_only and not taker:
        return 0.0
    rate = settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE if fee_rate is None else max(0.0, float(fee_rate))
    curve = price * (1.0 - price)
    fee = shares * rate * (curve ** max(0.0, float(fee_exponent)))
    return round(fee, 5) if fee >= 0.00001 else 0.0


def builder_fee_usdc(notional_usdc: float) -> float:
    return round(max(0.0, notional_usdc) * settings.POLYMARKET_BUILDER_FEE_BPS / 10_000, 5)


def total_fee_usdc(shares: float, price: float, *, taker: bool = True, **fee_parameters) -> float:
    return platform_fee_usdc(shares, price, taker=taker, **fee_parameters) + builder_fee_usdc(shares * price)


def state_fee_usdc(state, shares: float, price: float, *, taker: bool = True) -> float:
    """Комиссия по feeSchedule, сохранённому для конкретного события."""
    return total_fee_usdc(
        shares,
        price,
        taker=taker,
        fee_rate=getattr(state, "fee_rate", settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE),
        fee_exponent=getattr(state, "fee_exponent", 1.0),
        taker_only=getattr(state, "fee_taker_only", True),
        fees_enabled=getattr(state, "fees_enabled", True),
    )


def net_buy_edge(probability: float, ask: float, *, taker: bool = True) -> float:
    """Expected USDC profit per share after entry fee, before exit/settlement."""
    return probability - ask - total_fee_usdc(1.0, ask, taker=taker)
