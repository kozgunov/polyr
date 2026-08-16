"""Polymarket fee helpers used by paper/backtest economics."""

from __future__ import annotations

import app_config as settings


def platform_fee_usdc(shares: float, price: float, *, taker: bool = True) -> float:
    """Current crypto fee curve; makers pay zero platform fee."""
    if not taker or shares <= 0 or not 0 < price < 1:
        return 0.0
    fee = shares * settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE * price * (1.0 - price)
    return round(fee, 5) if fee >= 0.00001 else 0.0


def builder_fee_usdc(notional_usdc: float) -> float:
    return round(max(0.0, notional_usdc) * settings.POLYMARKET_BUILDER_FEE_BPS / 10_000, 5)


def total_fee_usdc(shares: float, price: float, *, taker: bool = True) -> float:
    return platform_fee_usdc(shares, price, taker=taker) + builder_fee_usdc(shares * price)


def net_buy_edge(probability: float, ask: float, *, taker: bool = True) -> float:
    """Expected USDC profit per share after entry fee, before exit/settlement."""
    return probability - ask - total_fee_usdc(1.0, ask, taker=taker)
