"""Validated order intents for paper tests and a future audited live adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True, slots=True)
class OrderPlan:
    order_type: str
    side: str
    price: float
    size: float
    expiration: int | None = None
    price_cap: float | None = None


def limit_order(order_type: str, side: str, price: float, size: float, lifetime_seconds: int = 20) -> OrderPlan:
    if order_type not in {"GTC", "GTD"}:
        raise ValueError("limit order_type must be GTC or GTD")
    if side not in {"BUY", "SELL"} or not 0 < price < 1 or size <= 0:
        raise ValueError("invalid limit order")
    expiration = None
    if order_type == "GTD":
        # Polymarket requires a 60-second security-threshold buffer.
        expiration = int((datetime.now(UTC) + timedelta(seconds=60 + lifetime_seconds)).timestamp())
    return OrderPlan(order_type, side, price, size, expiration=expiration)


def immediate_order(order_type: str, side: str, price_cap: float, size: float) -> OrderPlan:
    if order_type not in {"FAK", "FOK"}:
        raise ValueError("immediate order_type must be FAK or FOK")
    if side not in {"BUY", "SELL"} or not 0 < price_cap < 1 or size <= 0:
        raise ValueError("invalid immediate order")
    return OrderPlan(order_type, side, price_cap, size, price_cap=price_cap)
