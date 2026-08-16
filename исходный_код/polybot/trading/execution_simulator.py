"""Детерминированный реалистичный paper-fill: очередь, глубина, latency и non-fill."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import app_config as settings


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    status: str
    filled_shares: float
    filled_price: float | None
    fill_probability: float
    latency_ms: int
    slippage_bps: float
    reason: str


def _unit(key: str) -> float:
    digest = hashlib.sha256(f"{settings.EXECUTION_RANDOM_SEED}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def limit_buy(key: str, requested_price: float, best_ask: float | None, ask_size: float | None,
              shares: float, spread: float | None) -> SimulatedFill:
    latency = int(settings.EXECUTION_BASE_LATENCY_MS + _unit(key + ":latency") * settings.EXECUTION_LATENCY_JITTER_MS)
    if best_ask is None or requested_price + 1e-9 < best_ask:
        return SimulatedFill("unfilled", 0.0, None, 0.0, latency, 0.0, "limit_below_current_ask")
    depth = max(0.0, float(ask_size or 0.0)) * settings.EXECUTION_MAX_BOOK_PARTICIPATION
    queue_penalty = max(0.0, float(ask_size or 0.0)) * settings.EXECUTION_QUEUE_AHEAD_FRACTION
    executable = max(0.0, depth - min(depth, queue_penalty * 0.10))
    depth_ratio = min(1.0, executable / max(shares, 1e-9))
    spread_penalty = min(0.45, float(spread or 1.0) * 3.0)
    probability = max(settings.EXECUTION_MIN_FILL_PROBABILITY, min(0.98, 0.25 + 0.70 * depth_ratio - spread_penalty))
    if _unit(key + ":fill") > probability or executable <= 0:
        return SimulatedFill("unfilled", 0.0, None, probability, latency, 0.0, "queue_or_liquidity_nonfill")
    filled = min(shares, executable)
    status = "filled" if filled >= shares * 0.999 else "partially_filled"
    return SimulatedFill(status, filled, min(requested_price, best_ask), probability, latency, 0.0, "gtd_limit_simulation")


def fak_sell(key: str, requested_bid: float | None, bid_size: float | None, shares: float,
             price_cap: float) -> SimulatedFill:
    latency = int(settings.EXECUTION_BASE_LATENCY_MS + _unit(key + ":latency") * settings.EXECUTION_LATENCY_JITTER_MS)
    if requested_bid is None or requested_bid < price_cap:
        return SimulatedFill("unfilled", 0.0, None, 0.0, latency, 0.0, "bid_below_fak_price_cap")
    available = max(0.0, float(bid_size or 0.0)) * settings.EXECUTION_MAX_BOOK_PARTICIPATION
    filled = min(shares, available)
    if filled <= 0:
        return SimulatedFill("unfilled", 0.0, None, 0.0, latency, 0.0, "no_executable_bid_depth")
    depth_ratio = filled / max(shares, 1e-9)
    slippage_bps = min(settings.FAK_PRICE_CAP_SLIPPAGE_BPS, (1.0 - depth_ratio) * settings.FAK_PRICE_CAP_SLIPPAGE_BPS)
    price = max(price_cap, requested_bid * (1.0 - slippage_bps / 10_000))
    status = "filled" if depth_ratio >= 0.999 else "partially_filled"
    return SimulatedFill(status, filled, price, depth_ratio, latency, slippage_bps, "fak_depth_simulation")
