"""Единые доменные проверки исполнения входных заявок BTC Up/Down 5m."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import app_config as settings


@dataclass(frozen=True)
class ExecutionValidity:
    valid: bool
    reason: str | None
    elapsed_seconds: float
    remaining_seconds: float


def event_start(event_slug: str) -> datetime:
    return datetime.fromtimestamp(int(event_slug.rsplit("-", 1)[-1]), UTC)


def validate_entry_execution(
    event_slug: str,
    executed_at: datetime,
    price: float,
) -> ExecutionValidity:
    """Проверяет фактическое исполнение, а не только решение модели."""
    if executed_at.tzinfo is None:
        executed_at = executed_at.replace(tzinfo=UTC)
    start = event_start(event_slug)
    elapsed = (executed_at.astimezone(UTC) - start).total_seconds()
    remaining = 300.0 - elapsed
    reasons: list[str] = []
    if remaining <= 0:
        reasons.append("event_ended")
    elif settings.MODEL_TIME_GATES_ENABLED:
        if elapsed < float(settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN):
            reasons.append("entry_before_window")
        if remaining < float(settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE):
            reasons.append("entry_after_cutoff")
    if not float(settings.PAPER_MIN_ENTRY_PRICE) <= float(price) <= float(settings.PAPER_MAX_ENTRY_PRICE):
        reasons.append("entry_price_out_of_domain")
    reason = ";".join(reasons) or None
    return ExecutionValidity(not reasons, reason, elapsed, remaining)


def validate_entry_state(state, price: float) -> ExecutionValidity:
    """Проверяет актуальный MarketState; используется непосредственно перед submit/fill."""
    elapsed = float(state.elapsed_seconds)
    remaining = float(state.remaining_seconds)
    reasons: list[str] = []
    if remaining <= 0:
        reasons.append("event_ended")
    elif settings.MODEL_TIME_GATES_ENABLED:
        if elapsed < float(settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN):
            reasons.append("entry_before_window")
        if remaining < float(settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE):
            reasons.append("entry_after_cutoff")
    if not float(settings.PAPER_MIN_ENTRY_PRICE) <= float(price) <= float(settings.PAPER_MAX_ENTRY_PRICE):
        reasons.append("entry_price_out_of_domain")
    reason = ";".join(reasons) or None
    return ExecutionValidity(not reasons, reason, elapsed, remaining)
