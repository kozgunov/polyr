import time
from datetime import UTC, datetime

import pytest
from polybot.trading.order_plans import immediate_order, limit_order
from polybot.trading.execution_validity import validate_entry_execution


def test_gtd_has_required_security_buffer() -> None:
    order = limit_order("GTD", "BUY", 0.45, 10, lifetime_seconds=20)
    assert order.expiration is not None
    assert order.expiration >= int(time.time()) + 79


def test_fak_carries_price_cap() -> None:
    order = immediate_order("FAK", "SELL", 0.39, 5)
    assert order.price_cap == 0.39


def test_rejects_wrong_order_family() -> None:
    with pytest.raises(ValueError):
        limit_order("FAK", "BUY", 0.5, 1)


def test_published_future_market_can_accept_pre_event_limit_entry() -> None:
    future_start = int(time.time() // 300) * 300 + 300
    result = validate_entry_execution(
        f"btc-updown-5m-{future_start}", datetime.now(UTC), 0.45,
    )
    assert result.valid
    assert result.elapsed_seconds < 0


def test_finished_event_is_never_tradeable() -> None:
    past_start = int(time.time() // 300) * 300 - 600
    result = validate_entry_execution(
        f"btc-updown-5m-{past_start}", datetime.now(UTC), 0.45,
    )
    assert not result.valid
    assert result.reason == "event_ended"
