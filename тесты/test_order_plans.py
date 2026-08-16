import time

import pytest
from polybot.trading.order_plans import immediate_order, limit_order


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
