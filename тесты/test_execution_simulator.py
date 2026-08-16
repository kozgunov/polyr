from polybot.trading.execution_simulator import fak_sell, limit_buy


def test_limit_below_ask_is_not_filled():
    fill = limit_buy("test", 0.60, 0.61, 100, 5, 0.02)
    assert fill.status == "unfilled"
    assert fill.filled_shares == 0


def test_limit_fill_never_exceeds_requested_price_or_depth():
    fill = limit_buy("deep-book", 0.61, 0.61, 1000, 5, 0.01)
    assert fill.filled_price is None or fill.filled_price <= 0.61
    assert fill.filled_shares <= 5


def test_fak_respects_price_cap_and_available_depth():
    fill = fak_sell("sell", 0.55, 8, 10, 0.54)
    assert fill.filled_price is None or fill.filled_price >= 0.54
    assert fill.filled_shares <= 2
