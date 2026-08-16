import pytest
from polybot.trading.fees import net_buy_edge, platform_fee_usdc
from polybot.trading.policy import confidence_for
from test_trading_policy import state


def test_crypto_taker_fee_is_symmetric_around_half() -> None:
    assert platform_fee_usdc(100, 0.30) == pytest.approx(platform_fee_usdc(100, 0.70))
    assert platform_fee_usdc(100, 0.50) == pytest.approx(1.75)


def test_maker_has_no_platform_fee() -> None:
    assert platform_fee_usdc(100, 0.50, taker=False) == 0.0


def test_probability_equal_to_price_is_negative_edge_after_fee() -> None:
    assert net_buy_edge(0.80, 0.80) < 0


def test_legacy_confidence_is_direction_symmetric() -> None:
    up = confidence_for(state(source_returns_pct={"bybit": 0.2, "okx": 0.2, "pyth": 0.2}))
    down = confidence_for(state(source_returns_pct={"bybit": -0.2, "okx": -0.2, "pyth": -0.2}))
    assert up == pytest.approx(down)
