import math

from polybot.models.timeseries_challenger import forecast


def test_arima_garch_challenger_is_shadow_feature_only() -> None:
    prices = [60_000 * math.exp(0.00015 * index + 0.0005 * math.sin(index / 5)) for index in range(100)]
    result = forecast(prices, horizon_steps=3)
    assert result.execution_role == "shadow_feature_only"
    assert result.arima_price > 0
    assert result.garch_sigma_per_step >= 0
    assert result.arima_direction in {"Up", "Down"}
