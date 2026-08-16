from polybot.models.market_regime import RegimeThresholds, classify, fit_thresholds, one_hot


def test_regime_uses_trend_volatility_phase_and_target_distance() -> None:
    thresholds = RegimeThresholds(
        volatility_low=0.01,
        volatility_high=0.03,
        momentum_flat=0.005,
        distance_near=0.02,
        distance_far=0.08,
    )
    regime = classify(
        {
            "realized_volatility_60s_pct": 0.04,
            "target_momentum_30s_pct": -0.02,
            "distance_to_target_pct": 0.01,
            "remaining_seconds": 40,
        },
        thresholds,
    )
    assert regime == {
        "trend": "downtrend",
        "volatility": "high",
        "phase": "late",
        "distance": "near",
        "expert": "downtrend__high",
    }
    assert len(one_hot(regime)) == 12
    assert sum(one_hot(regime)) == 4


def test_thresholds_are_fitted_from_the_passed_training_rows_only() -> None:
    training = [
        {
            "realized_volatility_60s_pct": value,
            "target_momentum_30s_pct": value / 2,
            "distance_to_target_pct": value * 2,
        }
        for value in (0.01, 0.02, 0.03, 0.04, 0.05)
    ]
    thresholds = fit_thresholds(training)
    assert 0.01 <= thresholds.volatility_low < thresholds.volatility_high <= 0.05
    assert 0.005 <= thresholds.momentum_flat <= 0.025
    assert 0.02 <= thresholds.distance_near < thresholds.distance_far <= 0.10
