import pytest

from polybot.models.bidirectional_entry import canonical_features, feature_names


def _book(distance: float, bid: float, ask: float) -> dict:
    sign = 1 if distance >= 0 else -1
    return {
        "best_bid": bid, "best_ask": ask, "midpoint": (bid + ask) / 2, "spread": ask - bid,
        "best_bid_size": 10, "best_ask_size": 12, "distance_to_target_pct": distance,
        "bybit_to_target_pct": distance, "okx_to_target_pct": distance, "pyth_to_target_pct": distance,
        "realized_volatility_60s_pct": .02, "distance_time_score": distance / .02,
        "distance_lag_15s_pct": distance - sign * .01, "distance_lag_30s_pct": distance - sign * .02,
        "distance_lag_60s_pct": distance - sign * .03, "target_momentum_15s_pct": sign * .01,
        "target_momentum_30s_pct": sign * .02, "target_momentum_60s_pct": sign * .03,
        "bybit_price": 100, "okx_price": 100.01, "pyth_price": 99.99,
    }


def test_mirrored_market_has_identical_candidate_representation() -> None:
    up = _book(.10, .58, .60)
    down = _book(.10, .38, .40)
    history = {}
    original = canonical_features(
        "btc-updown-5m-1788799200", "Up", "2026-09-07T16:42:00+00:00",
        up, down, history, (),
    )
    mirrored_up = _book(-.10, .38, .40)
    mirrored_down = _book(-.10, .58, .60)
    mirrored = canonical_features(
        "btc-updown-5m-1788799200", "Down", "2026-09-07T16:42:00+00:00",
        mirrored_down, mirrored_up, history, (),
    )
    assert feature_names(()) == list(original)
    assert original == mirrored


def test_candidate_history_is_oriented_to_selected_side() -> None:
    history = {
        "history_3_available_fraction": 1, "history_3_up_rate": 2 / 3,
        "history_3_last_up": 1, "history_3_signed_streak": 2,
    }
    up = canonical_features("btc-updown-5m-1788799200", "Up", "2026-09-07T16:42:00+00:00",
                            _book(.1, .5, .51), _book(.1, .48, .49), history, (3,))
    down = canonical_features("btc-updown-5m-1788799200", "Down", "2026-09-07T16:42:00+00:00",
                              _book(.1, .48, .49), _book(.1, .5, .51), history, (3,))
    assert up["history_3_candidate_win_rate"] == 2 / 3
    assert down["history_3_candidate_win_rate"] == pytest.approx(1 / 3)
    assert up["history_3_last_candidate_win"] == 1
    assert down["history_3_last_candidate_win"] == 0
