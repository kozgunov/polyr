from polybot.models.exit_features import FEATURES, feature_map, vector


def _state(distance: float) -> dict:
    return {
        "distance_to_target_pct": distance,
        "remaining_seconds": 120,
        "realized_volatility_60s_pct": 0.02,
        "source_disagreement_pct": 0.01,
        "target_distance_lags_pct": {"15": distance - 0.01},
        "book_json": {
            "Up": {"spread": 0.02, "best_bid_size": 10, "best_ask_size": 20},
            "Down": {"spread": 0.02, "best_bid_size": 20, "best_ask_size": 10},
        },
    }


def test_distance_is_oriented_to_held_contract() -> None:
    up = feature_map(_state(0.10), "Up", 0.60, 10, 5)
    down = feature_map(_state(0.10), "Down", 0.40, 10, 5)
    assert up["oriented_distance_to_target_pct"] == 0.10
    assert down["oriented_distance_to_target_pct"] == -0.10
    assert up["held_side_currently_winning"] == 1.0
    assert down["held_side_currently_winning"] == 0.0


def test_feature_vector_has_stable_schema_and_finite_values() -> None:
    values = feature_map(_state(-0.10), "Down", 0.70, 10, 5)
    result = vector(values)
    assert len(result) == len(FEATURES)
    assert all(value == value for value in result)
