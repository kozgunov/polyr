from polybot.models.retraining import _reward


def test_sub_ten_cent_net_profit_is_penalized() -> None:
    score, tier = _reward(0.05, 10.0)
    assert score < 0
    assert tier == "inadequate_net_profit"


def test_ten_cent_net_profit_is_a_positive_training_example() -> None:
    score, tier = _reward(0.10, 10.0)
    assert score > 0
    assert tier == "small_profit"
