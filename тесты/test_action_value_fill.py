import pytest

from polybot.models import action_value
from test_trading_policy import state


def test_fill_probability_weights_analytical_limit_value(monkeypatch: pytest.MonkeyPatch) -> None:
    market = state(
        book_json={
            "Up": {"best_bid": 0.45, "best_ask": 0.47, "midpoint": 0.46},
            "Down": {"best_bid": 0.53, "best_ask": 0.55, "midpoint": 0.54},
        },
        up_bid=0.45, up_ask=0.47, down_bid=0.53, down_ask=0.55,
    )
    monkeypatch.setattr(action_value, "_artifact", lambda: None)
    monkeypatch.setattr(action_value, "_fill_artifact", lambda: {"fill_model": object()})
    monkeypatch.setattr(action_value, "_predict_fill", lambda *_: 0.4)
    value, parts = action_value.expected_pnl(market, "Up", 0.75, 0.47, 3.0)
    assert parts["fill_probability"] == pytest.approx(0.4)
    assert value == pytest.approx(parts["analytical"] * 0.4)
