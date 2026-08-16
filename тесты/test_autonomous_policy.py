from __future__ import annotations

import pytest

from polybot.models import autonomous_policy
from polybot.trading.policy import PositionState
from test_trading_policy import state


@pytest.fixture(autouse=True)
def historical_fixture_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autonomous_policy.settings, "ML_POLICY_REQUIRE_FRESH_DATA", False)


def market_state():
    return state(
        target_price=63000.0,
        reference_price=63010.0,
        book_json={
            "Up": {"best_bid": 0.45, "midpoint": 0.46, "best_ask": 0.47},
            "Down": {"best_bid": 0.53, "midpoint": 0.54, "best_ask": 0.55},
        },
        up_bid=0.45, up_ask=0.47, down_bid=0.53, down_ask=0.55,
    )


def test_ml_policy_selects_best_action_price_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    def utility(_state, outcome, _probability, price, notional):
        score = (2.0 if outcome == "Down" else 1.0) + (0.6 - price) + notional / 100
        return score, {"learned": score, "fill_probability": 0.8, "pnl_if_filled": score}

    monkeypatch.setattr(autonomous_policy, "expected_pnl", utility)
    decision = autonomous_policy.decide(market_state(), None, {"Up": 0.75, "Down": 0.25}, "custom")
    assert decision.action == "BUY_UP"
    assert decision.direction == "Up"
    assert decision.limit_price == 0.45
    # При слабой вероятности $5 запрещены: обычный размер должен оставаться $2–3.
    assert decision.notional_usdc == 3.0
    assert decision.confidence == 0.75
    assert "direction_preserved_from_entry_model" in decision.tags
    assert any(tag.startswith("utility_margin_score=") for tag in decision.tags)


def test_ml_policy_waits_below_direction_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        autonomous_policy,
        "expected_pnl",
        lambda *_: (10.0, {"learned": 10.0, "fill_probability": 1.0, "pnl_if_filled": 10.0}),
    )
    decision = autonomous_policy.decide(market_state(), None, {"Up": 0.70, "Down": 0.30}, "custom")
    assert decision.action == "WAIT"
    assert decision.confidence == 0.70
    assert "entry_model_low_confidence_wait" in decision.tags


def test_ml_policy_never_buys_after_event_entry_window_closed() -> None:
    closed = market_state()
    closed.remaining_seconds = -30
    decision = autonomous_policy.decide(closed, None, {"Up": 0.99, "Down": 0.01}, "custom")
    assert decision.action == "WAIT"
    assert "entry_window_closed" in decision.tags


def test_ml_policy_allows_five_dollars_only_for_very_high_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def utility(_state, outcome, _probability, _price, notional):
        score = (1.0 if outcome == "Up" else 0.0) + notional / 10
        return score, {"learned": score, "fill_probability": 0.9, "pnl_if_filled": score}

    monkeypatch.setattr(autonomous_policy, "expected_pnl", utility)
    decision = autonomous_policy.decide(market_state(), None, {"Up": 0.90, "Down": 0.10}, "custom")
    assert decision.action == "BUY_UP"
    assert decision.notional_usdc == 5.0
    assert "ml_entry_argmax" in decision.tags


def test_ml_policy_waits_when_wait_has_highest_utility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autonomous_policy, "expected_pnl", lambda *_: (-0.1, {"learned": -0.1}))
    decision = autonomous_policy.decide(market_state(), None, {"Up": 0.75, "Down": 0.25}, "custom")
    assert decision.action == "WAIT"
    assert "ml_entry_argmax" in decision.tags


def test_ml_exit_directly_chooses_close_without_manual_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autonomous_policy, "compare_exit_value", lambda *_: {
        "close_pnl": 0.2, "hold_pnl": -0.4, "close_advantage": 0.6,
        "learned_close_advantage": 0.7, "learned_close_probability": 0.9, "exit": True,
    })
    held = PositionState(1, market_state().event_slug, "Up", "token", 5, 3, 0.6, 0.5)
    decision = autonomous_policy.decide(market_state(), held, {"Up": 0.4, "Down": 0.6}, "catboost")
    assert decision.action == "CLOSE"
    assert decision.exit_fraction == 1.0
    assert "ml_exit_argmax" in decision.tags
