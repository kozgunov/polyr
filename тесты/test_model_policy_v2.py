import pytest
from polybot.models import model_policy
from polybot.trading.policy import PositionState
from test_trading_policy import state


@pytest.fixture(autouse=True)
def legacy_policy_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Эти тесты документируют прежнюю rule-policy, доступную для сравнений."""
    monkeypatch.setattr(model_policy.settings, "ML_AUTONOMOUS_POLICY_ENABLED", False)
    monkeypatch.setattr(model_policy.settings, "ACTION_VALUE_MODEL_WEIGHT", 0.35)


def market_state(**overrides):
    books = {
        "Up": {"best_bid": 0.68, "best_ask": 0.70, "spread": 0.02},
        "Down": {"best_bid": 0.28, "best_ask": 0.30, "spread": 0.02},
    }
    values = {"book_json": books, "up_bid": 0.68, "up_ask": 0.70, "down_bid": 0.28, "down_ask": 0.30}
    values.update(overrides)
    return state(**values)


def position(outcome: str = "Up") -> PositionState:
    return PositionState(1, "btc-updown-5m-1785756900", outcome, "token", 5, 3.5, 0.70, 0.68)


def adverse_up_state(**overrides):
    values = {
        "reference_price": 62990.0,
        "source_prices": {"bybit": 62991.0, "okx": 62990.0, "pyth": 62989.0},
        "source_returns_pct": {"bybit": -0.02, "okx": -0.02, "pyth": -0.02},
    }
    values.update(overrides)
    return market_state(**values)


def test_holds_without_opposite_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    decision = model_policy.decide_with_model(market_state(), position("Up"))
    assert decision.action == "HOLD"
    assert "hold_held_direction" in decision.tags


def test_catboost_probability_uses_its_metadata_without_name_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def predict_proba(self, rows):
            probability = 0.8 if rows[0][0] == 1.0 else 0.2
            return [[1.0 - probability, probability]]

    monkeypatch.setattr(model_policy, "_catboost_artifact", lambda: (FakeModel(), {"history_windows": []}))
    assert model_policy.probability_up(market_state(), "catboost") == pytest.approx(0.8)


def test_closes_only_on_opposite_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.17)
    decision = model_policy.decide_with_model(adverse_up_state(), position("Up"))
    assert decision.action == "PARTIAL_CLOSE"
    assert "held_direction_exit" in decision.tags


def test_strong_opposite_signal_never_flips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.10)
    decision = model_policy.decide_with_model(adverse_up_state(), position("Up"))
    assert decision.action == "PARTIAL_CLOSE"
    assert "flip_disabled" in decision.tags


def test_opposite_probability_does_not_exit_while_target_side_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.10)
    decision = model_policy.decide_with_model(market_state(), position("Up"))
    assert decision.action == "HOLD"
    assert "five_stage_hold" in decision.tags


def test_profit_alone_does_not_trigger_early_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    cheap = PositionState(1, "btc-updown-5m-1785756900", "Up", "token", 30, 3.0, 0.10, 0.20, 0)
    decision = model_policy.decide_with_model(market_state(up_bid=0.20), cheap)
    assert decision.action == "HOLD"
    assert "five_stage_hold" in decision.tags


def test_profit_exit_can_be_enabled_for_control_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    monkeypatch.setattr(model_policy.settings, "EXIT_ON_PROFIT_ALONE", True)
    cheap = PositionState(1, "btc-updown-5m-1785756900", "Up", "token", 30, 3.0, 0.10, 0.20, 0)
    decision = model_policy.decide_with_model(market_state(up_bid=0.20), cheap)
    assert decision.action == "PARTIAL_CLOSE"
    assert decision.exit_fraction == pytest.approx(0.20)
    assert "staged_profit_exit" in decision.tags


def test_fifth_stage_closes_remainder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.01)
    last_rung = PositionState(1, "btc-updown-5m-1785756900", "Up", "token", 6, 0.6, 0.10, 0.05, 4)
    decision = model_policy.decide_with_model(adverse_up_state(up_bid=0.05), last_rung)
    assert decision.action == "CLOSE"
    assert decision.exit_fraction == 1.0


def test_entry_is_allowed_until_last_30_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    monkeypatch.setattr(model_policy, "expected_pnl", lambda *_: (1.0, {"analytical": 1.0, "learned": 1.0}))
    decision = model_policy.decide_with_model(market_state(remaining_seconds=59))
    assert decision.action == "BUY_UP"


def test_entry_is_allowed_with_29_seconds_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    decision = model_policy.decide_with_model(market_state(remaining_seconds=29))
    assert decision.action == "BUY_UP"


def test_entry_is_blocked_inside_last_15_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    decision = model_policy.decide_with_model(market_state(remaining_seconds=14))
    assert decision.action == "WAIT"
    assert "late_entry_block" in decision.tags


def test_trade_is_blocked_when_target_side_is_not_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.90)
    decision = model_policy.decide_with_model(market_state(reference_price=62990.0))
    assert decision.action == "WAIT"
    assert "target_side_conflict" in decision.tags


def test_balanced_entry_requires_positive_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.84)
    monkeypatch.setattr(model_policy, "expected_pnl", lambda *_: (1.0, {"analytical": 1.0, "learned": 1.0}))
    decision = model_policy.decide_with_model(market_state())
    assert decision.action == "BUY_UP"


def test_action_value_never_inverts_entry_model_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дешёвый противоположный контракт может отклонить вход, но не заменить Down на Up."""
    monkeypatch.setattr(model_policy, "probability_up", lambda *_: 0.10)
    books = {
        "Up": {"best_bid": 0.04, "best_ask": 0.05, "midpoint": 0.05, "spread": 0.01},
        "Down": {"best_bid": 0.94, "best_ask": 0.95, "midpoint": 0.95, "spread": 0.01},
    }
    decision = model_policy.decide_with_model(market_state(
        book_json=books, up_bid=0.04, up_ask=0.05, down_bid=0.94, down_ask=0.95,
    ))
    assert decision.action != "BUY_UP"
    assert "direction_preserved_from_entry_model" in decision.tags


def test_consensus_enters_only_when_models_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "model_is_ready", lambda _: True)

    def signal(_state, _position, key):
        if key == "custom":
            return "Up", 0.82, 0.91, ["numeric"]
        return "Up", 0.88, 0.88, ["llm"]

    monkeypatch.setattr(model_policy, "_signal_for", signal)
    decision = model_policy.decide_with_model(market_state(), model_key="consensus_qwen_custom")
    assert decision.action == "BUY_UP"
    assert "consensus" in decision.tags


def test_consensus_disagreement_does_not_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "model_is_ready", lambda _: True)

    def signal(_state, _position, key):
        if key == "custom":
            return "Up", 0.90, 0.95, ["numeric"]
        return "Down", 0.90, 0.05, ["llm"]

    monkeypatch.setattr(model_policy, "_signal_for", signal)
    decision = model_policy.decide_with_model(market_state(), model_key="consensus_qwen_custom")
    assert decision.action == "WAIT"
    assert "consensus_disagreement" in decision.tags


def test_consensus_opposite_signal_closes_without_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_policy, "model_is_ready", lambda _: True)
    monkeypatch.setattr(
        model_policy,
        "_signal_for",
        lambda *_: ("Down", 0.90, 0.05, ["confirmed"]),
    )
    decision = model_policy.decide_with_model(
        adverse_up_state(), position("Up"), model_key="consensus_qwen_custom",
    )
    assert decision.action == "PARTIAL_CLOSE"
    assert "flip_disabled" in decision.tags
