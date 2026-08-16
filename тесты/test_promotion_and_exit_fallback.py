from __future__ import annotations

import joblib

from polybot.models import action_value, exit_value
from polybot.trading.policy import PositionState
from test_autonomous_policy import market_state


def test_unpromoted_action_value_artifact_never_executes(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "failed_value.joblib"
    joblib.dump({
        "fill_model": object(), "conditional_pnl_model": object(),
        "report": {"promotion_gate": {"passed": False}},
    }, artifact_path)
    monkeypatch.setattr(action_value.settings, "PNL_MODEL_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(action_value.settings, "PAPER_ACTION_VALUE_EXPERIMENT_ENABLED", True)
    action_value._artifact.cache_clear()
    try:
        assert action_value._artifact() is None
    finally:
        action_value._artifact.cache_clear()


def test_exit_fallback_requires_uncertainty_margin(monkeypatch) -> None:
    monkeypatch.setattr(exit_value, "_artifact", lambda: None)
    monkeypatch.setattr(exit_value.settings, "EXIT_VALUE_ENABLED", True)
    monkeypatch.setattr(exit_value.settings, "EXIT_FALLBACK_MIN_ADVANTAGE_USDC", 0.50)
    held = PositionState(1, market_state().event_slug, "Up", "token", 10, 4, 0.4, 0.5)

    weak = exit_value.compare(market_state(), held, held_probability=0.45)
    strong = exit_value.compare(market_state(), held, held_probability=0.30)

    assert weak["close_advantage"] < 0.50
    assert weak["exit"] is False
    assert strong["close_advantage"] >= 0.50
    assert strong["exit"] is True
