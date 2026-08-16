import pytest

import app_config as settings
from polybot.trading.live_executor import _gtd_expiration, submit_limit_order


class _Clock:
    def get_server_time(self) -> int:
        return 1_700_000_000


def test_gtd_expiration_uses_clob_time_and_meets_minimum() -> None:
    """CLOB V2 отклоняет GTD, если expiration ближе трёх минут."""
    expiration = _gtd_expiration(_Clock(), lifetime_seconds=20)  # type: ignore[arg-type]
    assert expiration >= 1_700_000_185


def test_live_submission_requires_explicit_flag() -> None:
    with pytest.raises(RuntimeError, match="EXPLICIT_SUBMIT"):
        submit_limit_order(token_id="1", side="BUY", price=0.5, size=2.0)


def test_live_submission_obeys_configuration_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LIVE_TRADING_ENABLED", False)
    monkeypatch.setattr(settings, "KILL_SWITCH", True)
    with pytest.raises(RuntimeError, match="BLOCKED_BY_CONFIGURATION"):
        submit_limit_order(token_id="1", side="BUY", price=0.5, size=2.0, submit=True)
