import pytest
from polybot.runtime import run_async, run_sync


def test_sync_success_marker(capsys):
    run_sync("worker.py", lambda: None)
    assert capsys.readouterr().out.strip() == "all_doneworker.py"


def test_sync_error_marker(capsys):
    def fail() -> None:
        raise ValueError("test failure")

    with pytest.raises(ValueError):
        run_sync("worker.py", fail)
    assert capsys.readouterr().out.strip() == "error_in_worker.py"


def test_help_exit_is_success(capsys):
    def help_exit() -> None:
        raise SystemExit(0)

    with pytest.raises(SystemExit) as raised:
        run_sync("worker.py", help_exit)
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == "all_doneworker.py"


def test_async_success_marker(capsys):
    async def finish() -> None:
        return None

    run_async("async_worker.py", finish)
    assert capsys.readouterr().out.strip() == "all_doneasync_worker.py"
