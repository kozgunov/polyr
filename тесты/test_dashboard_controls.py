from pathlib import Path

from fastapi.testclient import TestClient

from polybot.dashboard import app as dashboard
from polybot.trading.paper_engine import PaperEngine


def test_dashboard_buttons_have_working_local_endpoints(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "dashboard.sqlite3"
    engine = PaperEngine(database)
    engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)
    client = TestClient(dashboard.app)

    assert client.get("/health").status_code == 200
    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["models"]["entry_model"]
    assert overview.json()["models"]["exit_model"]

    for action in ("stop", "resume"):
        response = client.post("/api/trading-control", json={"action": action})
        assert response.status_code == 200

    for role in ("entry", "exit"):
        response = client.post("/api/model-selection", json={"model": "custom", "role": role})
        assert response.status_code == 200
        assert response.json()["role"] == role

    assert client.post("/api/live-scale", json={"scale": 0.1}).status_code == 200
    assert client.get("/api/trades").status_code == 200
    assert client.get("/api/table/runtime_controls?limit=20").status_code == 200


def test_dashboard_html_contains_every_primary_control() -> None:
    html = dashboard.STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")
    for control_id in (
        "modeToggle", "applyLiveScale", "stopTrading", "resumeTrading", "newPaperRun",
        "applyEntryModel", "applyExitModel", "refreshNow", "exportNow", "loadTrade", "loadTable",
    ):
        assert f'id="{control_id}"' in html


def test_one_click_live_is_not_blocked_by_paper_experiment(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "dashboard-live.sqlite3"
    engine = PaperEngine(database)
    engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)
    monkeypatch.setattr(dashboard.settings, "PAPER_ACTION_VALUE_EXPERIMENT_ENABLED", True)
    monkeypatch.setattr(dashboard.settings, "LIVE_EXECUTOR_IMPLEMENTED", True)
    monkeypatch.setattr(dashboard.settings, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(dashboard.settings, "KILL_SWITCH", False)
    client = TestClient(dashboard.app)

    response = client.post("/api/trading-mode", json={"mode": "live", "canary": True})

    assert response.status_code == 200
    assert response.json()["mode"] == "live"
    overview = client.get("/api/overview").json()
    assert overview["trading"]["mode"] == "live"
    assert overview["trading"]["requested_mode"] == "live"


def test_live_mode_overview_uses_live_positions_not_paper(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "dashboard-live-data.sqlite3"
    engine = PaperEngine(database)
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('trading_mode','live','2026-01-01T00:00:00+00:00','test')"
    )
    engine.db.execute(
        """INSERT INTO live_positions(event_slug,token_id,outcome,status,opened_at,average_price,
           shares,cost_usdc,current_price,realized_pnl_usdc,entry_decision_id,exit_stage,
           exit_timing,had_early_exit,early_exit_pnl_usdc)
           VALUES('btc-test','token','Down','open','2026-01-01T00:00:01+00:00',0.4,
                  5,2,0.6,0,NULL,0,'open',0,0)"""
    )
    engine.db.commit(); engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)
    dashboard.overview.cache_clear()

    payload = TestClient(dashboard.app).get("/api/overview").json()

    assert payload["trading"]["mode"] == "live"
    assert payload["active_trading"]["source"] == "live"
    assert len(payload["active_trading"]["open_positions"]) == 1
    assert payload["active_trading"]["unrealized_pnl"] == 1.0
