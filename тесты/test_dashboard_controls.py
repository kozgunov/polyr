from pathlib import Path
from datetime import UTC, datetime, timedelta

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
        "modeToggle", "applyLiveScale", "stopTrading", "resumeTrading", "manualUnlock", "newPaperRun",
        "applyEntryModel", "applyExitModel", "refreshNow", "exportNow", "loadTrade", "loadTable",
        "tradeRangeFull", "tradeRangeEntry", "tradeRangePosition", "tradeResetZoom",
        "tradeBtcChart", "tradeMarketChart", "tradeEntryConfidenceChart",
        "tradeExitConfidenceChart", "tradePnlChart", "tradeChartStatus",
    ):
        assert f'id="{control_id}"' in html

    javascript = dashboard.STATIC_DIR.joinpath("app.js").read_text(encoding="utf-8")
    assert "без синтетической интерполяции" in javascript
    assert "Вход · средняя цена" in javascript
    assert "Вход · VWAP" in javascript
    assert "Выход · средняя цена" in javascript
    assert "Выход · VWAP" in javascript
    assert "if(!currentTradeKey&&select.value)loadTrade()" not in javascript


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
    engine.db.executemany(
        """INSERT INTO live_orders(
             decision_id,event_slug,token_id,outcome,side,order_id,order_type,
             requested_price,requested_size,matched_size,status,created_at,
             execution_valid,average_fill_price,fill_notional_usdc,fee_usdc)
           VALUES(1,'btc-test','token','Down',?,?,'GTD',?,?,?,'filled',
                  '2026-01-01T00:00:01+00:00',1,?,?,0)""",
        (
            ("BUY", "order-1", 0.40, 2.0, 2.0, 0.40, 0.80),
            ("BUY", "order-2", 0.60, 3.0, 3.0, 0.60, 1.80),
            ("SELL", "order-3", 0.80, 1.0, 1.0, 0.79, 0.79),
        ),
    )
    engine.db.commit(); engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)
    dashboard.overview.cache_clear()

    payload = TestClient(dashboard.app).get("/api/overview").json()

    assert payload["trading"]["mode"] == "live"
    assert payload["active_trading"]["source"] == "live"
    assert len(payload["active_trading"]["open_positions"]) == 1
    assert payload["active_trading"]["unrealized_pnl"] == 1.0
    execution = payload["active_trading"]["execution"]
    assert execution["average_entry_price"] == 0.5
    assert execution["weighted_average_entry_price"] == 0.52
    assert execution["entry_fills"] == 2
    assert execution["entry_filled_shares"] == 5.0
    assert execution["average_exit_price"] == 0.79
    assert execution["weighted_average_exit_price"] == 0.79
    assert execution["exit_fills"] == 1
    assert execution["exit_filled_shares"] == 1.0


def test_dashboard_does_not_show_ended_live_market_as_open(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "dashboard-ended-live.sqlite3"
    engine = PaperEngine(database)
    end_epoch = int(datetime.now(UTC).timestamp()) - 30
    slug = f"btc-updown-5m-{end_epoch - 300}"
    quote_at = datetime.fromtimestamp(end_epoch, UTC) - timedelta(seconds=1)
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('trading_mode','live',?,'test')",
        (datetime.fromtimestamp(end_epoch - 301, UTC).isoformat(),),
    )
    engine.db.execute(
        """CREATE TABLE IF NOT EXISTS market_snapshots(
           id INTEGER PRIMARY KEY,event_slug TEXT,outcome TEXT,best_bid REAL,best_ask REAL,collected_at TEXT)"""
    )
    engine.db.execute(
        "INSERT INTO market_snapshots(event_slug,outcome,best_bid,best_ask,collected_at) VALUES(?,?,?,?,?)",
        (slug, "Up", .99, None, quote_at.isoformat()),
    )
    engine.db.execute(
        "INSERT INTO market_snapshots(event_slug,outcome,best_bid,best_ask,collected_at) VALUES(?,?,?,?,?)",
        (slug, "Down", None, .01, quote_at.isoformat()),
    )
    engine.db.execute(
        """INSERT INTO live_positions(event_slug,token_id,outcome,status,opened_at,average_price,
           shares,cost_usdc,current_price,ledger_validated) VALUES(?,?,?,?,?,?,?,?,?,1)""",
        (slug, "down", "Down", "open", datetime.fromtimestamp(end_epoch - 300, UTC).isoformat(),
         .49, 5.0, 2.45, .49),
    )
    engine.db.commit(); engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)
    dashboard.overview.cache_clear()

    payload = TestClient(dashboard.app).get("/api/overview").json()["active_trading"]

    assert payload["open_positions"] == []
    assert payload["realized_pnl"] == -2.45
    assert payload["recent_positions"][0]["status"] == "provisionally_resolved"
    assert payload["recent_positions"][0]["dashboard_provisional"] is True


def test_manual_unlock_resumes_same_paper_session_only_when_runtime_is_healthy(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "dashboard-unlock.sqlite3"
    engine = PaperEngine(database)
    now = datetime.now(UTC).isoformat()
    for key, value in (
        ("trading_mode", "paper"),
        ("validation_status", "healthy"),
        ("engine_heartbeat", "alive"),
        ("engine_state", "paused"),
    ):
        engine.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
            (key, value, now, "test"),
        )
    engine.db.commit()
    session_id = engine.session_id
    engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)

    response = TestClient(dashboard.app).post(
        "/api/trading-control", json={"action": "manual_unlock"}
    )

    assert response.status_code == 200
    with dashboard.connect() as connection:
        assert dashboard.runtime_control(connection, "engine_state", "") == "running"
        assert "cooldown unlock" in (dashboard.runtime_control_reason(connection, "engine_state") or "")
        session = connection.execute(
            "SELECT session_id,status FROM paper_sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        assert tuple(session) == (session_id, "running")


def test_manual_unlock_never_bypasses_live_or_degraded_validation(
    tmp_path: Path, monkeypatch
) -> None:
    database = tmp_path / "dashboard-unlock-blocked.sqlite3"
    engine = PaperEngine(database)
    now = datetime.now(UTC).isoformat()
    for key, value in (
        ("trading_mode", "live"),
        ("validation_status", "degraded"),
        ("engine_heartbeat", "alive"),
    ):
        engine.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
            (key, value, now, "test"),
        )
    engine.db.commit()
    engine.close()
    monkeypatch.setattr(dashboard.settings, "DATABASE_PATH", database)

    response = TestClient(dashboard.app).post(
        "/api/trading-control", json={"action": "manual_unlock"}
    )

    assert response.status_code == 409
