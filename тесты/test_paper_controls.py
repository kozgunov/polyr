from pathlib import Path
from datetime import UTC, datetime, timedelta

import pytest
from polybot.trading.paper_engine import PaperEngine


@pytest.fixture(autouse=True)
def legacy_confirmation_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Старые confirmation-тесты остаются как контроль legacy-policy."""
    monkeypatch.setattr("polybot.trading.paper_engine.settings.ML_AUTONOMOUS_POLICY_ENABLED", False)
from polybot.trading.policy import Decision
from polybot.trading.policy import PositionState
from polybot.trading.execution_simulator import SimulatedFill
from test_model_policy_v2 import market_state


def test_signal_must_persist_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    ticks = iter((0.0, 2.0, 4.0, 6.0))
    monkeypatch.setattr("polybot.trading.paper_engine.time.monotonic", lambda: next(ticks))
    candidate = Decision("BUY_UP", 0.8, "candidate", ["model_driven"], "Up", 0.70, 3.0)
    actions = [engine._confirmed(candidate, market_state(), None).action for _ in range(4)]
    assert actions == ["WAIT", "WAIT", "WAIT", "BUY_UP"]
    engine.close()


def test_entry_grid_respects_model_budget_clob_minimum_and_better_prices() -> None:
    plan = PaperEngine._entry_grid_plan(10.0, 0.69, 0.01, 5.0)
    assert [price for price, _ in plan] == [0.66, 0.63, 0.60]
    assert sum(notional for _, notional in plan) == pytest.approx(10.0)
    assert all(notional >= 1.0 and notional / price >= 5.0 for price, notional in plan)


def test_entry_grid_does_not_increase_small_model_position() -> None:
    assert PaperEngine._entry_grid_plan(3.0, 0.69, 0.01, 5.0) == []


def test_three_losses_pause_same_session_for_thirty_minutes(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    for index in range(3):
        engine.db.execute(
            """INSERT INTO paper_positions(session_id,event_slug,token_id,outcome,status,opened_at,
               average_price,shares,cost_usdc,closed_at,realized_pnl_usdc,execution_valid)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,1)""",
            (engine.session_id, f"btc-updown-5m-{4102440000 + index * 300}", f"token-{index}",
             "Up", "resolved", f"2026-01-01T00:0{index}:00+00:00", 0.5, 2.0, 1.0,
             f"2026-01-01T00:0{index}:30+00:00", -1.0),
        )
    engine.db.commit()

    assert engine._enforce_loss_streak() is True
    status = engine.db.execute(
        "SELECT status,consecutive_losses FROM paper_sessions WHERE session_id=?", (engine.session_id,)
    ).fetchone()
    assert tuple(status) == ("paused", 3)
    until = datetime.fromisoformat(engine._control("paper_cooldown_until", ""))
    remaining = (until - datetime.now(UTC)).total_seconds()
    assert 29 * 60 < remaining <= 30 * 60
    engine.close()


def test_new_session_archives_history_and_resets_budget(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    previous = engine.session_id
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','new_session_requested','now','test')"
    )
    engine.db.commit()
    assert engine._handle_runtime_control() == "running"
    assert engine.session_id != previous
    old = engine.db.execute("SELECT status FROM paper_sessions WHERE session_id=?", (previous,)).fetchone()
    current = engine.db.execute(
        "SELECT initial_balance_usdc,cash_balance_usdc,status FROM paper_sessions WHERE session_id=?",
        (engine.session_id,),
    ).fetchone()
    assert old[0] == "archived"
    assert tuple(current) == (300.0, 300.0, "running")
    engine.close()


def test_preopen_shadow_order_is_replaced_without_deleting_history(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    next_slug = "btc-updown-5m-4102444800"
    engine._upsert_preopen_shadow_order(1, "source", next_slug, "Up", "up", .40, 2.0, .75)
    engine._upsert_preopen_shadow_order(2, "source", next_slug, "Down", "down", .35, 3.0, .80)
    rows = engine.db.execute(
        "SELECT outcome,status FROM shadow_preopen_orders WHERE next_event_slug=? ORDER BY id",
        (next_slug,),
    ).fetchall()
    assert [tuple(row) for row in rows] == [("Up", "cancelled_replaced"), ("Down", "working")]
    engine.close()


def _ended_event_with_quotes(engine: PaperEngine, held_bid: float, held_ask: float) -> tuple[int, str]:
    end_epoch = int(datetime.now(UTC).timestamp()) - 11
    slug = f"btc-updown-5m-{end_epoch - 300}"
    engine.db.execute(
        """CREATE TABLE IF NOT EXISTS market_snapshots(
           id INTEGER PRIMARY KEY,event_slug TEXT,outcome TEXT,best_bid REAL,best_ask REAL,collected_at TEXT)"""
    )
    quote_at = datetime.fromtimestamp(end_epoch, UTC) - timedelta(seconds=1)
    engine.db.execute(
        "INSERT INTO market_snapshots(event_slug,outcome,best_bid,best_ask,collected_at) VALUES(?,?,?,?,?)",
        (slug, "Up", held_bid, held_ask, quote_at.isoformat()),
    )
    engine.db.execute(
        "INSERT INTO market_snapshots(event_slug,outcome,best_bid,best_ask,collected_at) VALUES(?,?,?,?,?)",
        (slug, "Down", max(0.0, 1.0 - held_ask), min(1.0, 1.0 - held_bid), quote_at.isoformat()),
    )
    cursor = engine.db.execute(
        """INSERT INTO paper_positions(session_id,event_slug,token_id,outcome,status,opened_at,
           average_price,shares,cost_usdc) VALUES(?,?,?,?,?,?,?,?,?)""",
        (engine.session_id, slug, "up", "Up", "open",
         datetime.fromtimestamp(end_epoch - 300, UTC).isoformat(), .80, 5.0, 4.0),
    )
    engine.db.execute(
        "UPDATE paper_sessions SET cash_balance_usdc=296.0 WHERE session_id=?", (engine.session_id,),
    )
    engine.db.commit()
    return int(cursor.lastrowid), slug


def test_post_event_extreme_quote_provisionally_releases_paper_position(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    position_id, _ = _ended_event_with_quotes(engine, .99, 1.0)
    engine._provisionally_settle_ended_paper_positions()
    position = engine.db.execute(
        "SELECT status,close_price,provisional_label,realized_pnl_usdc FROM paper_positions WHERE id=?",
        (position_id,),
    ).fetchone()
    session = engine.db.execute(
        "SELECT cash_balance_usdc,realized_pnl_usdc FROM paper_sessions WHERE session_id=?",
        (engine.session_id,),
    ).fetchone()
    assert tuple(position) == ("provisionally_resolved", 1.0, 1, 1.0)
    assert tuple(session) == (301.0, 1.0)
    engine.close()


def test_post_event_non_extreme_quote_waits_for_official_resolution(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    position_id, _ = _ended_event_with_quotes(engine, .96, .97)
    engine._provisionally_settle_ended_paper_positions()
    status = engine.db.execute("SELECT status FROM paper_positions WHERE id=?", (position_id,)).fetchone()[0]
    assert status == "open"
    engine.close()


def test_official_resolution_reconciles_provisional_mismatch(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    position_id, slug = _ended_event_with_quotes(engine, .99, 1.0)
    engine._provisionally_settle_ended_paper_positions()
    engine.db.execute(
        """CREATE TABLE training_examples(
           id INTEGER PRIMARY KEY,event_slug TEXT,outcome TEXT,token_id TEXT,label INTEGER)"""
    )
    engine.db.execute(
        "INSERT INTO event_resolutions(event_slug,outcome,token_id,label,resolved_at,source) VALUES(?,?,?,?,?,?)",
        (slug, "Up", "up", 0, datetime.now(UTC).isoformat(), "test_official"),
    )
    engine.db.commit()
    engine.settle_resolved()
    position = engine.db.execute(
        "SELECT status,close_price,official_label,provisional_mismatch,realized_pnl_usdc FROM paper_positions WHERE id=?",
        (position_id,),
    ).fetchone()
    session = engine.db.execute(
        "SELECT cash_balance_usdc,realized_pnl_usdc FROM paper_sessions WHERE session_id=?",
        (engine.session_id,),
    ).fetchone()
    assert tuple(position) == ("resolved", 0.0, 0, 1, -4.0)
    assert tuple(session) == (296.0, -4.0)
    engine.close()


def test_unsettled_previous_event_does_not_mask_current_event_position(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    current = market_state()
    previous_slug = current.event_slug.rsplit("-", 1)[0] + "-" + str(int(current.event_slug.rsplit("-", 1)[1]) - 300)
    for slug, outcome in ((previous_slug, "Down"), (current.event_slug, "Up")):
        engine.db.execute(
            """INSERT INTO paper_positions(session_id,event_slug,token_id,outcome,status,opened_at,
               average_price,shares,cost_usdc) VALUES(?,?,?,?,?,?,?,?,?)""",
            (engine.session_id, slug, f"{slug}-token", outcome, "open", current.observed_at, .7, 2, 1.4),
        )
    engine.db.commit()
    position = engine.open_position(current)
    assert position is not None
    assert position.event_slug == current.event_slug
    assert position.outcome == "Up"
    engine.close()


def test_risk_exit_requires_five_confirmations_and_ten_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    ticks = iter((0.0, 2.0, 4.0, 6.0, 10.0))
    monkeypatch.setattr("polybot.trading.paper_engine.time.monotonic", lambda: next(ticks))
    candidate = Decision(
        "PARTIAL_CLOSE", 0.9, "risk rung",
        ["staged_risk_exit", "exit_stage=1"], exit_fraction=0.2,
    )
    held = PositionState(1, market_state().event_slug, "Up", "token", 10, 3, 0.3, 0.2)
    actions = [engine._confirmed(candidate, market_state(), held).action for _ in range(5)]
    assert actions == ["HOLD", "HOLD", "HOLD", "HOLD", "PARTIAL_CLOSE"]
    engine.close()


def test_entry_limit_fill_has_no_adverse_slippage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polybot.trading.paper_engine.settings.EXECUTION_SIMULATION_ENABLED", False)
    monkeypatch.setattr("polybot.trading.paper_engine.settings.ENTRY_GRID_ENABLED", False)
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    current = market_state()
    current.book_json["Up"]["token_id"] = "up-token"
    current.book_json["Up"]["collected_at"] = datetime.now(UTC).isoformat()
    decision = Decision("BUY_UP", 0.85, "limit", ["model"], "Up", 0.70, 3.0)
    engine.db.execute(
        "INSERT INTO model_decisions(session_id,observed_at,event_slug,action,confidence,reason,tags_json,market_state_json,provider,model_name) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (engine.session_id, current.observed_at, current.event_slug, "BUY_UP", 0.85, "limit", "[]", "{}", "test", "custom"),
    )
    decision_id = int(engine.db.execute("SELECT last_insert_rowid()").fetchone()[0])
    engine._buy(decision_id, current, decision)
    order = engine.db.execute(
        "SELECT requested_price,filled_price,slippage_bps,order_type FROM paper_orders ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert tuple(order) == (0.70, 0.70, 0.0, "GTD")
    engine.close()


def test_stale_book_blocks_entry_at_execution_layer(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    current = market_state()
    current.book_json["Up"].update({"token_id": "up-token", "collected_at": "2020-01-01T00:00:00+00:00"})
    decision = Decision("BUY_UP", 0.85, "stale", ["model"], "Up", 0.70, 3.0)
    engine._buy(1, current, decision)
    assert engine.db.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0] == 0
    engine.close()


def test_partial_resting_entry_keeps_and_fills_remainder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    current = market_state()
    current.book_json["Up"]["token_id"] = "up-token"
    engine.db.execute(
        """INSERT INTO model_decisions(session_id,observed_at,event_slug,action,confidence,reason,
           tags_json,market_state_json,provider,model_name) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (engine.session_id, current.observed_at, current.event_slug, "BUY_UP", 0.85,
         "partial", "[]", "{}", "test", "catboost"),
    )
    decision_id = int(engine.db.execute("SELECT last_insert_rowid()").fetchone()[0])
    engine.db.execute(
        """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,
           requested_price,shares,notional_usdc,fee_usdc,slippage_bps,status,created_at,
           expiration_at,requested_shares,requested_notional_usdc,check_count)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (engine.session_id, decision_id, current.event_slug, "BUY_UP", "GTD", 0.70,
         0, 0, 0, 0, "unfilled", current.observed_at, "2999-01-01T00:00:00+00:00", 10.0, 7.0, 1),
    )
    fills = iter((2.0, 3.0))
    monkeypatch.setattr(
        "polybot.trading.paper_engine.limit_buy",
        lambda *args, **kwargs: SimulatedFill("partially_filled", next(fills), 0.69, 0.9, 420, 0.0, "partial"),
    )
    engine._reconcile_paper_entry_orders(current)
    engine._reconcile_paper_entry_orders(current)
    position = engine.db.execute("SELECT shares FROM paper_positions WHERE status='open'").fetchone()
    order = engine.db.execute("SELECT shares,status FROM paper_orders").fetchone()
    assert float(position[0]) == pytest.approx(5.0)
    assert float(order[0]) == pytest.approx(5.0)
    assert order[1] == "partially_filled"
    engine.close()


def test_limit_exit_cannot_exceed_visible_bid_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    current = market_state()
    current.book_json["Up"].update({
        "token_id": "up-token", "collected_at": datetime.now(UTC).isoformat(), "best_bid_size": 2.0,
    })
    engine.db.execute(
        """INSERT INTO paper_positions(session_id,event_slug,token_id,outcome,status,opened_at,
           average_price,shares,cost_usdc,current_price,entry_decision_id)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (engine.session_id, current.event_slug, "up-token", "Up", "open", current.observed_at,
         0.5, 10.0, 5.0, 0.68, 1),
    )
    position = engine.open_position(current)
    monkeypatch.setattr(
        "polybot.trading.paper_engine.fak_sell",
        lambda *args, **kwargs: SimulatedFill("partially_filled", 2.0, 0.68, 0.2, 400, 0.0, "depth"),
    )
    engine._close(2, current, position, "limit exit", 1.0, 1, limit_exit=True)
    row = engine.db.execute("SELECT status,shares FROM paper_positions").fetchone()
    order = engine.db.execute("SELECT status,shares,requested_price,filled_price FROM paper_orders").fetchone()
    assert tuple(row) == ("open", 8.0)
    assert tuple(order) == ("partially_filled", 2.0, 0.68, 0.68)
    engine.close()


def test_unfilled_gtd_order_can_fill_later_as_maker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    current = market_state()
    current.book_json["Up"]["token_id"] = "up-token"
    engine.db.execute(
        """INSERT INTO model_decisions(session_id,observed_at,event_slug,action,confidence,reason,
           tags_json,market_state_json,provider,model_name) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (engine.session_id, current.observed_at, current.event_slug, "BUY_UP", 0.85,
         "resting limit", "[]", "{}", "test", "catboost"),
    )
    decision_id = int(engine.db.execute("SELECT last_insert_rowid()").fetchone()[0])
    engine.db.execute(
        """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,
           requested_price,shares,notional_usdc,fee_usdc,slippage_bps,status,created_at,
           expiration_at,requested_shares,requested_notional_usdc,check_count)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (engine.session_id, decision_id, current.event_slug, "BUY_UP", "GTD", 0.70,
         0, 0, 0, 0, "unfilled", current.observed_at, "2999-01-01T00:00:00+00:00",
         4.0, 2.8, 1),
    )
    engine.db.commit()
    monkeypatch.setattr(
        "polybot.trading.paper_engine.limit_buy",
        lambda *args, **kwargs: SimulatedFill("filled", 4.0, 0.69, 0.9, 420, 0.0, "test_fill"),
    )
    engine._reconcile_paper_entry_orders(current)
    position = engine.db.execute(
        "SELECT outcome,shares,average_price,fees_usdc FROM paper_positions WHERE status='open'"
    ).fetchone()
    order = engine.db.execute("SELECT status,execution_reason FROM paper_orders").fetchone()
    assert tuple(position) == ("Up", 4.0, 0.69, 0.0)
    assert tuple(order) == ("filled", "gtd_resting_maker_fill")
    engine.close()


def test_entry_and_exit_models_are_independent(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('selected_entry_model','catboost','now','test')"
    )
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('selected_exit_model','qwen','now','test')"
    )
    engine.db.commit()
    assert engine._control("selected_entry_model", "") == "catboost"
    assert engine._control("selected_exit_model", "") == "qwen"
    engine.close()


def test_queued_model_switch_applies_immediately_when_flat(tmp_path: Path) -> None:
    engine = PaperEngine(tmp_path / "paper.sqlite3")
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('pending_entry_model','catboost','now','test')"
    )
    engine.db.execute(
        "INSERT OR REPLACE INTO runtime_controls VALUES('pending_entry_model_apply_at','2999-01-01T00:00:00+00:00','now','test')"
    )
    engine.db.commit()
    engine._apply_pending_models()
    assert engine._control("selected_entry_model", "") == "catboost"
    assert engine._control("pending_entry_model", "") == ""
    engine.close()
