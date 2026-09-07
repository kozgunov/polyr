from pathlib import Path

import pytest

from polybot.trading.paper_engine import PaperEngine


def test_clob_history_restores_fill_after_local_order_was_marked_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = PaperEngine(tmp_path / "live-sync.sqlite3")
    decision = engine.db.execute(
        """INSERT INTO model_decisions(session_id,observed_at,event_slug,action,confidence,reason,
           tags_json,market_state_json,provider,model_name,executed)
           VALUES(?,?,?,?,?,?,?,?,?,?,1)""",
        (engine.session_id, "2026-09-06T10:00:00+00:00", "btc-updown-5m-1788702600",
         "BUY_DOWN", .8, "test", "[]", "{}", "model_registry", "custom"),
    ).lastrowid
    engine.db.execute(
        """INSERT INTO live_orders(decision_id,event_slug,token_id,outcome,side,order_id,
           order_type,requested_price,requested_size,status,created_at,execution_valid,invalid_reason)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (decision, "btc-updown-5m-1788702600", "down-token", "Down", "BUY", "order-1",
         "GTD", .80, 5.0, "cancelled_domain_guard", "2026-09-06T10:00:00+00:00", 0, "event_ended"),
    )
    engine.db.commit()
    monkeypatch.setattr(
        "polybot.trading.paper_engine.get_account_trades",
        lambda: [{"taker_order_id": "order-1", "size": "5", "price": ".79",
                  "fee_rate_bps": "20", "match_time": "1788702682", "maker_orders": []}],
    )

    assert engine._sync_live_fills_from_history() == 1
    order = engine.db.execute(
        "SELECT status,matched_size,average_fill_price,fill_notional_usdc,fee_usdc,fill_source,execution_valid,invalid_reason "
        "FROM live_orders WHERE order_id='order-1'"
    ).fetchone()
    assert order[0] == "filled"
    expected_fee = 5.0 * (20 / 10_000) * .79 * (1 - .79)
    assert tuple(order[1:5]) == pytest.approx((5.0, .79, 3.95, expected_fee), abs=1e-5)
    assert tuple(order[5:]) == ("clob_account_trade_history", 1, None)
    position = engine.db.execute(
        "SELECT outcome,status,shares,average_price,cost_usdc,fees_usdc,ledger_validated "
        "FROM live_positions WHERE event_slug='btc-updown-5m-1788702600'"
    ).fetchone()
    assert position[0:2] == ("Down", "open")
    assert tuple(position[2:]) == pytest.approx((5.0, .79, 3.95 + expected_fee, expected_fee, 1), abs=1e-5)
    engine.close()


def test_clob_v2_fee_corrects_an_already_matched_live_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Нулевой fee_rate_bps из V2 trade history не означает fee-free.

    Это регрессия для случая: CLOB fill=0.500, 5 shares, а UI Polymarket
    показывает эффективную цену 0.5175 после taker fee $0.0875.
    """
    engine = PaperEngine(tmp_path / "live-v2-fee-sync.sqlite3")
    decision = engine.db.execute(
        """INSERT INTO model_decisions(session_id,observed_at,event_slug,action,confidence,reason,
           tags_json,market_state_json,provider,model_name,executed)
           VALUES(?,?,?,?,?,?,?,?,?,?,1)""",
        (engine.session_id, "2026-09-07T16:43:00+00:00", "btc-updown-5m-1788799200",
         "BUY_DOWN", .8, "test", "[]", "{}", "model_registry", "custom"),
    ).lastrowid
    engine.db.execute(
        """INSERT INTO live_orders(decision_id,event_slug,token_id,outcome,side,order_id,
           order_type,requested_price,requested_size,matched_size,average_fill_price,
           fill_notional_usdc,fee_usdc,fill_source,status,created_at,execution_valid)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        (decision, "btc-updown-5m-1788799200", "down-token", "Down", "BUY", "order-v2",
         "GTD", .52, 5.0, 5.0, .50, 2.50, 0.0, "clob_account_trade_history",
         "filled", "2026-09-07T16:43:00+00:00"),
    )
    engine.db.execute(
        """INSERT INTO live_positions(event_slug,token_id,outcome,status,opened_at,average_price,
           shares,cost_usdc,current_price,entry_decision_id,fees_usdc,ledger_validated,execution_valid)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        ("btc-updown-5m-1788799200", "down-token", "Down", "open",
         "2026-09-07T16:43:00+00:00", .50, 5.0, 2.50, .50, decision, 0.0, 1),
    )
    engine.db.commit()
    monkeypatch.setattr(
        "polybot.trading.paper_engine.get_account_trades",
        lambda: [{"taker_order_id": "order-v2", "size": "5", "price": ".50",
                  "fee_rate_bps": "0", "match_time": "1788799391", "maker_orders": []}],
    )

    assert engine._sync_live_fills_from_history() == 1
    expected_fee = 5.0 * .07 * .50 * (1.0 - .50)
    order = engine.db.execute(
        "SELECT average_fill_price,fill_notional_usdc,fee_usdc FROM live_orders WHERE order_id='order-v2'"
    ).fetchone()
    assert tuple(order) == pytest.approx((.50, 2.50, expected_fee), abs=1e-5)
    position = engine.db.execute(
        "SELECT average_price,cost_usdc,fees_usdc FROM live_positions WHERE event_slug='btc-updown-5m-1788799200'"
    ).fetchone()
    assert tuple(position) == pytest.approx((.50, 2.50 + expected_fee, expected_fee), abs=1e-5)
    engine.close()
