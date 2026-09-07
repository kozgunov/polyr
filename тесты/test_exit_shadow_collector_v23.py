import sqlite3

from polybot.trading.exit_shadow_collector import SCHEMA


def test_shadow_exit_schema_is_idempotent():
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA)
    connection.executescript(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(shadow_exit_orders_v23)")}
    assert {"horizon_seconds", "filled_size", "strategy_net_pnl_usdc", "advantage_vs_hold_usdc"} <= columns


def test_shadow_exit_unique_sampling_key():
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA)
    values = ("paper", 1, "s", "btc-updown-5m-1", "Up", "m", "2026-01-01T00:00:00+00:00",
              1, 5, "2026-01-01T00:00:05+00:00", .5, 10, 5, .5)
    sql = """INSERT OR IGNORE INTO shadow_exit_orders_v23(
             source,position_id,session_id,event_slug,outcome,entry_model,observed_at,sample_bucket,
             horizon_seconds,expires_at,limit_price,position_shares,position_cost_usdc,average_entry_price)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    assert connection.execute(sql, values).rowcount == 1
    assert connection.execute(sql, values).rowcount == 0
