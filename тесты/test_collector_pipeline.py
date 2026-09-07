import json
from datetime import UTC, datetime

from polybot.collectors.pipeline import Collector, Storage, best, parse_json, slug_from


def test_slug_from_event_url() -> None:
    assert slug_from("https://polymarket.com/event/btc-updown-5m-1785756900") == "btc-updown-5m-1785756900"
    assert slug_from("btc-updown-5m-1785756900") == "btc-updown-5m-1785756900"


def test_parse_json_accepts_gamma_serialized_arrays() -> None:
    assert parse_json('["Yes", "No"]') == ["Yes", "No"]
    assert parse_json("not-json") == []


def test_best_orderbook_levels() -> None:
    levels = [{"price": "0.48", "size": "12"}, {"price": "0.51", "size": "7"}]
    assert best(levels, True) == (0.51, 7.0)
    assert best(levels, False) == (0.48, 12.0)


def test_best_handles_empty_book() -> None:
    assert best([], True) == (None, None)


def test_disabled_raw_market_stream_does_not_grow_database(tmp_path) -> None:
    storage = Storage(str(tmp_path / "collector.sqlite3"))
    try:
        storage.raw("polymarket_market_ws", "event", {"event_type": "price_change"})
        assert storage.db.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0] == 0
    finally:
        storage.close()


def test_cached_preopen_preview_restores_current_token_mapping(tmp_path) -> None:
    storage = Storage(str(tmp_path / "collector.sqlite3"))
    epoch = int(datetime.now(UTC).timestamp() // 300) * 300
    slug = f"btc-updown-5m-{epoch}"
    try:
        for outcome, token in (("Up", "up-token"), ("Down", "down-token")):
            storage.write(
                """INSERT INTO future_event_snapshots(
                   collected_at,source_event_slug,next_event_slug,next_event_title,next_event_url,
                   next_start_time,seconds_before_start,market_id,token_id,outcome,raw_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now(UTC).isoformat(), f"btc-updown-5m-{epoch-300}", slug, "BTC test",
                 f"https://polymarket.com/event/{slug}", datetime.fromtimestamp(epoch, UTC).isoformat(),
                 30.0, "market", token, outcome, json.dumps({"test": True})),
            )
        collector = Collector(slug, storage)
        assert collector._discover_from_cached_preview() is True
        assert {(row["outcome"], row["token_id"]) for row in collector.tokens} == {
            ("Up", "up-token"), ("Down", "down-token"),
        }
        storage.flush()
        event = storage.db.execute("SELECT active,closed FROM events WHERE slug=?", (slug,)).fetchone()
        assert tuple(event) == (1, 0)
    finally:
        storage.close()
