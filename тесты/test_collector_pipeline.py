from polybot.collectors.pipeline import Storage, best, parse_json, slug_from


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
