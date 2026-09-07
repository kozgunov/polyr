import json

from polybot.models.exit_sequence_v22 import _book_close_pnl, _executable_close_pnl, _external_reference_at


def test_full_exit_requires_sufficient_depth():
    assert _executable_close_pnl(10, 7, .8, 9.99) is None
    assert _executable_close_pnl(10, 7, .8, 10) is not None


def test_exit_rejects_invalid_contract_prices():
    assert _executable_close_pnl(10, 7, 0, 100) is None
    assert _executable_close_pnl(10, 7, 1, 100) is None


def test_exit_pnl_is_net_of_fee():
    gross = 10 * .8 - 7
    result = _executable_close_pnl(10, 7, .8, 100)
    assert result is not None
    assert result < gross


def test_full_book_depth_is_used_for_vwap_exit():
    snapshot = {
        "best_bid": .80, "best_bid_size": 2,
        "raw_json": json.dumps({"bids": [{"price": ".80", "size": "2"}, {"price": ".798", "size": "8"}]}),
    }
    pnl, average = _book_close_pnl(10, 7, snapshot)
    assert pnl is not None
    assert average == .7984


def test_full_book_exit_rejects_depth_below_price_cap():
    snapshot = {
        "best_bid": .80, "best_bid_size": 2,
        "raw_json": json.dumps({"bids": [{"price": ".80", "size": "2"}, {"price": ".60", "size": "100"}]}),
    }
    pnl, average = _book_close_pnl(10, 7, snapshot)
    assert pnl is None
    assert average is None


def test_external_reference_is_causal_fresh_median():
    rows = {"bybit": [(100.0, 100.0), (110.0, 102.0)], "okx": [(99.0, 100.2)]}
    assert _external_reference_at(rows, 105.0, 15.0) == 100.1
    assert _external_reference_at(rows, 130.0, 15.0) is None
