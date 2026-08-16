from polybot.trading.live_guard import wilson_lower_bound
from polybot.trading.policy import MarketState, PositionState, decide


def state(**overrides) -> MarketState:
    values = {
        "event_slug": "btc-updown-5m-1785756900",
        "observed_at": "2026-08-03T10:00:00+00:00",
        "elapsed_seconds": 100.0,
        "remaining_seconds": 200.0,
        "source_returns_pct": {"bybit": 0.35, "okx": 0.34, "pyth": 0.33, "chainlink_rtds": 0.34},
        "source_prices": {"bybit": 63010, "okx": 63009, "pyth": 63005, "chainlink_rtds": 63006},
        "source_disagreement_pct": 0.02,
        "sharp_move_pct": 0.05,
        "up_bid": 0.39,
        "up_ask": 0.40,
        "down_bid": 0.59,
        "down_ask": 0.60,
        "book_json": {},
        "target_price": 63000.0,
        "reference_price": 63010.0,
        "target_source": "polymarket_crypto_price",
        "reference_observed_at": "2026-08-03T10:00:00+00:00",
        "realized_volatility_60s_pct": 0.02,
    }
    values.update(overrides)
    return MarketState(**values)


def test_conservative_policy_enters_only_high_confidence_direction() -> None:
    decision = decide(state())
    assert decision.action == "BUY_UP"
    assert decision.confidence >= 0.80
    assert "consensus_misalignment" in decision.tags


def test_sharp_move_blocks_entry() -> None:
    decision = decide(state(sharp_move_pct=0.30))
    assert decision.action == "WAIT"
    assert "sharp_move_block" in decision.tags


def test_source_conflict_blocks_entry() -> None:
    decision = decide(state(source_disagreement_pct=0.50))
    assert decision.action == "WAIT"
    assert "oracle_conflict" in decision.tags


def test_take_profit_closes_existing_position() -> None:
    position = PositionState(1, "btc-updown-5m-1785756900", "Up", "token", 10, 4, 0.4, 0.46)
    decision = decide(state(up_bid=0.46), position)
    assert decision.action == "CLOSE"
    assert "take_profit" in decision.tags


def test_wilson_gate_is_conservative() -> None:
    assert wilson_lower_bound(60, 100) < 0.60
    assert wilson_lower_bound(0, 0) == 0.0
