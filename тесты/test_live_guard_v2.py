from polybot.trading.live_guard import readiness_failures


def good_statistics():
    return {
        "resolved_events": 320,
        "wilson_win_rate": 0.55,
        "profit_factor": 1.35,
        "max_drawdown_pct": 0.06,
        "net_pnl_usdc": 45.0,
        "pnl_mean_ci95_lower": 0.04,
        "up_entries": 145,
        "down_entries": 175,
        "max_direction_share": 175 / 320,
    }


def test_balanced_positive_run_passes_statistical_gates():
    assert readiness_failures(good_statistics()) == []


def test_direction_collapse_and_negative_ci_block_live():
    statistics = good_statistics()
    statistics.update({
        "pnl_mean_ci95_lower": -0.01,
        "up_entries": 1,
        "down_entries": 319,
        "max_direction_share": 319 / 320,
    })
    failures = readiness_failures(statistics)
    assert any("CI lower" in failure for failure in failures)
    assert any("both Up and Down" in failure for failure in failures)
    assert any("single-direction" in failure for failure in failures)
