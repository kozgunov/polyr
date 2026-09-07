from polybot.analytics.timeseries_walk_forward import terminal_up_probability


def test_terminal_probability_is_symmetric_around_target():
    middle = terminal_up_probability(100.0, 100.0, 0.001, 10)
    above = terminal_up_probability(101.0, 100.0, 0.001, 10)
    below = terminal_up_probability(99.0, 100.0, 0.001, 10)
    assert middle == 0.5
    assert above > middle > below
