from __future__ import annotations

import json

from polybot.models.event_history import build_summaries, context, feature_names
from polybot.models.train_direction_model import FEATURE_NAMES, vector


def _row(slug: str, observed: str, label: int, distance: float):
    return (slug, "Up", observed, label, json.dumps({
        "distance_to_target_pct": distance, "reference_price": 100 + distance,
        "realized_volatility_60s_pct": abs(distance),
    }))


def test_neighbor_context_uses_only_completed_past_events():
    rows = [
        _row("btc-updown-5m-1000", "1970-01-01T00:16:50+00:00", 1, 0.1),
        _row("btc-updown-5m-1000", "1970-01-01T00:21:39+00:00", 1, 0.2),
        _row("btc-updown-5m-1300", "1970-01-01T00:21:50+00:00", 0, -0.1),
    ]
    summaries = build_summaries(rows)
    before_previous_end = context(summaries, "btc-updown-5m-1300", "1970-01-01T00:21:39+00:00", (3,))
    after_previous_end = context(summaries, "btc-updown-5m-1300", "1970-01-01T00:21:41+00:00", (3,))
    assert before_previous_end["history_3_available_fraction"] == 0
    assert after_previous_end["history_3_available_fraction"] == 1 / 3
    assert after_previous_end["history_3_up_rate"] == 1.0
    assert "history_3_mean_distance_at_240s_pct" in after_previous_end
    assert "history_3_mean_target_crossings" in after_previous_end


def test_legacy_vector_schema_stays_compatible():
    features = {"target_price": 100, "reference_price": 101, "history_3_up_rate": 1.0}
    legacy = vector("btc-updown-5m-1000", "Up", "1970-01-01T00:17:00+00:00", features)
    enriched = vector("btc-updown-5m-1000", "Up", "1970-01-01T00:17:00+00:00", features, (3,))
    assert len(legacy) == len(FEATURE_NAMES)
    assert len(enriched) == len(FEATURE_NAMES) + len(feature_names((3,)))
