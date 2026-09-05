"""Признаки завершённых соседних 5m-событий без утечки текущего исхода."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True)
class EventSummary:
    event_slug: str
    start_ts: int
    end_ts: int
    resolution_up: int
    final_distance_pct: float
    distance_range_pct: float
    mean_volatility_pct: float
    reference_return_pct: float
    distance_at_30s_pct: float
    distance_at_60s_pct: float
    distance_at_120s_pct: float
    distance_at_180s_pct: float
    distance_at_240s_pct: float
    distance_at_285s_pct: float
    target_crossings: int
    max_positive_distance_pct: float
    max_negative_distance_pct: float


BASE_HISTORY_METRICS = (
    "available_fraction", "up_rate", "flip_rate", "last_up", "signed_streak",
    "mean_final_distance_pct", "mean_abs_final_distance_pct", "mean_distance_range_pct",
    "mean_volatility_pct", "mean_reference_return_pct", "mean_target_crossings",
    "mean_max_positive_distance_pct", "mean_max_negative_distance_pct",
    "mean_distance_at_30s_pct", "mean_distance_at_60s_pct", "mean_distance_at_120s_pct",
    "mean_distance_at_180s_pct", "mean_distance_at_240s_pct", "mean_distance_at_285s_pct",
    "gap_seconds",
)

PATH_CHECKPOINTS = (30, 60, 120, 180, 240, 285)


def feature_names(windows: Iterable[int]) -> list[str]:
    return [f"history_{int(window)}_{metric}" for window in windows if int(window) > 0 for metric in BASE_HISTORY_METRICS]


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if result == result else 0.0
    except (TypeError, ValueError):
        return 0.0


def event_start(event_slug: str) -> int:
    return int(str(event_slug).rsplit("-", 1)[-1])


def build_summaries(rows: Iterable[Any]) -> list[EventSummary]:
    """Строит одну сводку на событие; используются только строки контракта Up."""
    grouped: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        outcome = str(row[1] if not isinstance(row, sqlite3.Row) else row["outcome"])
        if outcome != "Up":
            continue
        slug = str(row[0] if not isinstance(row, sqlite3.Row) else row["event_slug"])
        observed = str(row[2] if not isinstance(row, sqlite3.Row) else row["observed_at"])
        label = int(row[3] if not isinstance(row, sqlite3.Row) else row["label"])
        raw = row[4] if not isinstance(row, sqlite3.Row) else row["features_json"]
        grouped[slug].append((observed, label, json.loads(str(raw))))
    summaries: list[EventSummary] = []
    for slug, items in grouped.items():
        items.sort(key=lambda item: item[0])
        distances = [_number(item[2].get("distance_to_target_pct")) for item in items]
        start = event_start(slug)
        elapsed = []
        for observed, _, features in items:
            value = features.get("elapsed_seconds")
            if value is None:
                value = datetime.fromisoformat(observed).timestamp() - start
            elapsed.append(max(0.0, min(300.0, _number(value))))
        checkpoints = {
            second: distances[min(range(len(items)), key=lambda index: abs(elapsed[index] - second))]
            if items else 0.0
            for second in PATH_CHECKPOINTS
        }
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in distances]
        target_crossings = sum(
            signs[index] and signs[index - 1] and signs[index] != signs[index - 1]
            for index in range(1, len(signs))
        )
        references = [_number(item[2].get("reference_price")) for item in items]
        references = [value for value in references if value > 0]
        reference_return = (references[-1] / references[0] - 1.0) * 100.0 if len(references) >= 2 else 0.0
        summaries.append(EventSummary(
            slug, start, start + 300, int(items[-1][1]), distances[-1] if distances else 0.0,
            (max(distances) - min(distances)) if distances else 0.0,
            sum(_number(item[2].get("realized_volatility_60s_pct")) for item in items) / max(1, len(items)),
            reference_return,
            *(checkpoints[second] for second in PATH_CHECKPOINTS),
            target_crossings,
            max(distances) if distances else 0.0,
            min(distances) if distances else 0.0,
        ))
    return sorted(summaries, key=lambda item: item.start_ts)


def context(summaries: list[EventSummary], event_slug: str, observed_at: str, windows: Iterable[int]) -> dict[str, float]:
    """Возвращает только события, завершившиеся до времени принимаемого решения."""
    observed_ts = datetime.fromisoformat(observed_at).timestamp()
    current_start = event_start(event_slug)
    eligible = [item for item in summaries if item.start_ts < current_start and item.end_ts <= observed_ts]
    result: dict[str, float] = {}
    for raw_window in windows:
        window = int(raw_window)
        if window <= 0:
            continue
        selected = eligible[-window:]
        prefix = f"history_{window}_"
        labels = [item.resolution_up for item in selected]
        flips = sum(labels[index] != labels[index - 1] for index in range(1, len(labels)))
        streak = 0
        if labels:
            sign = 1 if labels[-1] else -1
            for label in reversed(labels):
                if (1 if label else -1) != sign:
                    break
                streak += sign
        result.update({
            prefix + "available_fraction": len(selected) / window,
            prefix + "up_rate": sum(labels) / max(1, len(labels)),
            prefix + "flip_rate": flips / max(1, len(labels) - 1),
            prefix + "last_up": float(labels[-1]) if labels else 0.5,
            prefix + "signed_streak": float(streak),
            prefix + "mean_final_distance_pct": sum(x.final_distance_pct for x in selected) / max(1, len(selected)),
            prefix + "mean_abs_final_distance_pct": sum(abs(x.final_distance_pct) for x in selected) / max(1, len(selected)),
            prefix + "mean_distance_range_pct": sum(x.distance_range_pct for x in selected) / max(1, len(selected)),
            prefix + "mean_volatility_pct": sum(x.mean_volatility_pct for x in selected) / max(1, len(selected)),
            prefix + "mean_reference_return_pct": sum(x.reference_return_pct for x in selected) / max(1, len(selected)),
            prefix + "mean_target_crossings": sum(x.target_crossings for x in selected) / max(1, len(selected)),
            prefix + "mean_max_positive_distance_pct": sum(x.max_positive_distance_pct for x in selected) / max(1, len(selected)),
            prefix + "mean_max_negative_distance_pct": sum(x.max_negative_distance_pct for x in selected) / max(1, len(selected)),
            **{
                prefix + f"mean_distance_at_{second}s_pct": sum(
                    getattr(x, f"distance_at_{second}s_pct") for x in selected
                ) / max(1, len(selected))
                for second in PATH_CHECKPOINTS
            },
            prefix + "gap_seconds": max(0.0, observed_ts - selected[-1].end_ts) if selected else 86_400.0,
        })
    return result


def summaries_from_connection(connection: sqlite3.Connection) -> list[EventSummary]:
    rows = connection.execute(
        """SELECT event_slug,outcome,observed_at,label,features_json
           FROM training_examples ORDER BY observed_at"""
    ).fetchall()
    return build_summaries(rows)


def context_from_connection(
    connection: sqlite3.Connection, event_slug: str, observed_at: str, windows: Iterable[int]
) -> dict[str, float]:
    return context(summaries_from_connection(connection), event_slug, observed_at, windows)
