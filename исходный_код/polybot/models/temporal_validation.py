"""Временные фолды без утечки соседних пятиминутных событий."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalFold:
    train_events: tuple[str, ...]
    validation_events: tuple[str, ...]
    test_events: tuple[str, ...]


def purged_expanding_folds(
    ordered_events: list[str], *, folds: int = 5, purge: int = 2, embargo: int = 2
) -> list[TemporalFold]:
    """Строит expanding-window фолды, сохраняя строгий порядок событий."""
    events = list(dict.fromkeys(ordered_events))
    if folds < 1 or len(events) < 15:
        return []
    test_size = max(1, len(events) // (folds + 2))
    first_test = len(events) - folds * test_size
    result: list[TemporalFold] = []
    for index in range(folds):
        test_start = first_test + index * test_size
        test_end = min(len(events), test_start + test_size)
        validation_end = max(0, test_start - purge)
        validation_start = max(0, validation_end - test_size)
        train_end = max(0, validation_start - embargo)
        if train_end and validation_end > validation_start and test_end > test_start:
            result.append(TemporalFold(
                tuple(events[:train_end]),
                tuple(events[validation_start:validation_end]),
                tuple(events[test_start:test_end]),
            ))
    return result

