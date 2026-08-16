"""Честное walk-forward сравнение направления без истории и с 3/12 предыдущими событиями."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.event_history import build_summaries, context as history_context
from polybot.models.temporal_validation import purged_expanding_folds
from polybot.models.train_direction_model import vector


def _load_raw(database: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    raw = connection.execute(
        """SELECT event_slug,outcome,observed_at,label,features_json
           FROM training_examples ORDER BY observed_at"""
    ).fetchall()
    connection.close()
    return raw


def _rows(raw: list[sqlite3.Row], windows: tuple[int, ...]) -> list[dict[str, Any]]:
    summaries = build_summaries(raw)
    result = []
    context_cache: dict[str, dict[str, float]] = {}
    for row in raw:
        features = json.loads(str(row["features_json"]))
        if features.get("target_price") is None or features.get("reference_price") is None:
            continue
        slug = str(row["event_slug"])
        if slug not in context_cache:
            context_cache[slug] = history_context(summaries, slug, str(row["observed_at"]), windows)
        features.update(context_cache[slug])
        result.append({
            "event": slug, "observed_at": str(row["observed_at"]),
            "x": vector(slug, str(row["outcome"]), str(row["observed_at"]), features, windows),
            "y": int(row["label"]),
        })
    return result


def _weights(groups: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(groups, return_counts=True)
    mapping = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([1.0 / mapping[group] for group in groups], dtype=np.float64)


def evaluate(database: Path = settings.DATABASE_PATH) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    # Один SQLite snapshot для всех вариантов: работающий collector не может изменить
    # состав событий между baseline и history-моделями.
    raw = _load_raw(database)
    for raw_window in settings.EVENT_HISTORY_WINDOWS:
        windows = () if int(raw_window) == 0 else (int(raw_window),)
        rows = _rows(raw, windows)
        events = list(dict.fromkeys(row["event"] for row in rows))
        folds = purged_expanding_folds(events, folds=5, purge=2, embargo=2)
        fold_reports = []
        for fold_index, fold in enumerate(folds, 1):
            train = [row for row in rows if row["event"] in set(fold.train_events)]
            validation = [row for row in rows if row["event"] in set(fold.validation_events)]
            test = [row for row in rows if row["event"] in set(fold.test_events)]
            if not train or not validation or not test or len({row["y"] for row in train}) < 2:
                continue
            x_train = np.asarray([row["x"] for row in train], dtype=np.float64)
            y_train = np.asarray([row["y"] for row in train], dtype=np.int8)
            g_train = np.asarray([row["event"] for row in train])
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=160, max_leaf_nodes=15, l2_regularization=1.0,
                random_state=settings.TRAINING_RANDOM_STATE,
            )
            model.fit(x_train, y_train, sample_weight=_weights(g_train))
            x_validation = np.asarray([row["x"] for row in validation], dtype=np.float64)
            y_validation = np.asarray([row["y"] for row in validation], dtype=np.int8)
            g_validation = np.asarray([row["event"] for row in validation])
            raw_validation = np.clip(model.predict_proba(x_validation)[:, 1], 1e-6, 1 - 1e-6)
            calibrator = LogisticRegression(C=0.25, random_state=settings.TRAINING_RANDOM_STATE)
            calibrator.fit(
                np.log(raw_validation / (1 - raw_validation)).reshape(-1, 1), y_validation,
                sample_weight=_weights(g_validation),
            )
            x_test = np.asarray([row["x"] for row in test], dtype=np.float64)
            y_test = np.asarray([row["y"] for row in test], dtype=np.int8)
            g_test = np.asarray([row["event"] for row in test])
            weights = _weights(g_test)
            raw_probability = np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)
            probability = calibrator.predict_proba(
                np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
            )[:, 1]
            prediction = (probability >= 0.5).astype(np.int8)
            fold_reports.append({
                "fold": fold_index, "train_events": len(fold.train_events), "test_events": len(fold.test_events),
                "roc_auc": float(roc_auc_score(y_test, probability, sample_weight=weights)),
                "brier": float(brier_score_loss(y_test, probability, sample_weight=weights)),
                "log_loss": float(log_loss(y_test, probability, sample_weight=weights, labels=[0, 1])),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction, sample_weight=weights)),
            })
        variants[f"history_{raw_window}"] = {
            "history_events": int(raw_window), "rows": len(rows), "events": len(events), "folds": fold_reports,
            "mean": {
                metric: float(np.mean([fold[metric] for fold in fold_reports])) if fold_reports else None
                for metric in ("roc_auc", "brier", "log_loss", "balanced_accuracy")
            },
        }
    report = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "protocol": "purged expanding walk-forward; identical folds; event-balanced weights; completed-neighbor context only",
        "variants": variants,
    }
    settings.HISTORY_CONTEXT_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
