from __future__ import annotations

import json
import sqlite3

import pyarrow.parquet as pq

import app_config as settings
from polybot.models import sequence_dataset
from polybot.models.temporal_validation import purged_expanding_folds
from polybot.tools.gpu_bootstrap import command_plan


def test_purged_folds_have_no_overlap_or_future_leakage():
    events = [f"event-{index:03d}" for index in range(100)]
    folds = purged_expanding_folds(events, folds=5, purge=2, embargo=2)
    assert len(folds) == 5
    for fold in folds:
        assert set(fold.train_events).isdisjoint(fold.validation_events)
        assert set(fold.validation_events).isdisjoint(fold.test_events)
        assert max(events.index(x) for x in fold.train_events) < min(events.index(x) for x in fold.validation_events)
        assert max(events.index(x) for x in fold.validation_events) < min(events.index(x) for x in fold.test_events)


def test_bootstrap_uses_isolated_python_and_explicit_cuda_wheel():
    commands = command_plan("cu128")
    flattened = [item for command in commands for item in command]
    assert ".venv-gpu" in " ".join(flattened)
    assert "https://download.pytorch.org/whl/cu128" in flattened
    assert any("requirements-gpu.txt" in item for item in flattened)


def test_sequence_dataset_is_fixed_length_parquet(tmp_path, monkeypatch):
    database = tmp_path / "market.sqlite3"
    output = tmp_path / "sequences.parquet"
    manifest_path = tmp_path / "manifest.json"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE training_examples(snapshot_id INTEGER,event_slug TEXT,outcome TEXT,observed_at TEXT,label INTEGER,features_json TEXT)"
    )
    for snapshot_id, second in enumerate((5, 10, 15), 1):
        connection.execute(
            "INSERT INTO training_examples VALUES(?,?,?,?,?,?)",
            (snapshot_id, "btc-updown-5m-1000", "Up", f"1970-01-01T00:16:{40 + second:02d}+00:00", 1,
             json.dumps({"best_bid": snapshot_id / 10, "remaining_seconds": 300 - second})),
        )
    connection.commit(); connection.close()
    monkeypatch.setattr(settings, "SEQUENCE_DATASET_MANIFEST_PATH", manifest_path)
    manifest = sequence_dataset.build(database, output)
    table = pq.read_table(output)
    assert manifest["rows"] == 3
    assert table.num_rows == 3
    assert len(table.column("best_bid")[0].as_py()) == 24
    assert manifest_path.exists()

