"""Строит snapshot базы, контрфактические действия и offline-кандидат value-модели."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import app_config as settings

from polybot.models.counterfactual_actions import build
from polybot.models.train_pnl_model import train


def main() -> None:
    snapshot = settings.DATA_DIR / "offline_counterfactual_training.sqlite3"
    source = sqlite3.connect(settings.DATABASE_PATH, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    destination = sqlite3.connect(snapshot)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    dataset = build(snapshot)
    version = datetime.now(UTC).strftime("counterfactual_entry_value_v2_%Y%m%dT%H%M%SZ")
    output = settings.MODEL_CANDIDATE_DIR / version
    report = train(snapshot, artifact_path=output / "btc_5m_action_value.joblib",
                   report_path=output / "report.json")
    summary = {"version": version, "database_snapshot": str(snapshot), "dataset": dataset,
               "model": report, "activation": "offline_candidate_not_promoted"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
