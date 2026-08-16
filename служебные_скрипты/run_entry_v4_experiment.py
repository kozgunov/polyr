"""Безопасно обучает и проверяет entry_value_v4 на отдельном снимке рабочей базы."""

from __future__ import annotations

import json
import sqlite3
import sys
import argparse
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as settings
from polybot.analytics.walk_forward_v9 import run as walk_forward
from polybot.models.counterfactual_actions import build as build_actions
from polybot.models.train_pnl_model import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-only", action="store_true")
    args = parser.parse_args()
    candidate = settings.MODEL_CANDIDATE_DIR / "entry_value_v4_tail_risk"
    candidate.mkdir(parents=True, exist_ok=True)
    snapshot = candidate / "training_snapshot.sqlite3"
    status_path = candidate / "experiment_status.json"
    source = sqlite3.connect(settings.DATABASE_PATH, timeout=30)
    live_events = int(source.execute("SELECT COUNT(DISTINCT event_slug) FROM training_examples").fetchone()[0])
    reusable = False
    if snapshot.exists():
        try:
            cached = sqlite3.connect(snapshot)
            cached_events = int(cached.execute("SELECT COUNT(DISTINCT event_slug) FROM training_examples").fetchone()[0])
            action_rows = int(cached.execute("SELECT COUNT(1) FROM counterfactual_action_examples").fetchone()[0])
            cached.close()
            reusable = cached_events >= live_events - 10 and action_rows > 0
        except sqlite3.Error:
            reusable = False
    if not reusable:
        status_path.write_text(json.dumps({"status":"snapshot","live_events":live_events}), encoding="utf-8")
        snapshot.unlink(missing_ok=True)
        target = sqlite3.connect(snapshot); source.backup(target); target.close()
    source.close()
    try:
        if reusable:
            db = sqlite3.connect(snapshot)
            dataset = {"rows": int(db.execute("SELECT COUNT(1) FROM counterfactual_action_examples").fetchone()[0]),
                       "events": int(db.execute("SELECT COUNT(DISTINCT event_slug) FROM counterfactual_action_examples").fetchone()[0]),
                       "cached": True}
            db.close()
        else:
            status_path.write_text(json.dumps({"status":"building_dataset","live_events":live_events}), encoding="utf-8")
            dataset = build_actions(snapshot)
        status_path.write_text(json.dumps({"status":"walk_forward","dataset":dataset}), encoding="utf-8")
        entry = ({"status": "reused", "report": str(settings.ACTION_VALUE_V4_REPORT_PATH)}
                 if args.walk_only else train(
                     snapshot, candidate / "btc_5m_net_pnl.joblib",
                     settings.ACTION_VALUE_V4_REPORT_PATH,
                 ))
        walk = walk_forward(snapshot)
        result = {
            "created_at": datetime.now(UTC).isoformat(), "dataset": dataset,
            "entry": entry, "walk_forward": walk,
            "activated": False,
            "activation_rule": "candidate activates only after entry and full-chain promotion gates pass",
        }
        (candidate / "experiment_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        status_path.write_text(json.dumps({"status":"completed","walk_forward":walk}, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("all_donerun_entry_v4_experiment.py")
    finally:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_run_entry_v4_experiment.py")
        raise
