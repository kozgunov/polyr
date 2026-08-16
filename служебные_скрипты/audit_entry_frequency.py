"""Объясняет, почему demo-модель редко открывает позиции."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as settings


def main() -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    session = connection.execute(
        "SELECT session_id,started_at FROM paper_sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    rows = connection.execute(
        """SELECT event_slug,action,confidence,reason,tags_json,predicted_up_probability,
                  predicted_down_probability,expected_net_edge,executed
           FROM model_decisions WHERE session_id=? ORDER BY id""", (session["session_id"],),
    ).fetchall()
    reasons = Counter(str(row["reason"]) for row in rows if row["action"] == "WAIT")
    tags = Counter(
        tag.split("=", 1)[0]
        for row in rows for tag in json.loads(row["tags_json"] or "[]")
        if row["action"] == "WAIT"
    )
    candidates = [
        row for row in rows
        if row["action"] == "WAIT"
        and row["predicted_up_probability"] is not None
        and row["expected_net_edge"] is not None
    ]
    thresholds = {}
    for confidence in (0.60, 0.65, 0.68, 0.70, 0.72):
        for edge in (0.00, 0.01, 0.02, 0.03, 0.04):
            eligible_events = {
                row["event_slug"] for row in candidates
                if max(float(row["predicted_up_probability"]), float(row["predicted_down_probability"])) >= confidence
                and float(row["expected_net_edge"]) >= edge
            }
            thresholds[f"p>={confidence:.2f},edge>={edge:.2f}"] = len(eligible_events)
    report = {
        "session_id": session["session_id"],
        "started_at": session["started_at"],
        "independent_events_seen": len({row["event_slug"] for row in rows}),
        "decisions": len(rows),
        "executed_entries": sum(row["executed"] and str(row["action"]).startswith("BUY_") for row in rows),
        "actions": Counter(str(row["action"]) for row in rows),
        "top_wait_reasons": reasons.most_common(15),
        "wait_tags": tags.most_common(20),
        "eligible_events_by_threshold": thresholds,
    }
    connection.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
        print(f"all_done{Path(__file__).name}")
    except Exception:
        print(f"error_in_{Path(__file__).name}")
        raise
