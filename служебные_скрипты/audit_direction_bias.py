"""Диагностика перекоса Up/Down у конкретной торговой модели."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings


def main(model: str = "qwen") -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT event_slug,action,executed,predicted_up_probability,
                  predicted_down_probability,confidence,tags_json
           FROM model_decisions WHERE model_name=? ORDER BY id""", (model,),
    ).fetchall()
    direction_votes = Counter()
    confidence_bins = Counter()
    for row in rows:
        p_up = row["predicted_up_probability"]
        if p_up is None:
            continue
        p_up = float(p_up)
        direction_votes["Up" if p_up > 0.5 else "Down" if p_up < 0.5 else "Hold"] += 1
        confidence = max(p_up, 1.0 - p_up)
        confidence_bins[f"{int(confidence * 10) / 10:.1f}"] += 1
    entries = connection.execute(
        """SELECT p.event_slug,p.outcome,p.status,p.realized_pnl_usdc,
                  d.predicted_up_probability,d.predicted_down_probability,d.confidence,d.reason
           FROM paper_positions p JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE d.model_name=? ORDER BY p.opened_at""", (model,),
    ).fetchall()
    event_labels = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            """SELECT event_slug,MAX(CASE WHEN outcome='Up' THEN label END)
               FROM training_examples GROUP BY event_slug"""
        ).fetchall() if row[1] is not None
    }
    entry_events = {str(row["event_slug"]) for row in entries}
    actual = Counter("Up" if event_labels[slug] else "Down" for slug in entry_events if slug in event_labels)
    result = {
        "model": model,
        "all_decisions": len(rows),
        "raw_direction_votes": dict(direction_votes),
        "confidence_bins": dict(sorted(confidence_bins.items())),
        "entries": len(entries),
        "entry_directions": dict(Counter(str(row["outcome"]) for row in entries)),
        "actual_resolution_on_entry_events": dict(actual),
        "mean_p_up": sum(float(row["predicted_up_probability"] or 0.5) for row in rows) / max(1, len(rows)),
        "up_entry_examples": [dict(row) for row in entries if row["outcome"] == "Up"][:5],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    print("all_done_audit_direction_bias.py")


if __name__ == "__main__":
    try:
        # Можно передать имя модели: ``python audit_direction_bias.py catboost``.
        main(sys.argv[1] if len(sys.argv) > 1 else "qwen")
    except Exception:
        print("error_in_audit_direction_bias.py")
        raise
