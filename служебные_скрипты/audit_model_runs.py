"""Read-only audit of paper sessions grouped by the model that opened each position."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings


def main() -> None:
    db = sqlite3.connect(settings.DATABASE_PATH)
    db.row_factory = sqlite3.Row
    sessions = [dict(row) for row in db.execute(
        """SELECT session_id,started_at,ended_at,status,strategy_version,model_name,run_label,
                  realized_pnl_usdc,total_wagered_usdc,total_fees_usdc
           FROM paper_sessions ORDER BY started_at"""
    )]
    by_model = [dict(row) for row in db.execute(
        """SELECT COALESCE(d.model_name,s.model_name,'unknown') AS model,
                  COALESCE(d.provider,'unknown') AS provider,COUNT(*) AS positions,
                  COUNT(DISTINCT p.event_slug) AS events,
                  SUM(CASE WHEN p.status IN ('closed','resolved') THEN 1 ELSE 0 END) AS completed,
                  ROUND(SUM(COALESCE(p.realized_pnl_usdc,0)),6) AS pnl,
                  ROUND(SUM(COALESCE(p.fees_usdc,0)),6) AS fees,
                  MIN(p.opened_at) AS first_open,MAX(p.closed_at) AS last_close
           FROM paper_positions p
           JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           GROUP BY model,provider ORDER BY positions DESC"""
    )]
    by_session_model = [dict(row) for row in db.execute(
        """SELECT p.session_id,COALESCE(d.model_name,s.model_name,'unknown') AS model,
                  COUNT(*) AS positions,COUNT(DISTINCT p.event_slug) AS events,
                  SUM(CASE WHEN COALESCE(p.realized_pnl_usdc,0)>0 THEN 1 ELSE 0 END) AS wins,
                  ROUND(SUM(COALESCE(p.realized_pnl_usdc,0)),6) AS pnl
           FROM paper_positions p JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           GROUP BY p.session_id,model ORDER BY MIN(p.opened_at)"""
    )]
    db.close()
    print(json.dumps({"sessions": sessions, "by_model": by_model, "by_session_model": by_session_model}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
