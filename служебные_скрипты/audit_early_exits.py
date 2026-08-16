"""Read-only аудит генерации, исполнения и разметки ранних выходов."""

from __future__ import annotations

import json
import sqlite3

import app_config as settings


def main() -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    result = {
        "positions_by_exit_timing": [dict(row) for row in connection.execute(
            "SELECT exit_timing,COUNT(*) positions,ROUND(SUM(COALESCE(realized_pnl_usdc,0)),4) pnl FROM paper_positions GROUP BY exit_timing"
        )],
        "positions_by_stage": [dict(row) for row in connection.execute(
            "SELECT exit_stage,COUNT(*) positions FROM paper_positions GROUP BY exit_stage ORDER BY exit_stage"
        )],
        "exit_decisions": [dict(row) for row in connection.execute(
            "SELECT action,executed,COUNT(*) decisions FROM model_decisions WHERE action IN ('CLOSE','PARTIAL_CLOSE') GROUP BY action,executed"
        )],
        "exit_orders": [dict(row) for row in connection.execute(
            "SELECT action,status,COUNT(*) orders FROM paper_orders WHERE action IN ('CLOSE','PARTIAL_CLOSE') GROUP BY action,status"
        )],
        "positions_since_last_early_exit": int(connection.execute(
            """SELECT COUNT(*) FROM paper_positions
               WHERE opened_at > COALESCE((SELECT MAX(closed_at) FROM paper_positions WHERE had_early_exit=1),'')"""
        ).fetchone()[0]),
        "latest_positions": [dict(row) for row in connection.execute(
            """SELECT id,event_slug,outcome,exit_timing,exit_stage,realized_pnl_usdc,opened_at,closed_at
               FROM paper_positions ORDER BY id DESC LIMIT 15"""
        )],
        "recent_position_decision_actions": [dict(row) for row in connection.execute(
            """SELECT action,executed,COUNT(*) decisions FROM model_decisions
               WHERE observed_at >= COALESCE((SELECT MAX(closed_at) FROM paper_positions WHERE had_early_exit=1),'')
               GROUP BY action,executed ORDER BY decisions DESC"""
        )],
        "recent_exit_candidate_tags": [dict(row) for row in connection.execute(
            """SELECT
                 SUM(CASE WHEN tags_json LIKE '%staged_risk_exit%' THEN 1 ELSE 0 END) risk_candidates,
                 SUM(CASE WHEN tags_json LIKE '%signal_confirmation_pending%' AND tags_json LIKE '%staged_risk_exit%' THEN 1 ELSE 0 END) pending_confirmations,
                 SUM(CASE WHEN tags_json LIKE '%five_stage_hold%' THEN 1 ELSE 0 END) holds
               FROM model_decisions
               WHERE observed_at >= COALESCE((SELECT MAX(closed_at) FROM paper_positions WHERE had_early_exit=1),'')"""
        )],
        "recent_early_exits": [dict(row) for row in connection.execute(
            """SELECT id,event_slug,outcome,exit_stage,exit_timing,early_exit_pnl_usdc,
                      realized_pnl_usdc,opened_at,closed_at FROM paper_positions
               WHERE had_early_exit=1 ORDER BY id DESC LIMIT 20"""
        )],
        # These counts intentionally avoid a multi-million-row self JOIN while
        # the live collector is writing. Exact pair validation is done by the
        # candidate trainer on a consistent database snapshot.
        "resolution_hold_labels": int(connection.execute(
            """SELECT COUNT(*) FROM action_counterfactuals
               WHERE action='HOLD' AND horizon_seconds>=300 AND status='resolved'
                 AND net_pnl_usdc IS NOT NULL"""
        ).fetchone()[0]),
        "legacy_all_horizon_hold_labels": int(connection.execute(
            """SELECT COUNT(*) FROM action_counterfactuals
               WHERE action='HOLD' AND status IN ('evaluated','resolved')
                 AND net_pnl_usdc IS NOT NULL"""
        ).fetchone()[0]),
        "resolution_feature_sample": dict(connection.execute(
            """SELECT outcome,current_bid,shares,cost_usdc,features_json
               FROM action_counterfactuals
               WHERE action='HOLD' AND horizon_seconds>=300 AND status='resolved'
               ORDER BY id DESC LIMIT 1"""
        ).fetchone() or {}),
    }
    connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
