"""Диагностика различий между симулятором PAPER и фактическим исполнением LIVE."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import app_config as settings


def _one(connection: sqlite3.Connection, query: str) -> dict:
    row = connection.execute(query).fetchone()
    return dict(row) if row else {}


def build(path: Path = settings.DATABASE_PATH) -> dict:
    connection = sqlite3.connect(path); connection.row_factory = sqlite3.Row
    paper = _one(connection, """SELECT COUNT(*) orders,
        SUM(status IN ('filled','partially_filled')) filled_orders,
        SUM(COALESCE(shares,0))/NULLIF(SUM(COALESCE(requested_shares,shares,0)),0) fill_ratio,
        AVG(latency_ms) avg_latency_ms,SUM(fee_usdc) recorded_fees_usdc
        FROM paper_orders""")
    live = _one(connection, """SELECT COUNT(*) orders,SUM(matched_size>0) filled_orders,
        SUM(matched_size)/NULLIF(SUM(requested_size),0) fill_ratio,
        SUM(side='BUY') buy_orders,SUM(side='SELL') sell_orders FROM live_orders""")
    live_positions = _one(connection, """SELECT COUNT(*) positions,
        SUM(status IN ('closed','resolved')) completed,SUM(realized_pnl_usdc) local_realized_pnl_usdc,
        SUM(had_early_exit) early_exits FROM live_positions""")
    errors = [dict(row) for row in connection.execute(
        """SELECT error_type,COUNT(*) count FROM live_execution_errors GROUP BY error_type ORDER BY count DESC"""
    )]
    report = {
        "created_at": datetime.now(UTC).isoformat(), "paper_execution": paper,
        "live_execution": live, "live_positions": live_positions, "live_errors": errors,
        "comparison": {
            "same_event_ab_test_available": False,
            "reason": "PAPER и LIVE исполнялись в разные интервалы; сравнение PnL один-к-одному некорректно.",
            "known_gaps": [
                "PAPER использует очередь, задержку и модель fill; LIVE получает реальный partial/cancel.",
                "LIVE ledger пока считает matched_size по состоянию заявки; UI Polymarket может включать claimable и уже погашенные позиции.",
                "Комиссия PAPER моделируется явно; фактическая комиссия/ребейт LIVE должна сверяться по trade fills.",
            ],
            "fixed_now": [
                "Ответы get_order/cancel_order нормализованы без падения на bool/list/model objects.",
                "Технические partial fills отделены от решения модели о частичном выходе.",
            ],
        },
    }
    connection.close()
    output = settings.MODEL_DIR / "paper_live_execution_gap_v14.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output"] = str(output)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
