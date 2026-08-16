"""Сверяет основные числа дашборда с первичными строками SQLite."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings
from polybot.dashboard.app import model_comparison, model_equity_curves, paper_overview


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    # Все проверки должны видеть один SQLite snapshot, даже когда engine пишет параллельно.
    connection.execute("BEGIN")
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, actual: object, expected: object) -> None:
        checks.append({"name": name, "ok": bool(condition), "actual": actual, "expected": expected})

    paper = paper_overview(connection)
    session_id = str(paper["session_id"])
    raw = connection.execute(
        """SELECT initial_balance_usdc,cash_balance_usdc,realized_pnl_usdc,total_fees_usdc
           FROM paper_sessions WHERE session_id=?""", (session_id,),
    ).fetchone()
    position_totals = connection.execute(
        """SELECT COALESCE(SUM(realized_pnl_usdc),0),COALESCE(SUM(fees_usdc),0),
                  COUNT(DISTINCT CASE WHEN status IN ('closed','resolved') THEN event_slug END),
                  SUM(CASE WHEN status='open' THEN 1 ELSE 0 END)
           FROM paper_positions WHERE session_id=?""", (session_id,),
    ).fetchone()
    check("session realized PnL", close(paper["realized_pnl"], raw["realized_pnl_usdc"]), paper["realized_pnl"], raw["realized_pnl_usdc"])
    check("positions sum equals session PnL", close(position_totals[0], raw["realized_pnl_usdc"]), position_totals[0], raw["realized_pnl_usdc"])
    check("dashboard fees", close(paper["fees"], raw["total_fees_usdc"]), paper["fees"], raw["total_fees_usdc"])
    check("dashboard resolved events", int(paper["resolved_events"]) == int(position_totals[2]), paper["resolved_events"], position_totals[2])
    check("dashboard open positions", len(paper["open_positions"]) == int(position_totals[3]), len(paper["open_positions"]), position_totals[3])
    if int(position_totals[3]) == 0:
        check("cash identity without open positions", close(raw["cash_balance_usdc"], raw["initial_balance_usdc"] + raw["realized_pnl_usdc"]), raw["cash_balance_usdc"], raw["initial_balance_usdc"] + raw["realized_pnl_usdc"])

    comparisons = model_comparison(connection)
    curves = model_equity_curves(connection)
    for item in comparisons:
        model = str(item["model"])
        points = curves.get(model, [])
        terminal = float(points[-1]["equity"]) if points else float(settings.PAPER_INITIAL_BALANCE_USDC)
        expected = float(settings.PAPER_INITIAL_BALANCE_USDC) + float(item["net_pnl_usdc"])
        check(f"equity terminal: {model}", close(terminal, expected), terminal, expected)
        auc = item.get("live_roc_auc")
        check(f"ROC-AUC range: {model}", auc is None or 0 <= float(auc) <= 1, auc, "None or [0,1]")

    result = {"ok": all(bool(item["ok"]) for item in checks), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    if not result["ok"]:
        raise SystemExit(1)
    print("all_done_audit_dashboard_consistency.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_audit_dashboard_consistency.py")
        raise
