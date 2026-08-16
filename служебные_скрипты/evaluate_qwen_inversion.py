"""Контрфактическая проверка Qwen Up/Down-инверсии на завершённых shadow-событиях."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config as settings
from polybot.trading.fees import total_fee_usdc


def _max_drawdown(values: list[float]) -> float:
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _ci95_lower(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return average - 1.96 * math.sqrt(variance / len(values))


def main() -> None:
    connection = sqlite3.connect(f"file:{settings.DATABASE_PATH.as_posix()}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """WITH ranked AS (
             SELECT *,ROW_NUMBER() OVER(
               PARTITION BY event_slug
               ORDER BY CASE WHEN action LIKE 'BUY_%' THEN 0 ELSE 1 END,observed_at,id
             ) event_rank
             FROM shadow_predictions
             WHERE model_name='qwen' AND status='resolved'
           )
           SELECT * FROM ranked
           WHERE event_rank=1 AND action LIKE 'BUY_%'
           ORDER BY evaluated_at,event_slug"""
    ).fetchall()
    original = [float(row["net_pnl_usdc"] or 0.0) for row in rows]
    inverted: list[float] = []
    missing_prices = 0
    up = down = 0
    for row in rows:
        inverse_direction = "Down" if row["direction"] == "Up" else "Up"
        snapshot = connection.execute(
            """SELECT best_ask,collected_at FROM market_snapshots
               WHERE event_slug=? AND outcome=? AND best_ask IS NOT NULL AND best_ask>0 AND best_ask<1
               ORDER BY ABS(julianday(collected_at)-julianday(?)),id LIMIT 1""",
            (row["event_slug"], inverse_direction, row["observed_at"]),
        ).fetchone()
        if snapshot is None:
            missing_prices += 1
            continue
        price = float(snapshot["best_ask"])
        notional = float(row["notional_usdc"] or settings.PAPER_ENTRY_NOTIONAL_USDC)
        shares = notional / price
        label_up = int(row["resolved_label"])
        won = label_up == int(inverse_direction == "Up")
        inverted.append(shares * int(won) - notional - total_fee_usdc(shares, price))
        up += inverse_direction == "Up"
        down += inverse_direction == "Down"
    connection.close()

    result = {
        "experiment": "qwen_inverted_v1_historical_counterfactual",
        "method": "one first executable BUY per event; opposite contract ask nearest decision time; Polymarket fee included",
        "events": len(rows),
        "evaluated": len(inverted),
        "missing_opposite_prices": missing_prices,
        "original": {
            "net_pnl_usdc": sum(original),
            "expectancy_usdc": mean(original) if original else 0.0,
            "max_drawdown_usdc": _max_drawdown(original),
        },
        "inverted": {
            "net_pnl_usdc": sum(inverted),
            "expectancy_usdc": mean(inverted) if inverted else 0.0,
            "expectancy_ci95_lower_usdc": _ci95_lower(inverted),
            "max_drawdown_usdc": _max_drawdown(inverted),
            "up": up,
            "down": down,
        },
        "warning": "Historical challenger only; not promoted to PAPER execution or LIVE.",
    }
    report = settings.MODEL_DIR / "qwen_inverted_v1_report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("all_doneevaluate_qwen_inversion.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_evaluate_qwen_inversion.py")
        raise
