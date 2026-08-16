"""Проверяет наивную гипотезу: после убыточной позиции следующая чаще прибыльная."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import numpy as np
from scipy.stats import fisher_exact


def _bootstrap_difference(after_loss: np.ndarray, unconditional: np.ndarray, samples: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(settings.TRAINING_RANDOM_STATE)
    differences = np.empty(samples)
    for index in range(samples):
        differences[index] = (
            rng.choice(after_loss, len(after_loss), replace=True).mean()
            - rng.choice(unconditional, len(unconditional), replace=True).mean()
        )
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def evaluate(database: Path = settings.DATABASE_PATH) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    positions = connection.execute(
        """SELECT event_slug,COALESCE(closed_at,opened_at),SUM(realized_pnl_usdc)
           FROM paper_positions
           WHERE status IN ('closed','resolved') AND COALESCE(execution_valid,1)=1
           GROUP BY session_id,event_slug ORDER BY COALESCE(closed_at,opened_at)"""
    ).fetchall()
    event_labels = connection.execute(
        """SELECT event_slug,MAX(CASE WHEN outcome='Up' THEN label END)
           FROM training_examples GROUP BY event_slug ORDER BY CAST(SUBSTR(event_slug,INSTR(event_slug,'5m-')+3) AS INTEGER)"""
    ).fetchall()
    connection.close()
    pnl = np.asarray([float(row[2] or 0.0) for row in positions], dtype=np.float64)
    wins = pnl > 0
    previous_loss = pnl[:-1] < 0
    next_win = wins[1:]
    after_loss = next_win[previous_loss].astype(np.int8)
    after_non_loss = next_win[~previous_loss].astype(np.int8)
    unconditional = wins[1:].astype(np.int8)
    table = [
        [int(after_loss.sum()), int(len(after_loss) - after_loss.sum())],
        [int(after_non_loss.sum()), int(len(after_non_loss) - after_non_loss.sum())],
    ]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    ci = _bootstrap_difference(after_loss, unconditional) if len(after_loss) and len(unconditional) else (None, None)
    labels = np.asarray([int(row[1]) for row in event_labels if row[1] is not None], dtype=np.int8)
    direction_flips = labels[1:] != labels[:-1] if len(labels) > 1 else np.asarray([], dtype=bool)
    report = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "trading_hypothesis": {
            "valid_event_positions": len(pnl), "previous_losses": len(after_loss),
            "p_next_win_after_loss": float(after_loss.mean()) if len(after_loss) else None,
            "baseline_next_win_probability": float(unconditional.mean()) if len(unconditional) else None,
            "difference": float(after_loss.mean() - unconditional.mean()) if len(after_loss) else None,
            "bootstrap_difference_ci95": list(ci), "fisher_exact_greater_p_value": float(p_value),
            "odds_ratio": float(odds_ratio),
            "supported_at_5pct": bool(p_value < 0.05 and ci[0] is not None and ci[0] > 0),
            "contingency_after_loss_vs_other": table,
        },
        "market_alternation_hypothesis": {
            "resolved_events": len(labels),
            "p_next_direction_differs": float(direction_flips.mean()) if len(direction_flips) else None,
            "excess_over_coin_flip": float(direction_flips.mean() - 0.5) if len(direction_flips) else None,
        },
        "warning": "Последовательность PnL зависит от выбранной моделью стороны и цены; это диагностика, а не торговое правило.",
    }
    settings.LOSS_REVERSAL_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
