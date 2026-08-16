"""Rolling-метрики, bootstrap CI, калибровка, direction collapse и shadow leaderboard."""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from statistics import mean
from typing import Any

import app_config as settings
from sklearn.metrics import brier_score_loss, roc_auc_score


def _bootstrap_lower(values: list[float], seed: int = 42) -> float | None:
    if len(values) < 10:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return float(means[int(len(means) * 0.025)])


def model_health(connection: sqlite3.Connection) -> dict[str, Any]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT p.event_slug,p.outcome,p.realized_pnl_usdc,p.closed_at,
                  COALESCE(d.model_name,s.model_name,'unknown') model_name,
                  d.predicted_up_probability
           FROM paper_positions p JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.status IN ('closed','resolved') ORDER BY COALESCE(p.closed_at,p.opened_at)"""
    ).fetchall()
    has_training = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_examples'"
    ).fetchone()
    labels = {
        str(row[0]): int(row[1]) for row in connection.execute(
            """SELECT event_slug,MAX(CASE WHEN outcome='Up' THEN label END)
               FROM training_examples GROUP BY event_slug"""
        ).fetchall() if row[1] is not None
    } if has_training else {}
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_name"])].append(row)
    result: dict[str, Any] = {}
    for model, items in grouped.items():
        model_result = {"windows": {}, "alerts": []}
        for window in settings.MODEL_HEALTH_WINDOWS:
            sample = items[-window:]
            pnls = [float(row["realized_pnl_usdc"] or 0) for row in sample]
            directions = [str(row["outcome"]) for row in sample]
            pairs = [
                (labels[str(row["event_slug"])], float(row["predicted_up_probability"]))
                for row in sample if row["predicted_up_probability"] is not None and str(row["event_slug"]) in labels
            ]
            auc = roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs]) if len({p[0] for p in pairs}) == 2 else None
            brier = brier_score_loss([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
            direction_share = max(directions.count("Up"), directions.count("Down")) / max(1, len(directions))
            model_result["windows"][str(window)] = {
                "events": len(sample), "net_pnl": sum(pnls), "expectancy": mean(pnls) if pnls else 0.0,
                "bootstrap_ci95_lower": _bootstrap_lower(pnls), "roc_auc": auc, "brier": brier,
                "up": directions.count("Up"), "down": directions.count("Down"),
                "max_direction_share": direction_share,
            }
        latest = model_result["windows"][str(settings.MODEL_HEALTH_WINDOWS[0])]
        if latest["roc_auc"] is not None and latest["roc_auc"] < settings.DRIFT_ALERT_ROC_AUC:
            model_result["alerts"].append("ROC-AUC drift")
        if latest["brier"] is not None and latest["brier"] > settings.DRIFT_ALERT_BRIER:
            model_result["alerts"].append("calibration drift")
        if latest["max_direction_share"] > settings.DRIFT_ALERT_DIRECTION_SHARE:
            model_result["alerts"].append("direction collapse")
        result[model] = model_result
    shadow: list[dict[str, Any]] = []
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='shadow_predictions'").fetchone():
        shadow_rows = connection.execute(
            """WITH ranked AS (
                 SELECT *,ROW_NUMBER() OVER(
                   PARTITION BY model_name,event_slug
                   ORDER BY CASE WHEN action LIKE 'BUY_%' THEN 0 ELSE 1 END,observed_at,id
                 ) event_rank
                 FROM shadow_predictions WHERE status='resolved'
               )
               SELECT model_name,event_slug,action,direction,predicted_up_probability,
                      resolved_label,COALESCE(net_pnl_usdc,0) net_pnl_usdc,evaluated_at
               FROM ranked WHERE event_rank=1 ORDER BY evaluated_at,event_slug"""
        ).fetchall()
        shadow_grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in shadow_rows:
            shadow_grouped[str(row["model_name"])].append(row)
        for model_name, items in shadow_grouped.items():
            trades = [row for row in items if str(row["action"]).startswith("BUY_")]
            pnls = [float(row["net_pnl_usdc"] or 0) for row in trades]
            equity = peak = drawdown = 0.0
            for pnl in pnls:
                equity += pnl
                peak = max(peak, equity)
                drawdown = max(drawdown, peak - equity)
            probability_pairs = [
                (int(row["resolved_label"]), float(row["predicted_up_probability"]))
                for row in items
                if row["resolved_label"] is not None and row["predicted_up_probability"] is not None
            ]
            y_true = [pair[0] for pair in probability_pairs]
            y_score = [pair[1] for pair in probability_pairs]
            shadow.append({
                "model_name": model_name,
                "samples": len(items),
                "trades": len(trades),
                "up": sum(str(row["direction"]) == "Up" for row in trades),
                "down": sum(str(row["direction"]) == "Down" for row in trades),
                "net_pnl": sum(pnls),
                "expectancy": mean(pnls) if pnls else 0.0,
                "max_drawdown": drawdown,
                "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None,
                "brier": brier_score_loss(y_true, y_score) if y_true else None,
                "first_event": str(items[0]["event_slug"]) if items else None,
                "last_event": str(items[-1]["event_slug"]) if items else None,
            })
        shadow.sort(key=lambda item: float(item["net_pnl"]), reverse=True)
    return {
        "models": result,
        "shadow_leaderboard": shadow,
        "shadow_method": "full resolved history; one decision per independent event; first executable BUY has priority",
    }
