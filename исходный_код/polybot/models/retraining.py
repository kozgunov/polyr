"""Champion/challenger retraining every N new independent resolved events."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings

from polybot.models.train_catboost_model import train as train_catboost
from polybot.models.train_direction_model import train as train_custom
from polybot.models.train_exit_model import train as train_exit
from polybot.models.train_pnl_model import train as train_pnl
from polybot.models.action_value_dataset import build as build_action_value_dataset
from polybot.models.counterfactual_actions import build as build_counterfactual_actions
from polybot.trading.fees import total_fee_usdc


def _reward(pnl: float, notional: float) -> tuple[float, str]:
    roi = pnl / max(0.01, notional)
    score = max(-1.0, min(3.0, roi))
    if roi <= -0.5:
        return score, "catastrophic_loss"
    if roi < 0:
        return score, "loss"
    if roi < 0.25:
        return score, "small_profit"
    if roi < 0.75:
        return score, "strong_profit"
    return score, "exceptional_profit"


def _qwen_dataset(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    rows = connection.execute(
        """SELECT event_slug,outcome,observed_at,label,features_json
           FROM training_examples ORDER BY observed_at"""
    ).fetchall()
    written = 0
    actual_trades = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for slug, outcome, observed_at, label, raw_features in rows:
            features = json.loads(raw_features)
            ask = features.get("best_ask")
            if ask is None or not 0 < float(ask) < 1:
                continue
            notional = float(settings.PAPER_ENTRY_NOTIONAL_USDC)
            shares = notional / float(ask)
            pnl = shares * int(label) - notional - total_fee_usdc(shares, float(ask))
            reward_score, reward_tier = _reward(pnl, notional)
            action = f"BUY_{str(outcome).upper()}" if pnl > 0 else "WAIT"
            distance = features.get("distance_to_target_pct")
            remaining = features.get("remaining_seconds")
            explanation = (
                f"Price to Beat distance={float(distance or 0):+.4f}%, remaining={float(remaining or 0):.0f}s, "
                f"entry={float(ask):.3f}, net PnL label={pnl:+.3f}; reward={reward_tier}."
            )
            compact = {
                key: features.get(key) for key in (
                    "target_price", "reference_price", "distance_to_target_pct", "remaining_seconds",
                    "best_bid", "best_ask", "spread", "best_bid_size", "best_ask_size",
                    "bybit_to_target_pct", "okx_to_target_pct", "pyth_to_target_pct",
                    "realized_volatility_60s_pct", "distance_time_score",
                    "target_momentum_15s_pct", "target_momentum_30s_pct", "target_momentum_60s_pct",
                )
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "BTC Up/Down 5m: maximize long-run net PnL after fees; abstain when uncertain. Return JSON action, confidence, explanation."},
                    {"role": "user", "content": json.dumps({"event": slug, "observed_at": observed_at, "outcome_contract": outcome, "features": compact}, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps({"action": action, "confidence": 1.0 if pnl > 0 else 0.0, "explanation": explanation, "net_pnl_target": pnl, "reward_score": reward_score, "reward_tier": reward_tier}, ensure_ascii=False)},
                ],
                "event_slug": slug, "net_pnl_target": pnl, "reward_score": reward_score,
                "reward_tier": reward_tier, "label_source": "resolved_outcome_counterfactual",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if {"paper_positions", "model_decisions"}.issubset(tables):
            actual_rows = connection.execute(
                """SELECT p.event_slug,p.outcome,p.opened_at,p.realized_pnl_usdc,
                          d.action,d.confidence,d.reason,d.market_state_json,d.model_name
                   FROM paper_positions p JOIN model_decisions d ON d.id=p.entry_decision_id
                   WHERE p.status IN ('closed','resolved') ORDER BY p.opened_at"""
            ).fetchall()
            for slug, outcome, opened_at, pnl, action, confidence, reason, raw_state, model_name in actual_rows:
                pnl = float(pnl or 0.0)
                reward_score, reward_tier = _reward(pnl, float(settings.PAPER_ENTRY_NOTIONAL_USDC))
                try:
                    state = json.loads(raw_state)
                except (TypeError, json.JSONDecodeError):
                    state = {}
                explanation = (
                    f"Фактическая demo-сделка {model_name}: {reason}. Итоговый net PnL={pnl:+.3f}; "
                    f"оценка результата={reward_tier}. Большая чистая прибыль предпочтительнее малого плюса."
                )
                payload = {
                    "messages": [
                        {"role": "system", "content": "BTC Up/Down 5m: maximize long-run net PnL after fees. Learn from actual paper outcomes and explain the decision briefly."},
                        {"role": "user", "content": json.dumps({"event": slug, "observed_at": opened_at, "market_state": state}, ensure_ascii=False)},
                        {"role": "assistant", "content": json.dumps({"action": action, "direction": outcome, "confidence": float(confidence), "explanation": explanation, "net_pnl_target": pnl, "reward_score": reward_score, "reward_tier": reward_tier}, ensure_ascii=False)},
                    ],
                    "event_slug": slug, "net_pnl_target": pnl, "reward_score": reward_score,
                    "reward_tier": reward_tier, "label_source": "actual_paper_trade",
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
                actual_trades += 1
    return {
        "rows": written, "counterfactual_rows": written - actual_trades,
        "actual_trade_rows": actual_trades, "events": len({str(row[0]) for row in rows}),
    }


def run_if_due(force: bool = False) -> dict[str, Any]:
    connection = sqlite3.connect(settings.DATABASE_PATH, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS model_training_cycles(
             id INTEGER PRIMARY KEY,started_at TEXT NOT NULL,completed_at TEXT,event_count INTEGER NOT NULL,
             new_events INTEGER NOT NULL,status TEXT NOT NULL,candidate_dir TEXT,report_json TEXT,error TEXT)"""
    )
    event_count = int(connection.execute("SELECT COUNT(DISTINCT event_slug) FROM training_examples").fetchone()[0])
    previous = connection.execute(
        "SELECT event_count FROM model_training_cycles WHERE status='completed' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    previous_count = int(previous[0]) if previous else 0
    new_events = event_count - previous_count
    if not force and new_events < settings.RETRAIN_EVERY_NEW_EVENTS:
        connection.close()
        return {"status": "not_due", "event_count": event_count, "new_events": new_events}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = settings.MODEL_CANDIDATE_DIR / f"cycle_{event_count}_{timestamp}"
    candidate.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    cursor = connection.execute(
        "INSERT INTO model_training_cycles(started_at,event_count,new_events,status,candidate_dir) VALUES(?,?,?,?,?)",
        (started, event_count, new_events, "running", str(candidate)),
    )
    cycle_id = int(cursor.lastrowid)
    connection.commit()
    snapshot_path = candidate / "training_snapshot.sqlite3"
    try:
        # Все тяжёлые построения датасетов пишут только в консистентный снимок.
        # Рабочая market_data.sqlite3 остаётся доступной collector/trading-engine.
        snapshot_connection = sqlite3.connect(snapshot_path)
        connection.backup(snapshot_connection)
        snapshot_connection.close()
        custom_metrics = train_custom(
            snapshot_path, artifact_path=candidate / "custom" / "btc_5m_direction.joblib",
        )
        catboost_metrics = train_catboost(snapshot_path, output=candidate / "catboost")
        action_value_dataset = build_action_value_dataset(snapshot_path)
        counterfactual_actions = build_counterfactual_actions(snapshot_path)
        try:
            pnl_metrics: dict[str, Any] = train_pnl(
                snapshot_path,
                artifact_path=candidate / "pnl_value" / "btc_5m_net_pnl.joblib",
                report_path=candidate / "pnl_value" / "pnl_model_report.json",
            )
        except (RuntimeError, ValueError) as exc:
            pnl_metrics = {"status": "blocked", "reason": str(exc)}
        try:
            exit_metrics: dict[str, Any] = train_exit(
                snapshot_path,
                artifact_path=candidate / "exit" / "btc_5m_exit_value.joblib",
                report_path=candidate / "exit" / "exit_model_report.json",
            )
        except RuntimeError as exc:
            exit_metrics = {"status": "blocked", "reason": str(exc)}
        snapshot_reader = sqlite3.connect(snapshot_path)
        try:
            qwen_dataset = _qwen_dataset(snapshot_reader, candidate / "qwen" / "supervised_reward_examples.jsonl")
        finally:
            snapshot_reader.close()
        report = {
            "cycle_id": cycle_id, "event_count": event_count, "new_events": new_events,
            "custom": custom_metrics, "catboost": catboost_metrics,
            "action_value_dataset": action_value_dataset,
            "counterfactual_actions": counterfactual_actions,
            "pnl_value": pnl_metrics,
            "exit_value": exit_metrics,
            "qwen_dataset": qwen_dataset,
            "activation": "candidate_only_manual_promotion_required",
            "warning": "Qwen weights were not fine-tuned automatically; dataset is prepared for a separate LoRA/QLoRA experiment.",
        }
        (candidate / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        connection.execute(
            "UPDATE model_training_cycles SET completed_at=?,status='completed',report_json=? WHERE id=?",
            (datetime.now(UTC).isoformat(), json.dumps(report, ensure_ascii=False), cycle_id),
        )
        connection.commit()
        return {"status": "completed", "candidate_dir": str(candidate), **report}
    except Exception as error:
        connection.execute(
            "UPDATE model_training_cycles SET completed_at=?,status='failed',error=? WHERE id=?",
            (datetime.now(UTC).isoformat(), f"{type(error).__name__}: {error}", cycle_id),
        )
        connection.commit()
        raise
    finally:
        try:
            snapshot_path.unlink(missing_ok=True)
        except OSError:
            pass
        connection.close()
