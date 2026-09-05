"""Continuous paper trading for BTC Up/Down 5m; never signs or submits orders."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median, pstdev

import app_config as settings
import joblib

from polybot.collectors.pipeline import now
from polybot.models.model_policy import decide_with_model
from polybot.models.action_value import position_notional
from polybot.models.model_registry import model_is_ready
from polybot.models.event_history import context as event_history_context, summaries_from_connection
from polybot.trading.fees import net_buy_edge, state_fee_usdc, total_fee_usdc
from polybot.trading.execution_simulator import fak_sell, limit_buy
from polybot.trading.execution_validity import validate_entry_execution, validate_entry_state
from polybot.trading.policy import Decision, MarketState, PositionState
from polybot.trading.live_executor import cancel_order as cancel_live_order
from polybot.trading.live_executor import get_order as get_live_order
from polybot.trading.live_executor import market_minimum_size
from polybot.trading.live_executor import summarize_order_fills
from polybot.trading.live_executor import submit_limit_order

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_sessions (
  session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
  initial_balance_usdc REAL NOT NULL, cash_balance_usdc REAL NOT NULL,
  realized_pnl_usdc REAL NOT NULL DEFAULT 0, total_wagered_usdc REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL, strategy_version TEXT NOT NULL, model_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_decisions (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, observed_at TEXT NOT NULL,
  event_slug TEXT, action TEXT NOT NULL, confidence REAL NOT NULL, reason TEXT NOT NULL,
  tags_json TEXT NOT NULL, market_state_json TEXT NOT NULL, position_state_json TEXT,
  provider TEXT NOT NULL, model_name TEXT NOT NULL, executed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS paper_positions (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, event_slug TEXT NOT NULL,
  token_id TEXT NOT NULL, outcome TEXT NOT NULL, status TEXT NOT NULL,
  opened_at TEXT NOT NULL, average_price REAL NOT NULL, shares REAL NOT NULL,
  cost_usdc REAL NOT NULL, current_price REAL, closed_at TEXT, close_price REAL,
  realized_pnl_usdc REAL, close_reason TEXT, entry_decision_id INTEGER, exit_decision_id INTEGER,
  exit_stage INTEGER NOT NULL DEFAULT 0,
  exit_timing TEXT NOT NULL DEFAULT 'open', had_early_exit INTEGER NOT NULL DEFAULT 0,
  early_exit_pnl_usdc REAL NOT NULL DEFAULT 0
  ,execution_valid INTEGER NOT NULL DEFAULT 1, invalid_reason TEXT
);
CREATE TABLE IF NOT EXISTS paper_orders (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, decision_id INTEGER NOT NULL,
  event_slug TEXT NOT NULL, action TEXT NOT NULL, order_type TEXT NOT NULL,
  requested_price REAL, filled_price REAL, shares REAL, notional_usdc REAL,
  fee_usdc REAL NOT NULL DEFAULT 0, slippage_bps REAL NOT NULL, status TEXT NOT NULL,
  created_at TEXT NOT NULL, requested_shares REAL, requested_notional_usdc REAL,
  last_checked_at TEXT, check_count INTEGER NOT NULL DEFAULT 0,
  submit_best_bid REAL, submit_best_ask REAL, submit_book_timestamp TEXT,
  fill_best_bid REAL, fill_best_ask REAL, fill_observed_at TEXT,
  observed_slippage_bps REAL, book_age_ms_at_submit REAL, book_age_ms_at_fill REAL,
  fill_probability_kind TEXT
  ,execution_valid INTEGER NOT NULL DEFAULT 1, invalid_reason TEXT
);
CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, observed_at TEXT NOT NULL,
  cash_usdc REAL NOT NULL, open_value_usdc REAL NOT NULL, equity_usdc REAL NOT NULL,
  realized_pnl_usdc REAL NOT NULL, unrealized_pnl_usdc REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS strategy_labels (
  decision_id INTEGER PRIMARY KEY, event_slug TEXT NOT NULL, resolved_label INTEGER,
  decision_was_correct INTEGER, realized_pnl_usdc REAL, resolution_source TEXT,
  labeled_at TEXT
);
CREATE TABLE IF NOT EXISTS event_resolutions (
  event_slug TEXT NOT NULL, outcome TEXT NOT NULL, token_id TEXT,
  label INTEGER NOT NULL, resolved_at TEXT NOT NULL, source TEXT NOT NULL,
  PRIMARY KEY(event_slug,outcome)
);
CREATE TABLE IF NOT EXISTS counterfactual_entries (
  id INTEGER PRIMARY KEY, decision_id INTEGER NOT NULL, event_slug TEXT NOT NULL,
  observed_at TEXT NOT NULL, outcome TEXT NOT NULL, entry_price REAL NOT NULL,
  confidence REAL NOT NULL, hypothetical_notional_usdc REAL NOT NULL,
  horizon_seconds INTEGER, evaluated_at TEXT, exit_price REAL,
  fee_usdc REAL NOT NULL DEFAULT 0, counterfactual_pnl_usdc REAL,
  resolved_label INTEGER, status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS action_counterfactuals (
  id INTEGER PRIMARY KEY,decision_id INTEGER,event_slug TEXT NOT NULL,observed_at TEXT NOT NULL,
  action TEXT NOT NULL,outcome TEXT,entry_price REAL,current_bid REAL,notional_usdc REAL,
  shares REAL,cost_usdc REAL,horizon_seconds INTEGER NOT NULL,status TEXT NOT NULL,
  evaluated_at TEXT,exit_price REAL,fee_usdc REAL NOT NULL DEFAULT 0,net_pnl_usdc REAL,
  resolved_label INTEGER,features_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_predictions (
  id INTEGER PRIMARY KEY,event_slug TEXT NOT NULL,observed_at TEXT NOT NULL,model_name TEXT NOT NULL,
  action TEXT NOT NULL,direction TEXT,confidence REAL NOT NULL,predicted_up_probability REAL,
  entry_price REAL,notional_usdc REAL,reason TEXT,status TEXT NOT NULL DEFAULT 'pending',
  resolved_label INTEGER,net_pnl_usdc REAL,evaluated_at TEXT
);
CREATE TABLE IF NOT EXISTS exit_shadow_predictions (
  id INTEGER PRIMARY KEY,source TEXT NOT NULL,position_id INTEGER NOT NULL,event_slug TEXT NOT NULL,
  outcome TEXT NOT NULL,policy TEXT NOT NULL,observed_at TEXT NOT NULL,action TEXT NOT NULL,
  predicted_advantage_usdc REAL,predicted_close_probability REAL,selected_pnl_usdc REAL,
  hold_pnl_usdc REAL,realized_advantage_usdc REAL,status TEXT NOT NULL DEFAULT 'pending',
  evaluated_at TEXT,UNIQUE(source,position_id,policy)
);
CREATE TABLE IF NOT EXISTS next_event_forecasts (
  id INTEGER PRIMARY KEY, source_event_slug TEXT NOT NULL, next_event_slug TEXT NOT NULL,
  observed_at TEXT NOT NULL, sample_bucket INTEGER NOT NULL, seconds_to_source_end REAL,
  current_target_distance_pct REAL, current_consensus_return_pct REAL,
  current_realized_volatility REAL, history_features_json TEXT NOT NULL,
  predictor_version TEXT NOT NULL, predicted_next_up_probability REAL NOT NULL,
  predicted_direction TEXT NOT NULL, confidence REAL NOT NULL, planned_token_id TEXT,
  planned_limit_price REAL, planned_notional_usdc REAL, planned_order_type TEXT,
  preopen_book_available INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
  next_resolved_label INTEGER, hypothetical_filled INTEGER,
  hypothetical_fill_price REAL, hypothetical_pnl_usdc REAL, evaluated_at TEXT,
  UNIQUE(source_event_slug,next_event_slug,sample_bucket)
);
CREATE TABLE IF NOT EXISTS shadow_preopen_orders (
  id INTEGER PRIMARY KEY, forecast_id INTEGER, source_event_slug TEXT NOT NULL,
  next_event_slug TEXT NOT NULL, outcome TEXT NOT NULL, token_id TEXT NOT NULL,
  limit_price REAL NOT NULL, notional_usdc REAL NOT NULL, confidence REAL NOT NULL,
  model_name TEXT NOT NULL, requested_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  expires_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'working',
  filled_at TEXT, fill_price REAL, cancelled_at TEXT, cancel_reason TEXT
);
CREATE TABLE IF NOT EXISTS runtime_controls (
  control_key TEXT PRIMARY KEY, control_value TEXT NOT NULL, updated_at TEXT NOT NULL,
  reason TEXT
);
CREATE TABLE IF NOT EXISTS live_orders (
  id INTEGER PRIMARY KEY, decision_id INTEGER NOT NULL, event_slug TEXT NOT NULL,
  token_id TEXT NOT NULL, outcome TEXT NOT NULL, side TEXT NOT NULL,
  order_id TEXT UNIQUE NOT NULL, order_type TEXT NOT NULL, requested_price REAL NOT NULL,
  requested_size REAL NOT NULL, matched_size REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL, created_at TEXT NOT NULL, expiration_at INTEGER,
  last_checked_at TEXT, error TEXT
  ,execution_valid INTEGER NOT NULL DEFAULT 1, invalid_reason TEXT
);
CREATE TABLE IF NOT EXISTS live_positions (
  id INTEGER PRIMARY KEY, event_slug TEXT UNIQUE NOT NULL, token_id TEXT NOT NULL,
  outcome TEXT NOT NULL, status TEXT NOT NULL, opened_at TEXT NOT NULL,
  average_price REAL NOT NULL, shares REAL NOT NULL, cost_usdc REAL NOT NULL,
  current_price REAL, closed_at TEXT, close_price REAL, realized_pnl_usdc REAL NOT NULL DEFAULT 0,
  entry_decision_id INTEGER, exit_decision_id INTEGER, exit_stage INTEGER NOT NULL DEFAULT 0,
  exit_timing TEXT NOT NULL DEFAULT 'open', had_early_exit INTEGER NOT NULL DEFAULT 0,
  early_exit_pnl_usdc REAL NOT NULL DEFAULT 0
  ,execution_valid INTEGER NOT NULL DEFAULT 1, invalid_reason TEXT
);
CREATE TABLE IF NOT EXISTS live_execution_errors (
  id INTEGER PRIMARY KEY, observed_at TEXT NOT NULL, decision_id INTEGER,
  event_slug TEXT, action TEXT NOT NULL, error_type TEXT NOT NULL, detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_decision_shadow_orders (
  id INTEGER PRIMARY KEY, decision_id INTEGER NOT NULL, live_order_id TEXT UNIQUE NOT NULL,
  event_slug TEXT NOT NULL, outcome TEXT NOT NULL, side TEXT NOT NULL,
  requested_price REAL NOT NULL, requested_size REAL NOT NULL, observed_at TEXT NOT NULL,
  submit_best_bid REAL, submit_best_ask REAL, status TEXT NOT NULL DEFAULT 'submitted',
  simulated_fill_price REAL, simulated_fill_size REAL NOT NULL DEFAULT 0,
  live_fill_price REAL, live_fill_size REAL NOT NULL DEFAULT 0,
  resolved_label INTEGER, paper_net_pnl_usdc REAL, live_net_pnl_usdc REAL,
  evaluated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON model_decisions(observed_at);
CREATE INDEX IF NOT EXISTS idx_decisions_event ON model_decisions(event_slug,id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON paper_positions(status,event_slug);
CREATE INDEX IF NOT EXISTS idx_counterfactual_pending ON counterfactual_entries(status,event_slug);
CREATE INDEX IF NOT EXISTS idx_strategy_labels_event ON strategy_labels(event_slug,decision_id);
CREATE INDEX IF NOT EXISTS idx_event_resolutions_token ON event_resolutions(event_slug,token_id);
CREATE INDEX IF NOT EXISTS idx_action_cf_pending ON action_counterfactuals(status,event_slug,horizon_seconds);
CREATE INDEX IF NOT EXISTS idx_shadow_pending ON shadow_predictions(status,event_slug,model_name);
CREATE INDEX IF NOT EXISTS idx_exit_shadow_pending ON exit_shadow_predictions(status,event_slug,policy);
CREATE INDEX IF NOT EXISTS idx_next_event_forecast_pending ON next_event_forecasts(status,next_event_slug);
CREATE INDEX IF NOT EXISTS idx_preopen_orders_active ON shadow_preopen_orders(status,next_event_slug,id);
CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live_orders(status,event_slug);
CREATE INDEX IF NOT EXISTS idx_live_positions_status ON live_positions(status,event_slug);
"""


def event_start(slug: str) -> datetime:
    return datetime.fromtimestamp(int(slug.rsplit("-", 1)[-1]), UTC)


class PaperEngine:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.db = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
        self.db.row_factory = sqlite3.Row
        # Все процессы проекта используют WAL: читатели дашборда не должны
        # блокировать короткие записи торгового движка и сборщика.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
        self.db.executescript(SCHEMA)
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_examples'").fetchone():
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_training_event_outcome ON training_examples(event_slug,outcome)")
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_snapshots'").fetchone():
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_event_outcome_time "
                "ON market_snapshots(event_slug,outcome,collected_at)"
            )
        self._migrate()
        self.db.commit()
        self.session_id = self._session()
        self.last_logged_decision = 0.0
        self.last_action_at = 0.0
        self.last_counterfactual_at = 0.0
        self.last_v8_counterfactual_at = 0.0
        self.last_shadow_at = 0.0
        self.confirmation_key: str | None = None
        self.confirmation_count = 0
        self.confirmation_started = 0.0
        self._history_slug: str | None = None
        self._history_features: dict[str, float] = {}
        self.validation_bad_cycles = 0
        self.last_next_forecast_at = 0.0
        self.exit_shadow_artifact = (
            joblib.load(settings.EXIT_SEQUENCE_SHADOW_ARTIFACT_PATH)
            if settings.EXIT_SEQUENCE_SHADOW_ARTIFACT_PATH.exists() else None
        )

    def _migrate(self) -> None:
        additions = {
            "paper_sessions": {
                "consecutive_losses": "INTEGER NOT NULL DEFAULT 0", "stopped_reason": "TEXT",
                "total_fees_usdc": "REAL NOT NULL DEFAULT 0", "run_label": "TEXT",
                "config_json": "TEXT", "parent_session_id": "TEXT",
            },
            "paper_positions": {
                "fees_usdc": "REAL NOT NULL DEFAULT 0", "gross_pnl_usdc": "REAL",
                "exit_stage": "INTEGER NOT NULL DEFAULT 0",
                "exit_timing": "TEXT NOT NULL DEFAULT 'open'",
                "had_early_exit": "INTEGER NOT NULL DEFAULT 0",
                "early_exit_pnl_usdc": "REAL NOT NULL DEFAULT 0",
                "execution_valid": "INTEGER NOT NULL DEFAULT 1", "invalid_reason": "TEXT",
                "resolution_labeled_at": "TEXT", "settlement_latency_seconds": "REAL",
                "provisional_resolution": "INTEGER NOT NULL DEFAULT 0",
                "provisional_label": "INTEGER", "provisional_resolved_at": "TEXT",
                "provisional_source": "TEXT", "official_label": "INTEGER",
                "official_reconciled_at": "TEXT", "provisional_mismatch": "INTEGER",
            },
            "paper_orders": {
                "expiration_at": "TEXT", "price_cap": "REAL", "fill_probability": "REAL",
                "latency_ms": "INTEGER", "execution_reason": "TEXT",
                "requested_shares": "REAL", "requested_notional_usdc": "REAL",
                "last_checked_at": "TEXT", "check_count": "INTEGER NOT NULL DEFAULT 0",
                "execution_valid": "INTEGER NOT NULL DEFAULT 1", "invalid_reason": "TEXT",
                "submit_best_bid": "REAL", "submit_best_ask": "REAL", "submit_book_timestamp": "TEXT",
                "fill_best_bid": "REAL", "fill_best_ask": "REAL", "fill_observed_at": "TEXT",
                "observed_slippage_bps": "REAL", "book_age_ms_at_submit": "REAL",
                "book_age_ms_at_fill": "REAL", "fill_probability_kind": "TEXT",
            },
            "model_decisions": {"predicted_up_probability": "REAL", "predicted_down_probability": "REAL", "expected_net_edge": "REAL"},
            "live_positions": {
                "exit_timing": "TEXT NOT NULL DEFAULT 'open'",
                "had_early_exit": "INTEGER NOT NULL DEFAULT 0",
                "early_exit_pnl_usdc": "REAL NOT NULL DEFAULT 0",
                "execution_valid": "INTEGER NOT NULL DEFAULT 1", "invalid_reason": "TEXT",
                "resolution_labeled_at": "TEXT", "settlement_latency_seconds": "REAL",
                "fees_usdc": "REAL NOT NULL DEFAULT 0", "gross_pnl_usdc": "REAL",
                "mark_price_updated_at": "TEXT", "ledger_validated": "INTEGER NOT NULL DEFAULT 0",
            },
            "live_orders": {
                "execution_valid": "INTEGER NOT NULL DEFAULT 1", "invalid_reason": "TEXT",
                "average_fill_price": "REAL", "fill_notional_usdc": "REAL NOT NULL DEFAULT 0",
                "fee_usdc": "REAL NOT NULL DEFAULT 0", "fill_source": "TEXT",
            },
        }
        for table, columns in additions.items():
            existing = {row[1] for row in self.db.execute(f'PRAGMA table_info("{table}")')}
            for name, definition in columns.items():
                if name not in existing:
                    self.db.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}')
        self._mark_historical_execution_validity()
        self.db.execute(
            "INSERT OR IGNORE INTO runtime_controls VALUES('trading_mode','paper',?,?)",
            (now(), "safe default"),
        )

    def _mark_historical_execution_validity(self) -> None:
        """Помечает невозможные исполнения, сохраняя исходную историю и raw PnL."""
        for table in ("paper_positions", "live_positions"):
            rows = self.db.execute(
                f"SELECT id,event_slug,opened_at,average_price,execution_valid,invalid_reason FROM {table}"
            ).fetchall()
            for row in rows:
                if int(row["execution_valid"] or 0) == 0 and "duplicate_cumulative_fill_reconciliation_bug" in str(row["invalid_reason"] or ""):
                    continue
                try:
                    validity = validate_entry_execution(
                        str(row["event_slug"]), datetime.fromisoformat(str(row["opened_at"])),
                        float(row["average_price"]),
                    )
                except (TypeError, ValueError, OverflowError):
                    validity = None
                self.db.execute(
                    f"UPDATE {table} SET execution_valid=?,invalid_reason=? WHERE id=?",
                    (int(validity.valid) if validity else 0,
                     validity.reason if validity else "invalid_execution_metadata", row["id"]),
                )
        for table, position_table in (("paper_orders", "paper_positions"), ("live_orders", "live_positions")):
            self.db.execute(
                f"""UPDATE {table} SET execution_valid=0,
                       invalid_reason=COALESCE((SELECT invalid_reason FROM {position_table} p
                         WHERE p.event_slug={table}.event_slug AND p.execution_valid=0 LIMIT 1),invalid_reason)
                    WHERE EXISTS(SELECT 1 FROM {position_table} p
                         WHERE p.event_slug={table}.event_slug AND p.execution_valid=0)"""
            )
        self.db.execute(
            "INSERT OR IGNORE INTO runtime_controls VALUES('engine_state','running',?,?)",
            (now(), "safe default"),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO runtime_controls VALUES('selected_model',?,?,?)",
            (settings.DEFAULT_TRADING_MODEL, now(), "safe default"),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO runtime_controls VALUES('selected_entry_model',?,?,?)",
            (self._control("selected_model", settings.DEFAULT_ENTRY_MODEL), now(), "safe default"),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO runtime_controls VALUES('selected_exit_model',?,?,?)",
            (self._control("selected_model", settings.DEFAULT_EXIT_MODEL), now(), "safe default"),
        )
        # Восстанавливаем ранние исполнения старой истории из фактических paper_orders.
        self.db.execute(
            """UPDATE paper_positions SET
                 had_early_exit=1,
                 early_exit_pnl_usdc=COALESCE((
                   SELECT SUM(COALESCE(o.notional_usdc,0)-COALESCE(o.fee_usdc,0)-COALESCE(o.shares,0)*paper_positions.average_price)
                   FROM paper_orders o
                   WHERE o.session_id=paper_positions.session_id AND o.event_slug=paper_positions.event_slug
                     AND o.action IN ('PARTIAL_CLOSE','CLOSE') AND o.status IN ('filled','partially_filled')
                 ),0)
               WHERE EXISTS(
                 SELECT 1 FROM paper_orders o
                 WHERE o.session_id=paper_positions.session_id AND o.event_slug=paper_positions.event_slug
                   AND o.action IN ('PARTIAL_CLOSE','CLOSE') AND o.status IN ('filled','partially_filled')
               )"""
        )
        # Старую историю размечаем без изменения итоговых финансовых результатов.
        self.db.execute(
            """UPDATE paper_positions SET exit_timing=CASE
                 WHEN close_reason='market_resolution' AND had_early_exit=1 THEN 'partial_early_then_resolution'
                 WHEN close_reason='market_resolution' THEN 'held_to_resolution'
                 WHEN status='closed' THEN 'early_full_exit'
                 ELSE 'open' END
               WHERE status IN ('closed','resolved')"""
        )
        # Не допускаем, чтобы позиции от завершённых, уже неактивных сессий
        # навсегда считались открытыми из-за старого сбоя разметки. Финансовый
        # результат не переписываем: позиция лишь исключается из активных и
        # помечается для отдельного разбора в истории.
        stale_before = int(time.time()) - 60 * 60
        self.db.execute(
            """UPDATE paper_positions
               SET status='stale_orphaned',
                   closed_at=COALESCE(closed_at, ?),
                   close_reason=COALESCE(close_reason, 'stale_orphaned_unresolved'),
                   exit_timing=CASE WHEN exit_timing='open' THEN 'stale_orphaned' ELSE exit_timing END
               WHERE status='open'
                 AND CAST(SUBSTR(event_slug, INSTR(event_slug, '5m-') + 3) AS INTEGER) < ?
                 AND session_id NOT IN (
                   SELECT session_id FROM paper_sessions WHERE status IN ('running','paused')
                 )""",
            (now(), stale_before),
        )
        self.db.commit()

    def _refresh_resolution_ledger(self) -> None:
        """Материализует один официальный label на outcome и отвязывает settlement от тяжёлых JOIN."""
        if not self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_examples'"
        ).fetchone():
            return
        self.db.execute(
            """INSERT OR IGNORE INTO event_resolutions(event_slug,outcome,token_id,label,resolved_at,source)
               SELECT event_slug,outcome,MAX(token_id),MAX(label),?, 'training_examples'
               FROM training_examples WHERE label IS NOT NULL GROUP BY event_slug,outcome""",
            (now(),),
        )

    @staticmethod
    def _strategy_config() -> str:
        keys = (
            "MIN_ENTRY_CONFIDENCE", "MAX_HELD_WIN_PROBABILITY_FOR_EXIT",
            "PAPER_MIN_ENTRY_NET_EDGE", "PAPER_MIN_ENTRY_PRICE",
            "PAPER_MAX_ENTRY_PRICE", "PAPER_ENTRY_NOTIONAL_USDC", "PAPER_MAX_EVENT_EXPOSURE_USDC",
            "PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN", "PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE",
            "ENTRY_SIGNAL_CONFIRMATIONS", "REVERSAL_SIGNAL_CONFIRMATIONS", "DEFAULT_TRADING_MODEL",
            "CONSENSUS_ENTRY_CONFIDENCE", "CONSENSUS_MAX_HELD_WIN_PROBABILITY_FOR_EXIT",
            "REQUIRE_OFFICIAL_EVENT_TARGET", "MAX_TARGET_REFERENCE_AGE_SECONDS",
            "FIVE_STAGE_EXIT_ENABLED", "EXIT_ON_PROFIT_ALONE", "EXIT_STAGE_REMAINING_FRACTIONS",
            "EXIT_STAGE_PROFIT_RETURN_PCT", "EXIT_STAGE_MAX_HELD_PROBABILITY",
            "ACTION_VALUE_ENABLED", "ACTION_VALUE_MIN_EXPECTED_PNL_USDC",
            "FILL_PROBABILITY_MODEL_ENABLED", "FILL_PROBABILITY_MIN_ROC_AUC",
            "FILL_PROBABILITY_MAX_BRIER",
            "ACTION_PROBABILITY_MODEL_WEIGHT", "ADAPTIVE_POSITION_SIZING_ENABLED",
            "POSITION_SIZE_EDGE_TIERS", "EXIT_VALUE_ENABLED", "EXIT_VALUE_MARGIN_USDC",
            "FULL_EXIT_ONLY_ENABLED", "ENTRY_GRID_ENABLED", "ENTRY_GRID_MIN_ORDERS",
            "ENTRY_GRID_MAX_ORDERS", "ENTRY_GRID_PRICE_STEP", "ENTRY_GRID_MIN_ORDER_USDC",
            "EXIT_FALLBACK_MIN_ADVANTAGE_USDC", "EXIT_FALLBACK_MIN_ADVANTAGE_FRACTION",
            "ML_POLICY_WAIT_UTILITY_USDC",
            "EXECUTION_SIMULATION_ENABLED", "EXECUTION_QUEUE_AHEAD_FRACTION",
            "EXECUTION_MAX_BOOK_PARTICIPATION", "EXECUTION_BASE_LATENCY_MS",
            "PAPER_POST_EVENT_SETTLEMENT_ENABLED", "PAPER_POST_EVENT_SETTLEMENT_DELAY_SECONDS",
            "PAPER_POST_EVENT_WIN_BID_THRESHOLD", "PAPER_POST_EVENT_LOSS_ASK_THRESHOLD",
            "PAPER_POST_EVENT_MAX_QUOTE_DISTANCE_SECONDS",
        )
        return json.dumps({key: getattr(settings, key) for key in keys}, ensure_ascii=False)

    def _new_session(self, parent_session_id: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO paper_sessions(session_id,started_at,ended_at,initial_balance_usdc,cash_balance_usdc,
               realized_pnl_usdc,total_wagered_usdc,status,strategy_version,model_name,run_label,config_json,
               parent_session_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, now(), None, settings.PAPER_INITIAL_BALANCE_USDC, settings.PAPER_INITIAL_BALANCE_USDC,
             0.0, 0.0, "running", settings.STRATEGY_VERSION,
             self._control("selected_entry_model", settings.DEFAULT_ENTRY_MODEL), settings.STRATEGY_RUN_LABEL,
             self._strategy_config(),
             parent_session_id),
        )
        self.db.commit()
        return session_id

    def _session(self) -> str:
        row = self.db.execute(
            "SELECT session_id FROM paper_sessions WHERE status IN ('running','paused') ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row:
            runtime_state = self._control("engine_state", "running")
            session_status = "paused" if runtime_state == "paused" else "running"
            self.db.execute("UPDATE paper_sessions SET status=? WHERE session_id=?", (session_status, row[0]))
            self.db.commit()
            return str(row[0])
        return self._new_session()

    def _control(self, key: str, default: str) -> str:
        row = self.db.execute("SELECT control_value FROM runtime_controls WHERE control_key=?", (key,)).fetchone()
        return str(row[0]) if row else default

    def _handle_runtime_control(self) -> str:
        state = self._control("engine_state", "running")
        if state != "new_session_requested":
            return state
        stale_cutoff = int(time.time() - 300 - settings.STALE_POSITION_NEW_SESSION_GRACE_SECONDS)
        open_count = int(self.db.execute(
            """SELECT COUNT(*) FROM paper_positions WHERE session_id=? AND status='open'
               AND CAST(SUBSTR(event_slug,INSTR(event_slug,'5m-')+3) AS INTEGER)>?""",
            (self.session_id, stale_cutoff),
        ).fetchone()[0])
        if open_count:
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('new_session_status','waiting_for_flat',?,?)",
                (now(), "queued: current position is awaiting official settlement"),
            )
            self.db.commit()
            # Сохраняем engine_state=new_session_requested. До расчёта текущей позиции
            # шаг работает как пауза и не открывает следующую сделку; после settlement
            # тот же запрос автоматически создаст чистую сессию.
            return "paused"
        previous = self.session_id
        previous_strategy = str(self.db.execute(
            "SELECT strategy_version FROM paper_sessions WHERE session_id=?", (previous,),
        ).fetchone()[0])
        try:
            from polybot.models.version_archive import archive_profitable_session

            archive_profitable_session(self.db, previous)
        except Exception as exc:
            # Архив не должен останавливать сбор данных или торговый цикл.
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('model_archive_status',?,?,?)",
                ("error", now(), type(exc).__name__),
            )
        retired_v1 = previous_strategy == "trained_numeric_model_v1"
        self.db.execute(
            """UPDATE paper_sessions SET status='archived',ended_at=?,stopped_reason=COALESCE(stopped_reason,?),
               run_label=COALESCE(run_label,?) WHERE session_id=?""",
            (now(), "retired_after_negative_pnl_analysis" if retired_v1 else "manual_new_session",
             "v1_negative_overactive_exit_flip" if retired_v1 else "archived_strategy_run", previous),
        )
        self.session_id = self._new_session(previous)
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','running',?,?)",
            (now(), f"new session {self.session_id}"),
        )
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('new_session_status','completed',?,?)",
            (now(), f"new session {self.session_id}"),
        )
        self.db.commit()
        self.confirmation_key = None
        self.confirmation_count = 0
        return "running"

    def _apply_requested_mode_if_flat(self) -> None:
        current = self._control("trading_mode", "paper")
        requested = self._control("requested_trading_mode", current)
        if requested == current:
            if self._control("mode_switch_state", "idle") == "waiting_current_event":
                self.db.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state',?,?,?)",
                    ("active" if current == "live" else "idle", now(), "stale queued switch cleared"),
                )
                self.db.commit()
            return
        if requested not in {"paper", "live"}:
            return
        if current == "live":
            blocked = self.db.execute("SELECT 1 FROM live_positions WHERE status='open' LIMIT 1").fetchone()
            blocked = blocked or self._live_pending()
        else:
            blocked = self.db.execute(
                "SELECT 1 FROM paper_positions WHERE session_id=? AND status='open' LIMIT 1", (self.session_id,)
            ).fetchone()
        if blocked:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('trading_mode',?,?,?)",
            (requested, now(), "queued mode applied by trading engine while flat"),
        )
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state',?,?,?)",
            ("active" if requested == "live" else "idle", now(), "queued switch completed"),
        )
        self.db.commit()

    def _apply_pending_models(self) -> None:
        """Применяет выбранные модели после позиции либо по истечении пяти минут."""
        paper_open = self.db.execute(
            "SELECT 1 FROM paper_positions WHERE session_id=? AND status='open' LIMIT 1", (self.session_id,)
        ).fetchone()
        live_open = self.db.execute("SELECT 1 FROM live_positions WHERE status='open' LIMIT 1").fetchone()
        flat = not (paper_open or live_open or self._live_pending())
        current = datetime.now(UTC)
        changed = False
        for role in ("entry", "exit"):
            pending = self._control(f"pending_{role}_model", "")
            deadline_raw = self._control(f"pending_{role}_model_apply_at", "")
            if pending not in {"catboost", "custom", "gemma", "qwen", "qwen_lora", "consensus_qwen_custom", "consensus_gemma_custom"}:
                continue
            try:
                deadline = datetime.fromisoformat(deadline_raw)
            except ValueError:
                deadline = current
            if not flat and current < deadline:
                continue
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
                (f"selected_{role}_model", pending, now(), "queued model switch applied by engine"),
            )
            if role == "entry":
                self.db.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('selected_model',?,?,?)",
                    (pending, now(), "legacy alias for selected_entry_model"),
                )
            self.db.execute(
                "DELETE FROM runtime_controls WHERE control_key IN (?,?)",
                (f"pending_{role}_model", f"pending_{role}_model_apply_at"),
            )
            changed = True
        if changed:
            self.db.commit()

    def _live_total_pnl(self, state: MarketState | None) -> float:
        realized = float(self.db.execute(
            "SELECT COALESCE(SUM(realized_pnl_usdc),0) FROM live_positions "
            "WHERE status IN ('closed','resolved') AND execution_valid=1"
        ).fetchone()[0])
        row = self.db.execute("SELECT * FROM live_positions WHERE status='open' LIMIT 1").fetchone()
        if not row:
            return realized
        mark = row["current_price"] or row["average_price"]
        if state is not None and row["event_slug"] == state.event_slug:
            mark = state.up_bid if row["outcome"] == "Up" else state.down_bid
            if mark is not None:
                self.db.execute(
                    "UPDATE live_positions SET current_price=?,mark_price_updated_at=? WHERE id=?",
                    (float(mark), now(), row["id"]),
                )
                self.db.commit()
        return realized + float(row["shares"]) * float(mark or 0) - float(row["cost_usdc"])

    def _confirmed(self, decision: Decision, state: MarketState, position: PositionState | None) -> Decision:
        if settings.ML_AUTONOMOUS_POLICY_ENABLED and settings.ML_POLICY_SKIP_SIGNAL_CONFIRMATION:
            self.confirmation_key = None
            self.confirmation_count = 0
            return decision
        if decision.action.startswith("BUY_"):
            required = settings.ENTRY_SIGNAL_CONFIRMATIONS
            minimum_seconds = settings.SIGNAL_CONFIRMATION_MIN_SECONDS
        elif decision.action in {"CLOSE", "PARTIAL_CLOSE"} and "staged_risk_exit" in decision.tags:
            required = settings.EXIT_RISK_SIGNAL_CONFIRMATIONS
            minimum_seconds = settings.EXIT_RISK_CONFIRMATION_MIN_SECONDS
        elif decision.action in {"CLOSE", "PARTIAL_CLOSE"} and "staged_profit_exit" in decision.tags:
            required = settings.EXIT_PROFIT_SIGNAL_CONFIRMATIONS
            minimum_seconds = settings.EXIT_PROFIT_CONFIRMATION_MIN_SECONDS
        else:
            self.confirmation_key = None
            self.confirmation_count = 0
            return decision
        stage_tag = next((tag for tag in decision.tags if tag.startswith("exit_stage=")), "")
        key = f"{state.event_slug}:{decision.action}:{decision.direction or ''}:{position.outcome if position else ''}:{stage_tag}"
        current = time.monotonic()
        if key != self.confirmation_key:
            self.confirmation_key = key
            self.confirmation_count = 1
            self.confirmation_started = current
        else:
            self.confirmation_count += 1
        elapsed = current - self.confirmation_started
        if self.confirmation_count >= required and elapsed >= minimum_seconds:
            return decision
        waiting_action = "HOLD" if position else "WAIT"
        return Decision(
            waiting_action, decision.confidence,
            f"Подтверждаем устойчивость сигнала: {self.confirmation_count}/{required}, {elapsed:.1f}/{minimum_seconds:.1f} сек",
            [*decision.tags, "signal_confirmation_pending"],
        )

    def _direction_collapse(self, model_key: str) -> tuple[str | None, float, int]:
        """Возвращает доминирующую сторону по одному последнему прогнозу на событие."""
        rows = self.db.execute(
            """WITH ranked AS (
                 SELECT event_slug,predicted_up_probability,
                        ROW_NUMBER() OVER(PARTITION BY event_slug ORDER BY observed_at DESC,id DESC) event_rank
                 FROM model_decisions
                 WHERE model_name=? AND predicted_up_probability IS NOT NULL
               )
               SELECT predicted_up_probability FROM ranked WHERE event_rank=1
               ORDER BY event_slug DESC LIMIT ?""",
            (model_key, settings.DIRECTION_COLLAPSE_WINDOW_EVENTS),
        ).fetchall()
        count = len(rows)
        if count < settings.DIRECTION_COLLAPSE_MIN_EVENTS:
            return None, 0.0, count
        up = sum(float(row[0]) >= 0.5 for row in rows)
        down = count - up
        dominant = "Up" if up >= down else "Down"
        return dominant, max(up, down) / count, count

    def _latest_event_slug(self) -> str | None:
        row = self.db.execute(
            "SELECT slug FROM events WHERE active=1 AND closed=0 ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else None

    def _price_rows(self, start_iso: str) -> tuple[dict[str, float], dict[str, float]]:
        enabled = ("bybit", "okx", "pyth")
        latest_rows = self.db.execute(
            """SELECT p.source,p.price,p.collected_at FROM external_prices p
               JOIN (SELECT source,MAX(id) id FROM external_prices WHERE source IN (?,?,?) GROUP BY source) x ON p.id=x.id""",
            enabled,
        ).fetchall()
        current = datetime.now(UTC)
        limits = {
            "bybit": settings.MAX_EXCHANGE_AGE_SECONDS,
            "okx": settings.MAX_EXCHANGE_AGE_SECONDS,
            "pyth": settings.MAX_PYTH_AGE_SECONDS,
        }
        latest: dict[str, float] = {}
        for row in latest_rows:
            source = str(row[0])
            try:
                age = max(0.0, (current - datetime.fromisoformat(str(row[2]))).total_seconds())
            except ValueError:
                continue
            if age <= float(limits[source]):
                latest[source] = float(row[1])
        starts: dict[str, float] = {}
        for source in latest:
            row = self.db.execute(
                "SELECT price FROM external_prices WHERE source=? AND collected_at>=? ORDER BY collected_at LIMIT 1",
                (source, start_iso),
            ).fetchone()
            if row:
                starts[source] = float(row[0])
        return latest, starts

    def _enforce_validation_stop(self, state: MarketState | None) -> bool:
        """Обязательная пауза при устойчиво сильно устаревшей валидации."""
        if not settings.VALIDATION_HARD_STOP_ENABLED:
            return False
        current = datetime.now(UTC)
        issues: list[str] = []
        checks = []
        if settings.ENABLE_BYBIT:
            checks.append(("bybit", "external_prices", "source='bybit'", settings.VALIDATION_HARD_STOP_AGE_SECONDS))
        if settings.ENABLE_OKX:
            checks.append(("okx", "external_prices", "source='okx'", settings.VALIDATION_HARD_STOP_AGE_SECONDS))
        if settings.ENABLE_PYTH:
            checks.append(("pyth", "external_prices", "source='pyth'", max(settings.MAX_PYTH_AGE_SECONDS, settings.VALIDATION_HARD_STOP_AGE_SECONDS)))
        for source, table, condition, limit in checks:
            row = self.db.execute(
                f"SELECT collected_at FROM {table} WHERE {condition} ORDER BY id DESC LIMIT 1"
            ).fetchone()
            try:
                age = (current - datetime.fromisoformat(str(row[0]))).total_seconds() if row else math.inf
            except ValueError:
                age = math.inf
            if age > float(limit):
                issues.append(f"{source}:{age:.1f}s")
        if state is not None:
            book_times = [
                str(book.get("collected_at")) for book in state.book_json.values() if book.get("collected_at")
            ]
            try:
                book_age = max((current - datetime.fromisoformat(value)).total_seconds() for value in book_times)
            except (ValueError, TypeError):
                book_age = math.inf
            if book_age > settings.VALIDATION_HARD_STOP_AGE_SECONDS:
                issues.append(f"polymarket_book:{book_age:.1f}s")
        if not issues:
            self.validation_bad_cycles = 0
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('validation_status','healthy',?,?)",
                (now(), "all required price sources and CLOB books are fresh"),
            )
            stopped = self.db.execute(
                "SELECT stopped_reason FROM paper_sessions WHERE session_id=?", (self.session_id,)
            ).fetchone()
            if stopped and str(stopped[0] or "").startswith("validation_latency:"):
                self.db.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','running',?,?)",
                    (now(), "automatic recovery: validations are fresh again"),
                )
                self.db.execute(
                    "UPDATE paper_sessions SET status='running',stopped_reason=NULL WHERE session_id=?",
                    (self.session_id,),
                )
            self.db.commit()
            return False
        self.validation_bad_cycles += 1
        detail = ",".join(issues)
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('validation_status','degraded',?,?)",
            (now(), f"cycle={self.validation_bad_cycles}; {detail}"),
        )
        if self.validation_bad_cycles < settings.VALIDATION_HARD_STOP_CONSECUTIVE_CYCLES:
            self.db.commit()
            return False
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','paused',?,?)",
            (now(), f"mandatory validation-latency stop: {detail}"),
        )
        self.db.execute(
            "UPDATE paper_sessions SET status='paused',stopped_reason=? WHERE session_id=?",
            (f"validation_latency:{detail}", self.session_id),
        )
        self.db.commit()
        return True

    def market_state(self) -> MarketState | None:
        slug = self._latest_event_slug()
        if not slug:
            return None
        started = event_start(slug)
        current = datetime.now(UTC)
        elapsed = (current - started).total_seconds()
        remaining = 300.0 - elapsed
        latest, starts = self._price_rows(started.isoformat())
        returns = {
            source: (price / starts[source] - 1.0) * 100.0
            for source, price in latest.items() if source in starts and starts[source] > 0
        }
        source_values = list(latest.values())
        disagreement = (
            (max(source_values) - min(source_values)) / median(source_values) * 100.0 if len(source_values) >= 2 else 999.0
        )
        cutoff = (current - timedelta(seconds=settings.PAPER_SHARP_MOVE_LOOKBACK_SECONDS)).isoformat()
        recent = [float(row[0]) for row in self.db.execute(
            "SELECT price FROM external_prices WHERE collected_at>=?", (cutoff,)
        ).fetchall()]
        sharp = (max(recent) - min(recent)) / median(recent) * 100.0 if len(recent) >= 2 else 0.0
        volatility_cutoff = (current - timedelta(seconds=60)).isoformat()
        volatility_rows = [float(row[0]) for row in self.db.execute(
            "SELECT price FROM external_prices WHERE source='bybit' AND collected_at>=? ORDER BY collected_at",
            (volatility_cutoff,),
        ).fetchall()]
        log_returns = [
            math.log(volatility_rows[index] / volatility_rows[index - 1]) * 100.0
            for index in range(1, len(volatility_rows)) if volatility_rows[index - 1] > 0
        ]
        volatility = pstdev(log_returns) if len(log_returns) >= 2 else 0.0

        target_price = reference_price = None
        target_source = reference_observed_at = None
        target_distance_lags: dict[str, float] = {}
        try:
            target_row = self.db.execute(
                "SELECT target_price,source FROM event_targets WHERE event_slug=?", (slug,),
            ).fetchone()
            reference_row = self.db.execute(
                """SELECT reference_price,collected_at FROM reference_price_snapshots
                   WHERE event_slug=? AND reference_price IS NOT NULL
                   AND source='polymarket_crypto_price' ORDER BY collected_at DESC LIMIT 1""",
                (slug,),
            ).fetchone()
            if target_row:
                target_price, target_source = float(target_row[0]), str(target_row[1])
            if reference_row:
                reference_observed_at = str(reference_row[1])
                reference_age = (current - datetime.fromisoformat(reference_observed_at)).total_seconds()
                if reference_age <= settings.MAX_TARGET_REFERENCE_AGE_SECONDS:
                    reference_price = float(reference_row[0])
            if reference_price is None and latest:
                reference_price = median(latest.values())
            if target_price:
                for lag_seconds in (15, 30, 60):
                    lag_cutoff = (current - timedelta(seconds=lag_seconds)).isoformat()
                    lag_row = self.db.execute(
                        """SELECT price FROM external_prices WHERE source='bybit' AND collected_at<=?
                           ORDER BY collected_at DESC LIMIT 1""", (lag_cutoff,),
                    ).fetchone()
                    if lag_row:
                        target_distance_lags[str(lag_seconds)] = (float(lag_row[0]) / target_price - 1.0) * 100.0
        except sqlite3.OperationalError:
            # Сборщик старой версии мог ещё не создать таблицы target/reference.
            pass
        books = self.db.execute(
            """SELECT m.* FROM market_snapshots m JOIN (
                 SELECT outcome,MAX(id) id FROM market_snapshots WHERE event_slug=? GROUP BY outcome
               ) x ON m.id=x.id""", (slug,),
        ).fetchall()
        by_outcome = {str(row["outcome"]): row for row in books}
        up, down = by_outcome.get("Up"), by_outcome.get("Down")
        if not up or not down:
            return None
        book_fields = (
            "collected_at", "event_slug", "market_id", "token_id", "outcome", "best_bid", "best_bid_size",
            "best_ask", "best_ask_size", "midpoint", "spread", "book_timestamp", "book_hash",
        )
        compact_books = {
            "Up": {key: up[key] for key in book_fields},
            "Down": {key: down[key] for key in book_fields},
        }
        fees_enabled = True
        fee_rate = settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE
        fee_exponent = 1.0
        fee_taker_only = True
        minimum_order_size = 0.0
        tick_size = 0.01
        try:
            market_row = self.db.execute(
                "SELECT raw_json FROM markets WHERE market_id=? LIMIT 1", (up["market_id"],),
            ).fetchone()
            market_payload = json.loads(str(market_row[0])) if market_row and market_row[0] else {}
            clob_info = market_payload.get("_clob_info") or {}
            fee_details = clob_info.get("fd") or {}
            schedule = market_payload.get("feeSchedule") or {}
            fees_enabled = bool(market_payload.get("feesEnabled", bool(fee_details or schedule)))
            fee_rate = float(fee_details.get("r", schedule.get("rate", fee_rate)))
            fee_exponent = float(fee_details.get("e", schedule.get("exponent", 1.0)))
            fee_taker_only = bool(fee_details.get("to", schedule.get("takerOnly", True)))
            minimum_order_size = float(clob_info.get("mos") or 0.0)
            tick_size = float(clob_info.get("mts") or 0.01)
        except (sqlite3.OperationalError, TypeError, ValueError, json.JSONDecodeError):
            pass
        if self._history_slug != slug:
            try:
                summaries = summaries_from_connection(self.db)
                self._history_features = event_history_context(
                    summaries, slug, current.isoformat(), settings.EVENT_HISTORY_LIVE_WINDOWS,
                )
            except (sqlite3.OperationalError, ValueError, TypeError):
                self._history_features = {}
            self._history_slug = slug
        return MarketState(
            slug, now(), elapsed, remaining, returns, latest, disagreement, sharp,
            up["best_bid"], up["best_ask"], down["best_bid"], down["best_ask"],
            compact_books, target_price, reference_price, target_source, reference_observed_at, volatility,
            target_distance_lags, dict(self._history_features), fees_enabled, fee_rate, fee_exponent, fee_taker_only,
            minimum_order_size, tick_size,
        )

    def open_position(self, state: MarketState) -> PositionState | None:
        if self._control("trading_mode", "paper") == "live":
            row = self.db.execute(
                "SELECT * FROM live_positions WHERE status='open' AND event_slug=? ORDER BY id DESC LIMIT 1",
                (state.event_slug,),
            ).fetchone()
            if not row:
                return None
            current_bid = state.up_bid if row["outcome"] == "Up" else state.down_bid
            return PositionState(
                row["id"], row["event_slug"], row["outcome"], row["token_id"], row["shares"],
                row["cost_usdc"], row["average_price"], current_bid, int(row["exit_stage"] or 0),
                str(row["opened_at"]), float(row["shares"]), float(row["cost_usdc"]),
                self._exit_runtime_features(row, state, current_bid),
            )
        row = self.db.execute(
            "SELECT * FROM paper_positions WHERE session_id=? AND event_slug=? AND status='open' ORDER BY id DESC LIMIT 1",
            (self.session_id, state.event_slug),
        ).fetchone()
        if not row:
            return None
        current_bid = state.up_bid if row["outcome"] == "Up" else state.down_bid
        buy_totals = self.db.execute(
            """SELECT COALESCE(SUM(shares),0),
                      COALESCE(SUM(shares*filled_price+fee_usdc),0)
               FROM paper_orders WHERE session_id=? AND event_slug=?
                 AND action LIKE 'BUY_%' AND status IN ('filled','partially_filled')""",
            (self.session_id, row["event_slug"]),
        ).fetchone()
        original_shares = float(buy_totals[0] or row["shares"])
        original_cost = float(buy_totals[1] or row["cost_usdc"])
        return PositionState(
            row["id"], row["event_slug"], row["outcome"], row["token_id"], row["shares"],
            row["cost_usdc"], row["average_price"], current_bid, int(row["exit_stage"] or 0),
            str(row["opened_at"]), original_shares, original_cost,
            {**self._exit_runtime_features(row, state, current_bid),
             "shares": original_shares, "original_cost_usdc": original_cost},
        )

    def _exit_runtime_features(
        self, row: sqlite3.Row, state: MarketState, current_bid: float | None,
    ) -> dict[str, float]:
        """Воспроизводит online-признаки exit-sequence без просмотра в будущее."""
        try:
            opened = datetime.fromisoformat(str(row["opened_at"]))
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            opened = datetime.now(UTC)
        observed = datetime.now(UTC)
        has_snapshots = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_snapshots'"
        ).fetchone()
        snapshots = (self.db.execute(
            """SELECT collected_at,COALESCE(best_bid,midpoint),spread,best_bid_size,best_ask_size
               FROM market_snapshots WHERE event_slug=? AND outcome=? AND collected_at>=?
               ORDER BY collected_at DESC,id DESC LIMIT 360""",
            (row["event_slug"], row["outcome"], row["opened_at"]),
        ).fetchall()[::-1] if has_snapshots else [])
        points: list[tuple[float, float]] = []
        for snapshot in snapshots:
            try:
                timestamp = datetime.fromisoformat(str(snapshot[0])).timestamp()
                bid = float(snapshot[1])
            except (TypeError, ValueError):
                continue
            if 0.0 <= bid <= 1.0:
                points.append((timestamp, bid))
        bid = float(current_bid if current_bid is not None else (points[-1][1] if points else 0.0))
        current_ts = observed.timestamp()

        def lag_bid(seconds: int) -> float:
            cutoff = current_ts - seconds
            candidates = [value for timestamp, value in points if timestamp <= cutoff]
            return candidates[-1] if candidates else (points[0][1] if points else bid)

        bids = [value for _, value in points] or [bid]
        times = [timestamp for timestamp, _ in points] or [current_ts]
        peak = max(bids); trough = min(bids)
        peak_index = max(range(len(bids)), key=bids.__getitem__)
        direction = 1.0 if str(row["outcome"]) == "Up" else -1.0
        distance = float(state.distance_to_target_pct or 0.0) * direction
        lag60 = state.target_distance_lags_pct.get("15")
        distance_momentum = distance - direction * float(lag60 if lag60 is not None else state.distance_to_target_pct or 0.0)
        latest_book = state.book_json.get(str(row["outcome"]), {}) or {}
        average = float(row["average_price"] or 0.0)
        return {
            "seconds_in_position": max(0.0, (observed - opened).total_seconds()),
            "remaining_seconds": float(state.remaining_seconds),
            "current_bid": bid,
            "average_price": average,
            "marked_return": bid / average - 1.0 if average > 0 else 0.0,
            "oriented_distance_to_target_pct": distance,
            "momentum_bid_3ticks": bid - bids[max(0, len(bids) - 4)],
            "momentum_distance_3ticks": distance_momentum,
            "momentum_bid_15s": bid - lag_bid(15),
            "momentum_bid_30s": bid - lag_bid(30),
            "momentum_bid_60s": bid - lag_bid(60),
            "bid_slope_15s": (bid - lag_bid(15)) / 15.0,
            "bid_slope_30s": (bid - lag_bid(30)) / 30.0,
            "target_distance_available": float(state.distance_to_target_pct is not None),
            "peak_bid_since_entry": peak,
            "trough_bid_since_entry": trough,
            "drawdown_from_peak": bid - peak,
            "recovery_from_trough": bid - trough,
            "seconds_since_peak": max(0.0, current_ts - times[peak_index]),
            "maximum_favorable_excursion": peak - average,
            "maximum_adverse_excursion": trough - average,
            "spread": float(latest_book.get("spread") or 0.0),
            "log_bid_size": math.log1p(max(0.0, float(latest_book.get("best_bid_size") or 0.0))),
            "log_ask_size": math.log1p(max(0.0, float(latest_book.get("best_ask_size") or 0.0))),
            "shares": float(row["shares"] or 0.0),
            "original_cost_usdc": float(row["cost_usdc"] or 0.0),
        }

    def _decision(self, state: MarketState, position: PositionState | None, decision: Decision, model_key: str) -> int:
        p_up = next((float(tag.split("=", 1)[1]) for tag in decision.tags if tag.startswith("p_up=")), None)
        edge = next((float(tag.split("=", 1)[1]) for tag in decision.tags if tag.startswith("net_edge=")), None)
        cursor = self.db.execute(
            """INSERT INTO model_decisions(session_id,observed_at,event_slug,action,confidence,reason,tags_json,
               market_state_json,position_state_json,provider,model_name,executed,predicted_up_probability,
               predicted_down_probability,expected_net_edge) VALUES(?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)""",
            (self.session_id, now(), state.event_slug, decision.action, decision.confidence, decision.reason,
             json.dumps(decision.tags, ensure_ascii=False), json.dumps(state.as_dict(), ensure_ascii=False, default=str),
             json.dumps(asdict(position), ensure_ascii=False) if position else None,
             settings.PAPER_DECISION_PROVIDER, model_key,
             p_up, 1.0 - p_up if p_up is not None else None, edge),
        )
        self.db.commit()
        return int(cursor.lastrowid)

    def _token(self, state: MarketState, outcome: str) -> str:
        return str(state.book_json[outcome]["token_id"])

    def _cash(self) -> float:
        return float(self.db.execute("SELECT cash_balance_usdc FROM paper_sessions WHERE session_id=?", (self.session_id,)).fetchone()[0])

    def _size_multiplier(self) -> float:
        try:
            return max(settings.TRADE_SIZE_MULTIPLIER_MIN, min(settings.TRADE_SIZE_MULTIPLIER_MAX,
                       float(self._control("trade_size_multiplier", str(settings.TRADE_SIZE_MULTIPLIER_DEFAULT)))))
        except ValueError:
            return settings.TRADE_SIZE_MULTIPLIER_DEFAULT

    def _live_pending(self) -> bool:
        return bool(self.db.execute(
            "SELECT 1 FROM live_orders WHERE status IN ('submitted','live','partial') LIMIT 1"
        ).fetchone())

    @staticmethod
    def _fresh_book(state: MarketState, outcome: str) -> bool:
        try:
            collected_at = datetime.fromisoformat(str(state.book_json[outcome]["collected_at"]))
            return (datetime.now(UTC) - collected_at).total_seconds() <= settings.MAX_EXECUTION_BOOK_AGE_SECONDS
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _entry_grid_plan(total_notional: float, maximum_price: float, tick_size: float,
                         minimum_shares: float) -> list[tuple[float, float]]:
        """Распределяет модельный notional по 3–5 более выгодным лимитам.

        Возвращает ``[(price, notional), ...]``. Если бюджета недостаточно для
        трёх валидных CLOB-заявок, возвращает пустой список и вызывающий код
        использует обычную единственную заявку, не увеличивая риск модели.
        """
        if not settings.ENTRY_GRID_ENABLED or total_notional <= 0 or maximum_price <= 0:
            return []
        tick = max(float(tick_size), 0.0001)
        step_ticks = max(1, round(float(settings.ENTRY_GRID_PRICE_STEP) / tick))
        minimum_count = max(3, int(settings.ENTRY_GRID_MIN_ORDERS))
        maximum_count = max(minimum_count, int(settings.ENTRY_GRID_MAX_ORDERS))
        for count in range(maximum_count, minimum_count - 1, -1):
            prices: list[float] = []
            for level in range(1, count + 1):
                units = round(maximum_price / tick) - step_ticks * level
                price = round(units * tick, 10)
                if price < settings.PAPER_MIN_ENTRY_PRICE or price > settings.PAPER_MAX_ENTRY_PRICE:
                    prices = []
                    break
                prices.append(price)
            if not prices:
                continue
            minimum_notionals = [
                max(float(settings.ENTRY_GRID_MIN_ORDER_USDC), price * minimum_shares)
                for price in prices
            ]
            required = sum(minimum_notionals)
            if required > total_notional + 1e-9:
                continue
            extra = (total_notional - required) / count
            return [(price, minimum_notional + extra)
                    for price, minimum_notional in zip(prices, minimum_notionals, strict=True)]
        return []

    def _live_buy(self, decision_id: int, state: MarketState, decision: Decision) -> None:
        if decision.direction is None or decision.limit_price is None or self._live_pending():
            return
        if not self._fresh_book(state, decision.direction):
            self.db.execute(
                "INSERT INTO live_execution_errors(observed_at,decision_id,event_slug,action,error_type,detail) VALUES(?,?,?,?,?,?)",
                (now(), decision_id, state.event_slug, decision.action, "STALE_ORDER_BOOK", "entry book is missing or stale"),
            )
            self.db.commit()
            return
        validity = validate_entry_state(state, float(decision.limit_price))
        if not validity.valid:
            self.db.execute(
                "INSERT INTO live_execution_errors(observed_at,decision_id,event_slug,action,error_type,detail) VALUES(?,?,?,?,?,?)",
                (now(), decision_id, state.event_slug, decision.action, "INVALID_ENTRY_DOMAIN", validity.reason or "invalid"),
            )
            self.db.commit()
            return
        maximum = settings.MAX_POSITION_USDC
        intended_notional = min(max(decision.notional_usdc * self._size_multiplier(), settings.LIVE_MIN_POSITION_USDC), maximum)
        if intended_notional <= 0:
            return
        token_id = self._token(state, decision.direction)
        try:
            size = max(intended_notional / float(decision.limit_price), market_minimum_size(token_id))
            actual_notional = size * float(decision.limit_price)
            if actual_notional > maximum + 1e-9:
                raise ValueError(f"MARKET_MINIMUM_EXCEEDS_MAX:need={actual_notional:.4f},max={maximum:.4f}")
            response = submit_limit_order(
                token_id=token_id, side="BUY", price=float(decision.limit_price), size=size,
                order_type=settings.ENTRY_ORDER_TYPE, lifetime_seconds=settings.GTD_EFFECTIVE_LIFETIME_SECONDS,
                # Это лимит с price cap: может исполниться сразу, но только <= указанной цены.
                post_only=False, max_notional=maximum, submit=True,
            )
        except Exception as exc:
            self.db.execute(
                "INSERT INTO live_execution_errors(observed_at,decision_id,event_slug,action,error_type,detail) VALUES(?,?,?,?,?,?)",
                (now(), decision_id, state.event_slug, decision.action, type(exc).__name__, str(exc)[:500]),
            )
            self.db.execute("INSERT OR REPLACE INTO runtime_controls VALUES('last_live_error',?,?,?)",
                            (type(exc).__name__, now(), "entry rejected; engine continues"))
            self.db.commit()
            return
        if not response.get("success") or not response.get("order_id"):
            raise RuntimeError(f"LIVE_ORDER_REJECTED:{response.get('error') or 'unknown'}")
        expiration = int(response.get("expiration") or (time.time() + settings.GTD_EFFECTIVE_LIFETIME_SECONDS))
        self.db.execute(
            """INSERT INTO live_orders(decision_id,event_slug,token_id,outcome,side,order_id,order_type,
               requested_price,requested_size,status,created_at,expiration_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, state.event_slug, self._token(state, decision.direction), decision.direction, "BUY",
             response["order_id"], settings.ENTRY_ORDER_TYPE, response.get("submitted_price", decision.limit_price), size,
             "submitted", now(), expiration),
        )
        shadow_bid = state.up_bid if decision.direction == "Up" else state.down_bid
        shadow_ask = state.up_ask if decision.direction == "Up" else state.down_ask
        self.db.execute(
            """INSERT OR IGNORE INTO live_decision_shadow_orders(
               decision_id,live_order_id,event_slug,outcome,side,requested_price,requested_size,
               observed_at,submit_best_bid,submit_best_ask) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, response["order_id"], state.event_slug, decision.direction, "BUY",
             float(response.get("submitted_price", decision.limit_price)), size, now(), shadow_bid, shadow_ask),
        )
        self.db.execute("UPDATE model_decisions SET executed=1 WHERE id=?", (decision_id,))
        self.db.commit()

    def _live_close(self, decision_id: int, state: MarketState, position: PositionState,
                    fraction: float, stage: int) -> None:
        if self._live_pending():
            return
        if not self._fresh_book(state, position.outcome):
            return
        bid = state.up_bid if position.outcome == "Up" else state.down_bid
        if bid is None:
            return
        size = position.shares * max(0.0, min(1.0, fraction))
        try:
            response = submit_limit_order(
                token_id=position.token_id, side="SELL", price=float(bid), size=size,
                order_type="GTD", lifetime_seconds=settings.GTD_EFFECTIVE_LIFETIME_SECONDS,
                post_only=False, submit=True,
            )
        except Exception as exc:
            self.db.execute(
                "INSERT INTO live_execution_errors(observed_at,decision_id,event_slug,action,error_type,detail) VALUES(?,?,?,?,?,?)",
                (now(), decision_id, state.event_slug, "SELL", type(exc).__name__, str(exc)[:500]),
            )
            self.db.commit()
            return
        if not response.get("success") or not response.get("order_id"):
            raise RuntimeError(f"LIVE_EXIT_REJECTED:{response.get('error') or 'unknown'}")
        expiration = int(response.get("expiration") or (time.time() + settings.GTD_EFFECTIVE_LIFETIME_SECONDS))
        self.db.execute(
            """INSERT INTO live_orders(decision_id,event_slug,token_id,outcome,side,order_id,order_type,
               requested_price,requested_size,status,created_at,expiration_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, state.event_slug, position.token_id, position.outcome, "SELL", response["order_id"],
             "GTD", response.get("submitted_price", bid), size, "submitted", now(), expiration),
        )
        self.db.execute(
            """INSERT OR IGNORE INTO live_decision_shadow_orders(
               decision_id,live_order_id,event_slug,outcome,side,requested_price,requested_size,
               observed_at,submit_best_bid,submit_best_ask) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, response["order_id"], state.event_slug, position.outcome, "SELL",
             float(response.get("submitted_price", bid)), size, now(), bid,
             state.up_ask if position.outcome == "Up" else state.down_ask),
        )
        self.db.execute(
            "UPDATE live_positions SET exit_stage=MAX(exit_stage,?),exit_timing='early_exit_submitted' WHERE id=?",
            (stage, position.position_id),
        )
        self.db.execute("UPDATE model_decisions SET executed=1 WHERE id=?", (decision_id,))
        self.db.commit()

    def _reconcile_live_orders(self) -> None:
        rows = self.db.execute(
            "SELECT * FROM live_orders WHERE status IN ('submitted','live','partial') ORDER BY id"
        ).fetchall()
        for row in rows:
            if str(row["side"]).upper() == "BUY":
                current_state = self.market_state()
                validity = (
                    validate_entry_state(current_state, float(row["requested_price"]))
                    if current_state is not None and current_state.event_slug == row["event_slug"]
                    else validate_entry_execution(str(row["event_slug"]), datetime.now(UTC), float(row["requested_price"]))
                )
                if not validity.valid:
                    try:
                        cancel_live_order(str(row["order_id"]))
                    finally:
                        self.db.execute(
                            "UPDATE live_orders SET status='cancelled_domain_guard',execution_valid=0,invalid_reason=?,last_checked_at=? WHERE id=?",
                            (validity.reason, now(), row["id"]),
                        )
                        self.db.commit()
                    continue
            if str(row["side"]).upper() == "SELL" and datetime.now(UTC) >= event_start(str(row["event_slug"])) + timedelta(seconds=300):
                try:
                    cancel_live_order(str(row["order_id"]))
                finally:
                    self.db.execute(
                        "UPDATE live_orders SET status='cancelled_event_ended',last_checked_at=? WHERE id=?",
                        (now(), row["id"]),
                    )
                    self.db.commit()
                continue
            try:
                actual = get_live_order(str(row["order_id"]))
                matched = max(0.0, float(actual["size_matched"]))
                original_size = max(float(actual["original_size"] or 0), float(row["requested_size"] or 0))
                delta = max(0.0, matched - float(row["matched_size"] or 0))
                price = float(actual["price"] or row["requested_price"])
                fill_summary = summarize_order_fills(str(row["order_id"])) if matched > float(row["matched_size"] or 0) else {}
                if float(fill_summary.get("size", 0)) > 0:
                    matched = float(fill_summary["size"])
                    delta = max(0.0, matched - float(row["matched_size"] or 0))
                    price = float(fill_summary["vwap"])
                cumulative_fee = float(fill_summary.get("fee", row["fee_usdc"] or 0))
                fee_delta = max(0.0, cumulative_fee - float(row["fee_usdc"] or 0))
                # Критично: cumulative matched_size сохраняется ДО любых сетевых действий.
                # Поэтому неудачная отмена остатка не сможет повторно начислить тот же fill.
                if delta > 0:
                    self.db.execute(
                        "UPDATE live_orders SET matched_size=?,average_fill_price=?,fill_notional_usdc=?,"
                        "fill_source=?,last_checked_at=?,error=NULL WHERE id=?",
                        (matched, price, float(fill_summary.get("notional", matched * price)),
                         "clob_trade_history" if fill_summary else "clob_order_cumulative_fallback", now(), row["id"]),
                    )
                    self.db.execute("UPDATE live_orders SET fee_usdc=? WHERE id=?",
                                    (float(fill_summary.get("fee", 0)), row["id"]))
                    self.db.commit()
                if delta > 0 and row["side"] == "BUY":
                    position = self.db.execute(
                        "SELECT * FROM live_positions WHERE event_slug=? AND status='open'", (row["event_slug"],)
                    ).fetchone()
                    if position:
                        shares = float(position["shares"]) + delta
                        cost = float(position["cost_usdc"]) + delta * price + fee_delta
                        self.db.execute("UPDATE live_positions SET shares=?,cost_usdc=?,average_price=?,fees_usdc=fees_usdc+? WHERE id=?",
                                        (shares, cost, cost / shares, fee_delta, position["id"]))
                    else:
                        self.db.execute(
                            """INSERT INTO live_positions(event_slug,token_id,outcome,status,opened_at,average_price,
                               shares,cost_usdc,current_price,entry_decision_id) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                            (row["event_slug"], row["token_id"], row["outcome"], "open", now(), price,
                             delta, delta * price + fee_delta, price, row["decision_id"]),
                        )
                        if fee_delta:
                            self.db.execute("UPDATE live_positions SET fees_usdc=? WHERE event_slug=?", (fee_delta, row["event_slug"]))
                    # После первого реального fill остаток входной заявки снимается: открытая позиция
                    # должна иметь возможность немедленно отправить защитный SELL.
                    if matched + 1e-9 < original_size:
                        try:
                            cancel_live_order(str(row["order_id"]))
                        except Exception as cancel_exc:
                            self.db.execute(
                                "UPDATE live_orders SET error=? WHERE id=?",
                                (f"cancel_after_partial:{type(cancel_exc).__name__}", row["id"]),
                            )
                elif delta > 0 and row["side"] == "SELL":
                    position = self.db.execute(
                        "SELECT * FROM live_positions WHERE event_slug=? AND status='open'", (row["event_slug"],)
                    ).fetchone()
                    if position:
                        sold = min(delta, float(position["shares"]))
                        fraction = sold / float(position["shares"])
                        pnl = sold * price - fee_delta - float(position["cost_usdc"]) * fraction
                        remaining = float(position["shares"]) - sold
                        if remaining <= 1e-8:
                            self.db.execute(
                                """UPDATE live_positions SET status='closed',shares=0,current_price=?,closed_at=?,
                                   close_price=?,realized_pnl_usdc=realized_pnl_usdc+?,exit_decision_id=?,
                                   exit_timing='early_full_exit',had_early_exit=1,
                                   early_exit_pnl_usdc=early_exit_pnl_usdc+? WHERE id=?""",
                                (price, now(), price, pnl, row["decision_id"], pnl, position["id"]),
                            )
                            self.db.execute("UPDATE live_positions SET fees_usdc=fees_usdc+? WHERE id=?", (fee_delta, position["id"]))
                        else:
                            self.db.execute(
                                """UPDATE live_positions SET shares=?,cost_usdc=cost_usdc*?,
                                   realized_pnl_usdc=realized_pnl_usdc+?,exit_timing='early_partial_exit',
                                   had_early_exit=1,early_exit_pnl_usdc=early_exit_pnl_usdc+? WHERE id=?""",
                                (remaining, 1.0 - fraction, pnl, pnl, position["id"]),
                            )
                            self.db.execute("UPDATE live_positions SET fees_usdc=fees_usdc+? WHERE id=?", (fee_delta, position["id"]))
                status_text = str(actual["status"]).upper()
                complete = matched + 1e-9 >= original_size
                status = "filled" if complete else ("live" if "LIVE" in status_text else "cancelled")
                if 0 < matched < original_size and status == "live":
                    status = "partial_cancelled" if row["side"] == "BUY" else "partial"
                if int(row["expiration_at"] or 0) and time.time() > int(row["expiration_at"]) and status in {"live", "partial"}:
                    cancel_live_order(str(row["order_id"]))
                    status = "cancelled"
                self.db.execute(
                    "UPDATE live_orders SET matched_size=?,status=?,last_checked_at=? WHERE id=?",
                    (matched, status, now(), row["id"]),
                )
                self.db.execute(
                    """UPDATE live_decision_shadow_orders SET live_fill_price=?,live_fill_size=?,
                       status=CASE WHEN ? > 0 THEN 'live_filled' ELSE status END WHERE live_order_id=?""",
                    (price if matched > 0 else None, matched, matched, row["order_id"]),
                )
            except Exception as exc:
                self.db.execute("UPDATE live_orders SET last_checked_at=?,error=? WHERE id=?",
                                (now(), type(exc).__name__, row["id"]))
        self.db.commit()

    def _settle_live_records(self) -> None:
        """Фиксирует итог в журнале; on-chain Redeem остаётся отдельной операцией Relayer."""
        rows = self.db.execute("SELECT * FROM live_positions WHERE status='open'").fetchall()
        for row in rows:
            label = self.db.execute(
                "SELECT label FROM training_examples WHERE event_slug=? AND token_id=? LIMIT 1",
                (row["event_slug"], row["token_id"]),
            ).fetchone()
            if not label:
                continue
            payout = float(row["shares"]) * int(label[0])
            pnl = payout - float(row["cost_usdc"])
            self.db.execute(
                """UPDATE live_positions SET status='resolved',current_price=?,closed_at=?,close_price=?,
                   realized_pnl_usdc=realized_pnl_usdc+?,
                   exit_timing=CASE WHEN had_early_exit=1 THEN 'partial_early_then_resolution' ELSE 'held_to_resolution' END
                   WHERE id=?""",
                (float(label[0]), now(), float(label[0]), pnl, row["id"]),
            )
        self.db.commit()

    def _reconcile_paper_entry_orders(self, state: MarketState | None) -> None:
        """Исполняет ожидающую GTD-limit заявку по последующим снимкам стакана."""
        rows = self.db.execute(
            """SELECT * FROM paper_orders
               WHERE session_id=? AND action LIKE 'BUY_%' AND status IN ('unfilled','partially_filled') ORDER BY id""",
            (self.session_id,),
        ).fetchall()
        current = datetime.now(UTC)
        for row in rows:
            expiration = datetime.fromisoformat(row["expiration_at"]) if row["expiration_at"] else current
            if state is None or row["event_slug"] != state.event_slug or current >= expiration:
                self.db.execute(
                    "UPDATE paper_orders SET status='expired',last_checked_at=?,check_count=check_count+1 WHERE id=?",
                    (now(), row["id"]),
                )
                continue
            validity = validate_entry_state(state, float(row["requested_price"] or 0))
            if not validity.valid:
                self.db.execute(
                    "UPDATE paper_orders SET status='cancelled_domain_guard',execution_valid=0,invalid_reason=?,last_checked_at=?,check_count=check_count+1 WHERE id=?",
                    (validity.reason, now(), row["id"]),
                )
                continue
            open_position = self.db.execute(
                "SELECT * FROM paper_positions WHERE session_id=? AND status='open' LIMIT 1", (self.session_id,),
            ).fetchone()
            if open_position and (
                str(open_position["event_slug"]) != str(row["event_slug"])
                or int(open_position["entry_decision_id"] or -1) != int(row["decision_id"])
            ):
                self.db.execute(
                    "UPDATE paper_orders SET status='cancelled_position_open',last_checked_at=? WHERE id=?",
                    (now(), row["id"]),
                )
                continue
            direction = "Up" if str(row["action"]).upper().endswith("UP") else "Down"
            already_filled = float(row["shares"] or 0)
            requested_shares = max(0.0, float(row["requested_shares"] or 0) - already_filled)
            if requested_shares <= 1e-9:
                self.db.execute("UPDATE paper_orders SET status='filled',last_checked_at=? WHERE id=?", (now(), row["id"]))
                continue
            requested_price = float(row["requested_price"] or 0)
            book = state.book_json.get(direction, {})
            fill_bid = book.get("best_bid")
            fill_ask = book.get("best_ask")
            try:
                fill_book_age_ms = max(
                    0.0, (datetime.now(UTC) - datetime.fromisoformat(str(book.get("collected_at")))).total_seconds() * 1000
                )
            except (TypeError, ValueError):
                fill_book_age_ms = None
            simulation = limit_buy(
                f"{state.event_slug}:{row['decision_id']}:resting:{row['check_count']}",
                requested_price, book.get("best_ask"), book.get("best_ask_size"),
                requested_shares, book.get("spread"),
            )
            if simulation.filled_shares <= 0:
                self.db.execute(
                    """UPDATE paper_orders SET fill_probability=?,latency_ms=?,execution_reason=?,
                              last_checked_at=?,check_count=check_count+1 WHERE id=?""",
                    (simulation.fill_probability, simulation.latency_ms, simulation.reason, now(), row["id"]),
                )
                continue
            fill = float(simulation.filled_price)
            shares = min(float(simulation.filled_shares), self._cash() / max(fill, 0.01))
            if shares <= 0:
                self.db.execute("UPDATE paper_orders SET status='cancelled_no_cash',last_checked_at=? WHERE id=?", (now(), row["id"]))
                continue
            # Заявка уже находилась в книге, поэтому при последующем касании
            # исполняется как maker. По правилам Polymarket maker fee равна нулю.
            fee = state_fee_usdc(state, shares, fill, taker=False)
            notional = shares * fill
            total_cost = notional + fee
            if open_position:
                new_shares = float(open_position["shares"]) + shares
                new_cost = float(open_position["cost_usdc"]) + total_cost
                self.db.execute(
                    "UPDATE paper_positions SET shares=?,cost_usdc=?,average_price=?,current_price=?,fees_usdc=fees_usdc+? WHERE id=?",
                    (new_shares, new_cost, new_cost / new_shares, fill, fee, open_position["id"]),
                )
            else:
                self.db.execute(
                    """INSERT INTO paper_positions(session_id,event_slug,token_id,outcome,status,opened_at,
                       average_price,shares,cost_usdc,current_price,entry_decision_id,fees_usdc)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self.session_id, state.event_slug, self._token(state, direction), direction, "open",
                     now(), fill, shares, total_cost, fill, row["decision_id"], fee),
                )
            cumulative_shares = already_filled + shares
            cumulative_notional = float(row["notional_usdc"] or 0) + notional
            cumulative_fee = float(row["fee_usdc"] or 0) + fee
            final_status = "filled" if cumulative_shares + 1e-9 >= float(row["requested_shares"] or 0) else "partially_filled"
            self.db.execute(
                """UPDATE paper_orders SET filled_price=?,shares=?,notional_usdc=?,fee_usdc=?,status=?,
                          fill_probability=?,latency_ms=?,execution_reason='gtd_resting_maker_fill',
                          last_checked_at=?,check_count=check_count+1,fill_best_bid=?,fill_best_ask=?,
                          fill_observed_at=?,observed_slippage_bps=?,book_age_ms_at_fill=?,
                          fill_probability_kind=? WHERE id=?""",
                (fill, cumulative_shares, cumulative_notional, cumulative_fee, final_status, simulation.fill_probability,
                 simulation.latency_ms, now(), fill_bid, fill_ask, now(),
                 max(0.0, (fill - float(row["submit_best_ask"])) / float(row["submit_best_ask"]) * 10_000)
                 if row["submit_best_ask"] else 0.0,
                 fill_book_age_ms, settings.EXECUTION_FILL_PROBABILITY_KIND, row["id"]),
            )
            self.db.execute(
                """UPDATE paper_sessions SET cash_balance_usdc=cash_balance_usdc-?,
                          total_wagered_usdc=total_wagered_usdc+?,total_fees_usdc=total_fees_usdc+?
                   WHERE session_id=?""",
                (total_cost, notional, fee, self.session_id),
            )
            self.db.execute("UPDATE model_decisions SET executed=1 WHERE id=?", (row["decision_id"],))
        self.db.commit()

    def _buy(self, decision_id: int, state: MarketState, decision: Decision, add: bool = False) -> None:
        if self._control("trading_mode", "paper") == "live":
            self._live_buy(decision_id, state, decision)
            return
        if decision.direction is None or decision.limit_price is None or decision.notional_usdc <= 0:
            return
        if not self._fresh_book(state, decision.direction):
            return
        validity = validate_entry_state(state, float(decision.limit_price))
        if not validity.valid:
            self.db.execute(
                "UPDATE model_decisions SET reason=reason||?,tags_json=? WHERE id=?",
                (f"; исполнение отклонено: {validity.reason}", json.dumps(["execution_domain_guard", validity.reason]), decision_id),
            )
            self.db.commit()
            return
        cash = self._cash()
        maximum = min(settings.PAPER_MAX_EVENT_EXPOSURE_USDC, settings.MAX_POSITION_USDC)
        notional = min(decision.notional_usdc * self._size_multiplier(), cash, maximum)
        if notional <= 0:
            return
        requested_shares = notional / decision.limit_price
        if requested_shares + 1e-9 < state.minimum_order_size:
            self.db.execute(
                "UPDATE model_decisions SET reason=reason||? WHERE id=?",
                (f"; отклонено: размер {requested_shares:.4f} < CLOB minimum {state.minimum_order_size:.4f}", decision_id),
            )
            self.db.commit()
            return
        tick_units = round(float(decision.limit_price) / max(state.tick_size, 0.0001))
        if abs(tick_units * state.tick_size - float(decision.limit_price)) > 1e-8:
            self.db.execute(
                "UPDATE model_decisions SET reason=reason||? WHERE id=?",
                (f"; отклонено: цена не кратна CLOB tick {state.tick_size}", decision_id),
            )
            self.db.commit()
            return
        book = state.book_json.get(decision.direction, {})
        submit_bid = book.get("best_bid")
        submit_ask = book.get("best_ask")
        submit_book_timestamp = book.get("book_timestamp")
        try:
            submit_book_age_ms = max(
                0.0, (datetime.now(UTC) - datetime.fromisoformat(str(book.get("collected_at")))).total_seconds() * 1000
            )
        except (TypeError, ValueError):
            submit_book_age_ms = None
        grid = self._entry_grid_plan(
            notional, float(decision.limit_price), float(state.tick_size), float(state.minimum_order_size),
        ) if not add else []
        if grid:
            expiration = (datetime.now(UTC) + timedelta(seconds=settings.GTD_EFFECTIVE_LIFETIME_SECONDS)).isoformat()
            self.db.execute(
                """UPDATE paper_orders SET status='cancelled_replaced',last_checked_at=?
                   WHERE session_id=? AND event_slug=? AND action LIKE 'BUY_%'
                     AND status IN ('unfilled','partially_filled')""",
                (now(), self.session_id, state.event_slug),
            )
            for level, (price, level_notional) in enumerate(grid, start=1):
                requested = level_notional / price
                simulation = limit_buy(
                    f"{state.event_slug}:{decision_id}:grid:{level}", price,
                    book.get("best_ask"), book.get("best_ask_size"), requested, book.get("spread"),
                )
                self.db.execute(
                    """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,
                       requested_price,filled_price,shares,notional_usdc,fee_usdc,slippage_bps,status,
                       created_at,expiration_at,price_cap,fill_probability,latency_ms,execution_reason,
                       requested_shares,requested_notional_usdc,last_checked_at,check_count,submit_best_bid,
                       submit_best_ask,submit_book_timestamp,book_age_ms_at_submit,fill_probability_kind)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self.session_id, decision_id, state.event_slug, decision.action, settings.ENTRY_ORDER_TYPE,
                     price, None, 0.0, 0.0, 0.0, 0.0, "unfilled", now(), expiration, price,
                     simulation.fill_probability, simulation.latency_ms, f"entry_grid_l{level}:{simulation.reason}",
                     requested, level_notional, now(), 1, submit_bid, submit_ask, submit_book_timestamp,
                     submit_book_age_ms, settings.EXECUTION_FILL_PROBABILITY_KIND),
                )
            levels = ",".join(f"{price:.3f}" for price, _ in grid)
            self.db.execute(
                "UPDATE model_decisions SET reason=reason||?,tags_json=? WHERE id=?",
                (f"; лимитная сетка {len(grid)} уровней: {levels}",
                 json.dumps([*decision.tags, "entry_limit_grid", f"grid_levels={len(grid)}"], ensure_ascii=False),
                 decision_id),
            )
            self.db.commit()
            return
        simulation = limit_buy(
            f"{state.event_slug}:{decision_id}:buy", decision.limit_price,
            book.get("best_ask"), book.get("best_ask_size"), requested_shares, book.get("spread"),
        ) if settings.EXECUTION_SIMULATION_ENABLED else None
        if simulation is not None and simulation.filled_shares <= 0:
            expiration = (datetime.now(UTC) + timedelta(seconds=settings.GTD_EFFECTIVE_LIFETIME_SECONDS)).isoformat()
            # На один контракт держим одну ожидающую GTD-заявку: новая сильная
            # оценка заменяет старую, как это делал бы реальный исполнитель.
            self.db.execute(
                """UPDATE paper_orders SET status='cancelled_replaced',last_checked_at=?
                   WHERE session_id=? AND event_slug=? AND action LIKE 'BUY_%' AND status='unfilled'""",
                (now(), self.session_id, state.event_slug),
            )
            self.db.execute(
                """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,requested_price,
                   filled_price,shares,notional_usdc,fee_usdc,slippage_bps,status,created_at,expiration_at,
                   price_cap,fill_probability,latency_ms,execution_reason,requested_shares,
                   requested_notional_usdc,last_checked_at,check_count,submit_best_bid,submit_best_ask,
                   submit_book_timestamp,book_age_ms_at_submit,fill_probability_kind)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.session_id, decision_id, state.event_slug, decision.action, settings.ENTRY_ORDER_TYPE,
                 decision.limit_price, None, 0.0, 0.0, 0.0, 0.0, simulation.status, now(), expiration,
                 decision.limit_price, simulation.fill_probability, simulation.latency_ms, simulation.reason,
                 requested_shares, notional, now(), 1, submit_bid, submit_ask, submit_book_timestamp,
                 submit_book_age_ms, settings.EXECUTION_FILL_PROBABILITY_KIND),
            )
            self.db.commit()
            return
        fill = float(simulation.filled_price) if simulation else min(0.99, decision.limit_price)
        shares = float(simulation.filled_shares) if simulation else requested_shares
        notional = shares * fill
        fee = state_fee_usdc(state, shares, fill)
        total_cost = notional + fee
        if total_cost > cash:
            notional = cash / (1.0 + settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE * (1.0 - fill))
            shares = notional / fill
            fee = state_fee_usdc(state, shares, fill)
            total_cost = notional + fee
        if add:
            row = self.db.execute(
                "SELECT * FROM paper_positions WHERE session_id=? AND event_slug=? AND status='open'",
                (self.session_id, state.event_slug),
            ).fetchone()
            if not row or row["outcome"] != decision.direction:
                return
            new_cost, new_shares = row["cost_usdc"] + total_cost, row["shares"] + shares
            self.db.execute(
                "UPDATE paper_positions SET cost_usdc=?,shares=?,average_price=?,current_price=?,fees_usdc=fees_usdc+? WHERE id=?",
                (new_cost, new_shares, new_cost / new_shares, fill, fee, row["id"]),
            )
        else:
            self.db.execute(
                """INSERT INTO paper_positions(session_id,event_slug,token_id,outcome,status,opened_at,average_price,
                   shares,cost_usdc,current_price,entry_decision_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (self.session_id, state.event_slug, self._token(state, decision.direction), decision.direction, "open",
                 now(), fill, shares, total_cost, fill, decision_id),
            )
            self.db.execute("UPDATE paper_positions SET fees_usdc=? WHERE id=last_insert_rowid()", (fee,))
        expiration = (datetime.now(UTC) + timedelta(seconds=settings.GTD_EFFECTIVE_LIFETIME_SECONDS)).isoformat()
        self.db.execute(
            """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,requested_price,filled_price,
               shares,notional_usdc,fee_usdc,slippage_bps,status,created_at,expiration_at,price_cap,
               fill_probability,latency_ms,execution_reason,requested_shares,requested_notional_usdc,
               submit_best_bid,submit_best_ask,submit_book_timestamp,fill_best_bid,fill_best_ask,fill_observed_at,
               observed_slippage_bps,book_age_ms_at_submit,book_age_ms_at_fill,fill_probability_kind)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session_id, decision_id, state.event_slug, decision.action, settings.ENTRY_ORDER_TYPE,
             decision.limit_price, fill, shares, notional, fee, 0.0,
             simulation.status if simulation else "filled", now(), expiration, decision.limit_price,
             simulation.fill_probability if simulation else 1.0, simulation.latency_ms if simulation else 0,
             simulation.reason if simulation else "legacy_exact_limit", requested_shares,
             decision.notional_usdc * self._size_multiplier(), submit_bid, submit_ask, submit_book_timestamp,
             submit_bid, submit_ask, now(),
             max(0.0, (fill - float(submit_ask)) / float(submit_ask) * 10_000) if submit_ask else 0.0,
             submit_book_age_ms, submit_book_age_ms, settings.EXECUTION_FILL_PROBABILITY_KIND),
        )
        self.db.execute(
            "UPDATE paper_sessions SET cash_balance_usdc=cash_balance_usdc-?,total_wagered_usdc=total_wagered_usdc+?,total_fees_usdc=total_fees_usdc+? WHERE session_id=?",
            (total_cost, notional, fee, self.session_id),
        )
        self.db.execute("UPDATE model_decisions SET executed=1 WHERE id=?", (decision_id,))
        self.db.commit()

    def _close(self, decision_id: int, state: MarketState, position: PositionState, reason: str, fraction: float = 1.0, stage: int = 0, limit_exit: bool = False) -> None:
        if self._control("trading_mode", "paper") == "live":
            self._live_close(decision_id, state, position, fraction, stage)
            return
        bid = state.up_bid if position.outcome == "Up" else state.down_bid
        if bid is None:
            return
        if not self._fresh_book(state, position.outcome):
            return
        fraction = max(0.0, min(1.0, fraction))
        requested_shares = position.shares * fraction
        book = state.book_json.get(position.outcome, {})
        submit_bid = book.get("best_bid")
        submit_ask = book.get("best_ask")
        submit_book_timestamp = book.get("book_timestamp")
        try:
            submit_book_age_ms = max(
                0.0, (datetime.now(UTC) - datetime.fromisoformat(str(book.get("collected_at")))).total_seconds() * 1000
            )
        except (TypeError, ValueError):
            submit_book_age_ms = None
        cap = bid if limit_exit else max(0.01, bid * (1 - settings.FAK_PRICE_CAP_SLIPPAGE_BPS / 10_000))
        simulation = (
            fak_sell(f"{state.event_slug}:{decision_id}:sell", bid, book.get("best_bid_size"), requested_shares, cap)
            if settings.EXECUTION_SIMULATION_ENABLED else None
        )
        if simulation is not None and simulation.filled_shares <= 0:
            self.db.execute(
                """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,requested_price,
                   filled_price,shares,notional_usdc,fee_usdc,slippage_bps,status,created_at,price_cap,
                   fill_probability,latency_ms,execution_reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.session_id, decision_id, state.event_slug, "CLOSE", settings.EXIT_ORDER_TYPE, bid, None,
                 0.0, 0.0, 0.0, 0.0, simulation.status, now(), cap, simulation.fill_probability,
                 simulation.latency_ms, simulation.reason),
            )
            self.db.commit()
            return
        fill = float(simulation.filled_price) if simulation else max(0.01, bid if limit_exit else bid * (1 - settings.ESTIMATED_SLIPPAGE_BPS / 10_000))
        sold_shares = float(simulation.filled_shares) if simulation else requested_shares
        fraction = sold_shares / position.shares
        allocated_cost = position.cost_usdc * fraction
        gross_proceeds = sold_shares * fill
        fee = state_fee_usdc(state, sold_shares, fill)
        proceeds, pnl = gross_proceeds - fee, gross_proceeds - allocated_cost - fee
        if fraction < 1.0:
            self.db.execute(
                """UPDATE paper_positions SET shares=shares-?,cost_usdc=cost_usdc-?,current_price=?,fees_usdc=fees_usdc+?,
                   realized_pnl_usdc=COALESCE(realized_pnl_usdc,0)+?,gross_pnl_usdc=COALESCE(gross_pnl_usdc,0)+?,
                   exit_stage=MAX(exit_stage,?),exit_timing='early_partial_exit',had_early_exit=1,
                   early_exit_pnl_usdc=early_exit_pnl_usdc+? WHERE id=?""",
                (sold_shares, allocated_cost, fill, fee, pnl, gross_proceeds - allocated_cost, stage, pnl, position.position_id),
            )
        else:
            self.db.execute(
                """UPDATE paper_positions SET status='closed',current_price=?,closed_at=?,close_price=?,
                   realized_pnl_usdc=COALESCE(realized_pnl_usdc,0)+?,gross_pnl_usdc=COALESCE(gross_pnl_usdc,0)+?,
                   fees_usdc=fees_usdc+?,close_reason=?,exit_decision_id=?,exit_stage=MAX(exit_stage,?),
                   exit_timing='early_full_exit',had_early_exit=1,
                   early_exit_pnl_usdc=early_exit_pnl_usdc+? WHERE id=?""",
                (fill, now(), fill, pnl, gross_proceeds - allocated_cost, fee, reason, decision_id, stage, pnl, position.position_id),
            )
        self.db.execute(
            "UPDATE paper_sessions SET cash_balance_usdc=cash_balance_usdc+?,realized_pnl_usdc=realized_pnl_usdc+?,total_fees_usdc=total_fees_usdc+? WHERE session_id=?",
            (proceeds, pnl, fee, self.session_id),
        )
        self.db.execute(
            """INSERT INTO paper_orders(session_id,decision_id,event_slug,action,order_type,requested_price,filled_price,
               shares,notional_usdc,fee_usdc,slippage_bps,status,created_at,price_cap,fill_probability,latency_ms,
               execution_reason,submit_best_bid,submit_best_ask,submit_book_timestamp,fill_best_bid,fill_best_ask,
               fill_observed_at,observed_slippage_bps,book_age_ms_at_submit,book_age_ms_at_fill,
               fill_probability_kind) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session_id, decision_id, state.event_slug, "PARTIAL_CLOSE" if fraction < 1 else "CLOSE",
             settings.TAKE_PROFIT_ORDER_TYPE if limit_exit else settings.EXIT_ORDER_TYPE, bid, fill, sold_shares,
             gross_proceeds, fee, simulation.slippage_bps if simulation else (0.0 if limit_exit else settings.ESTIMATED_SLIPPAGE_BPS),
             simulation.status if simulation else "filled", now(), cap,
             simulation.fill_probability if simulation else 1.0, simulation.latency_ms if simulation else 0,
             simulation.reason if simulation else "legacy_exit", submit_bid, submit_ask, submit_book_timestamp,
             submit_bid, submit_ask, now(),
             max(0.0, (float(submit_bid) - fill) / float(submit_bid) * 10_000) if submit_bid else 0.0,
             submit_book_age_ms, submit_book_age_ms, settings.EXECUTION_FILL_PROBABILITY_KIND),
        )
        self.db.execute("UPDATE model_decisions SET executed=1 WHERE id=?", (decision_id,))
        self.db.commit()

    def settle_resolved(self) -> None:
        self._refresh_resolution_ledger()
        positions = self.db.execute(
            "SELECT * FROM paper_positions WHERE status IN ('open','provisionally_resolved')"
        ).fetchall()
        for row in positions:
            label = self.db.execute(
                "SELECT label,resolved_at FROM event_resolutions WHERE event_slug=? AND token_id=? LIMIT 1",
                (row["event_slug"], row["token_id"]),
            ).fetchone()
            if not label:
                continue
            official_label = int(label[0])
            payout = float(row["shares"]) * official_label
            if str(row["status"]) == "provisionally_resolved":
                provisional_label = int(row["provisional_label"] or 0)
                credited_payout = float(row["shares"]) * provisional_label
                adjustment = payout - credited_payout
                self.db.execute(
                    """UPDATE paper_positions SET status='resolved',current_price=?,close_price=?,
                       realized_pnl_usdc=COALESCE(realized_pnl_usdc,0)+?,
                       close_reason=CASE WHEN had_early_exit=1
                         THEN COALESCE(NULLIF(close_reason,'' ) || '; ','') || 'remaining_shares_market_resolution_after_provisional'
                         ELSE 'market_resolution_after_provisional' END,
                       exit_timing=CASE WHEN had_early_exit=1 THEN 'partial_early_then_resolution' ELSE 'held_to_resolution' END,
                       official_label=?,official_reconciled_at=?,provisional_mismatch=?,
                       resolution_labeled_at=?,settlement_latency_seconds=MAX(0,unixepoch(?) -
                         (CAST(SUBSTR(event_slug,INSTR(event_slug,'5m-')+3) AS INTEGER)+300))
                       WHERE id=?""",
                    (float(official_label), float(official_label), adjustment, official_label, now(),
                     int(provisional_label != official_label), label[1], label[1], row["id"]),
                )
                self.db.execute(
                    """UPDATE paper_sessions SET cash_balance_usdc=cash_balance_usdc+?,
                       realized_pnl_usdc=realized_pnl_usdc+? WHERE session_id=?""",
                    (adjustment, adjustment, row["session_id"]),
                )
            else:
                pnl = payout - float(row["cost_usdc"])
                self.db.execute(
                    """UPDATE paper_positions SET status='resolved',current_price=?,closed_at=?,close_price=?,
                       realized_pnl_usdc=COALESCE(realized_pnl_usdc,0)+?,
                       close_reason=CASE WHEN had_early_exit=1
                         THEN COALESCE(NULLIF(close_reason,'' ) || '; ','') || 'remaining_shares_market_resolution'
                         ELSE 'market_resolution' END,
                       exit_timing=CASE WHEN had_early_exit=1 THEN 'partial_early_then_resolution' ELSE 'held_to_resolution' END,
                       official_label=?,official_reconciled_at=?,resolution_labeled_at=?,
                       settlement_latency_seconds=MAX(0,unixepoch(?) -
                         (CAST(SUBSTR(event_slug,INSTR(event_slug,'5m-')+3) AS INTEGER)+300))
                       WHERE id=?""",
                    (float(official_label), now(), float(official_label), pnl, official_label, now(),
                     label[1], label[1], row["id"]),
                )
                self.db.execute(
                    "UPDATE paper_sessions SET cash_balance_usdc=cash_balance_usdc+?,realized_pnl_usdc=realized_pnl_usdc+? WHERE session_id=?",
                    (payout, pnl, row["session_id"]),
                )
        pending_cf = self.db.execute(
            """SELECT c.*,MAX(t.label) AS joined_label
               FROM counterfactual_entries c
               JOIN training_examples t ON t.event_slug=c.event_slug AND t.outcome=c.outcome
               WHERE c.status='pending' GROUP BY c.id"""
        ).fetchall()
        for row in pending_cf:
            resolved_label = int(row["joined_label"])
            shares = row["hypothetical_notional_usdc"] / row["entry_price"]
            entry_fee = total_fee_usdc(shares, row["entry_price"])
            pnl = shares * resolved_label - row["hypothetical_notional_usdc"] - entry_fee
            self.db.execute(
                """UPDATE counterfactual_entries SET evaluated_at=?,exit_price=?,fee_usdc=?,counterfactual_pnl_usdc=?,
                   resolved_label=?,status='resolved' WHERE id=?""",
                (now(), float(resolved_label), entry_fee, pnl, resolved_label, row["id"]),
            )
        self._evaluate_next_event_forecasts()
        decisions = self.db.execute(
            """SELECT DISTINCT d.id,d.event_slug,d.action FROM model_decisions d
               JOIN training_examples t ON t.event_slug=d.event_slug
               LEFT JOIN strategy_labels l ON l.decision_id=d.id
               WHERE l.decision_id IS NULL"""
        ).fetchall()
        for row in decisions:
            outcomes = dict(self.db.execute(
                "SELECT outcome,label FROM training_examples WHERE event_slug=? GROUP BY outcome,label", (row["event_slug"],)
            ).fetchall())
            if not outcomes:
                continue
            chosen = "Up" if row["action"] in {"BUY_UP"} else "Down" if row["action"] in {"BUY_DOWN"} else None
            correct = int(outcomes.get(chosen, 0)) if chosen else None
            pnl_row = self.db.execute(
                "SELECT realized_pnl_usdc FROM paper_positions WHERE entry_decision_id=?", (row["id"],)
            ).fetchone()
            self.db.execute(
                """INSERT OR REPLACE INTO strategy_labels(decision_id,event_slug,resolved_label,decision_was_correct,
                   realized_pnl_usdc,resolution_source,labeled_at) VALUES(?,?,?,?,?,'training_examples',?)""",
                (row["id"], row["event_slug"], int(outcomes.get("Up", 0)), correct,
                 float(pnl_row[0]) if pnl_row and pnl_row[0] is not None else None, now()),
            )
        self._enforce_loss_streak()
        self.db.commit()

    def _provisionally_settle_ended_paper_positions(self) -> None:
        """Освобождает PAPER-позицию через T+10s только по экстремальной финальной котировке.

        Это оперативная бухгалтерская оценка, а не официальный label. Поздний
        официальный resolution всегда сверяет и при необходимости корректирует её.
        """
        if not settings.PAPER_POST_EVENT_SETTLEMENT_ENABLED:
            return
        current = datetime.now(UTC)
        rows = self.db.execute(
            "SELECT * FROM paper_positions WHERE status='open' ORDER BY id"
        ).fetchall()
        for row in rows:
            try:
                event_end = datetime.fromtimestamp(
                    int(str(row["event_slug"]).rsplit("-", 1)[-1]) + 300, UTC,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if (current - event_end).total_seconds() < settings.PAPER_POST_EVENT_SETTLEMENT_DELAY_SECONDS:
                continue
            cutoff = (event_end - timedelta(
                seconds=float(settings.PAPER_POST_EVENT_MAX_QUOTE_DISTANCE_SECONDS)
            )).isoformat()
            quotes = self.db.execute(
                """SELECT outcome,best_bid,best_ask,collected_at FROM market_snapshots
                   WHERE event_slug=? AND collected_at>=? ORDER BY collected_at DESC,id DESC""",
                (row["event_slug"], cutoff),
            ).fetchall()
            latest: dict[str, sqlite3.Row] = {}
            for quote in quotes:
                latest.setdefault(str(quote["outcome"]), quote)
            held = latest.get(str(row["outcome"]))
            other = latest.get("Down" if str(row["outcome"]) == "Up" else "Up")
            if held is None:
                continue
            held_bid = float(held["best_bid"] or 0.0)
            held_ask = float(held["best_ask"] or 1.0)
            other_bid = float(other["best_bid"] or 0.0) if other is not None else 0.0
            if held_bid >= settings.PAPER_POST_EVENT_WIN_BID_THRESHOLD:
                provisional_label = 1
            elif held_ask <= settings.PAPER_POST_EVENT_LOSS_ASK_THRESHOLD or other_bid >= settings.PAPER_POST_EVENT_WIN_BID_THRESHOLD:
                provisional_label = 0
            else:
                continue
            payout = float(row["shares"]) * provisional_label
            pnl = payout - float(row["cost_usdc"])
            self.db.execute(
                """UPDATE paper_positions SET status='provisionally_resolved',current_price=?,closed_at=?,
                   close_price=?,realized_pnl_usdc=COALESCE(realized_pnl_usdc,0)+?,
                   close_reason='post_event_quote_heuristic',
                   exit_timing=CASE WHEN had_early_exit=1 THEN 'partial_early_then_provisional'
                                    ELSE 'held_to_provisional_resolution' END,
                   provisional_resolution=1,provisional_label=?,provisional_resolved_at=?,
                   provisional_source='post_event_extreme_quote_t_plus_10s' WHERE id=?""",
                (float(provisional_label), now(), float(provisional_label), pnl,
                 provisional_label, now(), row["id"]),
            )
            self.db.execute(
                """UPDATE paper_sessions SET cash_balance_usdc=cash_balance_usdc+?,
                   realized_pnl_usdc=realized_pnl_usdc+? WHERE session_id=?""",
                (payout, pnl, row["session_id"]),
            )
        self.db.commit()

    def _record_next_event_forecast(self, state: MarketState) -> None:
        """Записывает shadow-прогноз следующей секции; реальную заявку не отправляет."""
        if not settings.NEXT_EVENT_CONTEXT_ENABLED:
            return
        # Stale market_state иногда живёт несколько секунд после закрытия окна.
        # Такой прогноз уже относится к начавшемуся, а не pre-open событию.
        if state.remaining_seconds <= 0 or state.elapsed_seconds < 0:
            return
        source_start = int(state.event_slug.rsplit("-", 1)[-1])
        next_start = source_start + 300
        if time.time() >= next_start:
            return
        sample_seconds = max(1.0, float(settings.NEXT_EVENT_FORECAST_SAMPLE_SECONDS))
        if time.monotonic() - self.last_next_forecast_at < sample_seconds:
            return
        self.last_next_forecast_at = time.monotonic()
        next_slug = f"{settings.COLLECTOR_BTC_5M_SLUG_PREFIX}-{source_start + 300}"
        momentum = math.tanh(state.consensus_return_pct / 0.12)
        target_signal = math.tanh(float(state.distance_to_target_pct or 0.0) / 0.06)
        # Базовый temporal predictor нужен для начала честной разметки. Он не
        # считается обученной production-моделью и работает только в shadow.
        predicted_up = min(0.95, max(0.05, 0.5 + 0.20 * momentum + 0.08 * target_signal))
        direction = "Up" if predicted_up >= 0.5 else "Down"
        confidence = max(predicted_up, 1.0 - predicted_up)
        preview = None
        if "future_event_snapshots" in {
            str(row[0]) for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            preview = self.db.execute(
                """SELECT token_id,best_bid,best_ask FROM future_event_snapshots
                   WHERE next_event_slug=? AND outcome=? AND seconds_before_start>=0
                   ORDER BY id DESC LIMIT 1""",
                (next_slug, direction),
            ).fetchone()
        token_id = str(preview[0]) if preview else None
        ask = float(preview[2]) if preview and preview[2] is not None else None
        bid = float(preview[1]) if preview and preview[1] is not None else None
        # Пассивный maker-like лимит ставится выгоднее текущего ask. Bid является
        # первой ценой в очереди; модель вправе заменить заявку новым прогнозом.
        reference_price = bid if bid is not None else ask
        planned_price = (
            min(float(settings.PAPER_MAX_ENTRY_PRICE), float(settings.NEXT_EVENT_LIMIT_PRICE_CAP),
                max(float(settings.PAPER_MIN_ENTRY_PRICE), float(settings.NEXT_EVENT_LIMIT_PRICE_FLOOR), reference_price))
            if reference_price is not None else None
        )
        selected_probability = predicted_up if direction == "Up" else 1.0 - predicted_up
        if planned_price is not None:
            probe_notional = float(settings.PAPER_ENTRY_NOTIONAL_USDC)
            probe_shares = probe_notional / planned_price
            probe_ev = probe_shares * selected_probability - probe_notional - total_fee_usdc(
                probe_shares, planned_price, taker=False,
            )
            planned_notional = position_notional(
                net_buy_edge(selected_probability, planned_price), probe_ev,
                win_probability=selected_probability, entry_price=planned_price,
                fill_probability=0.5,
            )
            if planned_notional > 0:
                minimum_notional = max(
                    float(settings.POSITION_SIZE_MIN_USDC),
                    float(settings.DEFAULT_CLOB_MIN_ORDER_SIZE_SHARES) * planned_price,
                )
                planned_notional = min(
                    max(float(planned_notional), minimum_notional),
                    float(settings.PAPER_MAX_EVENT_EXPOSURE_USDC),
                    float(settings.MAX_POSITION_USDC),
                )
        else:
            planned_notional = 0.0
        self.db.execute(
            """INSERT OR IGNORE INTO next_event_forecasts(
                 source_event_slug,next_event_slug,observed_at,sample_bucket,seconds_to_source_end,
                 current_target_distance_pct,current_consensus_return_pct,current_realized_volatility,
                 history_features_json,predictor_version,predicted_next_up_probability,
                 predicted_direction,confidence,planned_token_id,planned_limit_price,
                 planned_notional_usdc,planned_order_type,preopen_book_available)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (state.event_slug, next_slug, now(), int(time.time() // sample_seconds), state.remaining_seconds,
             state.distance_to_target_pct, state.consensus_return_pct, state.realized_volatility_60s_pct,
             json.dumps(state.history_features, ensure_ascii=False), "temporal_baseline_v1_shadow",
             predicted_up, direction, confidence, token_id, planned_price,
             planned_notional, "GTD_SHADOW", int(preview is not None)),
        )
        forecast = self.db.execute(
            """SELECT id FROM next_event_forecasts WHERE source_event_slug=? AND next_event_slug=?
               AND sample_bucket=?""",
            (state.event_slug, next_slug, int(time.time() // sample_seconds)),
        ).fetchone()
        if (
            settings.NEXT_EVENT_PREOPEN_SHADOW_ENABLED and forecast is not None
            and token_id and planned_price is not None and planned_notional > 0 and preview is not None
        ):
            self._upsert_preopen_shadow_order(
                int(forecast[0]), state.event_slug, next_slug, direction, token_id,
                float(planned_price), float(planned_notional), confidence,
            )
        self.db.commit()

    def _upsert_preopen_shadow_order(
        self, forecast_id: int, source_slug: str, next_slug: str, outcome: str,
        token_id: str, limit_price: float, notional: float, confidence: float,
    ) -> None:
        active = self.db.execute(
            """SELECT * FROM shadow_preopen_orders WHERE next_event_slug=? AND status='working'
               ORDER BY id DESC LIMIT 1""", (next_slug,),
        ).fetchone()
        replace = active is not None and (
            str(active["outcome"]) != outcome
            or abs(float(active["limit_price"]) - limit_price) >= .01 * settings.NEXT_EVENT_ORDER_REPLACE_TICKS
        )
        if active is not None and not replace:
            self.db.execute(
                """UPDATE shadow_preopen_orders SET forecast_id=?,confidence=?,notional_usdc=?,
                   updated_at=? WHERE id=?""",
                (forecast_id, confidence, notional, now(), active["id"]),
            )
            return
        if active is not None:
            self.db.execute(
                """UPDATE shadow_preopen_orders SET status='cancelled_replaced',cancelled_at=?,
                   cancel_reason='model_repriced_or_changed_direction',updated_at=? WHERE id=?""",
                (now(), now(), active["id"]),
            )
        start = event_start(next_slug)
        self.db.execute(
            """INSERT INTO shadow_preopen_orders(forecast_id,source_event_slug,next_event_slug,
               outcome,token_id,limit_price,notional_usdc,confidence,model_name,requested_at,
               updated_at,expires_at,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'working')""",
            (forecast_id, source_slug, next_slug, outcome, token_id, limit_price, notional,
             confidence, "temporal_baseline_v1_shadow", now(), now(), start.isoformat()),
        )

    def _update_preopen_shadow_orders(self) -> None:
        """Исполняет только shadow-лимиты до старта и архивирует неисполненные."""
        if not self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='future_event_snapshots'"
        ).fetchone():
            return
        current = datetime.now(UTC)
        rows = self.db.execute(
            "SELECT * FROM shadow_preopen_orders WHERE status='working' ORDER BY id"
        ).fetchall()
        for row in rows:
            start = event_start(str(row["next_event_slug"]))
            fill = self.db.execute(
                """SELECT collected_at,best_ask FROM future_event_snapshots
                   WHERE next_event_slug=? AND outcome=? AND collected_at>=?
                     AND collected_at<? AND best_ask IS NOT NULL AND best_ask<=?
                   ORDER BY collected_at LIMIT 1""",
                (row["next_event_slug"], row["outcome"], row["requested_at"],
                 start.isoformat(), float(row["limit_price"])),
            ).fetchone()
            if fill is not None:
                self.db.execute(
                    """UPDATE shadow_preopen_orders SET status='filled_shadow',filled_at=?,
                       fill_price=?,updated_at=? WHERE id=?""",
                    (fill[0], float(fill[1]), now(), row["id"]),
                )
            elif current >= start:
                self.db.execute(
                    """UPDATE shadow_preopen_orders SET status='expired_unfilled',cancelled_at=?,
                       cancel_reason='next_event_started_without_fill',updated_at=? WHERE id=?""",
                    (now(), now(), row["id"]),
                )

    def _evaluate_next_event_forecasts(self) -> None:
        """Размечает next-event forecast только после официального исхода будущего события."""
        self._update_preopen_shadow_orders()
        self.db.execute(
            """UPDATE next_event_forecasts SET status='invalid_after_start',evaluated_at=?
               WHERE status='pending' AND unixepoch(observed_at)>=
                     CAST(SUBSTR(next_event_slug,INSTR(next_event_slug,'5m-')+3) AS INTEGER)""",
            (now(),),
        )
        rows = self.db.execute(
            """SELECT f.*,MAX(t.label) AS next_up_label
               FROM next_event_forecasts f
               JOIN training_examples t ON t.event_slug=f.next_event_slug AND t.outcome='Up'
               WHERE f.status='pending' AND unixepoch(f.observed_at)<
                     CAST(SUBSTR(f.next_event_slug,INSTR(f.next_event_slug,'5m-')+3) AS INTEGER)
               GROUP BY f.id"""
        ).fetchall()
        has_preview = bool(self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='future_event_snapshots'"
        ).fetchone())
        for row in rows:
            label = int(row["next_up_label"])
            direction_won = label if row["predicted_direction"] == "Up" else 1 - label
            fill_row = None
            if has_preview and row["planned_limit_price"] is not None:
                fill_row = self.db.execute(
                    """SELECT collected_at,best_ask FROM future_event_snapshots
                       WHERE next_event_slug=? AND outcome=? AND seconds_before_start>=0
                         AND collected_at>=? AND best_ask IS NOT NULL AND best_ask<=?
                       ORDER BY collected_at LIMIT 1""",
                    (row["next_event_slug"], row["predicted_direction"], row["observed_at"],
                     float(row["planned_limit_price"])),
                ).fetchone()
            filled = int(fill_row is not None)
            fill_price = float(fill_row[1]) if fill_row else None
            pnl = None
            if fill_price and fill_price > 0:
                notional = float(row["planned_notional_usdc"] or settings.PAPER_ENTRY_NOTIONAL_USDC)
                shares = notional / fill_price
                # Касание лимита после постановки считается maker-fill; комиссия maker=0.
                fee = total_fee_usdc(shares, fill_price, taker=False)
                pnl = shares * direction_won - notional - fee
            self.db.execute(
                """UPDATE next_event_forecasts SET status='evaluated',next_resolved_label=?,
                     hypothetical_filled=?,hypothetical_fill_price=?,hypothetical_pnl_usdc=?,evaluated_at=?
                   WHERE id=?""",
                (label, filled, fill_price, pnl, now(), row["id"]),
            )

    def _enforce_loss_streak(self) -> bool:
        pnl_rows = self.db.execute(
            "SELECT id,realized_pnl_usdc FROM paper_positions WHERE session_id=? AND status IN ('closed','resolved') AND COALESCE(execution_valid,1)=1 ORDER BY closed_at DESC,id DESC LIMIT ?",
            (self.session_id, settings.MAX_CONSECUTIVE_LOSSES),
        ).fetchall()
        loss_streak = 0
        for row in pnl_rows:
            if float(row["realized_pnl_usdc"] or 0) < 0:
                loss_streak += 1
            else:
                break
        direction_rows = self.db.execute(
            """SELECT l.decision_was_correct FROM strategy_labels l
               JOIN model_decisions d ON d.id=l.decision_id
               WHERE d.session_id=? AND d.executed=1 AND d.action IN ('BUY_UP','BUY_DOWN')
                 AND l.decision_was_correct IS NOT NULL
               ORDER BY l.labeled_at DESC,l.decision_id DESC LIMIT ?""",
            (self.session_id, settings.MAX_CONSECUTIVE_WRONG_DIRECTIONS),
        ).fetchall()
        wrong_direction_streak = 0
        for row in direction_rows:
            if int(row[0]) == 0:
                wrong_direction_streak += 1
            else:
                break
        stop_reason = None
        if wrong_direction_streak >= settings.MAX_CONSECUTIVE_WRONG_DIRECTIONS:
            stop_reason = f"{wrong_direction_streak}_consecutive_wrong_directions"
        elif loss_streak >= settings.MAX_CONSECUTIVE_LOSSES:
            stop_reason = f"{loss_streak}_consecutive_losing_trades"
        if stop_reason:
            latest_position_id = int(pnl_rows[0]["id"]) if pnl_rows else 0
            acknowledged_id = int(self._control("loss_streak_trigger_position_id", "0") or 0)
            cooldown_until_raw = self._control("paper_cooldown_until", "")
            try:
                cooldown_until = datetime.fromisoformat(cooldown_until_raw).astimezone(UTC)
            except (TypeError, ValueError):
                cooldown_until = None
            current = datetime.now(UTC)
            # Обработанная серия не должна немедленно блокировать сессию снова
            # после ручной разблокировки или автоматического окончания паузы.
            if acknowledged_id == latest_position_id:
                if cooldown_until is not None and current < cooldown_until:
                    return True
                if self._control("engine_state", "running") == "paused" and cooldown_until is not None:
                    self.db.execute(
                        "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','running',?,?)",
                        (now(), "automatic PAPER cooldown completed"),
                    )
                    self.db.execute(
                        "INSERT OR REPLACE INTO runtime_controls VALUES('paper_cooldown_state','completed',?,?)",
                        (now(), stop_reason),
                    )
                    self.db.execute(
                        "UPDATE paper_sessions SET status='running',consecutive_losses=0,stopped_reason=NULL WHERE session_id=?",
                        (self.session_id,),
                    )
                    self.db.commit()
                return False
            cooldown_until = current + timedelta(seconds=settings.PAPER_LOSS_STREAK_COOLDOWN_SECONDS)
            self.db.execute(
                "UPDATE paper_sessions SET status='paused',consecutive_losses=?,stopped_reason=? WHERE session_id=?",
                (loss_streak, stop_reason, self.session_id),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','paused',?,?)",
                (now(), f"30-minute PAPER cooldown: {stop_reason}"),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('paper_cooldown_until',?,?,?)",
                (cooldown_until.isoformat(), now(), stop_reason),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('paper_cooldown_state','active',?,?)",
                (now(), stop_reason),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('loss_streak_trigger_position_id',?,?,?)",
                (str(latest_position_id), now(), stop_reason),
            )
            self.db.commit()
            return True
        self.db.execute("UPDATE paper_sessions SET consecutive_losses=? WHERE session_id=?", (loss_streak, self.session_id))
        return False

    def _counterfactuals(self, state: MarketState, decision_id: int, decision: Decision) -> None:
        if decision.action != "WAIT" or time.monotonic() - self.last_counterfactual_at < settings.WAIT_DECISION_SAMPLE_SECONDS:
            return
        p_up = next((float(tag.split("=", 1)[1]) for tag in decision.tags if tag.startswith("p_up=")), 0.5)
        for outcome, price, probability in (("Up", state.up_ask, p_up), ("Down", state.down_ask, 1 - p_up)):
            if price:
                self.db.execute(
                    """INSERT INTO counterfactual_entries(decision_id,event_slug,observed_at,outcome,entry_price,confidence,
                       hypothetical_notional_usdc,status) VALUES(?,?,?,?,?,?,?,'pending')""",
                    (decision_id, state.event_slug, now(), outcome, price, decision.confidence, settings.PAPER_ENTRY_NOTIONAL_USDC),
                )
        self.last_counterfactual_at = time.monotonic()
        self.db.commit()

    def _evaluate_counterfactuals(self, state: MarketState | None) -> None:
        if state is None:
            return
        rows = self.db.execute(
            "SELECT * FROM counterfactual_entries WHERE status='pending' AND event_slug=?", (state.event_slug,),
        ).fetchall()
        current = datetime.now(UTC)
        for row in rows:
            age = (current - datetime.fromisoformat(row["observed_at"])).total_seconds()
            horizon = next((h for h in settings.COUNTERFACTUAL_HORIZONS_SECONDS if age >= h), None)
            if horizon is None:
                continue
            bid = state.up_bid if row["outcome"] == "Up" else state.down_bid
            if bid is None:
                continue
            shares = row["hypothetical_notional_usdc"] / row["entry_price"]
            entry_fee = total_fee_usdc(shares, row["entry_price"])
            exit_fee = total_fee_usdc(shares, bid)
            pnl = shares * (bid - row["entry_price"]) - entry_fee - exit_fee
            self.db.execute(
                "UPDATE counterfactual_entries SET horizon_seconds=?,evaluated_at=?,exit_price=?,fee_usdc=?,counterfactual_pnl_usdc=?,status='evaluated' WHERE id=?",
                (horizon, now(), bid, entry_fee + exit_fee, pnl, row["id"]),
            )
        self.db.commit()

    def _record_v8_counterfactuals(
        self, state: MarketState, position: PositionState | None, decision_id: int | None,
    ) -> None:
        if time.monotonic() - self.last_v8_counterfactual_at < settings.COUNTERFACTUAL_SAMPLE_SECONDS:
            return
        features = json.dumps(state.as_dict(), ensure_ascii=False, default=str)
        horizons = tuple(settings.COUNTERFACTUAL_HORIZONS_SECONDS) + (300,)
        if position is None:
            for outcome, price in (("Up", state.up_ask), ("Down", state.down_ask)):
                if price is None:
                    continue
                shares = settings.PAPER_ENTRY_NOTIONAL_USDC / float(price)
                for horizon in horizons:
                    self.db.execute(
                        """INSERT INTO action_counterfactuals(decision_id,event_slug,observed_at,action,outcome,
                           entry_price,notional_usdc,shares,cost_usdc,horizon_seconds,status,features_json)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (decision_id, state.event_slug, now(), f"BUY_{outcome.upper()}", outcome, float(price),
                         settings.PAPER_ENTRY_NOTIONAL_USDC, shares, settings.PAPER_ENTRY_NOTIONAL_USDC,
                         horizon, "pending", features),
                    )
            self.db.execute(
                """INSERT INTO action_counterfactuals(decision_id,event_slug,observed_at,action,horizon_seconds,
                   status,evaluated_at,net_pnl_usdc,features_json) VALUES(?,?,?,?,?,'evaluated',?,0,?)""",
                (decision_id, state.event_slug, now(), "WAIT", 0, now(), features),
            )
        else:
            bid = state.up_bid if position.outcome == "Up" else state.down_bid
            if bid is not None:
                close_fee = total_fee_usdc(position.shares, float(bid))
                close_pnl = position.shares * float(bid) - close_fee - position.cost_usdc
                self.db.execute(
                    """INSERT INTO action_counterfactuals(decision_id,event_slug,observed_at,action,outcome,
                       current_bid,shares,cost_usdc,horizon_seconds,status,evaluated_at,exit_price,fee_usdc,
                       net_pnl_usdc,features_json) VALUES(?,?,?,?,?,?,?,?,0,'evaluated',?,?,?,?,?)""",
                    (decision_id, state.event_slug, now(), "CLOSE", position.outcome, float(bid), position.shares,
                     position.cost_usdc, now(), float(bid), close_fee, close_pnl, features),
                )
                for horizon in horizons:
                    self.db.execute(
                        """INSERT INTO action_counterfactuals(decision_id,event_slug,observed_at,action,outcome,
                           current_bid,shares,cost_usdc,horizon_seconds,status,features_json)
                           VALUES(?,?,?,?,?,?,?,?,?,'pending',?)""",
                        (decision_id, state.event_slug, now(), "HOLD", position.outcome, float(bid), position.shares,
                         position.cost_usdc, horizon, features),
                    )
        self.last_v8_counterfactual_at = time.monotonic()
        self.db.commit()

    def _evaluate_v8_counterfactuals(self, state: MarketState | None) -> None:
        current = datetime.now(UTC)
        if state is not None:
            rows = self.db.execute(
                "SELECT * FROM action_counterfactuals WHERE status='pending' AND event_slug=?", (state.event_slug,),
            ).fetchall()
            for row in rows:
                age = (current - datetime.fromisoformat(row["observed_at"])).total_seconds()
                if row["horizon_seconds"] >= 300 or age < row["horizon_seconds"]:
                    continue
                outcome = str(row["outcome"])
                bid = state.up_bid if outcome == "Up" else state.down_bid
                if bid is None:
                    continue
                shares = float(row["shares"] or 0)
                fee = total_fee_usdc(shares, float(bid))
                if str(row["action"]).startswith("BUY_"):
                    entry_fee = total_fee_usdc(shares, float(row["entry_price"]))
                    pnl = shares * float(bid) - float(row["notional_usdc"]) - entry_fee - fee
                    fee += entry_fee
                else:
                    pnl = shares * float(bid) - float(row["cost_usdc"] or 0) - fee
                self.db.execute(
                    """UPDATE action_counterfactuals SET status='evaluated',evaluated_at=?,exit_price=?,
                       fee_usdc=?,net_pnl_usdc=? WHERE id=?""", (now(), float(bid), fee, pnl, row["id"]),
                )
        resolved = self.db.execute(
            """SELECT c.*,MAX(t.label) label FROM action_counterfactuals c JOIN training_examples t
               ON t.event_slug=c.event_slug AND t.outcome=c.outcome
               WHERE c.status='pending' GROUP BY c.id"""
        ).fetchall()
        for row in resolved:
            label = int(row["label"])
            shares = float(row["shares"] or 0)
            if str(row["action"]).startswith("BUY_"):
                fee = total_fee_usdc(shares, float(row["entry_price"]))
                pnl = shares * label - float(row["notional_usdc"]) - fee
            else:
                fee = 0.0
                pnl = shares * label - float(row["cost_usdc"] or 0)
            self.db.execute(
                """UPDATE action_counterfactuals SET status='resolved',evaluated_at=?,exit_price=?,fee_usdc=?,
                   net_pnl_usdc=?,resolved_label=? WHERE id=?""", (now(), float(label), fee, pnl, label, row["id"]),
            )
        self.db.commit()

    def _shadow_tournament(self, state: MarketState, position: PositionState | None, active_model: str) -> None:
        if not settings.SHADOW_TOURNAMENT_ENABLED or time.monotonic() - self.last_shadow_at < settings.SHADOW_SAMPLE_SECONDS:
            return
        try:
            for model_key in settings.SHADOW_MODELS:
                if not model_is_ready(model_key):
                    continue
                try:
                    decision = decide_with_model(state, position, model_key)
                except Exception:
                    continue
                p_up = next((float(tag.split("=", 1)[1]) for tag in decision.tags if tag.startswith("p_up=")), None)
                self.db.execute(
                    """INSERT INTO shadow_predictions(event_slug,observed_at,model_name,action,direction,confidence,
                       predicted_up_probability,entry_price,notional_usdc,reason,status) VALUES(?,?,?,?,?,?,?,?,?,?,'pending')""",
                    (state.event_slug, now(), model_key, decision.action, decision.direction, decision.confidence,
                     p_up, decision.limit_price, decision.notional_usdc, decision.reason),
                )
            self.db.commit()
            self.last_shadow_at = time.monotonic()
        except sqlite3.OperationalError as error:
            self.db.rollback()
            if "locked" not in str(error).lower():
                raise
            print("SHADOW_SAMPLE_SKIPPED reason=database_locked")

    def _resolve_shadow(self) -> None:
        rows = self.db.execute(
            """SELECT s.*,MAX(CASE WHEN t.outcome='Up' THEN t.label END) label_up
               FROM shadow_predictions s JOIN training_examples t ON t.event_slug=s.event_slug
               WHERE s.status='pending' GROUP BY s.id"""
        ).fetchall()
        for row in rows:
            label_up = int(row["label_up"])
            pnl = 0.0
            if row["direction"] in {"Up", "Down"} and row["entry_price"]:
                won = label_up == int(row["direction"] == "Up")
                notional = float(row["notional_usdc"] or settings.PAPER_ENTRY_NOTIONAL_USDC)
                shares = notional / float(row["entry_price"])
                pnl = shares * int(won) - notional - total_fee_usdc(shares, float(row["entry_price"]))
            self.db.execute(
                """UPDATE shadow_predictions SET status='resolved',resolved_label=?,net_pnl_usdc=?,evaluated_at=?
                   WHERE id=?""", (label_up, pnl, now(), row["id"]),
            )
        self.db.commit()

    def _record_exit_shadow(
        self, state: MarketState, position: PositionState | None, decision: Decision, source: str,
    ) -> None:
        """Сравнивает exit-политики на одной позиции, не отправляя shadow-заявки."""
        if position is None or position.event_slug != state.event_slug:
            return
        table = "live_positions" if source == "live" else "paper_positions"
        row = self.db.execute(
            f"SELECT opened_at FROM {table} WHERE id=?", (position.position_id,),
        ).fetchone()
        bid = state.up_bid if position.outcome == "Up" else state.down_bid
        if row is None or bid is None or position.shares <= 0:
            return
        close_pnl = position.shares * float(bid) - total_fee_usdc(position.shares, float(bid)) - position.cost_usdc
        policies: list[tuple[str, str, float | None, float | None]] = [
            ("hold_to_resolution", "HOLD", None, None),
            ("active_exit_v13", "CLOSE" if decision.action == "CLOSE" else "HOLD", None, None),
        ]
        artifact = self.exit_shadow_artifact
        if artifact:
            snapshots = self.db.execute(
                """SELECT collected_at,COALESCE(best_bid,midpoint),spread,best_bid_size,best_ask_size
                   FROM market_snapshots WHERE event_slug=? AND outcome=? AND collected_at>=?
                   AND collected_at<=? ORDER BY collected_at""",
                (state.event_slug, position.outcome, row["opened_at"], state.observed_at),
            ).fetchall()
            bids = [float(item[1]) for item in snapshots if item[1] is not None]
            if bids:
                opened = datetime.fromisoformat(str(row["opened_at"]))
                observed = datetime.fromisoformat(state.observed_at)
                distance = float(state.distance_to_target_pct or 0) * (1 if position.outcome == "Up" else -1)
                last = snapshots[-1]
                values = {
                    "seconds_in_position": max(0.0, (observed - opened).total_seconds()),
                    "remaining_seconds": state.remaining_seconds, "current_bid": float(bid),
                    "average_price": position.average_price,
                    "marked_return": float(bid) / max(position.average_price, .001) - 1,
                    "oriented_distance_to_target_pct": distance,
                    "momentum_bid_3ticks": float(bid) - bids[max(0, len(bids) - 4)],
                    "momentum_distance_3ticks": 0.0, "peak_bid_since_entry": max(bids),
                    "trough_bid_since_entry": min(bids), "drawdown_from_peak": float(bid) - max(bids),
                    "recovery_from_trough": float(bid) - min(bids), "spread": float(last[2] or 0),
                    "log_bid_size": math.log1p(max(0.0, float(last[3] or 0))),
                    "log_ask_size": math.log1p(max(0.0, float(last[4] or 0))),
                    "shares": position.shares, "original_cost_usdc": position.cost_usdc,
                }
                vector = [[float(values.get(name, 0)) for name in artifact["features"]]]
                probability = float(artifact["close_classifier"].predict_proba(vector)[0, 1])
                advantage = float(artifact["advantage_model"].predict(vector)[0])
                action = "CLOSE" if (
                    probability >= float(artifact.get("shadow_probability_threshold", 1.01))
                    and advantage >= float(artifact.get("shadow_advantage_threshold", float("inf")))
                ) else "HOLD"
                policies.append(("exit_sequence_v14", action, advantage, probability))
        for policy, action, advantage, probability in policies:
            existing = self.db.execute(
                "SELECT id,action FROM exit_shadow_predictions WHERE source=? AND position_id=? AND policy=?",
                (source, position.position_id, policy),
            ).fetchone()
            if existing is None:
                self.db.execute(
                    """INSERT INTO exit_shadow_predictions(source,position_id,event_slug,outcome,policy,
                       observed_at,action,predicted_advantage_usdc,predicted_close_probability,selected_pnl_usdc)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (source, position.position_id, state.event_slug, position.outcome, policy,
                     state.observed_at, action, advantage, probability, close_pnl if action == "CLOSE" else None),
                )
            elif existing["action"] == "HOLD" and action == "CLOSE":
                self.db.execute(
                    """UPDATE exit_shadow_predictions SET action='CLOSE',observed_at=?,predicted_advantage_usdc=?,
                       predicted_close_probability=?,selected_pnl_usdc=? WHERE id=?""",
                    (state.observed_at, advantage, probability, close_pnl, existing["id"]),
                )
        self.db.commit()

    def _resolve_exit_shadow(self) -> None:
        rows = self.db.execute(
            """SELECT s.*,MAX(t.label) label FROM exit_shadow_predictions s JOIN training_examples t
               ON t.event_slug=s.event_slug AND t.outcome=s.outcome WHERE s.status='pending' GROUP BY s.id"""
        ).fetchall()
        for row in rows:
            table = "live_positions" if row["source"] == "live" else "paper_positions"
            position = self.db.execute(
                f"SELECT shares,cost_usdc FROM {table} WHERE id=?", (row["position_id"],),
            ).fetchone()
            if position is None:
                continue
            hold_pnl = float(position["shares"]) * int(row["label"]) - float(position["cost_usdc"])
            selected = float(row["selected_pnl_usdc"]) if row["action"] == "CLOSE" else hold_pnl
            self.db.execute(
                """UPDATE exit_shadow_predictions SET status='evaluated',hold_pnl_usdc=?,selected_pnl_usdc=?,
                   realized_advantage_usdc=?,evaluated_at=? WHERE id=?""",
                (hold_pnl, selected, selected - hold_pnl, now(), row["id"]),
            )
        if rows:
            report = {row["policy"]: dict(row) for row in self.db.execute(
                """SELECT policy,COUNT(*) events,SUM(selected_pnl_usdc) pnl,SUM(realized_advantage_usdc) advantage,
                   SUM(action='CLOSE') closes FROM exit_shadow_predictions WHERE status='evaluated' GROUP BY policy"""
            )}
            settings.EXIT_SHADOW_REPORT_PATH.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        self.db.commit()

    def equity_snapshot(self, state: MarketState | None) -> None:
        cash = self._cash()
        open_rows = self.db.execute("SELECT * FROM paper_positions WHERE session_id=? AND status='open'", (self.session_id,)).fetchall()
        open_value = 0.0
        for row in open_rows:
            mark = None
            if state and row["event_slug"] == state.event_slug:
                mark = state.up_bid if row["outcome"] == "Up" else state.down_bid
            mark = float(mark if mark is not None else row["current_price"] or row["average_price"])
            open_value += float(row["shares"]) * mark
            self.db.execute("UPDATE paper_positions SET current_price=? WHERE id=?", (mark, row["id"]))
        realized = float(self.db.execute("SELECT realized_pnl_usdc FROM paper_sessions WHERE session_id=?", (self.session_id,)).fetchone()[0])
        equity = cash + open_value
        self.db.execute(
            "INSERT INTO paper_equity_snapshots(session_id,observed_at,cash_usdc,open_value_usdc,equity_usdc,realized_pnl_usdc,unrealized_pnl_usdc) VALUES(?,?,?,?,?,?,?)",
            (self.session_id, now(), cash, open_value, equity, realized, equity - cash - sum(float(r["cost_usdc"]) for r in open_rows)),
        )
        self.db.commit()

    def step(self) -> Decision:
        self.db.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('engine_heartbeat','alive',?,?)",
            (now(), f"session={self.session_id}"),
        )
        self.db.commit()
        self.settle_resolved()
        self._provisionally_settle_ended_paper_positions()
        self._settle_live_records()
        self._resolve_shadow()
        self._resolve_exit_shadow()
        self._apply_requested_mode_if_flat()
        self._apply_pending_models()
        state = self.market_state()
        self._enforce_validation_stop(state)
        if state is not None:
            self._record_next_event_forecast(state)
        self._evaluate_counterfactuals(state)
        self._evaluate_v8_counterfactuals(state)
        engine_state = self._handle_runtime_control()
        trading_mode = self._control("trading_mode", "paper")
        if trading_mode == "live":
            self._reconcile_live_orders()
        elif trading_mode == "paper" and engine_state != "paused":
            self._reconcile_paper_entry_orders(state)
        if trading_mode not in {"paper", "live"} or engine_state == "paused":
            if state is not None:
                active_model = self._control("selected_entry_model", settings.DEFAULT_ENTRY_MODEL)
                # На паузе продолжаем только shadow/контрфактуальное наблюдение без заявок и капитала.
                self.db.commit()
                self._shadow_tournament(state, None, active_model)
                self._record_v8_counterfactuals(state, None, None)
            self.equity_snapshot(state)
            return Decision("WAIT", 1.0, "Торговый движок поставлен на паузу", ["engine_paused"])
        if state is None:
            self.equity_snapshot(None)
            return Decision("WAIT", 0.0, "Нет свежего активного BTC 5m рынка", ["no_market"])
        position = self.open_position(state)
        entry_model_key = self._control("selected_entry_model", settings.DEFAULT_ENTRY_MODEL)
        exit_model_key = self._control("selected_exit_model", settings.DEFAULT_EXIT_MODEL)
        model_key = exit_model_key if position is not None else entry_model_key
        # Не держим SQLite write-lock во время вычисления shadow-моделей.
        self.db.commit()
        self._shadow_tournament(state, position, entry_model_key)
        decision = decide_with_model(
            state, position, model_key,
            active_collection=(trading_mode == "paper"),
        )
        # До записи решения проверяем, что оно технически исполнимо. Иначе dashboard
        # показывал BUY, хотя _buy молча отклонял старый стакан или закрытое событие.
        if position is None and decision.action.startswith("BUY_"):
            if decision.direction is None or not self._fresh_book(state, decision.direction):
                decision = Decision(
                    "WAIT", decision.confidence,
                    "Вход не отправлен: стакан отсутствует или старше допустимого лимита",
                    [*decision.tags, "execution_book_not_fresh"], direction=decision.direction,
                )
            elif decision.limit_price is None:
                decision = Decision(
                    "WAIT", decision.confidence, "Вход не отправлен: модель не выбрала limit price",
                    [*decision.tags, "execution_missing_limit_price"], direction=decision.direction,
                )
            else:
                validity = validate_entry_state(state, float(decision.limit_price))
                if not validity.valid:
                    decision = Decision(
                        "WAIT", decision.confidence, f"Вход не отправлен: {validity.reason}",
                        [*decision.tags, "execution_domain_guard", str(validity.reason)], direction=decision.direction,
                    )
        if position is None and decision.action.startswith("BUY_") and settings.DIRECTION_COLLAPSE_BLOCK_ENABLED:
            dominant, share, events = self._direction_collapse(model_key)
            if dominant == decision.direction and share > settings.DIRECTION_COLLAPSE_MAX_SHARE:
                decision = Decision(
                    "WAIT", decision.confidence,
                    f"Вход заблокирован: {dominant} занимает {share:.0%} последних {events} независимых событий",
                    [*decision.tags, "directional_collapse", f"direction_share={share:.4f}", f"direction_events={events}"],
                    direction=decision.direction,
                )
        if trading_mode == "live" and self._live_total_pnl(state) <= -settings.LIVE_CANARY_MAX_LOSS_USDC:
            if position is not None:
                decision = Decision(
                    "CLOSE", 1.0, "Достигнут максимальный убыток LIVE-CANARY; закрываем позицию",
                    ["live_canary_loss_stop", "exit_stage=5"], direction=position.outcome,
                )
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('trading_mode','paper',?,?)",
                    (now(), "live canary maximum loss reached"),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('requested_trading_mode','paper',?,?)",
                    (now(), "automatic canary stop"),
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state','blocked_loss_limit',?,?)",
                    (now(), "automatic canary stop"),
                )
                self.db.commit()
                return Decision("WAIT", 1.0, "LIVE-CANARY остановлен по лимиту убытка", ["live_canary_loss_stop"])
        if position is None and decision.action.startswith("BUY_"):
            if trading_mode == "live":
                prior_entries = int(self.db.execute(
                    "SELECT COUNT(*) FROM live_orders WHERE event_slug=? AND side='BUY'", (state.event_slug,),
                ).fetchone()[0])
            else:
                prior_entries = int(self.db.execute(
                    "SELECT COUNT(*) FROM paper_positions WHERE session_id=? AND event_slug=?",
                    (self.session_id, state.event_slug),
                ).fetchone()[0])
            if prior_entries >= settings.MAX_FRESH_ENTRIES_PER_EVENT:
                decision = Decision("WAIT", decision.confidence, "Повторный свежий вход в это событие запрещён", [*decision.tags, "event_entry_limit"])
        self._record_exit_shadow(state, position, decision, trading_mode)
        decision = self._confirmed(decision, state, position)
        # Полный выход является инвариантом исполнения: даже старый артефакт или
        # кешированное решение PARTIAL_CLOSE преобразуется в полное CLOSE.
        if settings.FULL_EXIT_ONLY_ENABLED and decision.action == "PARTIAL_CLOSE":
            decision = Decision(
                "CLOSE", decision.confidence,
                f"{decision.reason}; частичный выход преобразован в полный",
                [*decision.tags, "full_exit_only"], direction=decision.direction,
                exit_fraction=1.0,
            )
        status = self.db.execute("SELECT status FROM paper_sessions WHERE session_id=?", (self.session_id,)).fetchone()[0]
        if status == "forced_stopped" or self._enforce_loss_streak():
            decision = Decision(
                "WAIT", 1.0,
                f"PAPER поставлен на {settings.PAPER_LOSS_STREAK_COOLDOWN_SECONDS // 60} минут после серии убытков",
                ["loss_streak_cooldown"],
            )
        day_cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        daily_realized = float(self.db.execute(
            """SELECT COALESCE(SUM(realized_pnl_usdc),0) FROM paper_positions
               WHERE session_id=? AND closed_at>=? AND status IN ('closed','resolved')
                 AND COALESCE(execution_valid,1)=1""",
            (self.session_id, day_cutoff),
        ).fetchone()[0])
        if decision.action in {"BUY_UP", "BUY_DOWN", "ADD"} and daily_realized <= -settings.MAX_DAILY_LOSS_USDC:
            decision = Decision("WAIT", 1.0, "Достигнут лимит дневного убытка", ["daily_loss_limit"])
        should_log = decision.action not in {"WAIT", "HOLD"} or time.monotonic() - self.last_logged_decision >= settings.PAPER_DECISION_LOG_SECONDS
        if not should_log:
            self.equity_snapshot(state)
            return decision
        decision_id = self._decision(state, position, decision, model_key)
        self._counterfactuals(state, decision_id, decision)
        self._record_v8_counterfactuals(state, position, decision_id)
        self.last_logged_decision = time.monotonic()
        action_allowed = time.monotonic() - self.last_action_at >= settings.PAPER_MIN_SECONDS_BETWEEN_ACTIONS
        if action_allowed and decision.action.startswith("BUY_"):
            self._buy(decision_id, state, decision)
            self.last_action_at = time.monotonic()
        elif action_allowed and settings.ALLOW_POSITION_ADD and decision.action == "ADD" and position:
            self._buy(decision_id, state, decision, add=True)
            self.last_action_at = time.monotonic()
        elif action_allowed and decision.action == "CLOSE" and position:
            stage = next((int(tag.split("=", 1)[1]) for tag in decision.tags if tag.startswith("exit_stage=")), 5)
            self._close(decision_id, state, position, decision.reason, 1.0, stage, "limit_exit" in decision.tags)
            self.last_action_at = time.monotonic()
        elif action_allowed and settings.PARTIAL_EXIT_ENABLED and decision.action == "PARTIAL_CLOSE" and position:
            stage = next((int(tag.split("=", 1)[1]) for tag in decision.tags if tag.startswith("exit_stage=")), position.exit_stage + 1)
            fraction = decision.exit_fraction or settings.PARTIAL_EXIT_FRACTION
            self._close(decision_id, state, position, decision.reason, fraction, stage, "limit_exit" in decision.tags)
            self.last_action_at = time.monotonic()
        self.equity_snapshot(state)
        return decision

    def close(self) -> None:
        try:
            self.db.execute("UPDATE paper_sessions SET status='paused' WHERE session_id=?", (self.session_id,))
            self.db.commit()
        except sqlite3.OperationalError:
            self.db.rollback()
        finally:
            self.db.close()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe continuous paper trading for BTC Up/Down 5m")
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=0)
    return parser.parse_args()


async def main() -> None:
    args = arguments()
    if not args.db.exists():
        raise RuntimeError(f"Database not found: {args.db}")
    engine = PaperEngine(args.db)
    started = time.monotonic()
    print(f"TRADING_ENGINE_READY session={engine.session_id} initial=${settings.PAPER_INITIAL_BALANCE_USDC:.2f}")
    try:
        while True:
            try:
                decision = engine.step()
                # Старая ошибка не должна продолжать отображаться как активная,
                # если следующий полный торговый тик завершился успешно.
                engine.db.execute(
                    "DELETE FROM runtime_controls WHERE control_key='engine_last_error'"
                )
                engine.db.commit()
                print(f"PAPER_DECISION action={decision.action} confidence={decision.confidence:.3f} reason={decision.reason}")
            except Exception as exc:
                engine.db.rollback()
                try:
                    engine.db.execute(
                        "INSERT OR REPLACE INTO runtime_controls VALUES('engine_last_error',?,?,?)",
                        (type(exc).__name__, now(), "step failed; loop continues"),
                    )
                    engine.db.commit()
                except sqlite3.OperationalError:
                    engine.db.rollback()
                print(f"TRADING_ENGINE_ERROR type={type(exc).__name__}")
            if args.once or args.max_seconds and time.monotonic() - started >= args.max_seconds:
                break
            await asyncio.sleep(settings.PAPER_POLL_SECONDS)
    finally:
        engine.close()


if __name__ == "__main__":
    from polybot.runtime import run_async

    run_async(__file__, main)
