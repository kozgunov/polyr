"""Shadow GTD SELL collector: наблюдает исполнимость выхода без отправки ордеров."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import app_config as settings
from polybot.collectors.pipeline import now
from polybot.trading.fees import total_fee_usdc


HORIZONS = (5, 10, 15)
SAMPLE_SECONDS = 5
SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_exit_orders_v23 (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  position_id INTEGER NOT NULL,
  session_id TEXT,
  event_slug TEXT NOT NULL,
  outcome TEXT NOT NULL,
  entry_model TEXT,
  observed_at TEXT NOT NULL,
  sample_bucket INTEGER NOT NULL,
  horizon_seconds INTEGER NOT NULL,
  expires_at TEXT NOT NULL,
  limit_price REAL NOT NULL,
  position_shares REAL NOT NULL,
  position_cost_usdc REAL NOT NULL,
  average_entry_price REAL NOT NULL,
  submit_best_bid REAL,
  submit_best_bid_size REAL,
  submit_best_ask REAL,
  submit_best_ask_size REAL,
  submit_spread REAL,
  submit_book_timestamp TEXT,
  submit_book_age_ms REAL,
  reference_price REAL,
  target_price REAL,
  oriented_distance_to_target_pct REAL,
  remaining_seconds REAL,
  immediate_fill_size REAL NOT NULL DEFAULT 0,
  resting_fill_size REAL NOT NULL DEFAULT 0,
  filled_size REAL NOT NULL DEFAULT 0,
  fill_price REAL,
  exit_fee_usdc REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'working',
  fill_reason TEXT,
  filled_at TEXT,
  evaluated_at TEXT,
  resolved_label INTEGER,
  strategy_net_pnl_usdc REAL,
  hold_net_pnl_usdc REAL,
  advantage_vs_hold_usdc REAL,
  UNIQUE(source,position_id,sample_bucket,horizon_seconds)
);
CREATE INDEX IF NOT EXISTS idx_shadow_exit_v23_working
  ON shadow_exit_orders_v23(status,expires_at,event_slug,outcome);
CREATE INDEX IF NOT EXISTS idx_shadow_exit_v23_position
  ON shadow_exit_orders_v23(source,position_id,observed_at);
"""


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _event_end(slug: str) -> datetime:
    return datetime.fromtimestamp(int(slug.rsplit("-", 1)[-1]) + 300, UTC)


def _position_rows(connection: sqlite3.Connection) -> list[dict]:
    rows: list[dict] = []
    for source, table in (("paper", "paper_positions"), ("live", "live_positions")):
        result = connection.execute(
            f"""SELECT p.*,COALESCE(d.model_name,'unknown') entry_model
                FROM {table} p LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                WHERE p.status='open' AND COALESCE(p.execution_valid,1)=1"""
        ).fetchall()
        rows.extend({**dict(row), "source": source} for row in result)
    return rows


def _latest_book(connection: sqlite3.Connection, event_slug: str, outcome: str) -> dict | None:
    row = connection.execute(
        """SELECT collected_at,best_bid,best_bid_size,best_ask,best_ask_size,spread,book_timestamp
           FROM market_snapshots WHERE event_slug=? AND outcome=?
           ORDER BY collected_at DESC,id DESC LIMIT 1""", (event_slug, outcome),
    ).fetchone()
    return dict(row) if row else None


def _latest_reference(connection: sqlite3.Connection, event_slug: str) -> dict | None:
    row = connection.execute(
        """SELECT collected_at,reference_price,target_price FROM reference_price_snapshots
           WHERE event_slug=? ORDER BY collected_at DESC,id DESC LIMIT 1""", (event_slug,),
    ).fetchone()
    return dict(row) if row else None


def _create_candidates(connection: sqlite3.Connection) -> int:
    created = 0; observed = datetime.now(UTC); bucket = int(observed.timestamp() // SAMPLE_SECONDS)
    for position in _position_rows(connection):
        slug, outcome = str(position["event_slug"]), str(position["outcome"])
        if observed >= _event_end(slug):
            continue
        book = _latest_book(connection, slug, outcome)
        reference = _latest_reference(connection, slug)
        if not book or book["best_bid"] is None:
            continue
        book_time = _timestamp(str(book["collected_at"])); age_ms = (observed - book_time).total_seconds() * 1000
        if age_ms < 0 or age_ms > float(settings.MAX_EXECUTION_BOOK_AGE_SECONDS) * 1000:
            continue
        shares = float(position.get("shares") or 0); cost = float(position.get("cost_usdc") or 0)
        if shares <= 0 or cost <= 0:
            continue
        bid = float(book["best_bid"]); bid_size = max(0.0, float(book["best_bid_size"] or 0))
        immediate = min(shares, bid_size)
        reference_price = float(reference["reference_price"]) if reference and reference["reference_price"] else None
        target_price = float(reference["target_price"]) if reference and reference["target_price"] else None
        distance = None
        if reference_price and target_price:
            raw_distance = (reference_price / target_price - 1) * 100
            distance = raw_distance if outcome == "Up" else -raw_distance
        for horizon in HORIZONS:
            expires = min(observed + timedelta(seconds=horizon), _event_end(slug))
            cursor = connection.execute(
                """INSERT OR IGNORE INTO shadow_exit_orders_v23(
                   source,position_id,session_id,event_slug,outcome,entry_model,observed_at,sample_bucket,
                   horizon_seconds,expires_at,limit_price,position_shares,position_cost_usdc,
                   average_entry_price,submit_best_bid,submit_best_bid_size,submit_best_ask,
                   submit_best_ask_size,submit_spread,submit_book_timestamp,submit_book_age_ms,
                   reference_price,target_price,oriented_distance_to_target_pct,remaining_seconds,
                   immediate_fill_size,filled_size,fill_price,exit_fee_usdc,status,fill_reason,filled_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (position["source"], int(position["id"]), position.get("session_id"), slug, outcome,
                 position.get("entry_model"), observed.isoformat(), bucket, horizon, expires.isoformat(),
                 bid, shares, cost, float(position.get("average_price") or cost / shares), bid, bid_size,
                 book["best_ask"], book["best_ask_size"], book["spread"], book["book_timestamp"], age_ms,
                 reference_price, target_price, distance, max(0.0, (_event_end(slug) - observed).total_seconds()),
                 immediate, immediate, bid if immediate > 0 else None,
                 total_fee_usdc(immediate, bid) if immediate > 0 else 0.0,
                 "filled" if immediate + 1e-9 >= shares else "working",
                 "immediate_best_bid" if immediate > 0 else "resting_at_limit",
                 observed.isoformat() if immediate + 1e-9 >= shares else None),
            )
            created += int(cursor.rowcount > 0)
    return created


def _advance_candidates(connection: sqlite3.Connection) -> int:
    changed = 0; current = datetime.now(UTC)
    rows = connection.execute(
        "SELECT * FROM shadow_exit_orders_v23 WHERE status='working' ORDER BY id LIMIT 500"
    ).fetchall()
    for raw in rows:
        row = dict(raw); remaining = float(row["position_shares"]) - float(row["filled_size"])
        # Если после размещения лучший bid стал выше нашего sell-limit, рынок прошёл
        # через цену заявки: оставшийся объём считаем maker-fill по limit price.
        crossed = connection.execute(
            """SELECT collected_at,best_bid FROM market_snapshots
               WHERE event_slug=? AND outcome=? AND collected_at>? AND collected_at<=?
                 AND best_bid>? ORDER BY collected_at,id LIMIT 1""",
            (row["event_slug"], row["outcome"], row["observed_at"],
             min(current, _timestamp(row["expires_at"])).isoformat(), float(row["limit_price"]) + 1e-9),
        ).fetchone()
        if remaining > 1e-9 and crossed is not None:
            filled = float(row["position_shares"])
            # Resting maker-часть не платит taker fee; комиссия immediate уже сохранена.
            connection.execute(
                """UPDATE shadow_exit_orders_v23 SET resting_fill_size=?,filled_size=?,fill_price=?,
                   status='filled',fill_reason='resting_limit_crossed',filled_at=? WHERE id=?""",
                (remaining, filled, float(row["limit_price"]), crossed["collected_at"], row["id"]),
            ); changed += 1
        elif current >= _timestamp(row["expires_at"]):
            status = "partial" if float(row["filled_size"]) > 0 else "expired_unfilled"
            connection.execute(
                "UPDATE shadow_exit_orders_v23 SET status=?,fill_reason=COALESCE(fill_reason,?) WHERE id=?",
                (status, "gtd_expired", row["id"]),
            ); changed += 1
    return changed


def _resolve_candidates(connection: sqlite3.Connection) -> int:
    rows = connection.execute(
        """SELECT s.*,MAX(t.label) resolved FROM shadow_exit_orders_v23 s
           JOIN training_examples t ON t.event_slug=s.event_slug AND t.outcome=s.outcome
           WHERE s.evaluated_at IS NULL AND s.status IN ('filled','partial','expired_unfilled')
           GROUP BY s.id LIMIT 1000"""
    ).fetchall()
    for raw in rows:
        row = dict(raw); label = int(row["resolved"]); shares = float(row["position_shares"])
        filled = float(row["filled_size"]); remaining = max(0.0, shares - filled)
        proceeds = filled * float(row["limit_price"])
        strategy = proceeds - float(row["exit_fee_usdc"] or 0) + remaining * label - float(row["position_cost_usdc"])
        hold = shares * label - float(row["position_cost_usdc"])
        connection.execute(
            """UPDATE shadow_exit_orders_v23 SET evaluated_at=?,resolved_label=?,
               strategy_net_pnl_usdc=?,hold_net_pnl_usdc=?,advantage_vs_hold_usdc=? WHERE id=?""",
            (now(), label, strategy, hold, strategy - hold, row["id"]),
        )
    return len(rows)


def cycle(connection: sqlite3.Connection) -> dict[str, int]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = {"created": _create_candidates(connection), "advanced": _advance_candidates(connection),
                  "resolved": _resolve_candidates(connection)}
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('exit_shadow_v23_heartbeat','alive',?,?)",
            (now(), json.dumps(result, ensure_ascii=False)),
        )
        connection.commit(); return result
    except Exception:
        connection.rollback(); raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow GTD exit fill collector v23")
    parser.add_argument("--db", type=Path, default=settings.DATABASE_PATH)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    connection = sqlite3.connect(args.db, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL"); connection.execute("PRAGMA busy_timeout=10000")
    connection.executescript(SCHEMA); connection.commit()
    print("EXIT_SHADOW_V23_READY", flush=True)
    while True:
        try:
            result = cycle(connection)
            if any(result.values()):
                print(f"EXIT_SHADOW_V23 {result}", flush=True)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower():
                raise
        if args.once:
            break
        time.sleep(1)
    connection.close()


if __name__ == "__main__":
    main()
