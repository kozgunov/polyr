"""FastAPI dashboard for local pipeline observability and SQLite browsing."""

from __future__ import annotations

import json
import math
import base64
import hmac
import secrets
import sqlite3
import threading
import time
from functools import wraps
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import api_config as api
import app_config as settings
import psutil
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.model_registry import MODEL_SPECS, get_model, model_is_ready, public_models
from polybot.analytics.model_health import model_health
from polybot.trading.live_guard import paper_statistics, readiness_failures
from polybot.trading.live_executor import preflight as live_technical_preflight

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Polybot Control Center", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _cached(seconds: float):
    """Не допускает лавину параллельных тяжёлых `/api/overview` запросов."""
    def decorate(function):
        lock = threading.Lock()
        state: dict[str, Any] = {"at": 0.0, "value": None}
        @wraps(function)
        def wrapped(*args, **kwargs):
            current = time.monotonic()
            if state["value"] is not None and current - state["at"] < seconds:
                return state["value"]
            with lock:
                current = time.monotonic()
                if state["value"] is None or current - state["at"] >= seconds:
                    state["value"] = function(*args, **kwargs)
                    state["at"] = time.monotonic()
                return state["value"]
        def cache_clear() -> None:
            with lock:
                state["at"] = 0.0
                state["value"] = None
        wrapped.cache_clear = cache_clear
        return wrapped
    return decorate


def action_value_overview() -> dict[str, Any]:
    report: dict[str, Any] = {}
    try:
        report = json.loads(settings.PNL_MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    test = (report.get("two_stage") or {}).get("test_policy") or {}
    fill = report.get("fill_model") or {}
    candidate_v4: dict[str, Any] = {}
    try:
        v4 = json.loads(settings.ACTION_VALUE_V4_REPORT_PATH.read_text(encoding="utf-8"))
        v4_test = (v4.get("two_stage") or {}).get("test_policy") or {}
        v4_tail = v4.get("tail_risk_model") or {}
        candidate_v4 = {
            "version": "entry_value_v4_tail_risk", "candidate_only": True,
            "promotion_passed": bool((v4.get("promotion_gate") or {}).get("passed")),
            "test_pnl_usdc": v4_test.get("net_pnl_usdc"),
            "test_expectancy_usdc": v4_test.get("expectancy_usdc"),
            "test_ci_lower_usdc": v4_test.get("expectancy_ci95_lower_usdc"),
            "test_up": v4_test.get("up"), "test_down": v4_test.get("down"),
            "tail_rate": v4_tail.get("test_tail_rate"),
            "tail_probability_mean": v4_tail.get("predicted_tail_probability_mean"),
            "expected_tail_loss_mean_usdc": v4_tail.get("expected_tail_loss_mean_usdc"),
            "risk_penalty": v4_tail.get("selected_risk_penalty"),
            "max_tail_probability": v4_tail.get("selected_max_tail_probability"),
        }
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "enabled": bool(settings.ACTION_VALUE_ENABLED and settings.PNL_MODEL_ARTIFACT_PATH.exists()),
        "paper_experiment": bool(settings.PAPER_ACTION_VALUE_EXPERIMENT_ENABLED),
        "name": settings.PAPER_ACTION_VALUE_EXPERIMENT_NAME,
        "promotion_passed": bool((report.get("promotion_gate") or {}).get("passed")),
        "formula": report.get("formula", "P(fill) × E(net PnL | fill)"),
        "fill_roc_auc": fill.get("roc_auc"), "fill_brier": fill.get("brier"),
        "test_pnl_usdc": test.get("net_pnl_usdc"),
        "test_expectancy_usdc": test.get("expectancy_usdc"),
        "test_ci_lower_usdc": test.get("expectancy_ci95_lower_usdc"),
        "candidate_v4": candidate_v4,
    }


def _dashboard_password() -> str:
    configured = str(getattr(api, "DASHBOARD_PASSWORD", "") or "").strip()
    if configured:
        return configured
    path = settings.DASHBOARD_PASSWORD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_urlsafe(18), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


DASHBOARD_PASSWORD = _dashboard_password()


def event_metadata(slug: str | None) -> dict[str, Any]:
    """Единое представление BTC 5m-события для ссылок и локального времени UI."""
    value = str(slug or "")
    try:
        start_ts = int(value.rsplit("-", 1)[1])
        start = datetime.fromtimestamp(start_ts, UTC)
        end = start + timedelta(minutes=5)
        return {
            "event_slug": value,
            "event_url": f"https://polymarket.com/event/{value}",
            "event_start": start.isoformat(),
            "event_end": end.isoformat(),
        }
    except (IndexError, ValueError, OSError):
        return {"event_slug": value, "event_url": None, "event_start": None, "event_end": None}


def enrich_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.update(event_metadata(row.get("event_slug")))
    return rows


@app.middleware("http")
async def protect_remote_dashboard(request: Request, call_next):
    """Локальный браузер не спрашивает пароль; телефон и другие LAN-клиенты — спрашивают."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        auth = request.headers.get("Authorization", "")
        valid = False
        if auth.startswith("Basic "):
            try:
                username, password = base64.b64decode(auth[6:]).decode("utf-8").split(":", 1)
                valid = hmac.compare_digest(username, settings.DASHBOARD_USERNAME) and hmac.compare_digest(password, DASHBOARD_PASSWORD)
            except (ValueError, UnicodeDecodeError):
                valid = False
        if not valid:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Polybot dashboard"'})
    return await call_next(request)


class TradingModeRequest(BaseModel):
    mode: str
    canary: bool = True


class LiveScaleRequest(BaseModel):
    scale: float


class LivePreflightRequest(BaseModel):
    token_id: str | None = None


class TradingControlRequest(BaseModel):
    action: str


class ModelSelectionRequest(BaseModel):
    model: str
    role: str = "entry"


def connect() -> sqlite3.Connection | None:
    if not settings.DATABASE_PATH.exists():
        return None
    connection = sqlite3.connect(settings.DATABASE_PATH, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    # Большие window-запросы model health не должны зависеть от доступности
    # системной TEMP-папки: временные B-tree держим в памяти процесса.
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def table_names(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def count(connection: sqlite3.Connection, table: str) -> int:
    if table not in table_names(connection):
        return 0
    # REPLACE меняет rowid, поэтому MAX(rowid) завышает число строк в таблицах состояния.
    if table in {"runtime_controls", "events", "markets", "event_targets", "model_training_cycles"}:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    # Для больших append-only таблиц MAX(rowid) остаётся быстрым и точным.
    value = connection.execute(f'SELECT MAX(rowid) FROM "{table}"').fetchone()[0]
    return int(value or 0)


def age_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, (datetime.now(UTC) - datetime.fromisoformat(value)).total_seconds())
    except ValueError:
        return None


def latest_sources(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if "external_prices" not in table_names(connection):
        return []
    enabled = [name for name, flag in (("bybit", settings.ENABLE_BYBIT), ("okx", settings.ENABLE_OKX),
                                        ("pyth", settings.ENABLE_PYTH),
                                        ("chainlink_twap_60s", settings.ENABLE_CHAINLINK_RTDS)) if flag]
    rows = []
    for source in enabled:
        row = connection.execute(
            """SELECT source,price,confidence,collected_at,source_timestamp
               FROM external_prices WHERE source=? ORDER BY collected_at DESC LIMIT 1""", (source,),
        ).fetchone()
        if row:
            rows.append(row)
    return [{**dict(row), "age_seconds": age_seconds(row["collected_at"])} for row in rows]


def current_target(connection: sqlite3.Connection, slug: str | None) -> dict[str, Any] | None:
    """Price to Beat Polymarket и валидированная live-медиана внешних источников."""
    tables = set(table_names(connection))
    if not slug or not {"event_targets", "reference_price_snapshots"}.issubset(tables):
        return None
    target = connection.execute(
        "SELECT target_price,source,fetched_at FROM event_targets WHERE event_slug=?", (slug,),
    ).fetchone()
    reference = connection.execute(
        """SELECT reference_price,collected_at,source FROM reference_price_snapshots
           WHERE event_slug=? AND reference_price IS NOT NULL AND source='polymarket_crypto_price'
           ORDER BY collected_at DESC LIMIT 1""", (slug,),
    ).fetchone()
    if not target:
        return None
    target_price = float(target["target_price"])
    enabled = [name for name, flag in (("bybit", settings.ENABLE_BYBIT), ("okx", settings.ENABLE_OKX),
                                        ("pyth", settings.ENABLE_PYTH)) if flag]
    external_rows = []
    for source in enabled:
        row = connection.execute(
            "SELECT price FROM external_prices WHERE source=? ORDER BY collected_at DESC LIMIT 1", (source,),
        ).fetchone()
        if row:
            external_rows.append(row)
    external_prices = sorted(float(row[0]) for row in external_rows)
    external_median = external_prices[len(external_prices) // 2] if external_prices else None
    chainlink = connection.execute(
        "SELECT price,collected_at FROM external_prices WHERE source='chainlink_twap_60s' "
        "ORDER BY collected_at DESC LIMIT 1"
    ).fetchone() if settings.ENABLE_CHAINLINK_RTDS else None
    chainlink_age = age_seconds(chainlink["collected_at"]) if chainlink else None
    chainlink_fresh = bool(chainlink and chainlink_age is not None
                           and chainlink_age <= settings.MAX_CHAINLINK_AGE_SECONDS)
    reference_price = (float(reference["reference_price"]) if reference else
                       float(chainlink["price"]) if chainlink_fresh else external_median)
    side_validated = (
        (reference_price >= target_price) == (external_median >= target_price)
        if reference_price is not None and external_median is not None else None
    )
    return {
        "event_slug": slug,
        "target_price": target_price,
        "reference_price": reference_price,
        "distance_usd": reference_price - target_price if reference_price is not None else None,
        "distance_pct": (reference_price / target_price - 1.0) * 100.0 if reference_price is not None else None,
        "target_source": target["source"],
        "reference_source": (reference["source"] if reference else
                             "chainlink_twap_60s" if chainlink_fresh else "external_median_proxy"),
        "reference_age_seconds": (age_seconds(reference["collected_at"]) if reference else
                                  chainlink_age if chainlink_fresh else None),
        "chainlink_twap_price": float(chainlink["price"]) if chainlink else None,
        "chainlink_twap_age_seconds": chainlink_age,
        "external_median_price": external_median,
        "target_side_validated": side_validated,
    }


def latency_metrics(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if "source_metrics" not in table_names(connection):
        return []
    cutoff = (datetime.now(UTC) - timedelta(minutes=15)).isoformat()
    rows = connection.execute(
        """SELECT source,operation,ROUND(AVG(latency_ms),1) avg_ms,ROUND(MAX(latency_ms),1) max_ms,
                  SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) ok_count,COUNT(*) total_count,MAX(measured_at) last_seen
           FROM source_metrics
           WHERE id>(SELECT MAX(id)-10000 FROM source_metrics) AND measured_at>=?
           GROUP BY source,operation ORDER BY source""",
        (cutoff,),
    ).fetchall()
    return [dict(row) for row in rows]


def system_metrics() -> dict[str, Any]:
    disk = psutil.disk_usage(str(settings.PROJECT_ROOT))
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / 1024**3, 2),
        "memory_total_gb": round(memory.total / 1024**3, 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "database_mb": round(settings.DATABASE_PATH.stat().st_size / 1024**2, 2) if settings.DATABASE_PATH.exists() else 0,
    }


def access_matrix(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {item["source"]: item for item in sources}
    return [
        {"name": "Polymarket: рынок и стакан", "configured": True, "observed": True},
        {"name": "Polymarket: торговая авторизация", "configured": all(bool(getattr(api, key, "")) for key in ("POLYMARKET_PRIVATE_KEY", "POLYMARKET_API_KEY", "POLYMARKET_API_SECRET", "POLYMARKET_API_PASSPHRASE")), "observed": False},
        {"name": "Bybit public price", "configured": settings.ENABLE_BYBIT, "observed": "bybit" in seen},
        {"name": "OKX public price", "configured": settings.ENABLE_OKX, "observed": "okx" in seen},
        {"name": "Chainlink BTC/USD TWAP 60s", "configured": settings.ENABLE_CHAINLINK_RTDS,
         "observed": "chainlink_twap_60s" in seen},
        {"name": "Pyth oracle", "configured": settings.ENABLE_PYTH, "observed": "pyth" in seen},
        {"name": "Telegram Bot", "configured": bool(getattr(api, "TELEGRAM_BOT_TOKEN", "")), "observed": False},
        {"name": "Hugging Face", "configured": bool(getattr(api, "HUGGINGFACE_API_TOKEN", "")) or not settings.QWEN_LOCAL_FILES_ONLY, "observed": False},
    ]


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator, (centre + margin) / denominator


def _extended_metrics(pnls: list[float], fees: float, started_at: str | None = None) -> dict[str, Any]:
    """Метрики сделки считаются по независимым событиям, а не по отдельным заявкам."""
    n = len(pnls)
    wins = sum(value > 0 for value in pnls)
    losses = sum(value < 0 for value in pnls)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    avg_win = gross_profit / wins if wins else 0.0
    avg_loss = gross_loss / losses if losses else 0.0
    net = sum(pnls)
    wilson_low, wilson_high = _wilson_interval(wins, n)
    pnl_mean = net / n if n else 0.0
    pnl_se = stdev(pnls) / math.sqrt(n) if n > 1 else 0.0
    elapsed_hours = 0.0
    if started_at:
        try:
            elapsed_hours = max(0.0, (datetime.now(UTC) - datetime.fromisoformat(started_at)).total_seconds() / 3600)
        except ValueError:
            pass
    peak = 0.0
    curve = 0.0
    max_drawdown_usdc = 0.0
    for pnl in pnls:
        curve += pnl
        peak = max(peak, curve)
        max_drawdown_usdc = max(max_drawdown_usdc, peak - curve)
    return {
        "trades": n, "wins": wins, "losses": losses,
        "gross_profit_usdc": gross_profit, "gross_loss_usdc": gross_loss,
        "net_pnl_usdc": net, "expectancy_usdc": pnl_mean,
        "win_rate": wins / n if n else 0.0,
        "wilson_lower": wilson_low, "wilson_upper": wilson_high,
        "pnl_mean_ci95_lower": pnl_mean - 1.96 * pnl_se,
        "pnl_mean_ci95_upper": pnl_mean + 1.96 * pnl_se,
        "profit_factor": gross_profit / gross_loss if gross_loss else (999.0 if gross_profit else 0.0),
        "average_win_usdc": avg_win, "average_loss_usdc": avg_loss,
        "payoff_ratio": avg_win / avg_loss if avg_loss else 0.0,
        "breakeven_win_rate": avg_loss / (avg_win + avg_loss) if avg_win + avg_loss else 0.0,
        "max_drawdown_usdc": max_drawdown_usdc,
        "recovery_factor": net / max_drawdown_usdc if max_drawdown_usdc else 0.0,
        "fees_usdc": fees, "fees_to_gross_profit": fees / gross_profit if gross_profit else 0.0,
        "trades_per_hour": n / elapsed_hours if elapsed_hours > 0 else 0.0,
        "statistically_positive": n >= settings.LIVE_MIN_RESOLVED_EVENTS and pnl_mean - 1.96 * pnl_se > 0,
    }


def model_comparison(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Сопоставляет paper-результаты моделей; каждая строка сделки знает entry-модель."""
    if not {"paper_positions", "paper_sessions", "model_decisions"}.issubset(table_names(connection)):
        return []
    rows = connection.execute(
        """SELECT p.event_slug,p.realized_pnl_usdc,p.fees_usdc,p.closed_at,
                  COALESCE(d.model_name,s.model_name,'unknown') AS bot_model,
                  COALESCE(d.provider,'unknown') AS bot_provider,
                  s.started_at,s.strategy_version,p.outcome,d.predicted_up_probability,d.predicted_down_probability
           FROM paper_positions p
           JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.status IN ('closed','resolved') AND COALESCE(p.execution_valid,1)=1
           ORDER BY COALESCE(p.closed_at,p.opened_at)"""
    ).fetchall()
    labels = {
        (str(row[0]), str(row[1])): int(row[2])
        for row in connection.execute(
            """SELECT event_slug,outcome,MAX(label)
               FROM training_examples GROUP BY event_slug,outcome"""
        ).fetchall()
    } if "training_examples" in table_names(connection) else {}
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        provider = str(row["bot_provider"])
        versioned_model = f"{row['bot_model']} · {row['strategy_version'] or 'legacy'} · {provider}"
        grouped[(versioned_model, provider)].append(row)
    result = []
    for (model, provider), items in grouped.items():
        by_event: dict[str, float] = {}
        for row in items:
            slug = str(row["event_slug"])
            by_event[slug] = by_event.get(slug, 0.0) + float(row["realized_pnl_usdc"] or 0)
        metrics = _extended_metrics(
            list(by_event.values()),
            sum(float(row["fees_usdc"] or 0) for row in items),
            min((str(row["started_at"]) for row in items), default=None),
        )
        event_directions: dict[str, str] = {}
        for row in items:
            event_directions.setdefault(str(row["event_slug"]), str(row["outcome"]))
        up_entries = sum(value == "Up" for value in event_directions.values())
        down_entries = sum(value == "Down" for value in event_directions.values())
        calibration = []
        for row in items:
            outcome = str(row["outcome"])
            probability = (
                row["predicted_up_probability"] if outcome == "Up"
                else row["predicted_down_probability"]
            )
            label = labels.get((str(row["event_slug"]), outcome))
            if probability is not None and label is not None:
                calibration.append((int(label), min(1 - 1e-6, max(1e-6, float(probability)))))
        live_quality: dict[str, Any] = {
            "calibration_events": len(calibration), "live_roc_auc": None, "live_pr_auc": None,
            "live_brier": None, "live_log_loss": None,
        }
        if calibration:
            y_true = [item[0] for item in calibration]
            y_prob = [item[1] for item in calibration]
            live_quality["live_brier"] = float(brier_score_loss(y_true, y_prob))
            live_quality["live_log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
            if len(set(y_true)) == 2:
                live_quality["live_roc_auc"] = float(roc_auc_score(y_true, y_prob))
                live_quality["live_pr_auc"] = float(average_precision_score(y_true, y_prob))
        result.append({
            "model": model, "provider": provider, **metrics, **live_quality,
            "up_entries": up_entries, "down_entries": down_entries,
            "max_direction_share": max(up_entries, down_entries) / max(1, up_entries + down_entries),
        })
    return sorted(result, key=lambda item: (item["net_pnl_usdc"], item["trades"]), reverse=True)


def model_equity_curves(connection: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Строит сопоставимые кривые моделей: каждая начинается с виртуальных $300."""
    if not {"paper_positions", "paper_sessions", "model_decisions"}.issubset(table_names(connection)):
        return {}
    rows = connection.execute(
        """SELECT p.event_slug,p.closed_at,p.opened_at,p.realized_pnl_usdc,
                  COALESCE(d.model_name,s.model_name,'unknown') AS bot_model,
                  COALESCE(d.provider,'unknown') AS bot_provider,s.strategy_version
           FROM paper_positions p
           JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.status IN ('closed','resolved') AND COALESCE(p.execution_valid,1)=1
           ORDER BY COALESCE(p.closed_at,p.opened_at),p.id"""
    ).fetchall()
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model = f"{row['bot_model']} · {row['strategy_version'] or 'legacy'} · {row['bot_provider']}"
        slug = str(row["event_slug"])
        point = grouped[model].setdefault(slug, {
            "t": str(row["closed_at"] or row["opened_at"]), "event": slug, "pnl": 0.0,
        })
        point["pnl"] += float(row["realized_pnl_usdc"] or 0.0)
    curves: dict[str, list[dict[str, Any]]] = {}
    for model, events in grouped.items():
        equity = float(settings.PAPER_INITIAL_BALANCE_USDC)
        points = [{"index": 0, "equity": equity, "pnl": 0.0, "event": "start", "t": None}]
        for index, point in enumerate(events.values(), start=1):
            equity += float(point["pnl"])
            points.append({"index": index, "equity": equity, **point})
        curves[model] = points
    return curves


_MODEL_ANALYTICS_CACHE: dict[str, Any] = {
    "at": 0.0,
    "value": {"paper_comparison": [], "equity_curves": {}, "health": {}},
    "refreshing": False,
}
_MODEL_ANALYTICS_LOCK = threading.Lock()


def _refresh_model_analytics() -> None:
    """Считает bootstrap/кривые вне HTTP-запроса и публикует готовый JSON."""
    connection = connect()
    try:
        value = ({"paper_comparison": [], "equity_curves": {}, "health": {}}
                 if connection is None else {
            "paper_comparison": model_comparison(connection),
            "equity_curves": model_equity_curves(connection),
            "health": model_health(connection),
        })
        with _MODEL_ANALYTICS_LOCK:
            _MODEL_ANALYTICS_CACHE["value"] = value
            _MODEL_ANALYTICS_CACHE["at"] = time.monotonic()
    finally:
        if connection is not None:
            connection.close()
        with _MODEL_ANALYTICS_LOCK:
            _MODEL_ANALYTICS_CACHE["refreshing"] = False


def cached_model_analytics() -> dict[str, Any]:
    """Немедленно отдаёт cache; просроченную аналитику обновляет в фоне."""
    with _MODEL_ANALYTICS_LOCK:
        stale = time.monotonic() - float(_MODEL_ANALYTICS_CACHE["at"]) >= 300.0
        if stale and not _MODEL_ANALYTICS_CACHE["refreshing"]:
            _MODEL_ANALYTICS_CACHE["refreshing"] = True
            threading.Thread(
                target=_refresh_model_analytics,
                name="dashboard-model-analytics",
                daemon=True,
            ).start()
        return dict(_MODEL_ANALYTICS_CACHE["value"])


def paper_overview(connection: sqlite3.Connection | None) -> dict[str, Any]:
    empty = {
        "status": "not_started", "initial_balance": settings.PAPER_INITIAL_BALANCE_USDC,
        "cash": settings.PAPER_INITIAL_BALANCE_USDC, "equity": settings.PAPER_INITIAL_BALANCE_USDC,
        "realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_wagered": 0.0,
        "open_positions": [], "recent_positions": [], "recent_actions": [], "resolved_events": 0,
        "misalignment_signals": 0, "misalignment_executed": 0, "misalignment_pnl": 0.0,
        "metrics": {}, "loss_streak": 0, "fees": 0.0, "counterfactual": {}, "ml_policy_metrics": {},
        "execution": {"orders": 0, "filled": 0, "partial": 0, "unfilled": 0, "fill_rate": 0.0,
                      "average_latency_ms": 0.0, "average_entry_price": None,
                      "weighted_average_entry_price": None, "entry_fills": 0,
                      "entry_filled_shares": 0.0, "average_entry_latency_ms": None,
                      "average_entry_slippage_bps": None, "average_exit_price": None,
                      "weighted_average_exit_price": None, "exit_fills": 0,
                      "exit_filled_shares": 0.0, "average_exit_latency_ms": None,
                      "average_exit_slippage_bps": None},
    }
    if connection is None or "paper_sessions" not in table_names(connection):
        return empty
    session = connection.execute("SELECT * FROM paper_sessions ORDER BY started_at DESC LIMIT 1").fetchone()
    if not session:
        return empty
    session_data = dict(session)
    equity = connection.execute(
        "SELECT * FROM paper_equity_snapshots WHERE session_id=? ORDER BY id DESC LIMIT 1",
        (session_data["session_id"],),
    ).fetchone()
    positions = [dict(row) for row in connection.execute(
        """SELECT p.*,COALESCE(d.model_name,s.model_name,'unknown') AS bot_model,
                  COALESCE(d.provider,'unknown') AS bot_provider
           FROM paper_positions p
           JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.session_id=? ORDER BY p.id DESC LIMIT 25""", (session_data["session_id"],)
    ).fetchall()]
    all_positions = [dict(row) for row in connection.execute(
        "SELECT * FROM paper_positions WHERE session_id=? ORDER BY id",
        (session_data["session_id"],),
    ).fetchall()]
    valid_positions = [row for row in all_positions if int(row.get("execution_valid", 1) or 0) == 1]
    invalid_positions = [row for row in all_positions if int(row.get("execution_valid", 1) or 0) == 0]
    actions = [dict(row) for row in connection.execute(
        """SELECT observed_at,event_slug,action,confidence,reason,tags_json,executed,
                  predicted_up_probability,predicted_down_probability,expected_net_edge
           FROM model_decisions WHERE session_id=? ORDER BY id DESC LIMIT 40""", (session_data["session_id"],)
    ).fetchall()]
    # Оперативная панель показывает rolling-окно. Полный исторический разбор
    # остаётся в offline-отчётах: скан миллионов WAIT каждые 15 секунд раньше
    # задерживал интерфейс и конкурировал за CPU с торговлей.
    ml_rows = [dict(row) for row in connection.execute(
        """SELECT action,confidence,tags_json,executed FROM (
             SELECT session_id,action,confidence,tags_json,executed
             FROM model_decisions ORDER BY id DESC LIMIT 10000
           ) WHERE session_id=? AND tags_json LIKE '%ml_autonomous_policy%'""",
        (session_data["session_id"],),
    ).fetchall()]
    ml_policy_metrics: dict[str, Any] = {
        "decisions": len(ml_rows), "executed": sum(int(row.get("executed") or 0) for row in ml_rows),
        "actions": {}, "average_confidence": 0.0, "average_utility_gap": 0.0,
        "average_best_utility": 0.0, "limit_levels": {},
    }
    gaps: list[float] = []
    best_values: list[float] = []
    confidences: list[float] = []
    for row in ml_rows:
        action = str(row.get("action") or "UNKNOWN")
        ml_policy_metrics["actions"][action] = ml_policy_metrics["actions"].get(action, 0) + 1
        confidences.append(float(row.get("confidence") or 0.0))
        try:
            tags = json.loads(row.get("tags_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            tags = []
        for tag in tags:
            if tag.startswith("utility_gap="):
                gaps.append(float(tag.split("=", 1)[1]))
            elif tag.startswith("best_utility="):
                best_values.append(float(tag.split("=", 1)[1]))
            elif tag.startswith("selected_limit_level="):
                level = tag.split("=", 1)[1]
                ml_policy_metrics["limit_levels"][level] = ml_policy_metrics["limit_levels"].get(level, 0) + 1
    ml_policy_metrics["average_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    ml_policy_metrics["average_utility_gap"] = sum(gaps) / len(gaps) if gaps else 0.0
    ml_policy_metrics["average_best_utility"] = sum(best_values) / len(best_values) if best_values else 0.0
    utility_pairs: list[tuple[float, float]] = []
    quality_rows = connection.execute(
        """SELECT d.tags_json,p.realized_pnl_usdc FROM paper_positions p
           JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.session_id=? AND p.status IN ('closed','resolved')
             AND COALESCE(p.execution_valid,1)=1
             AND d.tags_json LIKE '%ml_autonomous_policy%'""",
        (session_data["session_id"],),
    ).fetchall()
    for quality_row in quality_rows:
        try:
            quality_tags = json.loads(quality_row[0] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        predicted = next(
            (float(tag.split("=", 1)[1]) for tag in quality_tags if tag.startswith("best_utility=")),
            None,
        )
        if predicted is not None:
            utility_pairs.append((predicted, float(quality_row[1] or 0.0)))
    if utility_pairs:
        errors = [abs(predicted - actual) for predicted, actual in utility_pairs]
        ml_policy_metrics["quality"] = {
            "evaluated_trades": len(utility_pairs),
            "utility_mae_usdc": sum(errors) / len(errors),
            "utility_bias_usdc": sum(predicted - actual for predicted, actual in utility_pairs) / len(utility_pairs),
            "utility_sign_accuracy": sum((predicted > 0) == (actual > 0) for predicted, actual in utility_pairs) / len(utility_pairs),
            "average_predicted_utility_usdc": sum(predicted for predicted, _ in utility_pairs) / len(utility_pairs),
            "average_realized_pnl_usdc": sum(actual for _, actual in utility_pairs) / len(utility_pairs),
        }
    else:
        ml_policy_metrics["quality"] = {
            "evaluated_trades": 0, "utility_mae_usdc": None, "utility_bias_usdc": None,
            "utility_sign_accuracy": None, "average_predicted_utility_usdc": None,
            "average_realized_pnl_usdc": None,
        }
    resolved = connection.execute(
        "SELECT COUNT(DISTINCT event_slug) FROM paper_positions WHERE session_id=? AND status IN ('closed','resolved') AND execution_valid=1",
        (session_data["session_id"],),
    ).fetchone()[0]
    misalignment = connection.execute(
        """SELECT COUNT(*),SUM(executed) FROM (
             SELECT session_id,tags_json,executed FROM model_decisions
             ORDER BY id DESC LIMIT 10000
           ) WHERE session_id=? AND tags_json LIKE '%consensus_misalignment%'""",
        (session_data["session_id"],)
    ).fetchone()
    misalignment_pnl = connection.execute(
        """SELECT COALESCE(SUM(p.realized_pnl_usdc),0) FROM paper_positions p
           JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.session_id=? AND COALESCE(p.execution_valid,1)=1
             AND d.tags_json LIKE '%consensus_misalignment%'""", (session_data["session_id"],)
    ).fetchone()[0]
    stats = paper_statistics(connection, session_data["session_id"])
    closed_pnls = [float(row[0] or 0) for row in connection.execute(
        "SELECT realized_pnl_usdc FROM paper_positions WHERE session_id=? AND status IN ('closed','resolved') AND execution_valid=1",
        (session_data["session_id"],),
    ).fetchall()]
    avg_win = sum(x for x in closed_pnls if x > 0) / max(1, sum(x > 0 for x in closed_pnls))
    avg_loss = abs(sum(x for x in closed_pnls if x < 0) / max(1, sum(x < 0 for x in closed_pnls)))
    counterfactual = {}
    if "counterfactual_entries" in table_names(connection):
        # Полная counterfactual-агрегация читает гигабайты и не относится к
        # оперативному контуру. Она строится отдельным offline-отчётом.
        counterfactual = {"deferred_to_offline_report": True}
    total_wagered = sum(float(row.get("cost_usdc") or 0) for row in valid_positions)
    settled_statuses = {"closed", "resolved", "provisionally_resolved"}
    valid_realized = sum(
        float(row.get("realized_pnl_usdc") or 0) for row in valid_positions
        if row.get("status") in settled_statuses
    )
    invalid_realized = sum(
        float(row.get("realized_pnl_usdc") or 0) for row in invalid_positions
        if row.get("status") in settled_statuses
    )
    valid_open = [row for row in valid_positions if row.get("status") == "open"]
    valid_unrealized = sum(
        float(row.get("shares") or 0) * float(row.get("current_price") or row.get("average_price") or 0)
        - float(row.get("cost_usdc") or 0) for row in valid_open
    )
    valid_cash = float(session_data["initial_balance_usdc"]) + valid_realized - sum(
        float(row.get("cost_usdc") or 0) for row in valid_open
    )
    execution = {"orders": 0, "filled": 0, "partial": 0, "unfilled": 0, "fill_rate": 0.0,
                 "average_latency_ms": 0.0, "average_entry_price": None,
                 "weighted_average_entry_price": None, "entry_fills": 0,
                 "entry_filled_shares": 0.0, "average_entry_latency_ms": None,
                 "average_entry_slippage_bps": None, "average_exit_price": None,
                 "weighted_average_exit_price": None, "exit_fills": 0,
                 "exit_filled_shares": 0.0, "average_exit_latency_ms": None,
                 "average_exit_slippage_bps": None}
    order_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(paper_orders)")}
    if {"fill_probability", "latency_ms", "execution_reason"}.issubset(order_columns):
        execution_row = connection.execute(
            """SELECT COUNT(*) orders,
                      SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) filled,
                      SUM(CASE WHEN status IN ('partial','partially_filled','partial_cancelled') THEN 1 ELSE 0 END) partial,
                      SUM(CASE WHEN status NOT IN ('filled','partial','partially_filled','partial_cancelled') THEN 1 ELSE 0 END) unfilled,
                      1.0*SUM(CASE WHEN status IN ('filled','partial','partially_filled','partial_cancelled') THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) fill_rate,
                      AVG(fill_probability) average_fill_probability,
                      AVG(latency_ms) average_latency_ms,
                      AVG(observed_slippage_bps) average_observed_slippage_bps,
                      AVG(book_age_ms_at_fill) average_book_age_ms_at_fill
               FROM paper_orders WHERE session_id=?""",
            (session_data["session_id"],),
        ).fetchone()
        if execution_row:
            execution = {
                "orders": int(execution_row[0] or 0), "filled": int(execution_row[1] or 0),
                "partial": int(execution_row[2] or 0), "unfilled": int(execution_row[3] or 0),
                "fill_rate": float(execution_row[4] or 0),
                "average_fill_probability": float(execution_row[5] or 0),
                "average_latency_ms": float(execution_row[6] or 0),
                "average_observed_slippage_bps": float(execution_row[7] or 0),
                "average_book_age_ms_at_fill": float(execution_row[8] or 0),
                "fill_probability_kind": settings.EXECUTION_FILL_PROBABILITY_KIND,
            }
        entry_price_row = connection.execute(
            """SELECT COUNT(*) entry_fills,
                      AVG(COALESCE(filled_price,requested_price)) average_entry_price,
                      SUM(COALESCE(filled_price,requested_price)*shares)/NULLIF(SUM(shares),0)
                          weighted_average_entry_price,
                      SUM(shares) entry_filled_shares,
                      AVG(latency_ms) average_entry_latency_ms,
                      AVG(observed_slippage_bps) average_entry_slippage_bps
               FROM paper_orders
               WHERE session_id=? AND action LIKE 'BUY_%' AND shares>0
                 AND COALESCE(execution_valid,1)=1""",
            (session_data["session_id"],),
        ).fetchone()
        if entry_price_row:
            execution.update({
                "entry_fills": int(entry_price_row[0] or 0),
                "average_entry_price": (
                    float(entry_price_row[1]) if entry_price_row[1] is not None else None
                ),
                "weighted_average_entry_price": (
                    float(entry_price_row[2]) if entry_price_row[2] is not None else None
                ),
                "entry_filled_shares": float(entry_price_row[3] or 0.0),
                "average_entry_latency_ms": (
                    float(entry_price_row[4]) if entry_price_row[4] is not None else None
                ),
                "average_entry_slippage_bps": (
                    float(entry_price_row[5]) if entry_price_row[5] is not None else None
                ),
            })
        exit_price_row = connection.execute(
            """SELECT COUNT(*) exit_fills,
                      AVG(COALESCE(filled_price,requested_price)) average_exit_price,
                      SUM(COALESCE(filled_price,requested_price)*shares)/NULLIF(SUM(shares),0)
                          weighted_average_exit_price,
                      SUM(shares) exit_filled_shares,
                      AVG(latency_ms) average_exit_latency_ms,
                      AVG(observed_slippage_bps) average_exit_slippage_bps
               FROM paper_orders
               WHERE session_id=? AND action IN ('CLOSE','PARTIAL_CLOSE','SELL') AND shares>0
                 AND COALESCE(execution_valid,1)=1""",
            (session_data["session_id"],),
        ).fetchone()
        if exit_price_row:
            execution.update({
                "exit_fills": int(exit_price_row[0] or 0),
                "average_exit_price": (
                    float(exit_price_row[1]) if exit_price_row[1] is not None else None
                ),
                "weighted_average_exit_price": (
                    float(exit_price_row[2]) if exit_price_row[2] is not None else None
                ),
                "exit_filled_shares": float(exit_price_row[3] or 0.0),
                "average_exit_latency_ms": (
                    float(exit_price_row[4]) if exit_price_row[4] is not None else None
                ),
                "average_exit_slippage_bps": (
                    float(exit_price_row[5]) if exit_price_row[5] is not None else None
                ),
            })
    extended = _extended_metrics(
        closed_pnls, float(session_data.get("total_fees_usdc", 0) or 0), session_data.get("started_at")
    )
    recent_orders = [dict(row) for row in connection.execute(
        """SELECT id,event_slug,action,order_type,requested_price,filled_price,
                  requested_shares,shares,requested_notional_usdc,notional_usdc,fee_usdc,
                  status,created_at,expiration_at,price_cap,fill_probability,latency_ms,
                  execution_reason,execution_valid,invalid_reason,submit_best_bid,submit_best_ask,
                  fill_best_bid,fill_best_ask,fill_observed_at,observed_slippage_bps,
                  book_age_ms_at_submit,book_age_ms_at_fill,fill_probability_kind
           FROM paper_orders WHERE session_id=? ORDER BY id DESC LIMIT 20""",
        (session_data["session_id"],),
    ).fetchall()]
    enrich_events(positions)
    enrich_events(actions)
    enrich_events(valid_open)
    enrich_events(recent_orders)
    metrics = {
        **stats, "roi_pct": float(stats["net_pnl_usdc"]) / total_wagered * 100 if total_wagered else 0.0,
        "average_win_usdc": avg_win, "average_loss_usdc": avg_loss,
        "payoff_ratio": avg_win / avg_loss if avg_loss else 0.0,
        **extended,
    }
    exit_comparison = {
        str(row[0]): {"events": int(row[1] or 0), "net_pnl_usdc": float(row[2] or 0),
                     "average_pnl_usdc": float(row[3] or 0)}
        for row in connection.execute(
            """SELECT exit_timing,COUNT(*),SUM(COALESCE(realized_pnl_usdc,0)),
                      AVG(COALESCE(realized_pnl_usdc,0))
               FROM paper_positions WHERE session_id=? AND status IN ('closed','resolved')
                 AND COALESCE(execution_valid,1)=1 GROUP BY exit_timing""",
            (session_data["session_id"],),
        )
    }
    return {
        "status": session_data["status"], "session_id": session_data["session_id"],
        "initial_balance": session_data["initial_balance_usdc"], "cash": valid_cash,
        "equity": float(session_data["initial_balance_usdc"]) + valid_realized + valid_unrealized,
        "realized_pnl": valid_realized, "unrealized_pnl": valid_unrealized,
        "total_wagered": total_wagered, "open_positions": valid_open,
        "recent_positions": positions, "recent_actions": actions, "resolved_events": resolved,
        "execution_reality": {
            "raw_realized_pnl": float(session_data["realized_pnl_usdc"] or 0),
            "valid_realized_pnl": valid_realized, "invalid_realized_pnl": invalid_realized,
            "valid_positions": len(valid_positions), "invalid_positions": len(invalid_positions),
            "invalid_reasons": dict(Counter(str(row.get("invalid_reason") or "unknown") for row in invalid_positions)),
        },
        "provisional_events": sum(row.get("status") == "provisionally_resolved" for row in valid_positions),
        "provisional_realized_pnl": sum(
            float(row.get("realized_pnl_usdc") or 0) for row in valid_positions
            if row.get("status") == "provisionally_resolved"
        ),
        "misalignment_signals": int(misalignment[0] or 0), "misalignment_executed": int(misalignment[1] or 0),
        "misalignment_pnl": float(misalignment_pnl or 0.0),
        "strategy": session_data["strategy_version"], "model": session_data["model_name"],
        "metrics": metrics, "loss_streak": int(session_data.get("consecutive_losses", 0) or 0),
        "fees": float(session_data.get("total_fees_usdc", 0) or 0), "counterfactual": counterfactual,
        "execution": execution, "recent_orders": recent_orders, "ml_policy_metrics": ml_policy_metrics,
        "exit_comparison": exit_comparison,
        "stopped_reason": session_data.get("stopped_reason"), "run_label": session_data.get("run_label"),
    }


@_cached(60.0)
def next_event_overview(connection: sqlite3.Connection | None) -> dict[str, Any]:
    empty = {"enabled": settings.NEXT_EVENT_CONTEXT_ENABLED, "latest": None, "evaluated": 0,
             "direction_accuracy": None, "hypothetical_fills": 0, "hypothetical_pnl_usdc": 0.0,
             "fill_rate": None, "expectancy_per_fill_usdc": None, "raw_samples": 0,
             "live_preopen_enabled": settings.NEXT_EVENT_PREOPEN_LIVE_ENABLED,
             "active_shadow_orders": [], "shadow_order_history": {}}
    if connection is None or "next_event_forecasts" not in table_names(connection):
        return empty
    latest_row = connection.execute(
        "SELECT * FROM next_event_forecasts WHERE status!='invalid_after_start' ORDER BY observed_at DESC LIMIT 1"
    ).fetchone()
    # 15-секундные наблюдения одного будущего события — это временной ряд,
    # а не независимые сделки. Для честного отчёта берём последний прогноз
    # перед стартом каждого next_event ровно один раз.
    summary = connection.execute(
        """SELECT COUNT(*) events,
                  AVG(CASE WHEN (f.predicted_direction='Up' AND f.next_resolved_label=1)
                                OR (f.predicted_direction='Down' AND f.next_resolved_label=0)
                           THEN 1.0 ELSE 0.0 END) direction_accuracy,
                  SUM(COALESCE(f.hypothetical_filled,0)) fills,
                  SUM(CASE WHEN f.hypothetical_filled=1 THEN COALESCE(f.hypothetical_pnl_usdc,0) ELSE 0 END) pnl
           FROM next_event_forecasts f
           JOIN (SELECT next_event_slug,MAX(id) id FROM next_event_forecasts
                 WHERE status='evaluated' AND unixepoch(observed_at)<
                       CAST(SUBSTR(next_event_slug,INSTR(next_event_slug,'5m-')+3) AS INTEGER)
                 GROUP BY next_event_slug) last ON last.id=f.id"""
    ).fetchone()
    raw_samples = int(connection.execute(
        "SELECT COUNT(*) FROM next_event_forecasts WHERE status='evaluated'"
    ).fetchone()[0])
    latest = dict(latest_row) if latest_row else None
    if latest:
        latest.update(event_metadata(latest.get("next_event_slug")))
        latest["source_event"] = event_metadata(latest.get("source_event_slug"))
    events = int(summary[0] or 0)
    fills = int(summary[2] or 0)
    pnl = float(summary[3] or 0.0)
    active_orders: list[dict[str, Any]] = []
    lifecycle: dict[str, int] = {}
    if "shadow_preopen_orders" in table_names(connection):
        active_orders = [dict(row) for row in connection.execute(
            """SELECT * FROM shadow_preopen_orders
               WHERE status IN ('working','filled_shadow')
                 AND CAST(SUBSTR(next_event_slug,INSTR(next_event_slug,'5m-')+3) AS INTEGER)>unixepoch('now')
               ORDER BY id DESC LIMIT 20"""
        ).fetchall()]
        for order in active_orders:
            order.update(event_metadata(order.get("next_event_slug")))
        lifecycle = {str(row[0]): int(row[1]) for row in connection.execute(
            "SELECT status,COUNT(*) FROM shadow_preopen_orders GROUP BY status"
        )}
    return {
        **empty, "latest": latest, "evaluated": events,
        "direction_accuracy": float(summary[1]) if summary[1] is not None else None,
        "hypothetical_fills": fills, "hypothetical_pnl_usdc": pnl,
        "fill_rate": fills / events if events else None,
        "expectancy_per_fill_usdc": pnl / fills if fills else None,
        "raw_samples": raw_samples,
        "active_shadow_orders": active_orders, "shadow_order_history": lifecycle,
    }
def _display_settle_ended_live_positions(
    connection: sqlite3.Connection, positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Строит read-only economic settlement, пока engine ждёт официальный label."""
    current = datetime.now(UTC)
    tables = set(table_names(connection))
    if "market_snapshots" not in tables:
        return positions
    result: list[dict[str, Any]] = []
    for original in positions:
        row = dict(original)
        if row.get("status") != "open":
            result.append(row)
            continue
        try:
            event_end_epoch = int(str(row["event_slug"]).rsplit("-", 1)[-1]) + 300
            event_end = datetime.fromtimestamp(event_end_epoch, UTC)
        except (KeyError, TypeError, ValueError, OverflowError):
            result.append(row)
            continue
        if (current - event_end).total_seconds() < settings.PAPER_POST_EVENT_SETTLEMENT_DELAY_SECONDS:
            result.append(row)
            continue
        cutoff = (event_end - timedelta(
            seconds=float(settings.PAPER_POST_EVENT_MAX_QUOTE_DISTANCE_SECONDS)
        )).isoformat()
        quotes = connection.execute(
            """SELECT outcome,best_bid,best_ask FROM market_snapshots
               WHERE event_slug=? AND collected_at>=? ORDER BY collected_at DESC,id DESC""",
            (row["event_slug"], cutoff),
        ).fetchall()
        latest: dict[str, sqlite3.Row] = {}
        for quote in quotes:
            latest.setdefault(str(quote["outcome"]), quote)
        outcome = str(row.get("outcome"))
        held = latest.get(outcome)
        other = latest.get("Down" if outcome == "Up" else "Up")
        label: int | None = None
        source: str | None = None
        if held is not None:
            held_bid = float(held["best_bid"] or 0.0)
            held_ask = float(held["best_ask"] or 1.0)
            other_bid = float(other["best_bid"] or 0.0) if other is not None else 0.0
            if held_bid >= settings.PAPER_POST_EVENT_WIN_BID_THRESHOLD:
                label, source = 1, "dashboard_post_event_extreme_quote"
            elif (held_ask <= settings.PAPER_POST_EVENT_LOSS_ASK_THRESHOLD
                  or other_bid >= settings.PAPER_POST_EVENT_WIN_BID_THRESHOLD):
                label, source = 0, "dashboard_post_event_extreme_quote"
        if label is None and "reference_price_snapshots" in tables:
            reference = connection.execute(
                """SELECT target_price,reference_price FROM reference_price_snapshots
                   WHERE event_slug=? AND reference_price IS NOT NULL
                     AND ABS(unixepoch(collected_at)-?)<=?
                   ORDER BY ABS(unixepoch(collected_at)-?) ASC,id DESC LIMIT 1""",
                (row["event_slug"], event_end_epoch,
                 float(settings.PAPER_POST_EVENT_MAX_QUOTE_DISTANCE_SECONDS), event_end_epoch),
            ).fetchone()
            if reference is not None:
                up_label = int(float(reference["reference_price"]) >= float(reference["target_price"]))
                label = up_label if outcome == "Up" else 1 - up_label
                source = "dashboard_post_event_reference_price"
        if label is not None:
            row.update({
                "status": "provisionally_resolved", "current_price": float(label),
                "close_price": float(label),
                "realized_pnl_usdc": float(row.get("shares") or 0) * label
                                      - float(row.get("cost_usdc") or 0),
                "dashboard_provisional": True, "settlement_source": source,
            })
        result.append(row)
    return result


def live_overview(connection: sqlite3.Connection | None, initial_balance: float) -> dict[str, Any]:
    """Локальный LIVE-ledger, отделённый от demo-сессии и бумажного капитала."""
    empty = {
        "status": "live", "initial_balance": initial_balance, "cash": initial_balance,
        "equity": initial_balance, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
        "total_wagered": 0.0, "open_positions": [], "recent_positions": [],
        "recent_actions": [], "resolved_events": 0, "misalignment_executed": 0,
        "misalignment_pnl": 0.0, "metrics": {}, "loss_streak": 0, "fees": 0.0,
        "execution": {"orders": 0, "filled": 0, "partial": 0, "unfilled": 0, "fill_rate": 0.0,
                      "average_latency_ms": 0.0, "average_entry_price": None,
                      "weighted_average_entry_price": None, "entry_fills": 0,
                      "entry_filled_shares": 0.0, "average_entry_latency_ms": None,
                      "average_entry_slippage_bps": None, "average_exit_price": None,
                      "weighted_average_exit_price": None, "exit_fills": 0,
                      "exit_filled_shares": 0.0, "average_exit_latency_ms": None,
                      "average_exit_slippage_bps": None},
        "model": "unknown", "source": "live", "balance_note": "локальный ledger, не баланс кошелька",
    }
    if connection is None or "live_positions" not in table_names(connection):
        return empty
    mode_since = connection.execute(
        "SELECT updated_at FROM runtime_controls WHERE control_key='trading_mode' AND control_value='live'"
    ).fetchone()
    action_since = str(mode_since[0]) if mode_since else "1970-01-01T00:00:00+00:00"
    positions = [dict(row) for row in connection.execute(
        """SELECT p.*,COALESCE(d.model_name,'unknown') bot_model,
                  COALESCE(d.provider,'unknown') bot_provider
           FROM live_positions p LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.opened_at>=? ORDER BY p.opened_at DESC,p.id DESC""",
        (action_since,),
    ).fetchall()]
    positions = [row for row in positions if int(row.get("execution_valid", 1) or 0) == 1]
    positions = _display_settle_ended_live_positions(connection, positions)
    if not positions:
        return empty
    actions = [dict(row) for row in connection.execute(
        """SELECT observed_at,event_slug,action,confidence,reason,tags_json,executed,
                  predicted_up_probability,predicted_down_probability,expected_net_edge,
                  model_name,provider
           FROM model_decisions WHERE observed_at>=? ORDER BY id DESC LIMIT 100""",
        (action_since,),
    ).fetchall()]
    # LIVE-пятиминутка может иметь экономически однозначный результат по
    # финальному 0.99/0.01 раньше часовой официальной разметки. Такой результат
    # показываем отдельно как provisional и позднее автоматически сверяем.
    closed = [row for row in positions if row.get("status") in {
        "closed", "resolved", "provisionally_resolved",
    }]
    opened = [row for row in positions if row.get("status") == "open"]
    realized = sum(float(row.get("realized_pnl_usdc") or 0) for row in closed)
    unrealized = sum(
        float(row.get("shares") or 0) * float(row.get("current_price") or row.get("average_price") or 0)
        - float(row.get("cost_usdc") or 0) for row in opened
    )
    wagered = sum(float(row.get("cost_usdc") or 0) for row in positions)
    pnls = [float(row.get("realized_pnl_usdc") or 0) for row in closed]
    live_fees = sum(float(row.get("fees_usdc") or 0) for row in positions)
    extended = _extended_metrics(pnls, live_fees, min(str(row["opened_at"]) for row in positions))
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    metrics = {
        **extended, "resolved_events": len({row["event_slug"] for row in closed}),
        "net_pnl_usdc": realized, "win_rate": sum(value > 0 for value in pnls) / len(pnls) if pnls else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else gross_profit,
        "expectancy_usdc": sum(pnls) / len(pnls) if pnls else 0.0,
        "up_entries": sum(row.get("outcome") == "Up" for row in positions),
        "down_entries": sum(row.get("outcome") == "Down" for row in positions),
    }
    order_count = filled = partial = unfilled = 0
    entry_price_metrics = {
        "average_entry_price": None, "weighted_average_entry_price": None,
        "entry_fills": 0, "entry_filled_shares": 0.0,
        "average_entry_latency_ms": None, "average_entry_slippage_bps": None,
        "average_exit_price": None, "weighted_average_exit_price": None,
        "exit_fills": 0, "exit_filled_shares": 0.0,
        "average_exit_latency_ms": None, "average_exit_slippage_bps": None,
    }
    recent_orders: list[dict[str, Any]] = []
    if "live_orders" in table_names(connection):
        order_rows = connection.execute(
            "SELECT status,COUNT(*) FROM live_orders WHERE created_at>=? GROUP BY status", (action_since,),
        ).fetchall()
        recent_orders = [dict(row) for row in connection.execute(
            """SELECT id,event_slug,outcome,side,order_id,order_type,requested_price,
                      requested_size,matched_size,status,created_at,expiration_at,last_checked_at,
                      error,execution_valid,invalid_reason,average_fill_price,
                      fill_notional_usdc,fee_usdc,fill_source
               FROM live_orders WHERE created_at>=? ORDER BY id DESC LIMIT 30""",
            (action_since,),
        ).fetchall()]
        order_count = sum(int(row[1]) for row in order_rows)
        filled = sum(int(row[1]) for row in order_rows if row[0] == "filled")
        partial = sum(int(row[1]) for row in order_rows if row[0] in {"partial", "partially_filled", "partial_cancelled"})
        unfilled = order_count - filled - partial
        entry_price_row = connection.execute(
            """SELECT COUNT(*),
                      AVG(COALESCE(average_fill_price,requested_price)),
                      SUM(COALESCE(fill_notional_usdc,
                                   matched_size*COALESCE(average_fill_price,requested_price)))
                          /NULLIF(SUM(matched_size),0),
                      SUM(matched_size),
                      AVG((julianday(last_checked_at)-julianday(created_at))*86400000.0),
                      AVG((COALESCE(average_fill_price,requested_price)-requested_price)
                          /NULLIF(requested_price,0)*10000.0)
               FROM live_orders
               WHERE UPPER(side)='BUY' AND matched_size>0
                 AND COALESCE(execution_valid,1)=1 AND created_at>=?""",
            (action_since,),
        ).fetchone()
        if entry_price_row:
            entry_price_metrics = {
                "entry_fills": int(entry_price_row[0] or 0),
                "average_entry_price": (
                    float(entry_price_row[1]) if entry_price_row[1] is not None else None
                ),
                "weighted_average_entry_price": (
                    float(entry_price_row[2]) if entry_price_row[2] is not None else None
                ),
                "entry_filled_shares": float(entry_price_row[3] or 0.0),
                "average_entry_latency_ms": (
                    float(entry_price_row[4]) if entry_price_row[4] is not None else None
                ),
                "average_entry_slippage_bps": (
                    float(entry_price_row[5]) if entry_price_row[5] is not None else None
                ),
            }
        exit_price_row = connection.execute(
            """SELECT COUNT(*),
                      AVG(COALESCE(average_fill_price,requested_price)),
                      SUM(COALESCE(fill_notional_usdc,
                                   matched_size*COALESCE(average_fill_price,requested_price)))
                          /NULLIF(SUM(matched_size),0),
                      SUM(matched_size),
                      AVG((julianday(last_checked_at)-julianday(created_at))*86400000.0),
                      AVG((requested_price-COALESCE(average_fill_price,requested_price))
                          /NULLIF(requested_price,0)*10000.0)
               FROM live_orders
               WHERE UPPER(side)='SELL' AND matched_size>0
                 AND COALESCE(execution_valid,1)=1 AND created_at>=?""",
            (action_since,),
        ).fetchone()
        if exit_price_row:
            entry_price_metrics.update({
                "exit_fills": int(exit_price_row[0] or 0),
                "average_exit_price": (
                    float(exit_price_row[1]) if exit_price_row[1] is not None else None
                ),
                "weighted_average_exit_price": (
                    float(exit_price_row[2]) if exit_price_row[2] is not None else None
                ),
                "exit_filled_shares": float(exit_price_row[3] or 0.0),
                "average_exit_latency_ms": (
                    float(exit_price_row[4]) if exit_price_row[4] is not None else None
                ),
                "average_exit_slippage_bps": (
                    float(exit_price_row[5]) if exit_price_row[5] is not None else None
                ),
            })
    model = next((row.get("bot_model") for row in positions if row.get("bot_model") != "unknown"), "unknown")
    return {
        **empty, "initial_balance": initial_balance, "cash": initial_balance + realized - sum(float(row.get("cost_usdc") or 0) for row in opened),
        "equity": initial_balance + realized + unrealized, "realized_pnl": realized,
        "unrealized_pnl": unrealized, "total_wagered": wagered, "open_positions": opened,
        "recent_positions": positions[:50], "recent_actions": actions,
        "resolved_events": len({row["event_slug"] for row in closed}), "metrics": metrics,
        "execution": {"orders": order_count, "filled": filled, "partial": partial, "unfilled": unfilled,
                      "fill_rate": (filled + partial) / order_count if order_count else 0.0,
                      "average_fill_probability": None, "average_latency_ms": 0.0,
                      **entry_price_metrics},
        "recent_orders": recent_orders,
        "fees": live_fees,
        "model": model,
    }


def runtime_mode(connection: sqlite3.Connection | None) -> str:
    if connection is None or "runtime_controls" not in table_names(connection):
        return "paper"
    row = connection.execute("SELECT control_value FROM runtime_controls WHERE control_key='trading_mode'").fetchone()
    return str(row[0]) if row else "paper"


def runtime_control(connection: sqlite3.Connection | None, key: str, default: str) -> str:
    if connection is None or "runtime_controls" not in table_names(connection):
        return default
    row = connection.execute("SELECT control_value FROM runtime_controls WHERE control_key=?", (key,)).fetchone()
    return str(row[0]) if row else default


def runtime_control_reason(connection: sqlite3.Connection | None, key: str) -> str | None:
    if connection is None or "runtime_controls" not in table_names(connection):
        return None
    row = connection.execute("SELECT reason FROM runtime_controls WHERE control_key=?", (key,)).fetchone()
    return str(row[0]) if row and row[0] else None


def _active_position_count(connection: sqlite3.Connection) -> int:
    """Считает только актуальные paper/live-позиции, не включая старые архивные хвосты."""
    tables = set(table_names(connection))
    paper_open = 0
    if {"paper_sessions", "paper_positions"}.issubset(tables):
        latest = connection.execute(
            "SELECT session_id FROM paper_sessions WHERE status IN ('running','paused') ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if latest:
            paper_open = int(connection.execute(
                "SELECT COUNT(*) FROM paper_positions WHERE session_id=? AND status='open'", (str(latest[0]),)
            ).fetchone()[0])
    live_open = int(connection.execute(
        "SELECT COUNT(*) FROM live_positions WHERE status='open'"
    ).fetchone()[0]) if "live_positions" in tables else 0
    live_pending = int(connection.execute(
        "SELECT COUNT(*) FROM live_orders WHERE status IN ('submitted','live','partial')"
    ).fetchone()[0]) if "live_orders" in tables else 0
    return paper_open + live_open + live_pending


def _ensure_model_selection_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS model_selection_history(
             id INTEGER PRIMARY KEY,session_id TEXT,changed_at TEXT NOT NULL,
             previous_model TEXT,new_model TEXT NOT NULL,source TEXT NOT NULL,
             role TEXT NOT NULL DEFAULT 'entry',state TEXT NOT NULL DEFAULT 'applied',applied_at TEXT)"""
    )
    existing = {str(row[1]) for row in connection.execute("PRAGMA table_info(model_selection_history)")}
    for name, definition in {
        "role": "TEXT NOT NULL DEFAULT 'entry'",
        "state": "TEXT NOT NULL DEFAULT 'applied'",
        "applied_at": "TEXT",
    }.items():
        if name not in existing:
            try:
                connection.execute(f"ALTER TABLE model_selection_history ADD COLUMN {name} {definition}")
            except sqlite3.OperationalError as exc:
                # Два одновременных refresh могут увидеть старую схему до COMMIT первого запроса.
                if "duplicate column name" not in str(exc).lower():
                    raise


def _apply_due_model_selections(connection: sqlite3.Connection) -> None:
    """Применяет очередь после закрытия позиции или не позднее чем через пять минут."""
    flat = _active_position_count(connection) == 0
    current_time = datetime.now(UTC)
    for role in ("entry", "exit"):
        pending_key = f"pending_{role}_model"
        deadline_key = f"pending_{role}_model_apply_at"
        pending = runtime_control(connection, pending_key, "").strip()
        deadline_raw = runtime_control(connection, deadline_key, "").strip()
        if pending not in MODEL_SPECS:
            continue
        try:
            deadline = datetime.fromisoformat(deadline_raw)
        except ValueError:
            deadline = current_time
        if not flat and current_time < deadline:
            continue
        active_key = f"selected_{role}_model"
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
            (active_key, pending, current_time.isoformat(), "queued model switch applied"),
        )
        if role == "entry":
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('selected_model',?,?,?)",
                (pending, current_time.isoformat(), "legacy alias for selected_entry_model"),
            )
        connection.execute(
            "UPDATE model_selection_history SET state='applied',applied_at=? WHERE id=(SELECT MAX(id) FROM model_selection_history WHERE role=? AND new_model=? AND state='queued')",
            (current_time.isoformat(), role, pending),
        )
        connection.execute("DELETE FROM runtime_controls WHERE control_key IN (?,?)", (pending_key, deadline_key))
    connection.commit()


def engine_alive(connection: sqlite3.Connection | None) -> bool:
    if connection is None or "runtime_controls" not in table_names(connection):
        return False
    row = connection.execute(
        "SELECT updated_at FROM runtime_controls WHERE control_key='engine_heartbeat'"
    ).fetchone()
    age = age_seconds(str(row[0])) if row else None
    # Локальная Qwen на CPU/iGPU может отвечать десятки секунд; короткий порог
    # создавал ложный статус offline прямо во время inference.
    return age is not None and age <= max(120.0, settings.PAPER_POLL_SECONDS * 4)


def model_overview(connection: sqlite3.Connection | None) -> dict[str, Any]:
    if connection is not None:
        _ensure_model_selection_schema(connection)
        _apply_due_model_selections(connection)
    legacy = runtime_control(connection, "selected_model", settings.DEFAULT_TRADING_MODEL)
    entry_model = runtime_control(connection, "selected_entry_model", legacy)
    exit_model = runtime_control(connection, "selected_exit_model", entry_model)
    if entry_model not in MODEL_SPECS:
        entry_model = settings.DEFAULT_ENTRY_MODEL
    if exit_model not in MODEL_SPECS:
        exit_model = settings.DEFAULT_EXIT_MODEL
    report: dict[str, Any] = {}
    if settings.MODEL_COMPARISON_REPORT_PATH.exists():
        try:
            report = json.loads(settings.MODEL_COMPARISON_REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {"error": "Не удалось прочитать offline-отчёт"}
    walk_forward_v8: dict[str, Any] = {}
    if settings.WALK_FORWARD_V8_REPORT_PATH.exists():
        try:
            walk_forward_v8 = json.loads(settings.WALK_FORWARD_V8_REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            walk_forward_v8 = {"error": "walk-forward v8 report is unreadable"}
    walk_forward_v9: dict[str, Any] = {}
    if settings.WALK_FORWARD_V9_REPORT_PATH.exists():
        try:
            walk_forward_v9 = json.loads(settings.WALK_FORWARD_V9_REPORT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            walk_forward_v9 = {"error": "walk-forward v9 report is unreadable"}
    regime_entry_v5: dict[str, Any] = {}
    if settings.REGIME_ENTRY_V5_REPORT_PATH.exists():
        try:
            regime_entry_v5 = json.loads(
                settings.REGIME_ENTRY_V5_REPORT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            regime_entry_v5 = {"error": "regime-aware entry v5 report is unreadable"}
    history_context: dict[str, Any] = {}
    loss_reversal: dict[str, Any] = {}
    for path, target in (
        (settings.HISTORY_CONTEXT_REPORT_PATH, history_context),
        (settings.LOSS_REVERSAL_REPORT_PATH, loss_reversal),
    ):
        if path.exists():
            try:
                target.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                target["error"] = f"unreadable report: {path.name}"
    timeseries_challenger: dict[str, Any] = {}
    if settings.TIMESERIES_CHALLENGER_REPORT_PATH.exists():
        try:
            timeseries_challenger = json.loads(
                settings.TIMESERIES_CHALLENGER_REPORT_PATH.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            timeseries_challenger = {"error": "ARIMA/GARCH shadow report is unreadable"}
    latest_training: dict[str, Any] = {}
    candidates = sorted(
        settings.MODEL_CANDIDATE_DIR.glob("cycle_*/training_report.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        try:
            latest_training = json.loads(candidates[-1].read_text(encoding="utf-8"))
            latest_training["report_path"] = str(candidates[-1])
        except (OSError, json.JSONDecodeError):
            latest_training = {"error": "Не удалось прочитать отчёт challenger-обучения"}
    return {
        "active": entry_model,
        "active_name": get_model(entry_model).name,
        "entry_model": entry_model,
        "entry_model_name": get_model(entry_model).name,
        "entry_model_version": runtime_control_reason(connection, "selected_entry_model"),
        "exit_model": exit_model,
        "exit_model_name": get_model(exit_model).name,
        "exit_model_version": runtime_control_reason(connection, "selected_exit_model"),
        "pending_entry_model": runtime_control(connection, "pending_entry_model", "") or None,
        "pending_entry_apply_at": runtime_control(connection, "pending_entry_model_apply_at", "") or None,
        "pending_exit_model": runtime_control(connection, "pending_exit_model", "") or None,
        "pending_exit_apply_at": runtime_control(connection, "pending_exit_model_apply_at", "") or None,
        "available": public_models(),
        "offline_report": report,
        "walk_forward_v8": walk_forward_v8,
        "walk_forward_v9": walk_forward_v9,
        "regime_entry_v5": regime_entry_v5,
        "history_context": history_context,
        "loss_reversal": loss_reversal,
        "timeseries_challenger": timeseries_challenger,
        "latest_training_cycle": latest_training,
        "switch_allowed": connection is not None,
        "switch_policy": "immediate_when_flat_or_within_5_minutes",
        "qlora": {
            "dataset_ready": settings.QWEN_LORA_ADAPTER_PATH.parent.joinpath("train.jsonl").exists(),
            "adapter_ready": settings.QWEN_LORA_ADAPTER_PATH.joinpath("adapter_config.json").exists(),
            "status": "trained" if settings.QWEN_LORA_ADAPTER_PATH.joinpath("adapter_config.json").exists() else "dataset_only",
        },
    }


def dataset_overview(connection: sqlite3.Connection | None) -> dict[str, Any]:
    if connection is None or "training_examples" not in table_names(connection):
        return {"rows": 0, "independent_events": 0, "ready_for_training": False}
    events = int(connection.execute("SELECT COUNT(DISTINCT event_slug) FROM training_examples").fetchone()[0])
    return {
        "rows": count(connection, "training_examples"), "independent_events": events,
        "minimum_events": settings.TRAINING_MIN_INDEPENDENT_EVENTS,
        "ready_for_training": events >= settings.TRAINING_MIN_INDEPENDENT_EVENTS,
        "export_path": str(settings.EXPORT_DIR),
    }


def project_summary(
    database: dict[str, Any], paper: dict[str, Any], models: dict[str, Any],
    mode: str, engine_state: str, engine_process_alive: bool,
) -> dict[str, Any]:
    comparison = models.get("paper_comparison") or []
    most_biased = max(comparison, key=lambda row: float(row.get("max_direction_share") or 0), default=None)
    qlora = models.get("qlora") or {}
    counts = database.get("counts") or {}
    items = [
        {
            "label": "Процессы", "value": "работают" if engine_process_alive else "требуют запуска",
            "note": f"движок: {engine_state}; база: {'доступна' if database.get('exists') else 'нет'}",
            "level": "" if engine_process_alive else "bad",
        },
        {
            "label": "Торговля", "value": str(mode).upper(),
            "note": f"капитал ${float(paper.get('equity') or 0):.2f}; realized ${float(paper.get('realized_pnl') or 0):+.2f}; unrealized ${float(paper.get('unrealized_pnl') or 0):+.2f}",
            "level": "warn" if mode == "live" else "",
        },
        {
            "label": "Модели", "value": f"{models.get('entry_model_name','—')} → {models.get('exit_model_name','—')}",
            "note": "первая выбирает сторону входа, вторая — удержание и выход",
            "level": "",
        },
        {
            "label": "Данные", "value": f"{int(counts.get('training_examples') or 0):,} примеров",
            "note": f"{int(counts.get('events') or 0):,} событий; {int(counts.get('market_snapshots') or 0):,} снимков стакана",
            "level": "",
        },
        {
            "label": "QLoRA", "value": "адаптер готов" if qlora.get("adapter_ready") else "только датасет",
            "note": "обучение ещё не выполнено" if not qlora.get("adapter_ready") else "LoRA challenger доступен для теста",
            "level": "warn" if not qlora.get("adapter_ready") else "",
        },
    ]
    if most_biased:
        share = float(most_biased.get("max_direction_share") or 0)
        items.append({
            "label": "Баланс направлений", "value": f"{most_biased.get('model')}: {share:.0%} одна сторона",
            "note": "выше 80% считается диагностическим предупреждением, но не поводом искусственно инвертировать сигнал",
            "level": "bad" if share > 0.90 else "warn" if share > 0.80 else "",
        })
    return {"generated_at": datetime.now(UTC).isoformat(), "items": items}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "database": settings.DATABASE_PATH.exists(), "time": datetime.now(UTC).isoformat()}


def _fast_table_counts(connection: sqlite3.Connection, tables: list[str]) -> dict[str, int]:
    """Даёт O(1) оперативные оценки без полного сканирования многогигабайтных таблиц."""
    result: dict[str, int] = {}
    for table in tables:
        try:
            row = connection.execute(f'SELECT MAX(rowid) FROM "{table}"').fetchone()
            result[table] = int(row[0] or 0)
        except sqlite3.OperationalError:
            result[table] = 0
    return result


def decision_diagnostics(connection: sqlite3.Connection | None) -> dict[str, Any]:
    """Объясняет WAIT, reference и paper-fill без подмены оценки фактом."""
    empty = {"decisions": 0, "wait": 0, "executed": 0, "wait_categories": {},
             "ml_wait": {}, "reference": {}, "limit_orders": {}}
    if connection is None:
        return empty
    tables = set(table_names(connection))
    if "model_decisions" not in tables:
        return empty
    session = connection.execute(
        "SELECT session_id FROM paper_sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone() if "paper_sessions" in tables else None
    condition, values = ("WHERE session_id=?", (str(session[0]),)) if session else ("", ())
    rows = connection.execute(
        f"SELECT action,tags_json,executed FROM model_decisions {condition} ORDER BY id DESC LIMIT 5000",
        values,
    ).fetchall()
    categories: Counter[str] = Counter()
    buy_gaps: list[float] = []
    for row in rows:
        if str(row["action"]) != "WAIT":
            continue
        try:
            tags = [str(value) for value in json.loads(row["tags_json"] or "[]")]
        except (TypeError, json.JSONDecodeError):
            tags = []
        tag_set = set(tags)
        if "invalid_target_price" in tag_set:
            categories["Не получен Price to Beat"] += 1
        elif "invalid_reference_price" in tag_set or "invalid_market_data" in tag_set:
            categories["Нет свежей текущей цены BTC"] += 1
        elif "ml_no_buy_candidate" in tag_set:
            categories["Нет допустимой limit-цены/размера"] += 1
        elif "ml_entry_argmax" in tag_set:
            categories["ML: WAIT имеет максимальную utility"] += 1
        elif "entry_model_low_confidence_wait" in tag_set:
            categories["Низкая уверенность направления"] += 1
        elif "value_gate_reject" in tag_set or "no_positive_edge" in tag_set:
            categories["Нет net edge после цены и комиссии"] += 1
        elif "book_unavailable" in tag_set or "stale_market_data" in tag_set:
            categories["Стакан отсутствует или устарел"] += 1
        else:
            categories["Прочее / позиция не открыта"] += 1
        values_by_key = dict(tag.split("=", 1) for tag in tags if "=" in tag)
        if "ml_entry_argmax" in tag_set:
            try:
                wait_utility = float(values_by_key["best_utility"])
            except (KeyError, ValueError):
                continue
            buy_scores = []
            for score in values_by_key.get("top_action_scores", "").split(";"):
                if score.startswith("BUY_") and "=" in score:
                    try:
                        buy_scores.append(float(score.rsplit("=", 1)[1]))
                    except ValueError:
                        pass
            if buy_scores:
                buy_gaps.append(max(buy_scores) - wait_utility)
    result = {**empty, "decisions": len(rows),
              "wait": sum(str(row["action"]) == "WAIT" for row in rows),
              "executed": sum(int(row["executed"] or 0) for row in rows),
              "wait_categories": dict(categories)}
    if buy_gaps:
        result["ml_wait"] = {
            "observations": len(buy_gaps),
            "mean_buy_minus_wait_utility": mean(buy_gaps),
            "near_entry_share": sum(gap >= -0.02 for gap in buy_gaps) / len(buy_gaps),
            "meaning": "negative means WAIT was more useful; near-entry is within $0.02 utility",
        }
    if {"event_targets", "external_prices"}.issubset(tables):
        target = connection.execute(
            "SELECT event_slug,target_price,completed,latest_reference_price,fetched_at "
            "FROM event_targets ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        latest_sources = connection.execute(
            """SELECT p.source,p.collected_at,p.price FROM external_prices p JOIN
               (SELECT source,MAX(id) id FROM external_prices
                WHERE source IN ('bybit','okx','pyth','chainlink_twap_60s') GROUP BY source) x
               ON p.id=x.id"""
        ).fetchall()
        now_utc = datetime.now(UTC)
        enabled_sources = {
            "bybit": (bool(settings.ENABLE_BYBIT), float(settings.MAX_EXCHANGE_AGE_SECONDS)),
            "okx": (bool(settings.ENABLE_OKX), float(settings.MAX_EXCHANGE_AGE_SECONDS)),
            "pyth": (bool(settings.ENABLE_PYTH), float(settings.MAX_PYTH_AGE_SECONDS)),
            "chainlink_twap_60s": (bool(settings.ENABLE_CHAINLINK_RTDS), float(settings.MAX_CHAINLINK_AGE_SECONDS)),
        }
        fresh_sources = []
        source_status = []
        for row in latest_sources:
            source = str(row["source"])
            age = max(0.0, (now_utc - datetime.fromisoformat(str(row["collected_at"]))).total_seconds())
            enabled, limit = enabled_sources.get(source, (False, 0.0))
            source_status.append({"source": source, "price": float(row["price"]),
                                  "age_seconds": age, "enabled": enabled,
                                  "fresh": bool(enabled and age <= limit), "max_age_seconds": limit})
            if enabled and age <= limit:
                fresh_sources.append({"source": source, "price": float(row["price"]), "age_seconds": age})
        chainlink_row = next((row for row in source_status if row["source"] == "chainlink_twap_60s"), None)
        chainlink_observed = chainlink_row is not None
        chainlink_fresh = bool(chainlink_row and chainlink_row["fresh"])
        if not settings.ENABLE_CHAINLINK_RTDS:
            oracle_reason = "Chainlink TWAP отключён"
        elif not chainlink_observed:
            oracle_reason = "ожидается первый Chainlink TWAP 60s тик"
        elif not chainlink_fresh:
            oracle_reason = f"Chainlink TWAP устарел: {chainlink_row['age_seconds']:.1f}с"
        else:
            oracle_reason = f"Chainlink TWAP актуален: {chainlink_row['age_seconds']:.1f}с"
        live_proxy = ("Chainlink BTC/USD TWAP 60s" if chainlink_fresh else
                      "median(" + ", ".join(row["source"] for row in fresh_sources
                                             if row["source"] != "chainlink_twap_60s") + ") fallback")
        result["reference"] = {
            "event_slug": str(target["event_slug"]) if target else None,
            "price_to_beat": float(target["target_price"]) if target else None,
            "official_close_available": bool(target and target["latest_reference_price"] is not None),
            "official_close_expected_before_resolution": False,
            "live_proxy": live_proxy if fresh_sources else "нет свежего live reference",
            "sources": fresh_sources,
            "source_status": source_status,
            "oracle": {"name": "Chainlink TWAP 60s", "enabled": bool(settings.ENABLE_CHAINLINK_RTDS),
                       "credentials_required": False, "observed": chainlink_observed, "fresh": chainlink_fresh,
                       "reason": oracle_reason},
        }
    if "paper_orders" in tables:
        order_condition, order_values = ("WHERE session_id=?", (str(session[0]),)) if session else ("", ())
        order_rows = connection.execute(
            f"SELECT status,execution_reason,COUNT(*) n FROM paper_orders {order_condition} GROUP BY 1,2",
            order_values,
        ).fetchall()
        result["limit_orders"] = {
            "orders": sum(int(row["n"]) for row in order_rows),
            "by_result": [{"status": str(row["status"]), "reason": str(row["execution_reason"] or ""),
                           "count": int(row["n"])} for row in order_rows],
            "fill_probability_is_estimate": True,
        }
    return result


@app.get("/api/overview")
@_cached(15.0)
def overview() -> dict[str, Any]:
    connection = connect()
    mode = "paper"
    engine_state = "stopped"
    requested_mode = "paper"
    mode_switch_state = "idle"
    live_scale = settings.LIVE_SCALE_DEFAULT
    engine_process_alive = False
    validation_status = "unknown"
    live_advisory: dict[str, Any] = {"desirable": False, "failures": ["нет доступной торговой статистики"]}
    diagnostics = decision_diagnostics(None)
    if connection is None:
        sources: list[dict[str, Any]] = []
        database = {"exists": False, "path": str(settings.DATABASE_PATH), "counts": {}, "latest_run": None, "current_event": None}
        latency: list[dict[str, Any]] = []
        paper = paper_overview(None)
        live = live_overview(None, settings.PAPER_INITIAL_BALANCE_USDC)
        dataset = dataset_overview(None)
        models = model_overview(None)
    else:
        try:
            tables = table_names(connection)
            counts = _fast_table_counts(connection, tables)
            latest_run = dict(connection.execute("SELECT * FROM collector_runs ORDER BY id DESC LIMIT 1").fetchone() or {}) if "collector_runs" in tables else None
            current_event = dict(connection.execute("SELECT slug,title,active,closed,end_date,fetched_at FROM events ORDER BY fetched_at DESC LIMIT 1").fetchone() or {}) if "events" in tables else None
            if current_event:
                current_event.update(event_metadata(current_event.get("slug")))
            sources = latest_sources(connection)
            latency = latency_metrics(connection)
            target = current_target(connection, current_event.get("slug") if current_event else None)
            database = {"exists": True, "path": str(settings.DATABASE_PATH), "counts": counts, "latest_run": latest_run, "current_event": current_event, "current_target": target, "tables": tables}
            paper = paper_overview(connection)
            next_event = next_event_overview(connection)
            dataset = dataset_overview(connection)
            mode = runtime_mode(connection)
            requested_mode = runtime_control(connection, "requested_trading_mode", mode)
            mode_switch_state = runtime_control(connection, "mode_switch_state", "idle")
            try:
                live_scale = float(runtime_control(connection, "trade_size_multiplier", str(settings.TRADE_SIZE_MULTIPLIER_DEFAULT)))
            except ValueError:
                live_scale = settings.LIVE_SCALE_DEFAULT
            engine_state = runtime_control(connection, "engine_state", "paused")
            validation_status = runtime_control(connection, "validation_status", "unknown")
            engine_process_alive = engine_alive(connection)
            if not engine_process_alive:
                engine_state = "stopped"
            models = model_overview(connection)
            diagnostics = decision_diagnostics(connection)
            live = live_overview(connection, settings.PAPER_INITIAL_BALANCE_USDC * live_scale)
            models.update(cached_model_analytics())
            latest_session = connection.execute(
                "SELECT session_id FROM paper_sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone() if "paper_sessions" in tables else None
            advisory_stats = paper_statistics(connection, str(latest_session[0]) if latest_session else None)
            advisory_failures = readiness_failures(advisory_stats)
            live_advisory = {
                "desirable": not advisory_failures,
                "failures": advisory_failures,
                "statistics": advisory_stats,
                "checks": [
                    {"name": "Завершённые независимые события", "passed": advisory_stats["resolved_events"] >= settings.LIVE_MIN_RESOLVED_EVENTS, "actual": int(advisory_stats["resolved_events"]), "required": f"≥ {settings.LIVE_MIN_RESOLVED_EVENTS}"},
                    {"name": "Нижняя граница Wilson win rate", "passed": advisory_stats["wilson_win_rate"] >= settings.LIVE_MIN_WILSON_WIN_RATE, "actual": f"{advisory_stats['wilson_win_rate']:.1%}", "required": f"≥ {settings.LIVE_MIN_WILSON_WIN_RATE:.1%}"},
                    {"name": "Profit factor", "passed": advisory_stats["profit_factor"] >= settings.LIVE_MIN_PROFIT_FACTOR, "actual": f"{advisory_stats['profit_factor']:.2f}", "required": f"≥ {settings.LIVE_MIN_PROFIT_FACTOR:.2f}"},
                    {"name": "Максимальная просадка", "passed": advisory_stats["max_drawdown_pct"] <= settings.LIVE_MAX_DRAWDOWN_PCT, "actual": f"{advisory_stats['max_drawdown_pct']:.1%}", "required": f"≤ {settings.LIVE_MAX_DRAWDOWN_PCT:.1%}"},
                    {"name": "Чистый PAPER PnL", "passed": advisory_stats["net_pnl_usdc"] >= settings.LIVE_MIN_NET_PNL_USDC, "actual": f"${advisory_stats['net_pnl_usdc']:.2f}", "required": f"≥ ${settings.LIVE_MIN_NET_PNL_USDC:.2f}"},
                    {"name": "Нижняя граница 95% CI среднего PnL", "passed": advisory_stats["pnl_mean_ci95_lower"] > 0, "actual": f"${advisory_stats['pnl_mean_ci95_lower']:.3f}", "required": "> $0"},
                    {"name": "Сделки в каждом направлении", "passed": min(advisory_stats["up_entries"], advisory_stats["down_entries"]) >= settings.LIVE_MIN_DIRECTION_ENTRIES, "actual": f"Up {advisory_stats['up_entries']} / Down {advisory_stats['down_entries']}", "required": f"≥ {settings.LIVE_MIN_DIRECTION_ENTRIES} в каждом"},
                    {"name": "Доля одного направления", "passed": advisory_stats["max_direction_share"] <= settings.LIVE_MAX_SINGLE_DIRECTION_SHARE, "actual": f"{advisory_stats['max_direction_share']:.1%}", "required": f"≤ {settings.LIVE_MAX_SINGLE_DIRECTION_SHARE:.1%}"},
                ],
                "note": "Рекомендательные условия не блокируют ручную LIVE-кнопку; технический preflight обязателен.",
            }
        finally:
            connection.close()
    active_trading = paper
    if connection is None:
        next_event = next_event_overview(None)
    if mode == "live":
        live_connection = connect()
        try:
            active_trading = live_overview(live_connection, settings.PAPER_INITIAL_BALANCE_USDC * live_scale)
        finally:
            if live_connection is not None:
                live_connection.close()
    summary = project_summary(database, paper, models, mode, engine_state, engine_process_alive)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "refresh_seconds": settings.DASHBOARD_REFRESH_SECONDS,
        "database": database,
        "sources": sources,
        "latency": latency,
        "system": system_metrics(),
        "llm": {"enabled": settings.LLM_ENABLED, "provider": settings.LLM_PRIMARY_PROVIDER, "model": settings.QWEN_MODEL_ID, "device": settings.QWEN_DEVICE, "min_confidence": settings.QWEN_MIN_CONFIDENCE, "training_enabled": settings.TRAINING_ENABLED, "minimum_examples": settings.TRAINING_MIN_EXAMPLES},
        "models": models,
        "ml_policy": {
            "enabled": settings.ML_AUTONOMOUS_POLICY_ENABLED,
            "paper_only": settings.ML_AUTONOMOUS_POLICY_PAPER_ONLY,
            "manual_trade_gates_enabled": not settings.ML_AUTONOMOUS_POLICY_ENABLED,
            "actions": ("WAIT", "BUY_UP", "BUY_DOWN", "HOLD", "CLOSE"),
            "limit_levels": settings.ML_POLICY_LIMIT_LEVELS,
            "notionals_usdc": settings.ML_POLICY_NOTIONALS_USDC,
            "metrics": paper.get("ml_policy_metrics", {}),
        },
        "decision_diagnostics": diagnostics,
        "project_summary": summary,
        "trading": {"mode": mode if connection is not None else "paper", "requested_mode": requested_mode, "mode_switch_state": mode_switch_state, "engine_state": engine_state if connection is not None else "stopped", "engine_process_alive": engine_process_alive if connection is not None else False, "live_enabled": mode == "live", "live_advisory": live_advisory, "live_executor_implemented": settings.LIVE_EXECUTOR_IMPLEMENTED, "live_keys_rotated": settings.LIVE_KEYS_ROTATED_AFTER_AUDIT, "live_scale": live_scale, "live_scale_min": settings.LIVE_SCALE_MIN, "live_scale_max": settings.LIVE_SCALE_MAX, "live_effective_budget_usdc": settings.PAPER_INITIAL_BALANCE_USDC * live_scale, "live_effective_max_position_usdc": settings.MAX_POSITION_USDC * live_scale, "live_canary_max_loss_usdc": settings.LIVE_CANARY_MAX_LOSS_USDC, "kill_switch": settings.KILL_SWITCH, "max_position_usdc": settings.MAX_POSITION_USDC, "max_daily_loss_usdc": settings.MAX_DAILY_LOSS_USDC, "max_consecutive_losses": settings.MAX_CONSECUTIVE_LOSSES, "max_consecutive_wrong_directions": settings.MAX_CONSECUTIVE_WRONG_DIRECTIONS, "validation_hard_stop_age_seconds": settings.VALIDATION_HARD_STOP_AGE_SECONDS, "validation_hard_stop_cycles": settings.VALIDATION_HARD_STOP_CONSECUTIVE_CYCLES, "validation_status": validation_status, "entry_order_type": settings.ENTRY_ORDER_TYPE, "exit_order_type": settings.EXIT_ORDER_TYPE, "next_event_shadow_enabled": settings.NEXT_EVENT_PREOPEN_SHADOW_ENABLED, "next_event_live_enabled": settings.NEXT_EVENT_PREOPEN_LIVE_ENABLED, "early_entry": settings.EARLY_ENTRY_ENABLED, "early_exit": settings.EARLY_EXIT_ENABLED, "early_exit_execution": settings.EARLY_EXIT_EXECUTION_ENABLED, "full_exit_only": settings.FULL_EXIT_ONLY_ENABLED, "entry_grid_enabled": settings.ENTRY_GRID_ENABLED, "entry_grid_live_enabled": settings.ENTRY_GRID_LIVE_ENABLED, "entry_grid_orders": [settings.ENTRY_GRID_MIN_ORDERS, settings.ENTRY_GRID_MAX_ORDERS], "entry_grid_price_step": settings.ENTRY_GRID_PRICE_STEP, "entry_grid_min_order_usdc": settings.ENTRY_GRID_MIN_ORDER_USDC, "entry_confidence": settings.MIN_ENTRY_CONFIDENCE, "max_held_win_probability_for_exit": settings.MAX_HELD_WIN_PROBABILITY_FOR_EXIT, "partial_exit_confidence": settings.PARTIAL_EXIT_CONFIDENCE, "entry_notional_usdc": settings.PAPER_ENTRY_NOTIONAL_USDC, "max_entry_price": settings.PAPER_MAX_ENTRY_PRICE, "min_entry_price": settings.PAPER_MIN_ENTRY_PRICE, "min_entry_edge": settings.PAPER_MIN_ENTRY_NET_EDGE, "value_safety_margin": settings.ENTRY_VALUE_SAFETY_MARGIN, "min_expected_pnl": settings.ACTION_VALUE_MIN_EXPECTED_PNL_USDC, "probability_model_weight": settings.ACTION_PROBABILITY_MODEL_WEIGHT, "adaptive_position_sizing": settings.ADAPTIVE_POSITION_SIZING_ENABLED, "position_size_tiers": settings.POSITION_SIZE_EDGE_TIERS, "realistic_execution": settings.EXECUTION_SIMULATION_ENABLED, "model_time_gates_enabled": settings.MODEL_TIME_GATES_ENABLED, "entry_start_seconds": settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN, "last_entry_remaining_seconds": settings.PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE, "five_stage_exit": settings.FIVE_STAGE_EXIT_ENABLED, "exit_profit_steps": settings.EXIT_STAGE_PROFIT_RETURN_PCT, "exit_risk_steps": settings.EXIT_STAGE_MAX_HELD_PROBABILITY, "action_value": action_value_overview()},
        "next_event": next_event,
        "active_trading": active_trading,
        "dataset": dataset,
        "economics": {"realized_pnl": active_trading["realized_pnl"], "unrealized_pnl": active_trading["unrealized_pnl"], "gross_income": max(0.0, active_trading["realized_pnl"] + active_trading.get("fees", 0)), "fees": active_trading.get("fees", 0), "estimated_slippage": active_trading["total_wagered"] * settings.ESTIMATED_SLIPPAGE_BPS / 10_000, "status": ("LIVE: локальный ledger по фактически исполненным ордерам" if mode == "live" else "Net PnL includes simulated Polymarket crypto taker fees")},
        "access": access_matrix(sources),
        "metric_definitions": {
            "win_rate": "Доля прибыльных независимых 5-минутных событий. Без размера выборки сама по себе ненадёжна.",
            "wilson": "95% интервал Wilson для истинного win rate. Нижняя граница — консервативная оценка качества.",
            "profit_factor": "Валовая прибыль / абсолютный валовый убыток. Выше 1 — прибыльно на этой выборке.",
            "expectancy": "Средний чистый PnL одного завершённого события после комиссий.",
            "roi": "Чистый PnL / весь поставленный объём. Не доходность начального капитала.",
            "max_drawdown": "Наибольшее снижение капитала от локального максимума.",
            "payoff_ratio": "Средняя прибыль / средний убыток. Показывает цену одной победы относительно поражения.",
            "breakeven_win_rate": "Минимальная доля побед, нужная при текущем среднем выигрыше и проигрыше.",
            "pnl_ci": "95% интервал среднего PnL сделки. Если нижняя граница ≤ 0, положительное ожидание ещё не доказано.",
            "recovery_factor": "Чистый PnL / максимальная просадка в долларах.",
            "roc_auc": "Способность ранжировать выигрышные исходы выше проигрышных: 0.5 — случайно, 1.0 — идеально. Не задаёт прибыльность сама по себе.",
            "pr_auc": "Precision–Recall AUC: качество поиска положительного исхода при дисбалансе классов. Сравнивается с базовой долей положительных меток, а не автоматически с 0.5.",
            "brier": "Средняя квадратичная ошибка вероятности. Ниже лучше; строго наказывает чрезмерную уверенность.",
            "log_loss": "Логарифмическая ошибка вероятности. Ниже лучше; особенно сильно наказывает уверенные ошибки.",
            "average_entry_price": "Простое среднее фактических цен исполненных BUY-заявок: каждый fill имеет одинаковый вес.",
            "weighted_average_entry_price": "Средневзвешенная цена входа (VWAP): фактическая цена каждого BUY-fill взвешена числом купленных контрактов.",
            "average_exit_price": "Простое среднее фактических цен исполненных SELL/CLOSE-заявок: каждый fill имеет одинаковый вес.",
            "weighted_average_exit_price": "Средневзвешенная цена выхода (VWAP): фактическая цена каждого SELL/CLOSE-fill взвешена числом проданных контрактов.",
        },
    }


@app.post("/api/trading-control")
def trading_control(request: TradingControlRequest) -> dict[str, Any]:
    if request.action not in {"stop", "resume", "new_session", "manual_unlock"}:
        raise HTTPException(400, "action must be stop, resume, new_session or manual_unlock")
    connection = connect()
    if connection is None:
        raise HTTPException(409, "Database is not initialized")
    try:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS runtime_controls(
                 control_key TEXT PRIMARY KEY,control_value TEXT NOT NULL,updated_at TEXT NOT NULL,reason TEXT)"""
        )
        latest = connection.execute("SELECT * FROM paper_sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        if request.action == "stop":
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','paused',?,?)",
                (datetime.now(UTC).isoformat(), "manual dashboard stop"),
            )
            if latest and latest["status"] == "running":
                connection.execute("UPDATE paper_sessions SET status='paused' WHERE session_id=?", (latest["session_id"],))
            message = "Торговля остановлена; сбор данных продолжается"
        elif request.action == "resume":
            if latest and latest["status"] == "forced_stopped":
                raise HTTPException(409, "Сессия остановлена risk-gate; начните новый demo-прогон")
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','running',?,?)",
                (datetime.now(UTC).isoformat(), "manual dashboard resume"),
            )
            if latest and latest["status"] == "paused":
                connection.execute("UPDATE paper_sessions SET status='running' WHERE session_id=?", (latest["session_id"],))
            message = "Торговля продолжена с текущим бюджетом и историей"
        elif request.action == "new_session":
            stale_cutoff = int(datetime.now(UTC).timestamp() - 300 - settings.STALE_POSITION_NEW_SESSION_GRACE_SECONDS)
            open_count = int(connection.execute(
                """SELECT COUNT(*) FROM paper_positions WHERE status='open'
                   AND CAST(SUBSTR(event_slug,INSTR(event_slug,'5m-')+3) AS INTEGER)>?""",
                (stale_cutoff,),
            ).fetchone()[0])
            if open_count:
                raise HTTPException(409, "Новый прогон заблокирован: сначала дождитесь расчёта открытой позиции")
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','new_session_requested',?,?)",
                (datetime.now(UTC).isoformat(), "manual dashboard new $300 session"),
            )
            message = "Запрошен новый demo-прогон с бюджетом $300; предыдущий будет архивирован"
        else:
            # Кнопка снимает только статистический PAPER risk-stop. Она не обходит
            # техническую валидацию и никогда не включает LIVE.
            if runtime_mode(connection) != "paper":
                raise HTTPException(409, "Ручная разблокировка доступна только в PAPER; LIVE она не включает")
            validation = runtime_control(connection, "validation_status", "unknown")
            if validation != "healthy":
                raise HTTPException(409, f"Разблокировка запрещена: validation_status={validation}")
            last_error = runtime_control(connection, "engine_last_error", "")
            if last_error:
                raise HTTPException(409, f"Разблокировка запрещена: trading-engine сообщает ошибку: {last_error}")
            heartbeat = connection.execute(
                "SELECT updated_at FROM runtime_controls WHERE control_key='engine_heartbeat'"
            ).fetchone()
            if not heartbeat:
                raise HTTPException(409, "Разблокировка запрещена: нет heartbeat trading-engine")
            try:
                heartbeat_at = datetime.fromisoformat(str(heartbeat[0]).replace("Z", "+00:00"))
                heartbeat_age = (datetime.now(UTC) - heartbeat_at.astimezone(UTC)).total_seconds()
            except (TypeError, ValueError):
                raise HTTPException(409, "Разблокировка запрещена: некорректный heartbeat trading-engine")
            heartbeat_limit = max(60.0, float(settings.VALIDATION_HARD_STOP_AGE_SECONDS) * 3.0)
            if heartbeat_age > heartbeat_limit:
                raise HTTPException(409, f"Разблокировка запрещена: trading-engine не отвечает {heartbeat_age:.0f} сек")
            if _active_position_count(connection):
                raise HTTPException(409, "Разблокировка запрещена: есть открытая позиция или активная LIVE-заявка")
            active_paper_orders = int(connection.execute(
                """SELECT COUNT(*) FROM paper_orders
                   WHERE status IN ('submitted','working','live','partial','partially_filled')
                     AND (expiration_at IS NULL OR expiration_at>?)""",
                (datetime.now(UTC).isoformat(),),
            ).fetchone()[0]) if "paper_orders" in table_names(connection) else 0
            if active_paper_orders:
                raise HTTPException(409, "Разблокировка запрещена: есть активная PAPER-заявка")
            changed_at = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('engine_state','running',?,?)",
                (changed_at, "manual PAPER cooldown unlock from dashboard"),
            )
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('paper_cooldown_state','manual_released',?,?)",
                (changed_at, "same session resumed by user"),
            )
            connection.execute("DELETE FROM runtime_controls WHERE control_key='paper_cooldown_until'")
            if latest:
                connection.execute(
                    "UPDATE paper_sessions SET status='running',consecutive_losses=0,stopped_reason=NULL WHERE session_id=?",
                    (latest["session_id"],),
                )
            message = "PAPER разблокирован: текущая сессия продолжена с тем же бюджетом и историей"
        connection.commit()
        return {"action": request.action, "message": message}
    finally:
        connection.close()


@app.post("/api/model-selection")
def select_model(request: ModelSelectionRequest) -> dict[str, Any]:
    if request.model not in MODEL_SPECS:
        raise HTTPException(400, "Неизвестная модель")
    if request.role not in {"entry", "exit"}:
        raise HTTPException(400, "role must be entry or exit")
    if not model_is_ready(request.model):
        raise HTTPException(409, "Модель ещё не установлена или отсутствуют локальные веса")
    connection = connect()
    if connection is None:
        raise HTTPException(409, "База данных ещё не создана")
    try:
        _ensure_model_selection_schema(connection)
        _apply_due_model_selections(connection)
        open_count = _active_position_count(connection)
        active_key = f"selected_{request.role}_model"
        fallback = runtime_control(connection, "selected_model", settings.DEFAULT_TRADING_MODEL)
        previous = runtime_control(connection, active_key, fallback)
        changed_at = datetime.now(UTC).isoformat()
        apply_at = (datetime.now(UTC) + timedelta(seconds=settings.MODEL_SWITCH_MAX_DELAY_SECONDS)).isoformat()
        latest = connection.execute(
            "SELECT session_id FROM paper_sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        state = "queued" if open_count else "applied"
        if open_count:
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
                (f"pending_{request.role}_model", request.model, changed_at, "selected from dashboard"),
            )
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
                (f"pending_{request.role}_model_apply_at", apply_at, changed_at, "maximum switch delay"),
            )
        else:
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES(?,?,?,?)",
                (active_key, request.model, changed_at, "selected from dashboard"),
            )
            if request.role == "entry":
                connection.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('selected_model',?,?,?)",
                    (request.model, changed_at, "legacy alias for selected_entry_model"),
                )
        connection.execute(
            """INSERT INTO model_selection_history(
                 session_id,changed_at,previous_model,new_model,source,role,state,applied_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (str(latest[0]) if latest else None, changed_at, previous, request.model,
             "dashboard", request.role, state, None if open_count else changed_at),
        )
        connection.commit()
        role_name = "входа" if request.role == "entry" else "выхода"
        return {
            "model": request.model,
            "role": request.role,
            "queued": bool(open_count),
            "apply_at": apply_at if open_count else changed_at,
            "name": get_model(request.model).name,
            "message": (
                f"Модель {role_name} {get_model(request.model).name} поставлена в очередь"
                if open_count else f"Выбрана модель {role_name}: {get_model(request.model).name}"
            ),
        }
    finally:
        connection.close()


@app.post("/api/trading-mode")
def set_trading_mode(request: TradingModeRequest) -> dict[str, Any]:
    if request.mode not in {"paper", "live"}:
        raise HTTPException(400, "mode must be paper or live")
    connection = connect()
    if connection is None:
        raise HTTPException(409, "Database is not initialized")
    try:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS runtime_controls(
                 control_key TEXT PRIMARY KEY,control_value TEXT NOT NULL,updated_at TEXT NOT NULL,reason TEXT)"""
        )
        changed_at = datetime.now(UTC).isoformat()
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('requested_trading_mode',?,?,?)",
            (request.mode, changed_at, "one-button dashboard request"),
        )
        latest = connection.execute(
            "SELECT session_id FROM paper_sessions WHERE status IN ('running','paused') ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        session_id = str(latest[0]) if latest else None
        open_count = _active_position_count(connection)
        if request.mode == "live":
            stats = paper_statistics(connection, session_id)
            # Canary проверяет техническое исполнение на малом капитале; production-gate остаётся отдельным.
            failures = [] if request.canary else readiness_failures(stats)
            # ML-эксперимент остаётся предупреждением, но не отменяет явный
            # ручной LIVE-запрос после успешного технического preflight.
            if not request.canary and not settings.LIVE_KEYS_ROTATED_AFTER_AUDIT:
                failures.append("Polymarket credentials must be rotated after the security audit")
            if not settings.LIVE_EXECUTOR_IMPLEMENTED:
                failures.append("audited live order executor is not implemented")
            if not settings.LIVE_TRADING_ENABLED:
                failures.append("LIVE_TRADING_ENABLED is false")
            if settings.KILL_SWITCH:
                failures.append("kill switch is active")
            if failures:
                stats_safe = {
                    key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                    for key, value in stats.items()
                }
                connection.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state','blocked_preflight',?,?)",
                    (changed_at, "; ".join(failures)),
                )
                connection.commit()
                raise HTTPException(409, {
                    "message": "Live-запрос сохранён, но активация заблокирована",
                    "failures": failures, "statistics": stats_safe, "open_position": bool(open_count),
                })
            if open_count:
                connection.execute(
                    "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state','waiting_current_event',?,?)",
                    (changed_at, "will switch after current position resolves"),
                )
                connection.commit()
                return {"mode": "paper", "queued": True, "message": "LIVE включится после текущего события"}
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('trading_mode','live',?,?)",
                (changed_at, "manual one-click live request passed technical preflight"),
            )
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state','active',?,?)",
                (changed_at, "live mode active"),
            )
            connection.commit()
            return {"mode": "live", "queued": False, "message": "Live-режим включён"}
        if open_count:
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state','waiting_current_event',?,?)",
                (changed_at, "paper switch waits for current position"),
            )
            connection.commit()
            return {"mode": runtime_mode(connection), "queued": True, "message": "PAPER включится после текущего события"}
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('trading_mode','paper',?,?)",
            (changed_at, "selected from dashboard"),
        )
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('mode_switch_state','idle',?,?)",
            (changed_at, "paper mode active"),
        )
        connection.commit()
        return {"mode": "paper", "queued": False, "message": "Демо-режим включён"}
    finally:
        connection.close()
        overview.cache_clear()


@app.post("/api/live-preflight")
def run_live_preflight(request: LivePreflightRequest) -> dict[str, Any]:
    """Проверяет настоящий CLOB и подпись, но гарантированно не отправляет ордер."""
    connection = connect()
    if connection is None:
        raise HTTPException(409, "Database is not initialized")
    try:
        token_id = str(request.token_id or "").strip()
        if not token_id:
            row = connection.execute(
                "SELECT token_id FROM market_snapshots WHERE token_id IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            token_id = str(row[0]) if row else ""
        if not token_id:
            raise HTTPException(409, "No current Polymarket token is available")
        checked_at = datetime.now(UTC).isoformat()
        try:
            result = live_technical_preflight(token_id)
        except Exception as exc:
            connection.execute(
                "INSERT OR REPLACE INTO runtime_controls VALUES('live_preflight_state','failed',?,?)",
                (checked_at, type(exc).__name__),
            )
            connection.commit()
            raise HTTPException(409, {"message": "Live preflight failed", "error_type": type(exc).__name__})
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('live_preflight_state',?,?,?)",
            ("passed" if result.ready else "failed", checked_at, "sign-only; no order submitted"),
        )
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('live_preflight_at',?,?,?)",
            (checked_at, checked_at, "technical CLOB preflight"),
        )
        connection.commit()
        return {"message": "Live preflight passed; no order was submitted", **result.public_dict()}
    finally:
        connection.close()


@app.post("/api/live-scale")
def set_live_scale(request: LiveScaleRequest) -> dict[str, Any]:
    scale = float(request.scale)
    if not settings.TRADE_SIZE_MULTIPLIER_MIN <= scale <= settings.TRADE_SIZE_MULTIPLIER_MAX:
        raise HTTPException(400, f"scale must be between {settings.TRADE_SIZE_MULTIPLIER_MIN} and {settings.TRADE_SIZE_MULTIPLIER_MAX}")
    connection = connect()
    if connection is None:
        raise HTTPException(409, "Database is not initialized")
    try:
        open_count = _active_position_count(connection)
        if open_count:
            raise HTTPException(409, "Нельзя менять коэффициент во время открытой live-позиции")
        connection.execute(
            "INSERT OR REPLACE INTO runtime_controls VALUES('trade_size_multiplier',?,?,?)",
            (f"{scale:.4f}", datetime.now(UTC).isoformat(), "paper/live order multiplier"),
        )
        connection.commit()
        return {
            "scale": scale,
            "effective_budget_usdc": settings.PAPER_INITIAL_BALANCE_USDC,
            "effective_max_position_usdc": min(settings.MAX_POSITION_USDC, settings.PAPER_ENTRY_NOTIONAL_USDC * scale),
            "message": "Коэффициент live-капитала сохранён",
        }
    finally:
        connection.close()


@app.get("/api/timeseries")
def timeseries(minutes: int = Query(default=15, ge=1, le=1440)) -> dict[str, Any]:
    connection = connect()
    if connection is None:
        return {"series": {}}
    if "external_prices" not in table_names(connection):
        connection.close()
        return {"series": {}}
    cutoff = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()
    try:
        rows = connection.execute(
            """SELECT collected_at,source,price FROM external_prices
               WHERE collected_at>=? AND source IN ('bybit','okx','pyth') ORDER BY collected_at""", (cutoff,),
        ).fetchall()
    finally:
        connection.close()
    series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        series.setdefault(row["source"], []).append({"t": row["collected_at"], "v": row["price"]})
    return {"series": series}


@app.get("/api/trades")
def trade_list(limit: int = Query(default=40, ge=1, le=200)) -> dict[str, Any]:
    """Последние позиции и пропущенные события для офлайн-разбора."""
    connection = connect()
    if connection is None:
        return {"trades": []}
    try:
        tables = set(table_names(connection))
        rows: list[dict[str, Any]] = []
        for source, table in (("paper", "paper_positions"), ("live", "live_positions")):
            if table not in tables:
                continue
            source_rows = [dict(row) for row in connection.execute(
                f"""SELECT p.id,p.event_slug,p.outcome,p.status,p.opened_at,p.closed_at,
                           p.average_price,p.current_price,p.close_price,p.shares,p.cost_usdc,
                           p.realized_pnl_usdc,p.fees_usdc,p.entry_decision_id,p.exit_decision_id,
                           p.exit_timing,p.had_early_exit,p.early_exit_pnl_usdc,
                           p.execution_valid,p.invalid_reason,
                           COALESCE(d.model_name,'unknown') entry_model,COALESCE(d.reason,'') entry_reason
                    FROM {table} p LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                    ORDER BY p.opened_at DESC LIMIT ?""",
                (limit,),
            )]
            if source == "live":
                source_rows = _display_settle_ended_live_positions(connection, source_rows)
            for item in source_rows:
                item["source"] = source
                mark = next((item.get(key) for key in ("current_price", "close_price", "average_price") if item.get(key) is not None), 0)
                item["marked_pnl_usdc"] = (
                    float(item.get("realized_pnl_usdc") or 0)
                    if item.get("status") in {"closed", "resolved", "provisionally_resolved"}
                    else float(item.get("shares") or 0) * float(mark) - float(item.get("cost_usdc") or 0)
                )
                rows.append(item)
        rows.sort(key=lambda row: str(row.get("opened_at") or ""), reverse=True)
        enrich_events(rows)

        # Одно событие может содержать сотни WAIT-наблюдений. Для панели оставляем
        # только последнее объяснение и явно отделяем его от фактической позиции.
        skipped: list[dict[str, Any]] = []
        if "model_decisions" in tables:
            recent_decision_span = max(2_000, limit * 200)
            skipped = [dict(row) for row in connection.execute(
                """WITH latest AS (
                       SELECT event_slug,MAX(id) decision_id,MAX(observed_at) observed_at
                       FROM model_decisions
                       WHERE event_slug IS NOT NULL
                         AND id>=(SELECT MAX(id)-? FROM model_decisions)
                       GROUP BY event_slug
                   )
                   SELECT d.event_slug,d.observed_at,d.action,d.confidence,d.reason,
                          d.model_name,d.provider,d.predicted_up_probability,
                          d.predicted_down_probability,d.expected_net_edge
                   FROM latest l JOIN model_decisions d ON d.id=l.decision_id
                   WHERE NOT EXISTS (SELECT 1 FROM paper_positions p WHERE p.event_slug=d.event_slug)
                     AND NOT EXISTS (SELECT 1 FROM live_positions p WHERE p.event_slug=d.event_slug)
                   ORDER BY d.observed_at DESC LIMIT ?""",
                (recent_decision_span, limit),
            )]
            for item in skipped:
                item.update({"kind": "skipped", "entered": False})
            enrich_events(skipped)
        for item in rows:
            item.update({"kind": "position", "entered": True})
        return {"trades": rows[:limit], "skipped_events": skipped}
    finally:
        connection.close()


@app.get("/api/trade/{source}/{position_id}")
def trade_detail(source: str, position_id: int) -> dict[str, Any]:
    if source not in {"paper", "live"}:
        raise HTTPException(400, "source must be paper or live")
    connection = connect()
    if connection is None:
        raise HTTPException(404, "Database is not initialized")
    table = "paper_positions" if source == "paper" else "live_positions"
    try:
        position_row = connection.execute(f"SELECT * FROM {table} WHERE id=?", (position_id,)).fetchone()
        if not position_row:
            raise HTTPException(404, "Position not found")
        position = dict(position_row)
        slug, outcome = str(position["event_slug"]), str(position["outcome"])
        target_row = connection.execute(
            "SELECT target_price,start_time,end_time,source FROM event_targets WHERE event_slug=?", (slug,)
        ).fetchone()
        try:
            event_epoch = int(slug.rsplit("-", 1)[-1])
            event_start = datetime.fromtimestamp(event_epoch, UTC).isoformat()
            event_end = datetime.fromtimestamp(event_epoch + 300, UTC).isoformat()
        except (TypeError, ValueError):
            event_start = str(target_row["start_time"]) if target_row else str(position["opened_at"])
            event_end = str(target_row["end_time"]) if target_row else str(position.get("closed_at") or datetime.now(UTC).isoformat())
        decision_row = connection.execute(
            "SELECT model_name,confidence,reason,predicted_up_probability,predicted_down_probability FROM model_decisions WHERE id=?",
            (position.get("entry_decision_id"),),
        ).fetchone() if position.get("entry_decision_id") else None
        reference = [dict(row) for row in connection.execute(
            """SELECT collected_at,reference_price,target_price,source
               FROM reference_price_snapshots WHERE event_slug=? ORDER BY collected_at""",
            (slug,),
        )]
        btc = [dict(row) for row in connection.execute(
            """SELECT collected_at,source,price,confidence,source_timestamp
               FROM external_prices
               WHERE collected_at>=? AND collected_at<=?
                 AND source IN ('chainlink_twap_60s','bybit','okx','pyth')
               ORDER BY collected_at,id""",
            (event_start, event_end),
        )]
        market = [dict(row) for row in connection.execute(
            """SELECT collected_at,outcome,best_bid,best_ask,midpoint,best_bid_size,best_ask_size
               FROM market_snapshots WHERE event_slug=? ORDER BY collected_at,outcome""",
            (slug,),
        )]
        contract = [row for row in market if str(row.get("outcome")) == outcome]
        decisions = [dict(row) for row in connection.execute(
            """SELECT observed_at,action,confidence,reason,provider,model_name,executed,
                      predicted_up_probability,predicted_down_probability,expected_net_edge,
                      position_state_json,tags_json
               FROM model_decisions WHERE event_slug=? ORDER BY observed_at,id""",
            (slug,),
        )]
        for row in decisions:
            row["role"] = "exit" if row.get("position_state_json") or str(row.get("action") or "").upper() in {"HOLD", "CLOSE", "PARTIAL_CLOSE", "SELL"} else "entry"
        contract_legacy = [dict(row) for row in connection.execute(
            """SELECT collected_at,best_bid,best_ask,midpoint FROM market_snapshots
               WHERE event_slug=? AND outcome=? ORDER BY collected_at""",
            (slug, outcome),
        )]
        winning_row = connection.execute(
            "SELECT outcome FROM training_examples WHERE event_slug=? AND label=1 LIMIT 1", (slug,),
        ).fetchone()
        outcome_label_row = connection.execute(
            "SELECT label FROM training_examples WHERE event_slug=? AND outcome=? LIMIT 1", (slug, outcome),
        ).fetchone()
        orders: list[dict[str, Any]] = []
        if source == "paper" and "paper_orders" in table_names(connection):
            orders = [dict(row) for row in connection.execute(
                """SELECT action,order_type,requested_price,filled_price,shares,notional_usdc,
                          fee_usdc,slippage_bps,status,created_at,fill_probability,latency_ms,
                          execution_reason,fill_observed_at
                   FROM paper_orders WHERE session_id=? AND event_slug=? ORDER BY id""",
                (position.get("session_id"), slug),
            )]
        elif source == "live" and "live_orders" in table_names(connection):
            orders = [dict(row) for row in connection.execute(
                """SELECT side action,order_type,requested_price,
                          CASE WHEN matched_size>0 THEN COALESCE(average_fill_price,requested_price) END filled_price,
                          matched_size shares,COALESCE(fill_notional_usdc,matched_size*requested_price) notional_usdc,
                          COALESCE(fee_usdc,0.0) fee_usdc,0.0 slippage_bps,status,created_at,
                          NULL fill_probability,NULL latency_ms,error execution_reason,
                          CASE WHEN matched_size>0 THEN last_checked_at END fill_observed_at
                   FROM live_orders WHERE event_slug=? ORDER BY id""", (slug,),
            )]
        for row in orders:
            row["submitted_at"] = row.get("created_at")
            row["filled_at"] = row.get("fill_observed_at") if float(row.get("shares") or 0) > 0 else None
        entry_orders = [row for row in orders if str(row.get("action") or "").startswith("BUY") and float(row.get("shares") or 0) > 0]
        if not entry_orders and float(position.get("shares") or 0) > 0:
            entry_orders = [{
                "shares": float(position.get("shares") or 0),
                "notional_usdc": float(position.get("cost_usdc") or 0),
                "fee_usdc": 0.0,
                "filled_at": position.get("opened_at"),
                "fill_observed_at": position.get("opened_at"),
                "filled_price": position.get("average_price"),
                "requested_price": position.get("average_price"),
            }]
        original_shares = sum(float(row.get("shares") or 0) for row in entry_orders) or float(position.get("shares") or 0)
        original_cost = sum(float(row.get("notional_usdc") or 0) + float(row.get("fee_usdc") or 0) for row in entry_orders) or float(position.get("cost_usdc") or 0)
        # Свежая позиция получает финальную обучающую разметку не мгновенно.
        # До её появления используем только уже записанный движком официальный
        # или provisional-исход. Это позволяет честно дорисовать HOLD до конца
        # события, не выводя результат из последней котировки на стороне UI.
        resolved_label_source = "training_examples" if outcome_label_row else None
        resolved_label = int(outcome_label_row[0]) if outcome_label_row else None
        if resolved_label is None and position.get("official_label") is not None:
            resolved_label = int(position["official_label"])
            resolved_label_source = "official_label"
        if resolved_label is None and position.get("provisional_label") is not None:
            resolved_label = int(position["provisional_label"])
            resolved_label_source = "provisional_label"
        winning_outcome = str(winning_row[0]) if winning_row else None
        if winning_outcome is None and resolved_label is not None:
            winning_outcome = outcome if resolved_label == 1 else ("Down" if outcome == "Up" else "Up")
        hold_to_resolution_pnl = (
            original_shares * resolved_label - original_cost if resolved_label is not None else None
        )
        actual_pnl = position.get("realized_pnl_usdc")
        had_early_exit = bool(position.get("had_early_exit"))
        early_exit_vs_hold = (
            float(actual_pnl) - float(hold_to_resolution_pnl)
            if had_early_exit and actual_pnl is not None and hold_to_resolution_pnl is not None else None
        )
        hold_pnl_series = []
        actual_pnl_series = []
        exit_orders = [row for row in orders if str(row.get("action") or "").upper() in {"CLOSE", "PARTIAL_CLOSE", "SELL"} and float(row.get("shares") or 0) > 0]
        for stage, row in enumerate(exit_orders, start=1):
            row["exit_stage"] = min(stage, 5)
        for point in contract:
            mark = point.get("best_bid") if point.get("best_bid") is not None else point.get("midpoint")
            if mark is None:
                continue
            filled_entries = [order for order in entry_orders
                              if str(order.get("filled_at") or order.get("fill_observed_at") or "")
                              <= str(point["collected_at"])]
            entered_shares = sum(float(order.get("shares") or 0) for order in filled_entries)
            entered_cost = sum(float(order.get("notional_usdc") or 0) + float(order.get("fee_usdc") or 0)
                               for order in filled_entries)
            if entered_shares <= 0:
                continue
            hold_value = entered_shares * float(mark) - entered_cost
            hold_pnl_series.append({
                "t": point["collected_at"], "v": hold_value,
            })
            sold_shares = 0.0
            realized_proceeds = 0.0
            for order in exit_orders:
                fill_time = order.get("filled_at") or order.get("fill_observed_at")
                if not fill_time or str(fill_time) > str(point["collected_at"]):
                    continue
                shares = float(order.get("shares") or 0)
                fill = float(order.get("filled_price") or order.get("requested_price") or 0)
                sold_shares += shares
                realized_proceeds += shares * fill - float(order.get("fee_usdc") or 0)
            remaining = max(0.0, entered_shares - sold_shares)
            actual_pnl_series.append({
                "t": point["collected_at"],
                "v": realized_proceeds + remaining * float(mark) - entered_cost,
            })
        # Время исхода — конец самой пятиминутки. closed_at может быть позднее,
        # потому что отражает момент получения/записи расчёта, а не конец рынка.
        resolved_at = event_end
        if resolved_label is not None and resolved_at:
            hold_pnl_series.append({"t": resolved_at, "v": float(hold_to_resolution_pnl)})
            if actual_pnl is not None:
                actual_pnl_series.append({"t": resolved_at, "v": float(actual_pnl)})
        return {
            "source": source, "position": position,
            "decision": dict(decision_row) if decision_row else None,
            "target": {**(dict(target_row) if target_row else {}), **event_metadata(slug)},
            "event_window": {"start": event_start, "end": event_end, "duration_seconds": 300},
            "reference": reference, "btc": btc, "contract": contract_legacy, "market": market,
            "decisions": decisions, "pnl": actual_pnl_series,
            "hold_pnl": hold_pnl_series,
            "orders": orders,
            "fees": {
                "entry_usdc": sum(float(row.get("fee_usdc") or 0) for row in entry_orders),
                "exit_usdc": sum(float(row.get("fee_usdc") or 0) for row in exit_orders),
                "total_usdc": sum(float(row.get("fee_usdc") or 0) for row in orders),
                "source": "recorded fills",
            },
            "exit_plan": {
                "enabled": settings.FIVE_STAGE_EXIT_ENABLED,
                "completed_stage": int(position.get("exit_stage") or 0),
                "remaining_fractions": list(settings.EXIT_STAGE_REMAINING_FRACTIONS),
                "risk_probability_thresholds": list(settings.EXIT_STAGE_MAX_HELD_PROBABILITY),
                "profit_return_thresholds": list(settings.EXIT_STAGE_PROFIT_RETURN_PCT),
            },
            "resolution": {
                "winning_outcome": winning_outcome,
                "selected_outcome_won": bool(resolved_label) if resolved_label is not None else None,
                "resolved_label": resolved_label,
                "resolved_label_source": resolved_label_source,
                "resolved_at": resolved_at,
                "had_early_exit": had_early_exit,
                "hold_to_resolution_pnl_usdc": hold_to_resolution_pnl,
                "actual_pnl_usdc": float(actual_pnl) if actual_pnl is not None else None,
                "early_exit_vs_hold_usdc": early_exit_vs_hold,
                "original_shares": original_shares,
                "original_cost_usdc": original_cost,
            },
        }
    finally:
        connection.close()


@app.post("/api/export")
def export_current_data() -> dict[str, Any]:
    """Ручной компактный экспорт из панели; сборщик и trading-engine не останавливаются."""
    from polybot.exporting.data_export import export_data

    if not settings.DATABASE_PATH.exists():
        raise HTTPException(409, "База данных ещё не создана")
    result = export_data(limit=10_000)
    return {
        "message": f"CSV/JSONL обновлены: {settings.EXPORT_DIR}",
        "rows": result,
    }


@app.get("/api/table/{table}")
def browse_table(table: str, limit: int = Query(default=settings.DASHBOARD_DATABASE_ROWS, ge=1, le=500)) -> dict[str, Any]:
    connection = connect()
    if connection is None:
        raise HTTPException(404, "Database does not exist yet")
    try:
        allowed = table_names(connection)
        if table not in allowed:
            raise HTTPException(404, "Unknown table")
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
        rows = connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid DESC LIMIT ?', (limit,)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 500:
                    item[key] = value[:500] + "…"
            output.append(item)
        return {"table": table, "columns": columns, "rows": output}
    finally:
        connection.close()
