"""Hard gate for future live trading. No live order executor is enabled here."""

from __future__ import annotations

import math
import sqlite3
from statistics import stdev

import app_config as settings


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def paper_statistics(connection: sqlite3.Connection, session_id: str | None = None) -> dict[str, float | int]:
    where = "status IN ('closed','resolved')"
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(paper_positions)")}
    if "execution_valid" in columns:
        where += " AND execution_valid=1"
    parameters: tuple[str, ...] = ()
    if session_id is not None:
        where += " AND session_id=?"
        parameters = (session_id,)
    rows = connection.execute(
        f"SELECT event_slug,realized_pnl_usdc,outcome FROM paper_positions WHERE {where}", parameters,
    ).fetchall()
    by_event: dict[str, float] = {}
    event_direction: dict[str, str] = {}
    for slug, pnl, outcome in rows:
        by_event[str(slug)] = by_event.get(str(slug), 0.0) + float(pnl or 0.0)
        event_direction.setdefault(str(slug), str(outcome))
    outcomes = list(by_event.values())
    wins = sum(value > 0 for value in outcomes)
    gross_profit = sum(value for value in outcomes if value > 0)
    gross_loss = abs(sum(value for value in outcomes if value < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    pnl_mean = sum(outcomes) / len(outcomes) if outcomes else 0.0
    pnl_se = stdev(outcomes) / math.sqrt(len(outcomes)) if len(outcomes) > 1 else float("inf")
    up_entries = sum(direction == "Up" for direction in event_direction.values())
    down_entries = sum(direction == "Down" for direction in event_direction.values())
    peak, max_drawdown = float(settings.PAPER_INITIAL_BALANCE_USDC), 0.0
    equity = peak
    for value in outcomes:
        equity += value
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return {
        "resolved_events": len(outcomes), "wins": wins,
        "wilson_win_rate": wilson_lower_bound(wins, len(outcomes)),
        "net_pnl_usdc": sum(outcomes), "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown,
        "expectancy_usdc": pnl_mean,
        "pnl_mean_ci95_lower": pnl_mean - 1.96 * pnl_se,
        "up_entries": up_entries, "down_entries": down_entries,
        "max_direction_share": max(up_entries, down_entries) / max(1, up_entries + down_entries),
        "win_rate": wins / len(outcomes) if outcomes else 0.0,
    }


def readiness_failures(statistics: dict[str, float | int]) -> list[str]:
    failures = []
    if statistics["resolved_events"] < settings.LIVE_MIN_RESOLVED_EVENTS:
        failures.append("insufficient independent resolved events")
    if statistics["wilson_win_rate"] < settings.LIVE_MIN_WILSON_WIN_RATE:
        failures.append("Wilson win-rate lower bound is too low")
    if statistics["profit_factor"] < settings.LIVE_MIN_PROFIT_FACTOR:
        failures.append("profit factor is too low")
    if statistics["max_drawdown_pct"] > settings.LIVE_MAX_DRAWDOWN_PCT:
        failures.append("maximum drawdown is too high")
    if statistics["net_pnl_usdc"] < settings.LIVE_MIN_NET_PNL_USDC:
        failures.append("net paper PnL is below the minimum")
    if settings.LIVE_REQUIRE_POSITIVE_PNL_CI95 and statistics["pnl_mean_ci95_lower"] <= 0:
        failures.append("mean paper PnL 95% CI lower bound is not positive")
    if min(statistics["up_entries"], statistics["down_entries"]) < settings.LIVE_MIN_DIRECTION_ENTRIES:
        failures.append("insufficient independent entries in both Up and Down directions")
    if statistics["max_direction_share"] > settings.LIVE_MAX_SINGLE_DIRECTION_SHARE:
        failures.append("single-direction share is too high")
    return failures


def main() -> None:
    if not settings.DATABASE_PATH.exists():
        raise RuntimeError("Database does not exist")
    connection = sqlite3.connect(settings.DATABASE_PATH)
    try:
        statistics = paper_statistics(connection)
    finally:
        connection.close()
    print(f"LIVE_GATE_STATS {statistics}")
    failures = readiness_failures(statistics)
    if failures:
        raise RuntimeError("LIVE_TRADING_BLOCKED: " + "; ".join(failures))
    if not settings.LIVE_TRADING_ENABLED or settings.KILL_SWITCH:
        raise RuntimeError("LIVE_TRADING_BLOCKED: configuration lock is active")
    confirmation = input(f"Type {settings.LIVE_REQUIRED_CONFIRMATION!r} to continue: ").strip()
    if confirmation != settings.LIVE_REQUIRED_CONFIRMATION:
        raise RuntimeError("LIVE_TRADING_BLOCKED: confirmation mismatch")
    raise RuntimeError("LIVE_TRADING_BLOCKED: audited live order executor has not been implemented")
