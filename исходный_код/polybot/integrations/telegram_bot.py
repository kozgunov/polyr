"""Telegram Bot API listener with concise /data operational and paper-trading report."""

from __future__ import annotations

import sqlite3

import api_config as api
import app_config as settings
import httpx

from polybot.trading.live_guard import paper_statistics

BASE_URL = f"https://api.telegram.org/bot{api.TELEGRAM_BOT_TOKEN}"


async def call(client: httpx.AsyncClient, method: str, payload: dict | None = None) -> dict:
    response = await client.post(f"{BASE_URL}/{method}", json=payload or {})
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description', 'unknown error')}")
    return data["result"]


def _max_rowid(connection: sqlite3.Connection, table: str) -> int:
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if not exists:
        return 0
    return int(connection.execute(f'SELECT MAX(rowid) FROM "{table}"').fetchone()[0] or 0)


def data_report() -> str:
    """Latest active decisions, not a long infrastructure dump."""
    if not settings.DATABASE_PATH.exists():
        return "POLYBOT /data\nБаза ещё не создана."
    connection = sqlite3.connect(settings.DATABASE_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_decisions'").fetchone()
        if not exists:
            return "POLYBOT /data\nАктивных действий ещё нет."
        rows = connection.execute(
            """SELECT observed_at,action,confidence,reason,event_slug,executed FROM model_decisions
               WHERE executed=1 OR action NOT IN ('WAIT','HOLD') ORDER BY id DESC LIMIT 8"""
        ).fetchall()
        if not rows:
            return "POLYBOT /data\nАктивных действий ещё нет."
        lines = ["POLYBOT /data — последние активные действия"]
        for row in rows:
            mark = "EXEC" if row["executed"] else "DECISION"
            lines.append(f"{row['observed_at'][11:19]} {mark} {row['action']} c={row['confidence']:.3f}\n{row['reason']}")
        return "\n".join(lines)
    finally:
        connection.close()


def report() -> str:
    if not settings.DATABASE_PATH.exists():
        return "POLYBOT /report\nБаза ещё не создана."
    connection = sqlite3.connect(settings.DATABASE_PATH, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        session = connection.execute("SELECT * FROM paper_sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        if not session:
            return "POLYBOT /report\nДемо-торговля ещё не запускалась."
        equity = connection.execute("SELECT equity_usdc FROM paper_equity_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        stats = paper_statistics(connection, str(session["session_id"]))
        fees = float(session["total_fees_usdc"] or 0) if "total_fees_usdc" in session else 0.0
        streak = int(session["consecutive_losses"] or 0) if "consecutive_losses" in session else 0
        return (
            "POLYBOT /report\n"
            f"Статус: {session['status']}\n"
            f"Капитал: ${(equity[0] if equity else session['cash_balance_usdc']):.2f} / ${session['initial_balance_usdc']:.2f}\n"
            f"Net PnL: ${stats['net_pnl_usdc']:.2f}; fees: ${fees:.2f}\n"
            f"События: {stats['resolved_events']}; win rate: {stats['win_rate']:.1%}; Wilson: {stats['wilson_win_rate']:.1%}\n"
            f"Profit factor: {stats['profit_factor']:.2f}; expectancy: ${stats['expectancy_usdc']:.3f}\n"
            f"Max drawdown: {stats['max_drawdown_pct']:.2%}; loss streak: {streak}/{settings.MAX_CONSECUTIVE_LOSSES}"
        )
    finally:
        connection.close()


async def reply_to_command(client: httpx.AsyncClient, update: dict) -> None:
    message = update.get("message") or {}
    text = (message.get("text") or "").strip().split("@", 1)[0]
    chat_id = (message.get("chat") or {}).get("id")
    if not chat_id:
        return
    if text.startswith("/start"):
        response = "Бот подключён. /data — последние активные действия; /report — ключевые метрики."
    elif text.startswith("/data"):
        response = data_report()
    elif text.startswith("/report"):
        response = report()
    else:
        return
    await call(client, "sendMessage", {"chat_id": chat_id, "text": response})
    print(f"TELEGRAM_COMMAND_OK command={text} chat_id={chat_id}")


async def main() -> None:
    if not api.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty in api_config.py")
    offset: int | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        identity = await call(client, "getMe")
        print(f"BOT_OK username=@{identity['username']}; commands: /start /data /report")
        while True:
            payload: dict[str, int] = {"timeout": 30}
            if offset is not None:
                payload["offset"] = offset
            updates = await call(client, "getUpdates", payload)
            for update in updates:
                offset = int(update["update_id"]) + 1
                await reply_to_command(client, update)


if __name__ == "__main__":
    from polybot.runtime import run_async

    run_async(__file__, main)
