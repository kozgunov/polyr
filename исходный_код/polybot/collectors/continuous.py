"""Continuously discover and collect open BTC 5-minute Polymarket events."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import api_config as api
import app_config as settings
import httpx

from polybot.collectors.pipeline import Collector, Storage, now
from polybot.labeling.resolution import label_database
from polybot.storage.retention import prune
from polybot.storage.parquet_archive import archive_previous_hour
from polybot.models.retraining import run_if_due


async def _continuous_exchange_prices(storage: Storage) -> None:
    """Независимый горячий feed: не перезапускается на границе 5-минуток."""
    collector = Collector("btc-usd-continuous", storage)
    timeout = httpx.Timeout(3.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        while True:
            try:
                await collector.collect_prices(client)
                storage.flush()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - следующий тик остаётся полезен
                storage.raw("continuous_exchange_error", collector.slug, {"error": type(error).__name__})
            await asyncio.sleep(settings.COLLECTOR_POLL_SECONDS)


async def _continuous_chainlink(storage: Storage) -> None:
    """Независимый официальный Chainlink TWAP feed для текущего и следующего рынка."""
    collector = Collector("btc-usd-continuous", storage)
    await collector.chainlink_ws()


def _database_connection(path: str) -> sqlite3.Connection:
    """Отдельное соединение для фоновых задач, не блокирующее цикл тиков."""
    connection = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
    return connection


def _archive_in_background(path: str) -> dict:
    connection = _database_connection(path)
    try:
        return archive_previous_hour(connection)
    finally:
        connection.close()


def _prune_in_background(path: str) -> dict[str, int]:
    connection = _database_connection(path)
    try:
        return prune(connection)
    finally:
        connection.close()


def _label_in_background(path: str) -> tuple[int, int]:
    """Изолирует SQLite/HTTP-разметку от event loop критического сборщика."""
    return asyncio.run(label_database(Path(path)))


def _maintenance_batch(path: str) -> dict[str, object]:
    """Выполняет тяжёлые работы последовательно, чтобы не конкурировать за SQLite/CPU."""
    report: dict[str, object] = {}
    try:
        new_examples, pending = _label_in_background(path)
        report["label"] = {"new_examples": new_examples, "pending_events": pending}
    except Exception as error:  # noqa: BLE001 - следующая стадия всё равно полезна
        report["label_error"] = type(error).__name__
    if settings.PARQUET_ARCHIVE_ENABLED:
        try:
            report["parquet"] = _archive_in_background(path)
        except Exception as error:  # noqa: BLE001
            report["parquet_error"] = type(error).__name__
    try:
        report["retention"] = _prune_in_background(path)
    except Exception as error:  # noqa: BLE001
        report["retention_error"] = type(error).__name__
    if settings.AUTO_RETRAIN_ENABLED:
        try:
            report["retraining"] = run_if_due()
        except Exception as error:  # noqa: BLE001
            report["retraining_error"] = type(error).__name__
    return report


def _maintenance_entry_pause(storage: Storage, active: bool, reason: str) -> None:
    """Запрещает только новые входы; мониторинг и выход открытой позиции остаются активны."""
    until = ((datetime.now(UTC) + timedelta(seconds=settings.MAINTENANCE_ENTRY_PAUSE_MAX_SECONDS)).isoformat()
             if active else "")
    storage.write(
        "INSERT OR REPLACE INTO runtime_controls VALUES('maintenance_entry_pause_until',?,?,?)",
        (until, now(), reason),
    )


async def find_open_event(client: httpx.AsyncClient) -> tuple[str, float] | None:
    bucket = int(time.time() // 300) * 300
    # Текущий bucket проверяем первым: запрос предыдущего закрытого рынка раньше
    # добавлял лишнюю сетевую задержку ровно на границе 5-минуток.
    candidates = (bucket, bucket + 300, bucket - 300, bucket + 600)
    for timestamp in candidates:
        slug = f"{settings.COLLECTOR_BTC_5M_SLUG_PREFIX}-{timestamp}"
        response = await client.get(f"{api.POLYMARKET_GAMMA_URL}/events/slug/{slug}")
        if response.status_code == 404: continue
        response.raise_for_status()
        event = response.json()
        if event.get("closed") or not event.get("active"):
            continue
        # Для BTC 5m authoritative граница окна закодирована в slug. Поле
        # Gamma endDate иногда относится к более позднему settlement/status и
        # оставляло сборщик на завершённом событии ещё примерно на минуту.
        # Небольшая секунда перекрытия нужна только для бесшовного переключения.
        remaining = timestamp + 300 - time.time() + 1
        if remaining > 0: return slug, remaining
    return None


def find_cached_open_event(storage: Storage) -> tuple[str, float] | None:
    """Использует заранее собранный token mapping текущего окна без ожидания Gamma."""
    bucket = int(time.time() // 300) * 300
    slug = f"{settings.COLLECTOR_BTC_5M_SLUG_PREFIX}-{bucket}"
    cutoff = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    row = storage.db.execute(
        """SELECT COUNT(DISTINCT outcome) FROM future_event_snapshots
           WHERE next_event_slug=? AND collected_at>=?""",
        (slug, cutoff),
    ).fetchone()
    remaining = bucket + 300 - time.time() + 1
    return (slug, remaining) if row and int(row[0]) >= 2 and remaining > 0 else None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous read-only BTC 5m collector")
    parser.add_argument("--db", default=str(settings.DATABASE_PATH))
    parser.add_argument("--max-seconds", type=int, default=0, help="0 means run indefinitely")
    parser.add_argument("--no-ws", action="store_true")
    args = parser.parse_args()
    storage = Storage(args.db); started = time.monotonic()
    # Первый тяжёлый batch выполняется через полный интервал после старта, а не
    # сразу: запуск торговли всегда получает свободный горячий путь.
    last_label = last_maintenance = last_archive = time.monotonic()
    maintenance_task: asyncio.Task | None = None
    client = httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0))
    hot_feed_tasks = [asyncio.create_task(_continuous_exchange_prices(storage))]
    if settings.ENABLE_CHAINLINK_RTDS:
        hot_feed_tasks.append(asyncio.create_task(_continuous_chainlink(storage)))
    try:
        while args.max_seconds == 0 or time.monotonic() - started < args.max_seconds:
                # Забираем результаты тяжёлых фоновых работ, не задерживая открытие
                # следующего 5-минутного окна и получение свежих reference-тиков.
                if maintenance_task is not None and maintenance_task.done():
                    try:
                        print(f"MAINTENANCE_OK {maintenance_task.result()}")
                    except Exception as error:  # noqa: BLE001 - batch будет повторён через час
                        print(f"MAINTENANCE_DEFERRED error={type(error).__name__}")
                    maintenance_task = None
                if settings.MAINTENANCE_ENTRY_PAUSE_ENABLED and maintenance_task is None:
                    _maintenance_entry_pause(storage, False, "hourly maintenance completed")
                try:
                    # Предыдущая пятиминутка заранее сохраняет token mapping следующей.
                    # Это убирает Gamma из критического пути на границе событий.
                    found = find_cached_open_event(storage) or await find_open_event(client)
                except httpx.HTTPError as error:
                    # Краткий DNS/API-сбой не должен останавливать 24/7 pipeline.
                    print(f"DISCOVERY_DEFERRED error={type(error).__name__}")
                    # После сетевого/SSL-сбоя не переиспользуем потенциально
                    # повреждённый connection pool бесконечно.
                    await client.aclose()
                    client = httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0))
                    await asyncio.sleep(settings.COLLECTOR_DISCOVERY_SECONDS)
                    continue
                if not found:
                    print("WAITING_FOR_OPEN_BTC_5M_EVENT")
                    await asyncio.sleep(settings.COLLECTOR_DISCOVERY_SECONDS); continue
                slug, seconds = found
                if args.max_seconds:
                    seconds = min(seconds, args.max_seconds - (time.monotonic() - started))
                if seconds <= 0:
                    break
                run_started = now()
                storage.write("INSERT INTO collector_runs(started_at,event_slug,status,details_json) VALUES(?,?,?,?)", (run_started, slug, "running", json.dumps({"continuous": True, "read_only": True})))
                run_id = storage.db.execute("SELECT last_insert_rowid()").fetchone()[0]
                try:
                    collector = Collector(slug, storage)
                    # Bybit/OKX/Chainlink уже идут отдельным непрерывным контуром
                    # и не прерываются discovery/сменой slug этого collector.
                    await collector.run(
                        max(10, int(seconds)), not args.no_ws,
                        collect_external_feeds=False,
                    )
                    storage.write("UPDATE collector_runs SET ended_at=?,status=? WHERE id=?", (now(), "ok", run_id))
                    print(f"WINDOW_OK slug={slug} tokens={len(collector.tokens)}")
                except Exception as error:  # noqa: BLE001 - isolate failed window and keep 24/7 collection alive
                    try:
                        storage.write("UPDATE collector_runs SET ended_at=?,status=?,details_json=? WHERE id=?", (now(), "error", json.dumps({"error": type(error).__name__}), run_id))
                    except sqlite3.OperationalError as write_error:
                        # Диагностическая запись не должна повторно уронить 24/7 collector,
                        # если другой процесс временно держит write-lock.
                        storage.db.rollback()
                        print(f"WINDOW_STATUS_DEFERRED error={type(write_error).__name__}")
                    print(f"WINDOW_ERROR slug={slug} error={type(error).__name__}")
                    await asyncio.sleep(settings.COLLECTOR_DISCOVERY_SECONDS)
                maintenance_due = (
                    time.monotonic() - last_label >= settings.COLLECTOR_LABEL_INTERVAL_SECONDS
                    or settings.PARQUET_ARCHIVE_ENABLED
                    and time.monotonic() - last_archive >= settings.PARQUET_ROLLOVER_SECONDS
                    or time.monotonic() - last_maintenance >= settings.DATABASE_MAINTENANCE_SECONDS
                )
                if maintenance_due and maintenance_task is None:
                    if settings.TRADING_PRIORITY_MODE and settings.MAINTENANCE_ENTRY_PAUSE_ENABLED:
                        _maintenance_entry_pause(storage, True, "hourly labeling/parquet/retention")
                    storage.flush()
                    maintenance_task = asyncio.create_task(
                        asyncio.to_thread(_maintenance_batch, str(storage.path)),
                    )
                    last_label = last_archive = last_maintenance = time.monotonic()
    finally:
        for task in hot_feed_tasks:
            task.cancel()
        await asyncio.gather(*hot_feed_tasks, return_exceptions=True)
        if settings.MAINTENANCE_ENTRY_PAUSE_ENABLED:
            _maintenance_entry_pause(storage, False, "collector shutdown")
        await client.aclose()
        storage.close()


if __name__ == "__main__":
    from polybot.runtime import run_async
    run_async(__file__, main)
