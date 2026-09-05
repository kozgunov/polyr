"""Continuously discover and collect open BTC 5-minute Polymarket events."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime

import api_config as api
import app_config as settings
import httpx

from polybot.collectors.pipeline import Collector, Storage, now
from polybot.labeling.resolution import label_database
from polybot.storage.retention import prune
from polybot.storage.parquet_archive import archive_previous_hour
from polybot.models.retraining import run_if_due


async def find_open_event(client: httpx.AsyncClient) -> tuple[str, float] | None:
    bucket = int(time.time() // 300) * 300
    candidates = (bucket - 300, bucket, bucket + 300, bucket + 600)
    for timestamp in candidates:
        slug = f"{settings.COLLECTOR_BTC_5M_SLUG_PREFIX}-{timestamp}"
        response = await client.get(f"{api.POLYMARKET_GAMMA_URL}/events/slug/{slug}")
        if response.status_code == 404: continue
        response.raise_for_status()
        event = response.json()
        if event.get("closed") or not event.get("active"):
            continue
        end_date = event.get("endDate")
        try:
            remaining = datetime.fromisoformat(end_date).timestamp() - time.time() + 5
        except (AttributeError, ValueError):
            remaining = 300
        if remaining > 0: return slug, remaining
    return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous read-only BTC 5m collector")
    parser.add_argument("--db", default=str(settings.DATABASE_PATH))
    parser.add_argument("--max-seconds", type=int, default=0, help="0 means run indefinitely")
    parser.add_argument("--no-ws", action="store_true")
    args = parser.parse_args()
    storage = Storage(args.db); started = time.monotonic(); last_label = 0.0; last_maintenance = 0.0; last_archive = 0.0
    retraining_task: asyncio.Task | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            while args.max_seconds == 0 or time.monotonic() - started < args.max_seconds:
                try:
                    found = await find_open_event(client)
                except httpx.HTTPError as error:
                    # Краткий DNS/API-сбой не должен останавливать 24/7 pipeline.
                    print(f"DISCOVERY_DEFERRED error={type(error).__name__}")
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
                    await collector.run(max(10, int(seconds)), not args.no_ws)
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
                if time.monotonic() - last_label >= settings.COLLECTOR_LABEL_INTERVAL_SECONDS:
                    try:
                        new_examples, pending = await label_database(storage.path)
                        print(f"LABEL_PASS new_examples={new_examples} pending_events={pending}")
                    except (sqlite3.OperationalError, httpx.HTTPError) as error:
                        # Разметка будет повторена на следующем окне; сбор данных не останавливаем.
                        print(f"LABEL_DEFERRED error={type(error).__name__}")
                    if settings.AUTO_RETRAIN_ENABLED and (retraining_task is None or retraining_task.done()):
                        retraining_task = asyncio.create_task(asyncio.to_thread(run_if_due))
                    last_label = time.monotonic()
                if settings.PARQUET_ARCHIVE_ENABLED and time.monotonic() - last_archive >= settings.PARQUET_ROLLOVER_SECONDS:
                    storage.flush()
                    try:
                        print(f"PARQUET_ROLLOVER {archive_previous_hour(storage.db)}")
                    except (OSError, sqlite3.OperationalError) as error:
                        storage.db.rollback()
                        print(f"PARQUET_DEFERRED error={type(error).__name__}")
                    last_archive = time.monotonic()
                if time.monotonic() - last_maintenance >= settings.DATABASE_MAINTENANCE_SECONDS:
                    try:
                        print(f"RETENTION_OK deleted={prune(storage.db)}")
                    except sqlite3.OperationalError as error:
                        # Maintenance must never terminate 24/7 collection when the
                        # paper engine briefly owns SQLite's write lock.
                        storage.db.rollback()
                        print(f"RETENTION_DEFERRED error={type(error).__name__}")
                    last_maintenance = time.monotonic()
    finally:
        storage.close()


if __name__ == "__main__":
    from polybot.runtime import run_async
    run_async(__file__, main)
