"""Read-only Polymarket market-data collector: no order signing or trading."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import api_config as api
import app_config as settings
import httpx
import websockets

ROOT = settings.PROJECT_ROOT


def now() -> str:
    return datetime.now(UTC).isoformat()


def parse_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return []


def slug_from(value: str) -> str:
    value = value.strip().rstrip("/")
    if "/event/" not in value:
        return value
    slug = urlparse(value).path.rsplit("/event/", 1)[-1].split("/")[0]
    if not slug:
        raise ValueError("No event slug in URL")
    return slug


def best(levels: list[dict[str, Any]], maximum: bool) -> tuple[float | None, float | None]:
    values: list[tuple[float, float]] = []
    for level in levels:
        try:
            values.append((float(level["price"]), float(level["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None, None
    return (max if maximum else min)(values, key=lambda item: item[0])


class Storage:
    def __init__(self, path: str) -> None:
        self.path = ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000)
        self._last_raw_write: dict[str, float] = {}
        self._write_cache: list[tuple[str, tuple[Any, ...]]] = []
        self._cache_started = time.monotonic()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(f"PRAGMA busy_timeout={settings.SQLITE_BUSY_TIMEOUT_MS}")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS collector_runs (
              id INTEGER PRIMARY KEY, started_at TEXT, ended_at TEXT, event_slug TEXT,
              status TEXT, details_json TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY, slug TEXT, title TEXT, active INTEGER, closed INTEGER,
              end_date TEXT, resolution_source TEXT, fetched_at TEXT, raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS markets (
              market_id TEXT PRIMARY KEY, event_id TEXT, slug TEXT, question TEXT, active INTEGER,
              closed INTEGER, enable_order_book INTEGER, end_date TEXT, outcomes_json TEXT,
              token_ids_json TEXT, fetched_at TEXT, raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS market_snapshots (
              id INTEGER PRIMARY KEY, collected_at TEXT, event_slug TEXT, market_id TEXT,
              token_id TEXT, outcome TEXT, best_bid REAL, best_bid_size REAL, best_ask REAL,
              best_ask_size REAL, midpoint REAL, spread REAL, book_timestamp TEXT, book_hash TEXT,
              raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS external_prices (
              id INTEGER PRIMARY KEY, collected_at TEXT, source TEXT, symbol TEXT, price REAL,
              confidence REAL, source_timestamp TEXT, raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS event_targets (
              event_slug TEXT PRIMARY KEY, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
              target_price REAL NOT NULL, latest_reference_price REAL,
              source_timestamp TEXT, completed INTEGER NOT NULL DEFAULT 0,
              source TEXT NOT NULL, fetched_at TEXT NOT NULL, raw_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reference_price_snapshots (
              id INTEGER PRIMARY KEY, collected_at TEXT NOT NULL, event_slug TEXT NOT NULL,
              target_price REAL NOT NULL, reference_price REAL,
              source_timestamp TEXT, completed INTEGER NOT NULL DEFAULT 0,
              source TEXT NOT NULL, raw_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS raw_messages (
              id INTEGER PRIMARY KEY, received_at TEXT, stream TEXT, event_slug TEXT, payload_json TEXT
            );
            CREATE TABLE IF NOT EXISTS training_examples (
              snapshot_id INTEGER PRIMARY KEY, event_slug TEXT NOT NULL, market_id TEXT NOT NULL,
              token_id TEXT NOT NULL, outcome TEXT NOT NULL, observed_at TEXT NOT NULL,
              label INTEGER NOT NULL CHECK(label IN (0,1)), label_kind TEXT NOT NULL,
              features_json TEXT NOT NULL, labeled_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_metrics (
              id INTEGER PRIMARY KEY, measured_at TEXT NOT NULL, source TEXT NOT NULL,
              operation TEXT NOT NULL, status TEXT NOT NULL, latency_ms REAL,
              detail TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(collected_at);
            CREATE INDEX IF NOT EXISTS idx_prices_time ON external_prices(collected_at);
            CREATE INDEX IF NOT EXISTS idx_prices_source_time ON external_prices(source,collected_at);
            CREATE INDEX IF NOT EXISTS idx_reference_event_time
              ON reference_price_snapshots(event_slug,collected_at);
            """
        )
        self.db.commit()

    def write(self, query: str, values: tuple[Any, ...]) -> None:
        for attempt in range(settings.SQLITE_WRITE_RETRY_ATTEMPTS):
            try:
                self.db.execute(query, values)
                self.db.commit()
                return
            except sqlite3.OperationalError as error:
                self.db.rollback()
                if "locked" not in str(error).lower() or attempt + 1 >= settings.SQLITE_WRITE_RETRY_ATTEMPTS:
                    raise
                time.sleep(settings.SQLITE_WRITE_RETRY_DELAY_SECONDS * (attempt + 1))

    def buffered_write(self, query: str, values: tuple[Any, ...]) -> None:
        """Кэширует live-телеметрию только до конца текущего poll-цикла."""
        self._write_cache.append((query, values))
        if time.monotonic() - self._cache_started >= settings.COLLECTOR_WRITE_CACHE_MAX_SECONDS:
            self.flush()

    def flush(self) -> int:
        if not self._write_cache:
            self._cache_started = time.monotonic()
            return 0
        pending, self._write_cache = self._write_cache, []
        for attempt in range(settings.SQLITE_WRITE_RETRY_ATTEMPTS):
            try:
                for query, values in pending:
                    self.db.execute(query, values)
                self.db.commit()
                break
            except sqlite3.OperationalError as error:
                self.db.rollback()
                if "locked" not in str(error).lower() or attempt + 1 >= settings.SQLITE_WRITE_RETRY_ATTEMPTS:
                    self._write_cache = pending + self._write_cache
                    raise
                time.sleep(settings.SQLITE_WRITE_RETRY_DELAY_SECONDS * (attempt + 1))
            except Exception:
                self.db.rollback()
                self._write_cache = pending + self._write_cache
                raise
        self._cache_started = time.monotonic()
        return len(pending)

    def raw(self, stream: str, slug: str, payload: Any) -> None:
        if not settings.STORE_RAW_MESSAGES:
            return
        if stream == "polymarket_market_ws" and not settings.STORE_POLYMARKET_RAW_WS:
            return
        if stream in {"polymarket_market_ws", "polymarket_chainlink_rtds"}:
            current = time.monotonic()
            if current - self._last_raw_write.get(stream, 0.0) < settings.RAW_MESSAGE_SAMPLE_SECONDS:
                return
            self._last_raw_write[stream] = current
        self.buffered_write(
            "INSERT INTO raw_messages(received_at,stream,event_slug,payload_json) VALUES(?,?,?,?)",
            (now(), stream, slug, json.dumps(payload, ensure_ascii=False)),
        )

    def metric(self, source: str, operation: str, status: str, latency_ms: float | None, detail: str = "") -> None:
        self.buffered_write(
            "INSERT INTO source_metrics(measured_at,source,operation,status,latency_ms,detail) VALUES(?,?,?,?,?,?)",
            (now(), source, operation, status, latency_ms, detail[:300]),
        )

    def close(self) -> None:
        # При штатном перезапуске другой процесс может на секунды владеть
        # write-lock. Это не повод завершать сборщик с traceback: данные уже
        # были защищены повторными попытками, а незаписанный хвост кэша будет
        # собран повторно в следующем пяти-минутном окне.
        try:
            self.flush()
        except sqlite3.OperationalError as error:
            self.db.rollback()
            print(f"STORAGE_CLOSE_DEFERRED error={type(error).__name__}")
        finally:
            self.db.close()


class Collector:
    def __init__(self, slug: str, storage: Storage) -> None:
        self.slug, self.storage = slug, storage
        self.tokens: list[dict[str, str]] = []
        self.start_time: str | None = None
        self.end_time: str | None = None

    async def discover(self, client: httpx.AsyncClient) -> None:
        response = await client.get(f"{api.POLYMARKET_GAMMA_URL}/events/slug/{self.slug}")
        response.raise_for_status()
        event = response.json()
        if not isinstance(event, dict) or event.get("slug") != self.slug:
            raise RuntimeError("Gamma returned an unexpected event")
        event_id = str(event["id"])
        self.start_time = event.get("startTime")
        self.end_time = event.get("endDate")
        self.storage.buffered_write(
            "INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?,?,?)",
            (event_id, self.slug, event.get("title"), int(bool(event.get("active"))), int(bool(event.get("closed"))),
             event.get("endDate"), event.get("resolutionSource"), now(), json.dumps(event, ensure_ascii=False)),
        )
        for market in event.get("markets", []):
            outcomes, ids = parse_json(market.get("outcomes")), parse_json(market.get("clobTokenIds"))
            market_id = str(market["id"])
            if len(outcomes) != len(ids) or not ids:
                raise RuntimeError(f"Invalid outcome/token mapping for market {market_id}")
            self.storage.buffered_write(
                "INSERT OR REPLACE INTO markets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (market_id, event_id, market.get("slug"), market.get("question"), int(bool(market.get("active"))),
                 int(bool(market.get("closed"))), int(bool(market.get("enableOrderBook"))), market.get("endDate"),
                 json.dumps(outcomes), json.dumps(ids), now(), json.dumps(market, ensure_ascii=False)),
            )
            if market.get("enableOrderBook"):
                self.start_time = market.get("eventStartTime") or self.start_time
                self.end_time = market.get("endDate") or market.get("endDateIso") or self.end_time
                self.tokens.extend(
                    {"token_id": str(token_id), "outcome": str(outcome), "market_id": market_id}
                    for outcome, token_id in zip(outcomes, ids, strict=True)
                )
        if not self.tokens:
            raise RuntimeError("No order-book tokens found")
        await self.collect_reference(client)

    async def collect_reference(self, client: httpx.AsyncClient) -> None:
        """Сохраняет strike (Price to Beat) и текущую цену источника расчёта."""
        if not self.start_time or not self.end_time:
            raise RuntimeError("Event has no start/end time for Price to Beat")
        started = time.perf_counter()
        response = await client.get(
            api.POLYMARKET_CRYPTO_PRICE_URL,
            params={
                "symbol": "BTC", "eventStartTime": self.start_time,
                "variant": "fiveminute", "endDate": self.end_time,
            },
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://polymarket.com/"},
        )
        latency = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        target = float(data["openPrice"])
        reference = float(data["closePrice"]) if data.get("closePrice") is not None else None
        if target <= 0 or reference is not None and reference <= 0:
            raise RuntimeError("Invalid Price to Beat response")
        timestamp = str(data.get("timestamp") or "")
        completed = int(bool(data.get("completed")))
        raw = json.dumps(data, ensure_ascii=False)
        self.storage.buffered_write(
            """INSERT INTO event_targets(event_slug,start_time,end_time,target_price,latest_reference_price,
               source_timestamp,completed,source,fetched_at,raw_json) VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(event_slug) DO UPDATE SET target_price=excluded.target_price,
               latest_reference_price=excluded.latest_reference_price,source_timestamp=excluded.source_timestamp,
               completed=excluded.completed,fetched_at=excluded.fetched_at,raw_json=excluded.raw_json""",
            (self.slug, self.start_time, self.end_time, target, reference, timestamp, completed,
             "polymarket_crypto_price", now(), raw),
        )
        self.storage.buffered_write(
            """INSERT INTO reference_price_snapshots(collected_at,event_slug,target_price,reference_price,
               source_timestamp,completed,source,raw_json) VALUES(?,?,?,?,?,?,?,?)""",
            (now(), self.slug, target, reference, timestamp, completed, "polymarket_crypto_price", raw),
        )
        self.storage.metric("polymarket_crypto_price", "target", "ok", latency)

    async def collect_books(self, client: httpx.AsyncClient) -> None:
        async def get_book(token: dict[str, str]) -> tuple[dict[str, str], dict[str, Any] | None]:
            started = time.perf_counter()
            response = await client.get(f"{api.POLYMARKET_CLOB_URL}/book", params={"token_id": token["token_id"]})
            latency = (time.perf_counter() - started) * 1000
            if response.status_code == 404:
                self.storage.metric("polymarket_clob", "book", "unavailable", latency, "no_orderbook")
                self.storage.raw("polymarket_book_unavailable", self.slug, {"token_id": token["token_id"], "reason": "no_orderbook"})
                return token, None
            response.raise_for_status()
            self.storage.metric("polymarket_clob", "book", "ok", latency)
            return token, response.json()

        results = await asyncio.gather(*(get_book(token) for token in self.tokens), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self.storage.raw("polymarket_book_error", self.slug, {"error": type(result).__name__})
                continue
            token, book = result
            if book is None:
                continue
            bid, bid_size = best(book.get("bids", []), True)
            ask, ask_size = best(book.get("asks", []), False)
            if bid is not None and not 0 <= bid <= 1 or ask is not None and not 0 <= ask <= 1:
                raise RuntimeError("CLOB probability outside [0,1]")
            if bid is not None and ask is not None and bid > ask:
                raise RuntimeError("Crossed CLOB book")
            midpoint = (bid + ask) / 2 if bid is not None and ask is not None else None
            spread = ask - bid if bid is not None and ask is not None else None
            self.storage.buffered_write(
                """INSERT INTO market_snapshots(collected_at,event_slug,market_id,token_id,outcome,best_bid,
                   best_bid_size,best_ask,best_ask_size,midpoint,spread,book_timestamp,book_hash,raw_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now(), self.slug, token["market_id"], token["token_id"], token["outcome"], bid, bid_size, ask,
                 ask_size, midpoint, spread, book.get("timestamp"), book.get("hash"), json.dumps(book, ensure_ascii=False)),
            )

    async def collect_prices(self, client: httpx.AsyncClient) -> None:
        async def bybit() -> tuple[str, float, float | None, str | None, Any, float]:
            started = time.perf_counter()
            response = await client.get(f"{api.BYBIT_REST_URL}/v5/market/tickers", params={"category": "spot", "symbol": api.BYBIT_SYMBOL})
            response.raise_for_status(); data = response.json(); item = data["result"]["list"][0]
            return "bybit", float(item["lastPrice"]), None, item.get("time"), data, (time.perf_counter() - started) * 1000
        async def okx() -> tuple[str, float, float | None, str | None, Any, float]:
            started = time.perf_counter()
            response = await client.get(f"{api.OKX_REST_URL}/api/v5/market/ticker", params={"instId": api.OKX_SYMBOL})
            response.raise_for_status(); data = response.json(); item = data["data"][0]
            return "okx", float(item["last"]), None, item.get("ts"), data, (time.perf_counter() - started) * 1000
        async def pyth() -> tuple[str, float, float | None, str | None, Any, float]:
            started = time.perf_counter()
            headers = {"Authorization": f"Bearer {api.PYTH_API_KEY}"} if api.PYTH_API_KEY else None
            response = await client.get(f"{api.PYTH_HERMES_URL}/v2/updates/price/latest", params={"ids[]": api.PYTH_BTC_USD_FEED_ID, "parsed": "true"}, headers=headers)
            response.raise_for_status(); data = response.json(); item = data["parsed"][0]["price"]
            exponent = int(item["expo"]); price = int(item["price"]) * 10 ** exponent; confidence = int(item["conf"]) * 10 ** exponent
            timestamp = int(item["publish_time"])
            if price <= 0 or time.time() - timestamp > 90: raise RuntimeError("Pyth is stale or invalid")
            return "pyth", price, confidence, str(timestamp), data, (time.perf_counter() - started) * 1000
        calls = []
        if settings.ENABLE_BYBIT: calls.append(bybit())
        if settings.ENABLE_OKX: calls.append(okx())
        if settings.ENABLE_PYTH: calls.append(pyth())
        for result in await asyncio.gather(*calls, return_exceptions=True):
            if isinstance(result, Exception):
                self.storage.raw("external_price_error", self.slug, {"error": type(result).__name__}); continue
            source, price, confidence, timestamp, raw, latency = result
            self.storage.buffered_write("INSERT INTO external_prices(collected_at,source,symbol,price,confidence,source_timestamp,raw_json) VALUES(?,?,?,?,?,?,?)", (now(), source, "BTC/USD", price, confidence, timestamp, json.dumps(raw, ensure_ascii=False)))
            self.storage.metric(source, "price", "ok", latency)

    async def market_ws(self) -> None:
        subscription = {"assets_ids": [token["token_id"] for token in self.tokens], "type": "market", "custom_feature_enabled": True}
        async with websockets.connect(api.POLYMARKET_MARKET_WS, ping_interval=20) as socket:
            await socket.send(json.dumps(subscription))
            async for raw in socket:
                try: self.storage.raw("polymarket_market_ws", self.slug, json.loads(raw))
                except (TypeError, json.JSONDecodeError): pass

    async def chainlink_ws(self) -> None:
        subscription = {"action": "subscribe", "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": json.dumps({"symbol": api.CHAINLINK_SYMBOL})}]}
        async with websockets.connect(api.POLYMARKET_RTDS_WS, ping_interval=20) as socket:
            await socket.send(json.dumps(subscription))
            while True:
                try: raw = await asyncio.wait_for(socket.recv(), timeout=4)
                except TimeoutError: await socket.send("PING"); continue
                try: payload = json.loads(raw)
                except (TypeError, json.JSONDecodeError): continue
                self.storage.raw("polymarket_chainlink_rtds", self.slug, payload)
                if payload.get("topic") not in {"crypto_prices_chainlink", "crypto_prices"}:
                    continue
                price = payload.get("payload", {})
                points = price.get("data") if isinstance(price.get("data"), list) else [price]
                for point in points:
                    if price.get("symbol") != api.CHAINLINK_SYMBOL:
                        continue
                    try: value, timestamp = float(point["value"]), int(point["timestamp"])
                    except (KeyError, TypeError, ValueError): continue
                    if value > 0:
                        self.storage.buffered_write("INSERT INTO external_prices(collected_at,source,symbol,price,confidence,source_timestamp,raw_json) VALUES(?,?,?,?,?,?,?)", (now(), "chainlink_rtds", "BTC/USD", value, None, str(timestamp), json.dumps(payload, ensure_ascii=False)))
                        self.storage.metric("chainlink_rtds", "stream", "ok", max(0.0, time.time() * 1000 - timestamp))

    async def run(self, seconds: int, use_ws: bool) -> None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15), follow_redirects=True) as client:
            await self.discover(client)
            self.storage.flush()
            tasks = []
            if use_ws and settings.COLLECTOR_USE_WEBSOCKETS:
                if settings.ENABLE_POLYMARKET and settings.ENABLE_POLYMARKET_MARKET_WS:
                    tasks.append(asyncio.create_task(self.market_ws()))
                if settings.ENABLE_CHAINLINK_RTDS: tasks.append(asyncio.create_task(self.chainlink_ws()))
            try:
                finish = time.monotonic() + seconds
                while time.monotonic() < finish:
                    await asyncio.gather(
                        self.collect_books(client), self.collect_prices(client), self.collect_reference(client),
                    )
                    self.storage.flush()
                    await asyncio.sleep(settings.COLLECTOR_POLL_SECONDS)
            finally:
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Polymarket data collector")
    parser.add_argument("--event", default=settings.COLLECTOR_DEFAULT_EVENT_URL)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--db", default=str(settings.DATABASE_PATH))
    parser.add_argument("--no-ws", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0: parser.error("--duration must be positive")
    return args


async def main() -> None:
    args = arguments(); slug = slug_from(args.event); storage = Storage(args.db)
    storage.write("INSERT INTO collector_runs(started_at,event_slug,status,details_json) VALUES(?,?,?,?)", (now(), slug, "running", json.dumps({"read_only": True, "websockets": not args.no_ws})))
    run_id = storage.db.execute("SELECT last_insert_rowid()").fetchone()[0]
    try:
        collector = Collector(slug, storage); await collector.run(args.duration, not args.no_ws)
        storage.write("UPDATE collector_runs SET ended_at=?,status=? WHERE id=?", (now(), "ok", run_id))
        print(f"COLLECTOR_OK db={storage.path} tokens={len(collector.tokens)}")
    except Exception as error:
        storage.write("UPDATE collector_runs SET ended_at=?,status=?,details_json=? WHERE id=?", (now(), "error", json.dumps({"error": type(error).__name__}), run_id)); raise
    finally: storage.close()


if __name__ == "__main__":
    from polybot.runtime import run_async
    run_async(__file__, main)
