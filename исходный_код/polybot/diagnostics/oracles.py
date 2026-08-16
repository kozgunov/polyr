"""Validate live BTC/USD oracle data without exposing credentials."""

from __future__ import annotations

import asyncio
import json
import time

import api_config as config
import httpx
import websockets


async def check_pyth() -> None:
    headers = {}
    if config.PYTH_API_KEY:
        headers["Authorization"] = f"Bearer {config.PYTH_API_KEY}"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"{config.PYTH_HERMES_URL}/v2/updates/price/latest",
            params={"ids[]": config.PYTH_BTC_USD_FEED_ID, "parsed": "true"},
            headers=headers,
        )
        response.raise_for_status()
        item = response.json()["parsed"][0]
        price = item["price"]
        value = int(price["price"]) * (10 ** int(price["expo"]))
        confidence = int(price["conf"]) * (10 ** int(price["expo"]))
        age = time.time() - int(price["publish_time"])
        if value <= 0 or age > 60:
            raise RuntimeError(f"Pyth stale/invalid: age={age:.1f}s")
        print(f"PYTH_OK price={value:.2f} confidence=±{confidence:.2f} age={age:.1f}s")


async def check_chainlink_rtds() -> None:
    request = {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": json.dumps({"symbol": config.CHAINLINK_SYMBOL}),
            }
        ],
    }
    async with asyncio.timeout(25):
        async with websockets.connect(config.POLYMARKET_RTDS_WS, ping_interval=20) as socket:
            await socket.send(json.dumps(request))
            while True:
                try:
                    raw_message = await asyncio.wait_for(socket.recv(), timeout=4)
                except TimeoutError:
                    await socket.send("PING")
                    continue
                try:
                    message = json.loads(raw_message)
                except (json.JSONDecodeError, TypeError):
                    continue
                if message.get("topic") != "crypto_prices_chainlink":
                    continue
                payload = message.get("payload", {})
                timestamp = int(payload["timestamp"])
                age = time.time() - timestamp / 1000
                value = float(payload["value"])
                if payload.get("symbol") != config.CHAINLINK_SYMBOL or value <= 0 or age > 60:
                    raise RuntimeError(f"Chainlink RTDS stale/invalid: age={age:.1f}s")
                print(f"CHAINLINK_RTDS_OK price={value:.2f} age={age:.1f}s")
                return


async def main() -> None:
    results = await asyncio.gather(check_pyth(), check_chainlink_rtds(), return_exceptions=True)
    failed = False
    for name, result in zip(("Pyth", "Chainlink RTDS"), results, strict=True):
        if isinstance(result, Exception):
            failed = True
            print(f"{name.upper().replace(' ', '_')}_ERROR {type(result).__name__}: {result}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    from polybot.runtime import run_async
    run_async(__file__, main)
