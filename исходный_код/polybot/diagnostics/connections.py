"""Audit configured providers and write connection_audit.md without secrets."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import app_config as settings
import httpx
import websockets

ROOT = settings.PROJECT_ROOT
CONFIG_PATH = ROOT / "api_config.py"
REPORT_PATH = ROOT / "документация" / "connection_audit.md"


@dataclass
class Check:
    service: str
    status: str
    detail: str
    fix: str = "—"


def load_config():
    spec = importlib.util.spec_from_file_location("project_api_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def http_check(
    client: httpx.AsyncClient,
    service: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    validator=None,
    fix: str = "Проверить URL, доступ в сеть и лимиты API.",
) -> Check:
    started = time.perf_counter()
    try:
        # Некоторые сетевые драйверы/прокси игнорируют внутренний timeout
        # httpx на стадии DNS/TLS. Внешний deadline гарантирует, что аудит
        # завершается отчётом, а не зависает бесконечно.
        response = await asyncio.wait_for(
            client.get(url, params=params, headers=headers), timeout=12,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        if validator and not validator(payload):
            return Check(service, "WARN", "Ответ получен, но формат неожиданный.", fix)
        latency = round((time.perf_counter() - started) * 1000)
        return Check(service, "OK", f"HTTP {response.status_code}, {latency} ms")
    except httpx.HTTPStatusError as exc:
        return Check(service, "ERROR", f"HTTP {exc.response.status_code}", fix)
    except Exception as exc:  # noqa: BLE001 - diagnostics must report any provider/client failure
        return Check(service, "ERROR", type(exc).__name__, fix)


async def websocket_connect_check(service: str, url: str) -> Check:
    try:
        async with asyncio.timeout(10):
            async with websockets.connect(url, ping_interval=None):
                return Check(service, "OK", "WebSocket handshake successful")
    except Exception as exc:  # noqa: BLE001 - diagnostics must report any WebSocket failure
        return Check(
            service,
            "ERROR",
            f"{type(exc).__name__}: {exc}",
            "Проверить URL, TLS, firewall и доступность WebSocket.",
        )


def is_hex_address(value: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", value or ""))


def is_feed_id(value: str) -> bool:
    return bool(re.fullmatch(r"0x[a-fA-F0-9]{64}", value or ""))


async def run_checks(config) -> list[Check]:
    checks: list[Check] = []
    timeout = httpx.Timeout(12.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        checks.extend(
            await asyncio.gather(
                http_check(
                    client,
                    "Polymarket Gamma",
                    f"{config.POLYMARKET_GAMMA_URL}/events",
                    params={"limit": 1, "active": "true", "closed": "false"},
                    validator=lambda p: isinstance(p, (list, dict)),
                ),
                http_check(
                    client,
                    "Polymarket Data",
                    f"{config.POLYMARKET_DATA_URL}/trades",
                    params={"limit": 1},
                    validator=lambda p: isinstance(p, list),
                ),
                http_check(
                    client,
                    "Polymarket CLOB",
                    f"{config.POLYMARKET_CLOB_URL}/ok",
                    validator=lambda p: isinstance(p, (dict, str)),
                ),
                http_check(
                    client,
                    "Polymarket geoblock",
                    config.POLYMARKET_GEOBLOCK_URL,
                    validator=lambda p: isinstance(p, dict) and "blocked" in p,
                ),
                http_check(
                    client,
                    "Bybit public market data",
                    f"{config.BYBIT_REST_URL}/v5/market/tickers",
                    params={"category": "spot", "symbol": config.BYBIT_SYMBOL},
                    validator=lambda p: p.get("retCode") == 0,
                ),
                http_check(
                    client,
                    "OKX public market data",
                    f"{config.OKX_REST_URL}/api/v5/market/ticker",
                    params={"instId": config.OKX_SYMBOL},
                    validator=lambda p: p.get("code") == "0",
                ),
            )
        )

        pyth_key = str(getattr(config, "PYTH_API_KEY", "")).strip()
        pyth_headers = {"Authorization": f"Bearer {pyth_key}"} if pyth_key else None
        checks.append(
            await http_check(
                client,
                "Pyth BTC/USD oracle",
                f"{config.PYTH_HERMES_URL}/v2/updates/price/latest",
                params={"ids[]": config.PYTH_BTC_USD_FEED_ID, "parsed": "true"},
                headers=pyth_headers,
                validator=lambda p: isinstance(p, dict) and "parsed" in p,
                fix=(
                    "Получить Pyth API key и перейти на рекомендованный Hermes endpoint; "
                    "ключ станет обязательным 18.08.2026."
                ),
            )
        )

        bot_token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "")).strip()
        if bot_token:
            checks.append(
                await http_check(
                    client,
                    "Telegram Bot API",
                    f"https://api.telegram.org/bot{bot_token}/getMe",
                    validator=lambda p: p.get("ok") is True,
                    fix=(
                        "При HTTP 401 перевыпустить token у @BotFather; при timeout проверить "
                        "доступ к api.telegram.org, proxy/firewall и повторить запрос."
                    ),
                )
            )
        else:
            checks.append(Check("Telegram Bot API", "MISSING", "TELEGRAM_BOT_TOKEN пуст.", "Создать бота через @BotFather."))

        openai_key = str(getattr(config, "OPENAI_API_KEY", "")).strip()
        if openai_key:
            checks.append(
                await http_check(
                    client,
                    "OpenAI API",
                    f"{config.OPENAI_API_URL}/models",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    validator=lambda p: "data" in p,
                    fix="Проверить OPENAI_API_KEY и API billing; ChatGPT subscription не является API credit.",
                )
            )
        else:
            checks.append(Check("OpenAI API", "OPTIONAL", "OPENAI_API_KEY пуст.", "Создать API key на platform.openai.com."))

    checks.extend(
        await asyncio.gather(
            websocket_connect_check("Polymarket Market WebSocket", config.POLYMARKET_MARKET_WS),
            websocket_connect_check("Polymarket RTDS WebSocket", config.POLYMARKET_RTDS_WS),
            websocket_connect_check("Bybit WebSocket", config.BYBIT_WS_URL),
            websocket_connect_check("OKX WebSocket", config.OKX_WS_URL),
        )
    )

    private_key = str(getattr(config, "POLYMARKET_PRIVATE_KEY", "")).strip()
    funder = str(getattr(config, "POLYMARKET_FUNDER_ADDRESS", "")).strip()
    signature_type = getattr(config, "POLYMARKET_SIGNATURE_TYPE", None)
    clob_fields = [
        str(getattr(config, "POLYMARKET_API_KEY", "")).strip(),
        str(getattr(config, "POLYMARKET_API_SECRET", "")).strip(),
        str(getattr(config, "POLYMARKET_API_PASSPHRASE", "")).strip(),
    ]
    issues = []
    if not private_key.startswith("0x"):
        issues.append("private key отсутствует или не начинается с 0x")
    if not is_hex_address(funder):
        issues.append("funder address имеет неверный формат")
    if signature_type not in (0, 1, 2, 3):
        issues.append("signature type не выбран")
    if not all(clob_fields):
        issues.append("CLOB apiKey/secret/passphrase не созданы")
    checks.append(
        Check(
            "Polymarket trading auth",
            "OK" if not issues else "INCOMPLETE",
            "; ".join(issues) if issues else "Все обязательные поля заполнены.",
            "Запустить generate_polymarket_credentials.py и указать корректный POLYMARKET_SIGNATURE_TYPE.",
        )
    )

    chainlink_issues = []
    if not str(getattr(config, "CHAINLINK_STREAM_URL", "")).strip():
        chainlink_issues.append("CHAINLINK_STREAM_URL пуст")
    if not is_feed_id(str(getattr(config, "CHAINLINK_FEED_ID", ""))):
        chainlink_issues.append("CHAINLINK_FEED_ID имеет неожиданный формат")
    checks.append(
        Check(
            "Direct Chainlink Data Streams",
            "INCOMPLETE" if chainlink_issues else "CONFIGURED",
            "; ".join(chainlink_issues) if chainlink_issues else "Credentials and feed metadata are present; protocol-specific test is still required.",
            "Уточнить endpoint и схему auth в Chainlink Self Service. До этого использовать Chainlink через Polymarket RTDS.",
        )
    )

    telegram_fields = (
        getattr(config, "TELEGRAM_API_ID", ""),
        getattr(config, "TELEGRAM_API_HASH", ""),
        getattr(config, "TELEGRAM_PHONE", ""),
    )
    checks.append(
        Check(
            "Telegram MTProto",
            "CONFIGURED" if all(telegram_fields) else "INCOMPLETE",
            "MTProto credentials present." if all(telegram_fields) else "API_ID/API_HASH/PHONE заполнены не полностью.",
            "Получить api_id/api_hash на my.telegram.org/apps и указать отдельный номер collector-аккаунта.",
        )
    )

    checks.append(
        Check(
            "Local Qwen",
            "CONFIGURED",
            f"Model configured: {settings.QWEN_MODEL_ID}. Model files download on first use.",
            "При нехватке RAM уменьшить модель или использовать quantized runtime.",
        )
    )
    return checks


def write_report(checks: list[Check]) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [
        "# Аудит API-подключений",
        "",
        f"Проверено: `{now}`",
        "",
        "> Секреты в отчёт не записываются.",
        "",
        "| Подключение | Статус | Результат | Что исправить |",
        "|---|---|---|---|",
    ]
    for item in checks:
        detail = item.detail.replace("|", "\\|").replace("\n", " ")
        fix = item.fix.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item.service} | **{item.status}** | {detail} | {fix} |")
    lines += [
        "",
        "## Интерпретация",
        "",
        "- `OK` — сетевое подключение и базовый формат ответа проверены.",
        "- `CONFIGURED` — конфигурация заполнена, но полноценный authenticated flow ещё не выполнялся.",
        "- `INCOMPLETE` — обязательные поля отсутствуют или неоднозначны.",
        "- `OPTIONAL` — источник не блокирует основной Polymarket pipeline.",
        "- `ERROR` — сетевой запрос или проверка API завершились ошибкой.",
        "",
        "## Важные замечания",
        "",
        "- Binance полностью исключён из проекта.",
        "- CryptoRank исключён как платный необязательный источник.",
        "- Canonical resolution source для BTC Up/Down остаётся Chainlink через Polymarket RTDS.",
        "- Pyth BTC/USD добавлен как независимый oracle для сверки цены и confidence interval.",
        "- LLM формирует структурированный сигнал, но не имеет прямого доступа к размещению ордеров.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    config = load_config()
    checks = await run_checks(config)
    write_report(checks)
    counts: dict[str, int] = {}
    for item in checks:
        counts[item.status] = counts.get(item.status, 0) + 1
    print("Connection audit completed:", json.dumps(counts, ensure_ascii=False))
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    from polybot.runtime import run_async
    run_async(__file__, main)
