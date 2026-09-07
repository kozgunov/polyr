"""Безопасный исполнитель Polymarket CLOB для live-canary.

Preflight никогда не отправляет заявку. Реальная отправка требует одновременно
разрешения в конфигурации, снятого kill switch и явного ``submit=True``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from functools import lru_cache
from typing import Any

import api_config as api
import app_config as settings
from py_clob_client_v2 import ApiCreds, ClobClient, OrderArgs, PartialCreateOrderOptions
from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams, OrderPayload, OrderType, TradeParams

from polybot.trading.fees import platform_fee_usdc


@dataclass(frozen=True, slots=True)
class LivePreflight:
    authenticated: bool
    collateral_readable: bool
    allowance_readable: bool
    orderbook_readable: bool
    signed_order_created: bool
    token_id: str
    tick_size: str | None

    @property
    def ready(self) -> bool:
        return all((self.authenticated, self.collateral_readable, self.allowance_readable,
                    self.orderbook_readable, self.signed_order_created))

    def public_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ready": self.ready}


def _required(name: str) -> str:
    value = str(getattr(api, name, "") or "").strip()
    if not value:
        raise RuntimeError(f"{name} is empty")
    return value


def build_client() -> ClobClient:
    """Создаёт аутентифицированный клиент, не записывая секреты в журнал."""
    credentials = ApiCreds(
        api_key=_required("POLYMARKET_API_KEY"),
        api_secret=_required("POLYMARKET_API_SECRET"),
        api_passphrase=_required("POLYMARKET_API_PASSPHRASE"),
    )
    return ClobClient(
        host=_required("POLYMARKET_CLOB_URL"),
        chain_id=int(getattr(api, "POLYGON_CHAIN_ID", 137)),
        key=_required("POLYMARKET_PRIVATE_KEY"),
        creds=credentials,
        signature_type=int(getattr(api, "POLYMARKET_SIGNATURE_TYPE")),
        funder=_required("POLYMARKET_FUNDER_ADDRESS"),
        use_server_time=True,
        retry_on_error=True,
    )


def preflight(token_id: str) -> LivePreflight:
    """Проверяет L2, balance/allowance, стакан и локальную подпись без POST /order."""
    client = build_client()
    collateral = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    conditional = client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=str(token_id))
    )
    book = client.get_order_book(str(token_id))
    tick_size = str(client.get_tick_size(str(token_id)))
    neg_risk = bool(client.get_neg_risk(str(token_id)))
    expiration = int((datetime.now(UTC) + timedelta(seconds=90)).timestamp())
    signed = client.create_order(
        OrderArgs(token_id=str(token_id), price=0.50, size=2.0, side="BUY", expiration=expiration),
        PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk),
    )
    return LivePreflight(
        authenticated=True,
        collateral_readable=collateral is not None,
        allowance_readable=conditional is not None,
        orderbook_readable=book is not None,
        signed_order_created=signed is not None,
        token_id=str(token_id),
        tick_size=tick_size,
    )


def market_minimum_size(token_id: str) -> float:
    """Читает minimum shares непосредственно из текущего CLOB-стакана."""
    book = build_client().get_order_book(str(token_id))
    value = getattr(book, "min_order_size", None)
    if value is None and isinstance(book, dict):
        value = book.get("min_order_size")
    return max(1.0, float(value or 1.0))


def _gtd_expiration(client: ClobClient, lifetime_seconds: int) -> int:
    """Возвращает GTD-expiration по времени CLOB, а не по часам ноутбука.

    CLOB отклоняет GTD, если expiry уже наступил или находится слишком близко
    к серверному времени.  Поэтому оставляем обязательный минутный буфер и
    минимум две минуты жизни заявки после него. Добавлен пятисекундный запас,
    потому что CLOB требует expiry строго дальше трёх минут.
    """
    try:
        server_now = int(float(client.get_server_time()))
    except (TypeError, ValueError):
        # Резервный вариант нужен только при временной недоступности /time.
        server_now = int(datetime.now(UTC).timestamp())
    effective_lifetime = max(125, int(lifetime_seconds))
    expiration = server_now + 60 + effective_lifetime
    if expiration <= server_now + 60:
        raise RuntimeError("INVALID_GTD_EXPIRATION")
    return expiration


def submit_limit_order(*, token_id: str, side: str, price: float, size: float,
                       order_type: str = "GTD", lifetime_seconds: int = 20,
                       post_only: bool = False, max_notional: float | None = None,
                       submit: bool = False) -> dict[str, Any]:
    """Создаёт и отправляет только лимитную заявку с тройным явным разрешением."""
    if not submit:
        raise RuntimeError("LIVE_SUBMISSION_REQUIRES_EXPLICIT_SUBMIT")
    if not settings.LIVE_TRADING_ENABLED or settings.KILL_SWITCH:
        raise RuntimeError("LIVE_SUBMISSION_BLOCKED_BY_CONFIGURATION")
    if side not in {"BUY", "SELL"} or order_type not in {"GTC", "GTD"}:
        raise ValueError("Canary permits only GTC/GTD limit orders")
    if not 0 < float(price) < 1 or float(size) <= 0:
        raise ValueError("Invalid price or size")
    limit = float(max_notional) if max_notional is not None else settings.MAX_POSITION_USDC
    if side == "BUY" and float(price) * float(size) > limit + 1e-9:
        raise ValueError("Canary order exceeds configured maximum notional")
    client = build_client()
    tick_size = str(client.get_tick_size(str(token_id)))
    tick = Decimal(tick_size)
    raw_price = Decimal(str(price))
    rounding = ROUND_FLOOR if side == "BUY" else ROUND_CEILING
    normalized_price = float((raw_price / tick).to_integral_value(rounding=rounding) * tick)
    if not 0 < normalized_price < 1:
        raise ValueError("Price is outside market tick range")
    neg_risk = bool(client.get_neg_risk(str(token_id)))
    book = client.get_order_book(str(token_id))
    minimum = getattr(book, "min_order_size", None)
    if minimum is None and isinstance(book, dict):
        minimum = book.get("min_order_size")
    if float(size) + 1e-9 < float(minimum or 1.0):
        raise ValueError(f"ORDER_BELOW_MARKET_MINIMUM:size={size:.6f},minimum={float(minimum or 1.0):.6f}")
    expiration = 0
    if order_type == "GTD":
        expiration = _gtd_expiration(client, lifetime_seconds)
    signed = client.create_order(
        OrderArgs(token_id=str(token_id), price=normalized_price, size=float(size), side=side,
                  expiration=expiration),
        PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk),
    )
    response = client.post_order(signed, getattr(OrderType, order_type), post_only=post_only)
    if not isinstance(response, dict):
        response = dict(response)
    return {
        "success": bool(response.get("success")), "order_id": response.get("orderID"),
        "status": response.get("status"), "making_amount": response.get("makingAmount"),
        "taking_amount": response.get("takingAmount"), "error": response.get("errorMsg") or "",
        "expiration": expiration, "tick_size": tick_size, "submitted_price": normalized_price,
    }


def get_order(order_id: str) -> dict[str, Any]:
    """Возвращает нормализованное фактическое состояние CLOB-заявки."""
    raw = build_client().get_order(str(order_id))
    if not isinstance(raw, dict):
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif hasattr(raw, "__dict__"):
            raw = vars(raw)
        else:
            raw = {"status": str(raw)}
    return {
        "order_id": raw.get("id") or order_id,
        "status": str(raw.get("status") or "unknown"),
        "side": str(raw.get("side") or ""),
        "token_id": str(raw.get("asset_id") or ""),
        "price": float(raw.get("price") or 0),
        "original_size": float(raw.get("original_size") or 0),
        "size_matched": float(raw.get("size_matched") or 0),
        "expiration": int(raw.get("expiration") or 0),
    }


def cancel_order(order_id: str) -> dict[str, Any]:
    """Отменяет одну конкретную заявку; массовая отмена здесь намеренно недоступна."""
    raw = build_client().cancel_order(OrderPayload(orderID=str(order_id)))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        return {"canceled": list(raw)}
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    return {"result": raw}


def get_order_trades(order_id: str) -> list[dict[str, Any]]:
    """Возвращает реальные CLOB fills заявки для расчёта VWAP и комиссии."""
    rows = build_client().get_trades(TradeParams(order_id=str(order_id)))
    result: list[dict[str, Any]] = []
    for row in rows or []:
        if hasattr(row, "model_dump"):
            row = row.model_dump()
        elif hasattr(row, "to_dict"):
            row = row.to_dict()
        elif hasattr(row, "__dict__"):
            row = vars(row)
        if isinstance(row, dict):
            result.append(row)
    return result


def get_account_trades() -> list[dict[str, Any]]:
    """Возвращает фактическую историю fills текущего CLOB-аккаунта.

    Это авторитетный резервный источник: ордер мог успеть исполниться,
    даже если локальный polling get_order опоздал или завершился ошибкой.
    """
    rows = build_client().get_trades()
    result: list[dict[str, Any]] = []
    for row in rows or []:
        if hasattr(row, "model_dump"):
            row = row.model_dump()
        elif hasattr(row, "to_dict"):
            row = row.to_dict()
        elif hasattr(row, "__dict__"):
            row = vars(row)
        if isinstance(row, dict):
            result.append(row)
    return result


@lru_cache(maxsize=512)
def get_market_fee_schedule(condition_id: str) -> dict[str, float | bool | str]:
    """Возвращает актуальную feeSchedule CLOB V2 для конкретного рынка.

    В V2 комиссия задаётся оператором в момент match. Поэтому
    ``fee_rate_bps`` в trade history может быть нулём даже для рынка с
    включённой taker-комиссией. Авторитетен блок ``fd`` market info.
    """
    fallback = {
        "fee_rate": float(settings.POLYMARKET_CRYPTO_TAKER_FEE_RATE),
        "fee_exponent": 1.0,
        "taker_only": True,
        "source": "configured_crypto_fallback",
    }
    if not str(condition_id or "").strip():
        return fallback
    try:
        info = build_client().get_clob_market_info(str(condition_id)) or {}
        details = info.get("fd") or info.get("feeSchedule") or {}
        if not details:
            return fallback
        return {
            "fee_rate": max(0.0, float(details.get("r", details.get("rate", fallback["fee_rate"])) or 0)),
            "fee_exponent": max(0.0, float(details.get("e", details.get("exponent", 1.0)) or 1.0)),
            "taker_only": bool(details.get("to", details.get("takerOnly", True))),
            "source": "clob_market_info",
        }
    except Exception:
        # Сбой fee endpoint не должен останавливать reconciliation.
        return fallback


def summarize_order_fills(order_id: str) -> dict[str, float]:
    """Считает фактические size/VWAP/fee только по legs указанной заявки."""
    legs: list[tuple[float, float, float, bool]] = []
    for trade in get_order_trades(order_id):
        schedule = get_market_fee_schedule(str(trade.get("market") or ""))
        if str(trade.get("taker_order_id") or "") == str(order_id):
            legs.append((float(trade.get("size") or 0), float(trade.get("price") or 0),
                         float(trade.get("fee_rate_bps") or 0), True,
                         float(schedule["fee_rate"]), float(schedule["fee_exponent"]),
                         bool(schedule["taker_only"])))
        for maker in trade.get("maker_orders") or []:
            if str(maker.get("order_id") or "") == str(order_id):
                legs.append((float(maker.get("matched_amount") or 0), float(maker.get("price") or 0),
                             float(maker.get("fee_rate_bps") or 0), False,
                             float(schedule["fee_rate"]), float(schedule["fee_exponent"]),
                             bool(schedule["taker_only"])))
    size = sum(item[0] for item in legs)
    notional = sum(item[0] * item[1] for item in legs)
    fee = sum(matched_trade_fee_usdc(
        item[0], item[1], item[2], taker=item[3], fee_rate=item[4],
        fee_exponent=item[5], taker_only=item[6],
    ) for item in legs)
    return {"size": size, "notional": notional, "vwap": notional / size if size else 0.0, "fee": fee}


def matched_trade_fee_usdc(
    size: float,
    price: float,
    fee_rate_bps: float,
    *,
    taker: bool,
    fee_rate: float | None = None,
    fee_exponent: float = 1.0,
    taker_only: bool = True,
) -> float:
    """Фактическая CLOB fee по feeSchedule рынка.

    ``fee_rate_bps`` оставлен как fallback для старых V1 fills и тестов.
    Для V2 передаётся десятичная ставка из ``get_clob_market_info().fd``.
    """
    rate = float(fee_rate) if fee_rate is not None else float(fee_rate_bps or 0) / 10_000.0
    return platform_fee_usdc(
        size, price, taker=taker, fee_rate=rate, fee_exponent=fee_exponent,
        taker_only=taker_only, fees_enabled=rate > 0,
    )
