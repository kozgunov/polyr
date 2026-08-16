"""Безопасный исполнитель Polymarket CLOB для live-canary.

Preflight никогда не отправляет заявку. Реальная отправка требует одновременно
разрешения в конфигурации, снятого kill switch и явного ``submit=True``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

import api_config as api
import app_config as settings
from py_clob_client_v2 import ApiCreds, ClobClient, OrderArgs, PartialCreateOrderOptions
from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams, OrderType


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
    raw = build_client().cancel_order(str(order_id))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        return {"canceled": list(raw)}
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    return {"result": raw}
