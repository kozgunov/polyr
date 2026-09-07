"""ARIMA/GARCH-признаки для shadow-эксперимента; модуль не отправляет заявки."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class TimeSeriesForecast:
    last_price: float
    arima_price: float
    arima_log_return: float
    arima_direction: str
    garch_sigma_per_step: float
    horizon_steps: int
    observations: int
    execution_role: str = "shadow_feature_only"

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _returns(prices: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 40:
        raise ValueError("ARIMA/GARCH требует минимум 40 положительных цен")
    return np.diff(np.log(values))


def _garch11_sigma(returns: np.ndarray, horizon_steps: int) -> float:
    """MLE GARCH(1,1); прогнозирует риск, а не направление."""
    scaled = np.asarray(returns, dtype=float) * 10_000.0
    variance = max(float(np.var(scaled)), 1e-8)

    def logistic(value: float) -> float:
        # Optimizer may probe extreme unconstrained values; clipping preserves
        # the asymptote and prevents numeric overflow from invalidating an event.
        bounded = max(-50.0, min(50.0, float(value)))
        return 1.0 / (1.0 + math.exp(-bounded))

    def objective(raw: np.ndarray) -> float:
        omega = math.exp(max(-50.0, min(50.0, float(raw[0]))))
        alpha = logistic(float(raw[1])) * 0.45
        beta = logistic(float(raw[2])) * (0.995 - alpha)
        sigma2 = np.empty_like(scaled)
        sigma2[0] = variance
        for index in range(1, scaled.size):
            sigma2[index] = max(1e-8, omega + alpha * scaled[index - 1] ** 2 + beta * sigma2[index - 1])
        return float(0.5 * np.sum(np.log(sigma2) + scaled**2 / sigma2))

    fitted = minimize(objective, np.array([math.log(variance * .05), -1.4, 2.2]), method="L-BFGS-B")
    raw = fitted.x
    omega = math.exp(max(-50.0, min(50.0, float(raw[0]))))
    alpha = logistic(float(raw[1])) * 0.45
    beta = logistic(float(raw[2])) * (0.995 - alpha)
    sigma2 = variance
    for index in range(1, scaled.size):
        sigma2 = max(1e-8, omega + alpha * scaled[index - 1] ** 2 + beta * sigma2)
    for _ in range(max(1, int(horizon_steps))):
        sigma2 = max(1e-8, omega + (alpha + beta) * sigma2)
    return math.sqrt(sigma2) / 10_000.0


def forecast(prices: list[float], horizon_steps: int = 6) -> TimeSeriesForecast:
    """Строит независимые trend/risk признаки без принятия торгового решения."""
    returns = _returns(prices)
    order = (1, 0, 1) if returns.size >= 80 else (1, 0, 0)
    fitted = ARIMA(returns, order=order, trend="c").fit()
    predicted = np.asarray(fitted.forecast(steps=max(1, int(horizon_steps))), dtype=float)
    cumulative = float(predicted.sum())
    last = float(prices[-1])
    future = last * math.exp(cumulative)
    return TimeSeriesForecast(
        last_price=last,
        arima_price=future,
        arima_log_return=cumulative,
        arima_direction="Up" if future >= last else "Down",
        garch_sigma_per_step=_garch11_sigma(returns, horizon_steps),
        horizon_steps=max(1, int(horizon_steps)),
        observations=len(prices),
    )
