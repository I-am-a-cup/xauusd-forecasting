"""ARIMA с фиксированными параметрами: обновляется состояние, но не веса."""
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

CANDIDATES = ((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1))

def predict_arima(prices: pd.DataFrame, last_known_index: int,
                  origins: tuple[int, ...], order: tuple[int, int, int],
                  horizons=(1, 5)) -> tuple[pd.DataFrame, dict]:
    """Прогноз на каждой origin-дате из состоявшихся наблюдений <= origin.

    К моменту фиксации модели известны цены 0..last_known_index.
    На test добавляем цены последующих завершённых свечей через extend,
    не повторяем fit и не подбираем порядок по test.
    """
    if not origins or last_known_index < 10:
        raise ValueError("Недостаточно данных или нет дат прогноза")
    close = prices.close.to_numpy(dtype=float)
    dates = pd.to_datetime(prices.date).dt.strftime("%Y-%m-%d").to_numpy()
    origins = tuple(origins)
    if any(i <= last_known_index or i+max(horizons) >= len(close) for i in origins):
        raise ValueError("Недопустимые даты происхождения или цели")
    if tuple(sorted(set(origins))) != origins:
        raise ValueError("Даты прогнозов должны быть уникальными и упорядоченными")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Неверные исходные цены")

    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Явно фиксируем отсутствие детерминированного дрейфа.
        # Для d=1 это также значение по умолчанию statsmodels.
        state = ARIMA(close[:last_known_index+1], order=order, trend="n").fit()
    fit_seconds = perf_counter()-started
    result = []
    observed_last = last_known_index
    started = perf_counter()
    for origin in origins:
        new_known = close[observed_last+1:origin+1]
        if len(new_known):
            state = state.extend(new_known)
        observed_last = origin
        forecast = np.asarray(state.forecast(steps=max(horizons)), dtype=float)
        if not np.isfinite(forecast).all():
            raise ValueError("ARIMA дала NaN или бесконечность")
        for horizon in horizons:
            result.append(dict(origin_date=dates[origin],
                               target_date=dates[origin+horizon],
                               horizon=horizon, model="arima",
                               y_pred=float(forecast[horizon-1])))
    info = {
        "order": list(order),
        "last_known_date": str(dates[last_known_index]),
        "first_forecast_date": str(dates[origins[0]]),
        "train_observations": last_known_index+1,
        "fit_seconds": fit_seconds,
        "predict_seconds": perf_counter()-started,
        "warnings": [str(w.message) for w in caught],
        "refit_on_test": False,
    }
    return pd.DataFrame(result), info
