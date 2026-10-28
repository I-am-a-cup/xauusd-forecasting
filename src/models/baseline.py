"""Наивный прогноз сохранения последней известной цены в торговом ряду."""
from numbers import Integral
import pandas as pd


OUTPUT_COLUMNS = ["origin_date", "target_date", "horizon", "model", "y_pred"]


def predict_naive(prices: pd.DataFrame, origins, horizons=(1, 5)) -> pd.DataFrame:
    """Для даты формирования прогноза t и горизонта h возвращает цену C[t].

    origins — позиции в ряду, отсортированном по торговым датам.
    Будущие значения Close не используются: будущие даты нужны лишь для подписей.
    Если для горизонта нет целевой даты, функция явно сообщает об ошибке.
    """
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("В таблице цен должны быть столбцы date и close")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    if p["date"].duplicated().any():
        raise ValueError("Обнаружены повторяющиеся торговые даты")
    if p["close"].isna().any():
        raise ValueError("Обнаружены пропуски в ценах закрытия")
    p = p.sort_values("date").reset_index(drop=True)
    horizons = tuple(horizons)
    if not horizons or any(not isinstance(h, Integral) or h < 1 for h in horizons):
        raise ValueError("Горизонты должны быть положительными целыми числами")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Горизонты прогноза не должны повторяться")

    out = []
    for i in origins:
        if not isinstance(i, Integral) or not 0 <= i < len(p):
            raise IndexError(f"Недопустимая позиция даты прогноза: {i}")
        for h in horizons:
            if i + h >= len(p):
                raise ValueError(f"Нет доступной целевой даты для позиции {i}, горизонт {h}")
            out.append({
                "origin_date": p.at[i, "date"].date().isoformat(),
                "target_date": p.at[i+h, "date"].date().isoformat(),
                "horizon": int(h),
                "model": "naive",
                "y_pred": float(p.at[i, "close"]),
            })
    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)
