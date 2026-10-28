"""Persistence (naive) forecasts on ordered trading observations."""
from numbers import Integral
import pandas as pd


OUTPUT_COLUMNS = ["origin_date", "target_date", "horizon", "model", "y_pred"]


def predict_naive(prices: pd.DataFrame, origins, horizons=(1, 5)) -> pd.DataFrame:
    """For each forecast origin t and horizon h, return C[t] as the forecast.

    Origin positions are indices of the *sorted* trading observations. Forecast
    computation does not read any future Close, only future dates for labeling.
    Incomplete horizons are rejected instead of being silently dropped.
    """
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("prices must contain date and close")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    if p["date"].duplicated().any():
        raise ValueError("Duplicate trading dates")
    if p["close"].isna().any():
        raise ValueError("Missing Close values")
    p = p.sort_values("date").reset_index(drop=True)
    horizons = tuple(horizons)
    if not horizons or any(not isinstance(h, Integral) or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Duplicate horizons")

    out = []
    for i in origins:
        if not isinstance(i, Integral) or not 0 <= i < len(p):
            raise IndexError(f"Invalid forecast origin: {i}")
        for h in horizons:
            if i + h >= len(p):
                raise ValueError(f"No observed target date for origin {i}, horizon {h}")
            out.append({
                "origin_date": p.at[i, "date"].date().isoformat(),
                "target_date": p.at[i+h, "date"].date().isoformat(),
                "horizon": int(h),
                "model": "naive",
                "y_pred": float(p.at[i, "close"]),
            })
    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)
