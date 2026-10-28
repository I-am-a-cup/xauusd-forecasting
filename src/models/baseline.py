"""Persistence (naive) forecast on an ordered set of trading observations."""
import pandas as pd


OUTPUT_COLUMNS = ["origin_date", "target_date", "horizon", "model", "y_pred"]


def predict_naive(prices: pd.DataFrame, origins, horizons=(1, 5)) -> pd.DataFrame:
    """Forecast the last observed Close at t for horizons h in observations.

    origins: integer row positions in sorted input, after row t's Close is known.
    Does not fit parameters and never reads target Close to produce a forecast.
    """
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("prices must contain date and close")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    if p["date"].duplicated().any():
        raise ValueError("duplicate trading dates")
    if p["close"].isna().any():
        raise ValueError("missing close values")
    p = p.sort_values("date").reset_index(drop=True)
    out = []
    for i in origins:
        if not isinstance(i, int) or not 0 <= i < len(p):
            raise IndexError(f"invalid origin position: {i}")
        for h in horizons:
            if not isinstance(h, int) or h < 1:
                raise ValueError("horizons must be positive integers")
            if i + h >= len(p):
                continue
            out.append({
                "origin_date": p.at[i, "date"].date().isoformat(),
                "target_date": p.at[i+h, "date"].date().isoformat(),
                "horizon": h,
                "model": "naive",
                "y_pred": float(p.at[i, "close"]),
            })
    return pd.DataFrame(out, columns=OUTPUT_COLUMNS)
