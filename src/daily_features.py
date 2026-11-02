"""Лаговые признаки для дневного золота — только текущие и прошлые цены."""
import numpy as np
import pandas as pd


FEATURES = (
    "close", "lag_close_1", "lag_close_2", "lag_close_5",
    "return_1", "return_5", "range_pct",
    "mean_close_5", "std_return_5", "weekday",
)
WARMUP = 10


def model_table(prices: pd.DataFrame, last_known_index: int,
                forecast_origins: tuple[int, ...]) -> pd.DataFrame:
    """Подготовить малый набор признаков и цели, не допуская утечки.

    Цели рассчитываются ПОСЛЕ формирования календаря и до отбора строк.
    Разделение допускает только те train-цели, которые были известны
    на дату отсечения, в т.ч. для горизонта 5.
    """
    d = prices.copy().sort_values("date").reset_index(drop=True)
    if d["date"].duplicated().any():
        raise ValueError("Повторяются торговые дни")
    c = d["close"]
    d["lag_close_1"] = c.shift(1)
    d["lag_close_2"] = c.shift(2)
    d["lag_close_5"] = c.shift(5)
    d["return_1"] = c.pct_change(1, fill_method=None)
    d["return_5"] = c.pct_change(5, fill_method=None)
    d["range_pct"] = (d["high"] - d["low"]) / c
    d["mean_close_5"] = c.rolling(5).mean()
    d["std_return_5"] = d["return_1"].rolling(5).std()
    d["weekday"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["target_close_next_1d"] = c.shift(-1)
    d["target_close_next_5d"] = c.shift(-5)
    permitted = set(forecast_origins)
    max_origin_train = last_known_index - 5
    split = [
        "train" if i <= max_origin_train
        else "test" if i in permitted else "excluded"
        for i in range(len(d))
    ]
    d["split"] = split
    d = d[(d["split"] != "excluded") & d[list(FEATURES)].notna().all(axis=1)
          & d[["target_close_next_1d", "target_close_next_5d"]].notna().all(axis=1)].copy()
    if d.empty or not {"train", "test"}.issubset(set(d["split"])):
        raise ValueError("Невозможно сформировать train и test из выбранного ряда")
    if not np.isfinite(d[list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("В признаках есть бесконечные значения")
    return d[["date", "split", *FEATURES, "target_close_next_1d", "target_close_next_5d"]]
