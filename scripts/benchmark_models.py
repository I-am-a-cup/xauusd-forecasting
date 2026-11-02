"""Сопоставимое время финального обучения и прогноза четырёх методов.

Запуск: python scripts/benchmark_models.py
Только отобранные ДО теста гиперпараметры; CV не включено в замер.
Все модели получают один ряд, одинаковые 47 исходных дат и 94 прогноза.
Первый прогон — прогрев; итог — медиана 3 последующих прогонов.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily_features import model_table
from src.metrics import assert_comparable_predictions
from src.models.arima import predict_arima
from src.models.baseline import predict_naive
from src.models.mlp import fit_mlp, forecast_frame, load_selected_configs, predict_fitted
from src.models.random_forest import RFConfig, fit_rf, predict_rf_fitted
from src.validation import holdout_split


def _rf_config(obj: dict) -> RFConfig:
    return RFConfig(**obj)


def measure_one(prices, split, table, configs, arima_orders, rf_configs):
    train = table[table.split.eq("train")]
    test = table[table.split.eq("test")]
    durations = []
    predictions = []

    # Naive: нет обучения. Одна операция возвращает прогнозы для двух горизонтов.
    t = perf_counter()
    naive = predict_naive(prices, split.forecast_origins)
    naive_time = perf_counter() - t
    predictions.append(naive)
    durations.append(dict(model="naive", fit_seconds=0.0,
                          predict_seconds=naive_time, predictions=len(naive)))

    arima_frames = []
    arima_train = 0.0
    arima_predict = 0.0
    for horizon in (1, 5):
        frame, stats = predict_arima(
            prices, split.last_known_index, split.forecast_origins,
            arima_orders[horizon], horizons=(horizon,))
        arima_train += stats["fit_seconds"]
        # Включает обновление ARIMA-состояния на каждой новой сессии.
        arima_predict += stats["predict_seconds"]
        arima_frames.append(frame)
    arima_frame = pd.concat(arima_frames, ignore_index=True)
    predictions.append(arima_frame)
    durations.append(dict(model="arima", fit_seconds=arima_train,
                          predict_seconds=arima_predict, predictions=len(arima_frame)))

    mlp_frames = []
    mlp_train = 0.0
    mlp_predict = 0.0
    for horizon in (1, 5):
        model, info = fit_mlp(train, horizon, configs[horizon], random_state=42)
        mlp_train += info["fit_seconds"]
        start = perf_counter()
        vals = predict_fitted(model, test)
        mlp_predict += perf_counter() - start
        mlp_frames.append(forecast_frame(prices, test, horizon, vals))
    mlp_frame = pd.concat(mlp_frames, ignore_index=True)
    predictions.append(mlp_frame)
    durations.append(dict(model="mlp", fit_seconds=mlp_train,
                          predict_seconds=mlp_predict, predictions=len(mlp_frame)))

    dates = pd.to_datetime(prices.date).dt.strftime("%Y-%m-%d").to_numpy()
    pos = pd.Series(np.arange(len(dates)), index=pd.to_datetime(prices.date))
    positions = pd.to_datetime(test.date).map(pos).to_numpy(dtype=int)
    rf_frames = []
    rf_train = 0.0
    rf_predict = 0.0
    for horizon in (1, 5):
        model, fit_seconds = fit_rf(train, horizon, rf_configs[horizon], seed=42)
        rf_train += fit_seconds
        start = perf_counter()
        vals = predict_rf_fitted(model, test)
        rf_predict += perf_counter() - start
        rf_frames.append(pd.DataFrame({
            "origin_date": dates[positions], "target_date": dates[positions+horizon],
            "horizon": horizon, "model": "random_forest", "y_pred": vals,
        }))
    rf_frame = pd.concat(rf_frames, ignore_index=True)
    predictions.append(rf_frame)
    durations.append(dict(model="random_forest", fit_seconds=rf_train,
                          predict_seconds=rf_predict, predictions=len(rf_frame)))

    combined = pd.concat(predictions, ignore_index=True)
    assert_comparable_predictions(
        combined, expected_models=("naive", "arima", "mlp", "random_forest"))
    if len(combined) != 376:
        raise AssertionError("Ожидалось 4 модели × 2 горизонта × 47 дат")
    return durations


def run(repeats: int = 3) -> pd.DataFrame:
    source = ROOT / "data/processed/xauusd_daily_dukascopy.csv"
    prices = pd.read_csv(source)
    split = holdout_split(len(prices))
    table = model_table(prices, split.last_known_index, split.forecast_origins)
    configs = load_selected_configs(ROOT / "results/mlp/metadata.json")
    selected = json.loads((ROOT / "results/arima_rf/metadata.json").read_text(encoding="utf-8"))
    arima_orders = {h: tuple(selected["selected_arima"][str(h)]) for h in (1, 5)}
    rf_configs = {h: _rf_config(selected["selected_rf"][str(h)]) for h in (1, 5)}

    if repeats < 1:
        raise ValueError("Нужно как минимум одно измерение")
    records = []
    with threadpool_limits(limits=1):
        for attempt in range(repeats + 1):
            for row in measure_one(prices, split, table, configs, arima_orders, rf_configs):
                if attempt:  # первый проход — прогрев
                    records.append({"repeat": attempt, **row})
    frame = pd.DataFrame(records)
    summary = frame.groupby("model", sort=True).agg(
        fit_median_s=("fit_seconds", "median"),
        predict_median_s=("predict_seconds", "median"),
        predictions=("predictions", "first"),
        repeats=("repeat", "nunique"),
    ).reset_index()
    # Медиану суммы считаем по каждому прогону, не складываем медианы компонент.
    frame["total_seconds"] = frame.fit_seconds + frame.predict_seconds
    totals = frame.groupby("model").total_seconds.median()
    summary["total_median_s"] = summary.model.map(totals)
    summary["fit_min_s"] = summary.model.map(frame.groupby("model").fit_seconds.min())
    summary["fit_max_s"] = summary.model.map(frame.groupby("model").fit_seconds.max())
    summary["predict_min_s"] = summary.model.map(frame.groupby("model").predict_seconds.min())
    summary["predict_max_s"] = summary.model.map(frame.groupby("model").predict_seconds.max())

    target = ROOT / "results/benchmark"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "repeats.csv", index=False, encoding="utf-8")
    summary.to_csv(target / "model_timings.csv", index=False, encoding="utf-8")
    env = {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "threadpool_limit": 1,
        "random_seed": 42,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "models": ["naive", "arima", "mlp", "random_forest"],
        "warmup_runs": 1, "measured_runs": repeats,
        "forecasts_per_model": 94,
        "last_known_train_date": prices.date.iloc[split.last_known_index],
        "forecast_origin_start": prices.date.iloc[split.forecast_origins[0]],
        "forecast_origin_end": prices.date.iloc[split.forecast_origins[-1]],
        "timing_scope": "Final fit of two selected horizons and all 94 forecasts; no CV search, no plots, file writing or data download.",
        "prediction_scope": "ARIMA includes state updates without refit; RF and MLP do fixed-parameter prediction.",
        "warning": "Wall-clock times are local to this computer and current versions, not direct results from the original authors' environments.",
    }
    (target / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ONE-MACHINE BENCHMARK (seconds; three measured runs):")
    print(summary.round(6).to_string(index=False))
    print("ENV:", env["python"], env["os"], "threads=", env["threadpool_limit"])
    return summary


if __name__ == "__main__":
    run()
