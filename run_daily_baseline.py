"""Сбор дневных котировок Dukascopy и проверка наивного прогноза.

Запуск: python run_daily_baseline.py
Источники данных остаются раздельными. Здесь используется ТОЛЬКО Dukascopy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.daily_sessions import build_daily_sessions
from src.daily_features import model_table
from src.models.baseline import predict_naive
from src.metrics import attach_actuals, assert_comparable_predictions, evaluate
from src.validation import holdout_split, expanding_folds


RAW = Path("data/raw/xauusd-h1-bid-2025-10-27-2026-10-28.csv")


def run(raw_path: Path, save: bool = True) -> dict:
    if not raw_path.is_file():
        raise FileNotFoundError(f"Отсутствует исходный часовой CSV: {raw_path}")
    raw = pd.read_csv(raw_path, low_memory=False)
    daily, audit = build_daily_sessions(raw)
    price = daily[["date", "close"]].copy()
    split = holdout_split(len(price), test_share=0.20, horizons=(1, 5),
                          min_train_observations=90)
    folds = expanding_folds(
        len(price), initial_train_observations=105, validation_size=25,
        step_size=25, horizons=(1, 5),
        end_before=split.last_known_index+1,
    )
    samples = model_table(daily, split.last_known_index, split.forecast_origins)
    preds = predict_naive(price, list(split.forecast_origins))
    assert_comparable_predictions(preds)
    evaluated = attach_actuals(preds, price)
    metrics = evaluate(evaluated)
    cutoff = price.loc[split.last_known_index, "date"]
    assert pd.to_datetime(samples[samples.split == "train"]["date"]).max() < pd.to_datetime(cutoff)
    report = {
        **audit,
        "dataset_rows": len(samples),
        "train_rows": int(samples.split.eq("train").sum()),
        "test_rows": int(samples.split.eq("test").sum()),
        "last_observed_training_date": cutoff,
        "holdout_first_origin": price.loc[split.forecast_origins[0], "date"],
        "holdout_last_origin": price.loc[split.forecast_origins[-1], "date"],
        "holdout_predictions_each_horizon": len(split.forecast_origins),
        "expanding_validation_folds": len(folds),
        "folds": [
            {"last_train_date": price.loc[f.last_known_index, "date"],
             "first_validation_origin": price.loc[f.forecast_origins[0], "date"],
             "last_validation_origin": price.loc[f.forecast_origins[-1], "date"]}
            for f in folds
        ],
        "features_note": (
            "10 признаков, только золото. Нет внешних индексов или заполнения пропусков. "
            "Нормализация не применяется. Для обучаемых моделей её параметры "
            "следует подбирать только на обучающем окне."
        ),
        "conclusion_scope": (
            "Метрики относятся только к наивному прогнозу, не к ARIMA, Random Forest или MLP. "
            "Сокращённые праздничные сессии отмечены для отдельной проверки."
        ),
    }
    if save:
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("results/metrics").mkdir(parents=True, exist_ok=True)
        Path("results/predictions").mkdir(parents=True, exist_ok=True)
        daily.to_csv("data/processed/xauusd_daily_dukascopy.csv", index=False, encoding="utf-8-sig")
        samples.to_csv("data/processed/xauusd_daily_model_ready.csv", index=False, encoding="utf-8-sig")
        evaluated.to_csv("results/predictions/naive_daily.csv", index=False, encoding="utf-8-sig")
        metrics.to_csv("results/metrics/naive_daily.csv", index=False, encoding="utf-8-sig")
        Path("data/processed/xauusd_daily_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("Dukascopy D1, торговые сессии:", len(daily))
    print("Даты:", daily.date.iloc[0], "—", daily.date.iloc[-1])
    print("Сокращённые сессии:", len(audit["short_sessions_for_review"]))
    print("Обучение/тест:", report["train_rows"], "/", report["test_rows"])
    print("Временных окон валидации:", len(folds))
    print("NAIVE (фактические котировки):")
    print(metrics.to_string(index=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Дневные сессии Dukascopy и baseline")
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    run(args.raw)


if __name__ == "__main__":
    main()
