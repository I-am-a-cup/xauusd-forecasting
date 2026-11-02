"""Проверки полноты горизонтов и состава моделей при совместном сравнении."""
import pandas as pd
import pytest

from src.metrics import assert_comparable_predictions


def _forecast(model: str, horizon: int) -> dict:
    return {
        "origin_date": "2026-09-01",
        "target_date": "2026-09-02" if horizon == 1 else "2026-09-08",
        "horizon": horizon,
        "model": model,
        "y_pred": 100.0,
    }


def test_rejects_models_with_only_one_horizon_each():
    # Ранее проверка ошибочно принимала такую таблицу.
    preds = pd.DataFrame([_forecast("naive", 1), _forecast("arima", 5)])
    with pytest.raises(ValueError, match="Не все горизонты"):
        assert_comparable_predictions(preds)


def test_final_comparison_requires_all_four_models():
    preds = pd.DataFrame([_forecast("naive", 1), _forecast("naive", 5)])
    with pytest.raises(ValueError, match="Набор моделей"):
        assert_comparable_predictions(
            preds, expected_models=("naive", "arima", "random_forest", "mlp"))


def test_all_four_models_and_both_horizons_pass():
    models = ("naive", "arima", "random_forest", "mlp")
    preds = pd.DataFrame([_forecast(model, h) for model in models for h in (1, 5)])
    assert_comparable_predictions(preds, expected_models=models)
