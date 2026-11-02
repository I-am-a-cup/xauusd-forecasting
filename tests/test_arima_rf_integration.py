"""Страховка от утечки будущей цены и проверки общего контракта."""
import numpy as np
import pandas as pd

from src.daily_features import model_table, FEATURES
from src.validation import holdout_split
from src.models.arima import predict_arima
from src.models.random_forest import RFConfig, fit_rf, predict_rf_fitted
from src.metrics import assert_comparable_predictions


def _prices(n=125):
    close = np.array([3000.0 + .7*i + 12*np.sin(i/8) for i in range(n)])
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "open": close-1,
        "high": close+3,
        "low": close-3,
        "close": close,
    })


def test_original_dmitry_features_reveal_target_exactly():
    """Доказательство утечки в архивном исходном коде, не только корреляция."""
    p = _prices()
    ma3 = p.close.rolling(3).mean()
    reconstructed = 3*ma3 - p.close.shift(1) - p.close.shift(2)
    assert np.allclose(reconstructed.iloc[2:], p.close.iloc[2:])


def test_random_forest_cannot_see_next_day_close_in_input():
    p = _prices()
    split = holdout_split(len(p))
    table = model_table(p, split.last_known_index, split.forecast_origins)
    train = table[table.split.eq("train")]
    test = table[table.split.eq("test")]
    chosen = RFConfig(20, 3, 3, 0.8)
    model, _ = fit_rf(train, 1, chosen)
    first = predict_rf_fitted(model, test.iloc[:1])[0]
    future = p.copy()
    origin_idx = split.forecast_origins[0]
    future.loc[origin_idx+1:, "close"] += 10000.0
    future.loc[origin_idx+1:, "open"] += 10000.0
    future.loc[origin_idx+1:, "high"] += 10000.0
    future.loc[origin_idx+1:, "low"] += 10000.0
    changed = model_table(future, split.last_known_index, split.forecast_origins)
    same_origin = changed[changed.split.eq("test")].iloc[:1]
    assert list(test[list(FEATURES)].iloc[0]) == list(same_origin[list(FEATURES)].iloc[0])
    assert np.isclose(first, predict_rf_fitted(model, same_origin)[0])


def test_arima_first_forecast_does_not_depend_on_future_close():
    p = _prices()
    split = holdout_split(len(p))
    origin = split.forecast_origins[0]
    original, meta = predict_arima(p, split.last_known_index, (origin,), (0,1,0))
    changed = p.copy()
    changed.loc[origin+1:, "close"] += 10000.0
    other, _ = predict_arima(changed, split.last_known_index, (origin,), (0,1,0))
    np.testing.assert_allclose(original.y_pred, other.y_pred)
    assert meta["refit_on_test"] is False


def test_model_outputs_are_comparable_on_published_data():
    p = pd.read_csv("data/processed/xauusd_daily_dukascopy.csv")
    preds = pd.read_csv("results/arima_rf/all_four_predictions.csv")
    assert_comparable_predictions(
        preds, expected_models=("naive","mlp","arima","random_forest"))
    assert len(preds) == 376
    assert preds.groupby(["model","horizon"]).size().eq(47).all()
    assert preds.origin_date.min() == "2026-08-17"
    assert preds.origin_date.max() == "2026-10-20"


def test_selected_params_from_cv_not_test_scores():
    import json
    m = json.load(open("results/arima_rf/metadata.json", encoding="utf-8"))
    assert m["validation_folds"] == 4
    assert m["test_origins"] == 47
    assert set(m["selected_arima"]) == {"1","5"}
    assert set(m["selected_rf"]) == {"1","5"}
