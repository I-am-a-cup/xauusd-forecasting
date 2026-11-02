"""Регрессии для ARIMA: выбор по CV, Naive и проверка реальной динамики модели."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import attach_actuals, evaluate
from src.models.arima import CANDIDATES, predict_arima
from src.models.baseline import predict_naive
from src.validation import holdout_split

ROOT = Path(__file__).resolve().parents[1]


def test_arima_010_equals_naive_for_every_published_origin_and_horizon():
    prices = pd.read_csv(ROOT / "data/processed/xauusd_daily_dukascopy.csv")
    split = holdout_split(len(prices))
    arima, info = predict_arima(
        prices, split.last_known_index, split.forecast_origins, (0, 1, 0))
    naive = predict_naive(prices, split.forecast_origins)
    columns = ("origin_date", "target_date", "horizon")
    paired = arima.merge(
        naive, on=list(columns), suffixes=("_arima", "_naive"),
        validate="one_to_one")
    assert len(paired) == 47 * 2
    assert paired.groupby("horizon").size().to_dict() == {1: 47, 5: 47}
    # Здесь требуется точное равенство цен, а не равенство метрик после округления.
    np.testing.assert_array_equal(paired.y_pred_arima, paired.y_pred_naive)
    assert info["order"] == [0, 1, 0]
    assert info["refit_on_test"] is False

    for horizon in (1, 5):
        a = evaluate(attach_actuals(arima[arima.horizon.eq(horizon)], prices))
        n = evaluate(attach_actuals(naive[naive.horizon.eq(horizon)], prices))
        np.testing.assert_array_equal(a[["mae", "rmse"]], n[["mae", "rmse"]])


def test_arima_110_is_not_hardwired_to_naive_on_autocorrelated_series():
    # Фиксированный AR(1) процесс для приращений, чтобы существовал
    # прогнозируемый сигнал, отсутствующий у случайного блуждания.
    rng = np.random.default_rng(421)
    noise = rng.normal(size=180)
    increments = np.empty(180)
    increments[0] = 1.2
    for i in range(1, len(increments)):
        increments[i] = .84 * increments[i - 1] + noise[i]
    prices = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=180).strftime("%Y-%m-%d"),
        "close": 3000 + np.cumsum(increments),
    })
    origins = tuple(range(145, 171))
    correlated, _ = predict_arima(prices, 144, origins, (1, 1, 0))
    random_walk, _ = predict_arima(prices, 144, origins, (0, 1, 0))
    assert correlated[["origin_date", "target_date", "horizon"]].equals(
        random_walk[["origin_date", "target_date", "horizon"]])
    differences = np.abs(correlated.y_pred.to_numpy() - random_walk.y_pred.to_numpy())
    assert len(differences) == 52
    assert np.max(differences) > 1.0
    assert np.count_nonzero(differences > 1e-5) == 52


def test_arima_order_selected_only_from_pretest_validation():
    cv = pd.read_csv(ROOT / "results/arima_rf/validation.csv")
    meta = __import__("json").loads(
        (ROOT / "results/arima_rf/metadata.json").read_text(encoding="utf-8"))
    arima = cv[cv.model.eq("arima")]
    assert set(arima.horizon) == {1, 5}
    assert set(arima.candidate) == set(range(len(CANDIDATES)))
    assert arima.groupby(["horizon", "candidate"]).size().eq(4).all()
    for horizon in (1, 5):
        mean_scores = arima[arima.horizon.eq(horizon)].groupby("candidate").mae.mean()
        assert mean_scores.idxmin() == 0
        assert meta["selected_arima"][str(horizon)] == list(CANDIDATES[0])
        assert (arima[arima.horizon.eq(horizon)]["last_known_date"] < "2026-08-17").all()


def test_readme_cv_table_matches_saved_validation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")
    cv = pd.read_csv(ROOT / "results/arima_rf/validation.csv")
    means = cv[cv.model.eq("arima")].groupby(["horizon", "candidate"]).mae.mean()
    for candidate, order in enumerate(CANDIDATES):
        h1 = f"{means.loc[1, candidate]:.2f}".replace(".", ",")
        h5 = f"{means.loc[5, candidate]:.2f}".replace(".", ",")
        marker = f"| ({order[0]},{order[1]},{order[2]})"
        assert marker in readme
        assert h1 in readme and h5 in readme
