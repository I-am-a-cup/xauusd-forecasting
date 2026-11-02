"""Контракт MLP, доступность целей и независимость от будущих цен."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.daily_features import FEATURES, model_table
from src.metrics import assert_comparable_predictions, attach_actuals
from src.models.baseline import predict_naive
from src.models.mlp import MLPConfig, fit_mlp, predict_fitted, predict_mlp
from src.validation import expanding_folds, holdout_split


@pytest.fixture
def daily():
    return pd.read_csv(Path(__file__).resolve().parents[1] /
                       'data/processed/xauusd_daily_dukascopy.csv')


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def test_mlp_and_naive_have_identical_dates(daily):
    split = holdout_split(len(daily))
    result = predict_mlp(daily, split.forecast_origins,
                         last_known_index=split.last_known_index)
    assert len(result) == 94
    assert list(result.columns) == ['origin_date', 'target_date', 'horizon', 'model', 'y_pred']
    combined = pd.concat([result, predict_naive(daily, split.forecast_origins)])
    assert_comparable_predictions(combined, expected_models=('naive', 'mlp'))
    assert attach_actuals(combined, daily).y_true.notna().all()


def test_future_price_changes_cannot_change_first_forecast(daily):
    split = holdout_split(len(daily))
    first = split.forecast_origins[0]
    original = predict_mlp(daily, (first,), last_known_index=split.last_known_index)
    altered = daily.copy()
    altered.loc[first+1:, ['open', 'high', 'low', 'close']] *= 3
    changed = predict_mlp(altered, (first,), last_known_index=split.last_known_index)
    np.testing.assert_array_equal(original.y_pred, changed.y_pred)


def test_scalers_fit_train_only_and_prediction_does_not_refit(daily):
    split = holdout_split(len(daily))
    table = model_table(daily, split.last_known_index, split.forecast_origins)
    train, test = table[table.split.eq('train')], table[table.split.eq('test')]
    estimator, info = fit_mlp(train, 5, MLPConfig((4,), 10.0))
    scaler = estimator.regressor_.named_steps['standardscaler']
    np.testing.assert_allclose(scaler.mean_, train[list(FEATURES)].mean())
    expected = np.log(train.target_close_next_5d / train.close).mean()
    assert estimator.transformer_.mean_[0] == pytest.approx(expected)
    before = scaler.mean_.copy()
    net = estimator.regressor_.named_steps['mlpregressor']
    weights = [x.copy() for x in net.coefs_]
    without_targets = test.drop(columns=['target_close_next_1d', 'target_close_next_5d'])
    prediction = predict_fitted(estimator, without_targets)
    np.testing.assert_array_equal(prediction, predict_fitted(estimator, without_targets))
    np.testing.assert_array_equal(scaler.mean_, before)
    for old, new in zip(weights, net.coefs_):
        np.testing.assert_array_equal(old, new)
    assert info['train_rows'] == 196


def test_all_folds_have_known_targets_and_five_purged_origins(daily):
    split = holdout_split(len(daily))
    folds = expanding_folds(len(daily), 105, 25, 25,
                            end_before=split.last_known_index+1)
    assert len(folds) == 4
    for fold in folds:
        table = model_table(daily, fold.last_known_index, fold.forecast_origins)
        train = table[table.split.eq('train')]
        last_origin = daily.index[daily.date.eq(train.date.iloc[-1])][0]
        assert last_origin + 5 <= fold.last_known_index
        assert fold.forecast_origins[0] - last_origin - 1 >= 5
        for h in (1, 5):
            positions = daily.index[daily.date.isin(train.date)]
            np.testing.assert_allclose(train[f'target_close_next_{h}d'], daily.close.iloc[positions+h])
        assert max(fold.forecast_origins)+5 <= split.last_known_index


def test_reject_test_rows_and_bad_features(daily):
    split = holdout_split(len(daily))
    table = model_table(daily, split.last_known_index, split.forecast_origins)
    with pytest.raises(ValueError, match='train'):
        fit_mlp(table, 1)
    train = table[table.split.eq('train')].copy()
    train.loc[train.index[0], 'return_1'] = np.inf
    with pytest.raises(ValueError, match='конечные'):
        fit_mlp(train, 1)
