import pandas as pd
import pytest
from src.models.baseline import predict_naive
from src.metrics import attach_actuals, evaluate, assert_comparable_predictions


def sample_prices():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06",
            "2026-10-07", "2026-10-08", "2026-10-09"]),
        "close": [100, 101, 102, 103, 104, 105, 106],
    })


def test_naive_horizons_and_target_dates():
    prices = sample_prices()
    preds = predict_naive(prices, [0, 1], horizons=(1, 5))
    assert len(preds) == 4
    assert preds.loc[
        (preds.horizon == 5) & (preds.origin_date == "2026-10-01"),
        "target_date"].iloc[0] == "2026-10-08"
    assert set(preds.loc[preds.origin_date == "2026-10-01", "y_pred"]) == {100.0}
    assert_comparable_predictions(preds)
    scored = evaluate(attach_actuals(preds, prices))
    assert len(scored) == 2
    assert set(scored.horizon) == {1, 5}


def test_naive_does_not_consult_future_close():
    prices = sample_prices()
    p1 = predict_naive(prices, [0], horizons=(5,))
    prices.loc[5, "close"] = 9999.0
    p2 = predict_naive(prices, [0], horizons=(5,))
    assert p1.y_pred.iloc[0] == p2.y_pred.iloc[0] == 100.0


def test_baseline_rejects_incomplete_five_day_horizon():
    with pytest.raises(ValueError, match="Нет доступной целевой даты"):
        predict_naive(sample_prices(), [3], horizons=(1, 5))


def test_evaluator_rejects_wrong_target_date_for_horizon():
    preds = predict_naive(sample_prices(), [0], horizons=(1, 5))
    preds.loc[preds.horizon == 5, "target_date"] = "2026-10-02"
    with pytest.raises(ValueError, match="не соответствует заявленному горизонту"):
        attach_actuals(preds, sample_prices())


def test_comparison_rejects_missing_origin():
    preds = predict_naive(sample_prices(), [0, 1], horizons=(1, 5))
    preds = preds.loc[~((preds.horizon == 5) & (preds.origin_date == "2026-10-02"))]
    with pytest.raises(ValueError, match="Даты формирования прогнозов не совпадают"):
        assert_comparable_predictions(preds)


def test_comparison_rejects_different_targets_between_models():
    preds = predict_naive(sample_prices(), [0, 1], horizons=(1, 5))
    extra = preds.copy()
    extra["model"] = "another"
    extra.loc[(extra.horizon == 5) & (extra.origin_date == "2026-10-01"), "target_date"] = "2026-10-07"
    with pytest.raises(ValueError, match="Модели используют разные целевые даты"):
        assert_comparable_predictions(pd.concat([preds, extra], ignore_index=True))
