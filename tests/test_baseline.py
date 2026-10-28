import pandas as pd
from src.models.baseline import predict_naive
from src.metrics import attach_actuals, evaluate


def test_naive_horizons_and_target_dates():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05",
                                "2026-10-06", "2026-10-07", "2026-10-08",
                                "2026-10-09"]),
        "close": [100, 101, 102, 103, 104, 105, 106],
    })
    p = predict_naive(prices, [0, 1], horizons=(1, 5))
    assert len(p) == 4
    assert p.loc[(p.horizon==5)&(p.origin_date=="2026-10-01"), "target_date"].iloc[0] == "2026-10-08"
    assert set(p.loc[p.origin_date=="2026-10-01","y_pred"]) == {100.0}
    e = evaluate(attach_actuals(p, prices))
    assert len(e) == 2
    assert set(e.horizon) == {1,5}


def test_naive_does_not_consult_future_close():
    prices = pd.DataFrame({"date": pd.date_range("2026-01-01",periods=7), "close": [1.,2.,3.,4.,5.,6.,7.]})
    a = predict_naive(prices, [0], horizons=(5,))
    prices.loc[5,"close"] = 9999.
    b = predict_naive(prices, [0], horizons=(5,))
    assert a.y_pred.iloc[0] == b.y_pred.iloc[0] == 1.0
