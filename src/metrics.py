"""Common scoring and checks for fair, time-aligned forecast comparisons."""
import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {"origin_date", "target_date", "horizon", "model", "y_pred"}


def assert_comparable_predictions(predictions: pd.DataFrame, horizons=(1, 5)) -> None:
    """Require every model/horizon to predict the same set of origins and targets.

    This does not replace chronological training validation; it guards the
    evaluation contract and detects silent omissions and mismatched test dates.
    """
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("No predictions to compare")

    p = predictions.copy()
    for col in ("origin_date", "target_date"):
        p[col] = pd.to_datetime(p[col], errors="raise")
    if p.duplicated(["model", "horizon", "origin_date"]).any():
        raise ValueError("Duplicate model/horizon/origin forecast")
    if set(p["horizon"].unique()) != set(horizons):
        raise ValueError("Incomplete or unexpected forecast horizons")

    expected = None
    for (model, horizon), group in p.groupby(["model", "horizon"]):
        pairs = tuple(sorted(zip(group["origin_date"], group["target_date"])))
        if expected is None:
            # The target date differs between horizons, but the ORIGIN dates must match.
            expected = tuple(sorted(group["origin_date"].tolist()))
        if tuple(sorted(group["origin_date"].tolist())) != expected:
            raise ValueError(f"Unequal forecast origins for {model}, h={horizon}")

    # For the same origin and horizon, all models must target the same date.
    target_nunique = p.groupby(["origin_date", "horizon"])["target_date"].nunique()
    if (target_nunique != 1).any():
        raise ValueError("Models disagree on target dates for the same origin/horizon")


def attach_actuals(predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach Close at origin and target; validate trading-observation offsets."""
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("prices must contain date and close")

    price = prices[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="raise")
    if price["date"].duplicated().any():
        raise ValueError("Duplicate trading dates in prices")
    if price[["date", "close"]].isna().any().any():
        raise ValueError("Missing trading dates or Close values")

    price = price.sort_values("date").reset_index(drop=True)
    index_by_date = pd.Series(np.arange(len(price), dtype=int), index=price["date"])

    pred = predictions.copy()
    pred["origin_date"] = pd.to_datetime(pred["origin_date"], errors="raise")
    pred["target_date"] = pd.to_datetime(pred["target_date"], errors="raise")
    if pred.duplicated(["model", "horizon", "origin_date"]).any():
        raise ValueError("Duplicate model/horizon/origin forecast")
    if pred["horizon"].isna().any() or not pred["horizon"].isin([1, 5]).all():
        raise ValueError("Unexpected forecast horizon; expected 1 or 5")

    origin_pos = pred["origin_date"].map(index_by_date)
    target_pos = pred["target_date"].map(index_by_date)
    if origin_pos.isna().any() or target_pos.isna().any():
        raise ValueError("Forecast references an unknown origin or target date")
    if not ((target_pos - origin_pos).to_numpy() == pred["horizon"].to_numpy()).all():
        raise ValueError("Target date is inconsistent with declared trading-day horizon")

    pred = pred.merge(
        price.rename(columns={"date": "target_date", "close": "y_true"}),
        on="target_date", how="left", validate="many_to_one")
    pred = pred.merge(
        price.rename(columns={"date": "origin_date", "close": "origin_close"}),
        on="origin_date", how="left", validate="many_to_one")
    return pred


def evaluate(predictions_with_actuals: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE in USD; sMAPE and directional accuracy in percent."""
    required = {"model", "horizon", "y_pred", "y_true", "origin_close"}
    if not required.issubset(predictions_with_actuals.columns):
        raise ValueError(f"Metric columns missing: {sorted(required - set(predictions_with_actuals.columns))}")
    records = []
    for (model, horizon), group in predictions_with_actuals.groupby(["model", "horizon"]):
        actual = group["y_true"].to_numpy(dtype=float)
        predicted = group["y_pred"].to_numpy(dtype=float)
        origin = group["origin_close"].to_numpy(dtype=float)
        if not (np.isfinite(actual).all() and np.isfinite(predicted).all() and np.isfinite(origin).all()):
            raise ValueError("Non-finite metric inputs")
        denominator = np.abs(actual) + np.abs(predicted)
        smape_terms = np.divide(
            2 * np.abs(actual - predicted), denominator,
            out=np.zeros_like(actual), where=denominator != 0)
        records.append({
            "model": model,
            "horizon": int(horizon),
            "n": len(group),
            "mae": float(np.mean(np.abs(actual-predicted))),
            "rmse": float(np.sqrt(np.mean((actual-predicted)**2))),
            "smape_pct": float(100*np.mean(smape_terms)),
            "direction_accuracy_pct": float(
                100*np.mean(np.sign(predicted-origin) == np.sign(actual-origin))),
        })
    return pd.DataFrame(records)
