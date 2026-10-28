"""Shared evaluation. Only evaluation code reads targets."""
import numpy as np
import pandas as pd


def attach_actuals(predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join observed target and origin Close on canonical trading dates."""
    required = {"origin_date", "target_date", "horizon", "model", "y_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions missing: {required - set(predictions.columns)}")
    price = prices[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"])
    if price["date"].duplicated().any():
        raise ValueError("duplicate price dates")
    pred = predictions.copy()
    pred["origin_date"] = pd.to_datetime(pred["origin_date"])
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred = pred.merge(price.rename(columns={"date": "target_date", "close": "y_true"}),
                      on="target_date", how="left", validate="many_to_one")
    pred = pred.merge(price.rename(columns={"date": "origin_date", "close": "origin_close"}),
                      on="origin_date", how="left", validate="many_to_one")
    if pred[["y_true", "origin_close"]].isna().any().any():
        raise ValueError("Predictions reference unavailable target or origin Close")
    return pred


def evaluate(predictions_with_actuals: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE in price units; sMAPE and direction_accuracy in percent."""
    required = {"model", "horizon", "y_pred", "y_true", "origin_close"}
    if not required.issubset(predictions_with_actuals.columns):
        raise ValueError(f"Missing columns: {required - set(predictions_with_actuals.columns)}")
    records = []
    for (model, h), g in predictions_with_actuals.groupby(["model", "horizon"]):
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)
        origin = g["origin_close"].to_numpy(dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(origin).all()):
            raise ValueError("Non-finite metric inputs")
        denom = np.abs(y) + np.abs(p)
        smape_terms = np.divide(2*np.abs(y-p), denom, out=np.zeros_like(y), where=denom!=0)
        records.append({
            "model": model, "horizon": int(h), "n": len(g),
            "mae": float(np.mean(np.abs(y-p))),
            "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "smape_pct": float(100*np.mean(smape_terms)),
            "direction_accuracy_pct": float(100*np.mean(np.sign(p-origin)==np.sign(y-origin))),
        })
    return pd.DataFrame(records)
