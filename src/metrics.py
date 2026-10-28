"""Общие метрики и проверка сопоставимости прогнозов по временным датам."""
import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {"origin_date", "target_date", "horizon", "model", "y_pred"}


def assert_comparable_predictions(predictions: pd.DataFrame, horizons=(1, 5)) -> None:
    """Проверяет совпадение дат формирования прогноза для всех моделей и горизонтов.

    Это не заменяет временную валидацию обучения: проверка обнаруживает
    пропущенные прогнозы и несогласованные целевые даты.
    """
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы прогнозов: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Нет прогнозов для сравнения")

    p = predictions.copy()
    for col in ("origin_date", "target_date"):
        p[col] = pd.to_datetime(p[col], errors="raise")
    if p.duplicated(["model", "horizon", "origin_date"]).any():
        raise ValueError("Повторяется прогноз для модели, горизонта и даты")
    if set(p["horizon"].unique()) != set(horizons):
        raise ValueError("Неполный или некорректный набор горизонтов")

    expected = None
    for (model, horizon), group in p.groupby(["model", "horizon"]):
        if expected is None:
            # Целевые даты различаются по горизонту, но даты формирования должны совпадать.
            expected = tuple(sorted(group["origin_date"].tolist()))
        if tuple(sorted(group["origin_date"].tolist())) != expected:
            raise ValueError(f"Даты формирования прогнозов не совпадают для {model}, h={horizon}")

    # Для одной исходной даты и горизонта все модели прогнозируют одну дату.
    target_nunique = p.groupby(["origin_date", "horizon"])["target_date"].nunique()
    if (target_nunique != 1).any():
        raise ValueError("Модели используют разные целевые даты при одинаковом горизонте")


def attach_actuals(predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Добавляет цену закрытия на исходную и целевую даты и проверяет горизонт."""
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы прогнозов: {sorted(missing)}")
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("В таблице цен должны быть столбцы date и close")

    price = prices[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="raise")
    if price["date"].duplicated().any():
        raise ValueError("В таблице цен есть повторяющиеся торговые даты")
    if price[["date", "close"]].isna().any().any():
        raise ValueError("В таблице цен есть пропуски дат или цен закрытия")

    price = price.sort_values("date").reset_index(drop=True)
    index_by_date = pd.Series(np.arange(len(price), dtype=int), index=price["date"])

    pred = predictions.copy()
    pred["origin_date"] = pd.to_datetime(pred["origin_date"], errors="raise")
    pred["target_date"] = pd.to_datetime(pred["target_date"], errors="raise")
    if pred.duplicated(["model", "horizon", "origin_date"]).any():
        raise ValueError("Повторяется прогноз для модели, горизонта и даты")
    if pred["horizon"].isna().any() or not pred["horizon"].isin([1, 5]).all():
        raise ValueError("Неверный горизонт: допускается только 1 или 5")

    origin_pos = pred["origin_date"].map(index_by_date)
    target_pos = pred["target_date"].map(index_by_date)
    if origin_pos.isna().any() or target_pos.isna().any():
        raise ValueError("Прогноз ссылается на отсутствующую исходную или целевую дату")
    if not ((target_pos - origin_pos).to_numpy() == pred["horizon"].to_numpy()).all():
        raise ValueError("Целевая дата не соответствует заявленному горизонту в торговых днях")

    pred = pred.merge(
        price.rename(columns={"date": "target_date", "close": "y_true"}),
        on="target_date", how="left", validate="many_to_one")
    pred = pred.merge(
        price.rename(columns={"date": "origin_date", "close": "origin_close"}),
        on="origin_date", how="left", validate="many_to_one")
    return pred


def evaluate(predictions_with_actuals: pd.DataFrame) -> pd.DataFrame:
    """MAE и RMSE — в долларах, sMAPE и точность направления — в процентах."""
    required = {"model", "horizon", "y_pred", "y_true", "origin_close"}
    if not required.issubset(predictions_with_actuals.columns):
        raise ValueError(f"Отсутствуют столбцы для расчёта метрик: {sorted(required - set(predictions_with_actuals.columns))}")
    records = []
    for (model, horizon), group in predictions_with_actuals.groupby(["model", "horizon"]):
        actual = group["y_true"].to_numpy(dtype=float)
        predicted = group["y_pred"].to_numpy(dtype=float)
        origin = group["origin_close"].to_numpy(dtype=float)
        if not (np.isfinite(actual).all() and np.isfinite(predicted).all() and np.isfinite(origin).all()):
            raise ValueError("Метрики невозможно рассчитать: обнаружены NaN или бесконечности")
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
