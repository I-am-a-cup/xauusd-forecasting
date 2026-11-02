"""Direct Random Forest: изменение цены на h сессий по известным на t признакам."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.daily_features import FEATURES

@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 160
    max_depth: int = 5
    min_samples_leaf: int = 3
    max_features: float = 0.8

CANDIDATES = (
    RFConfig(120, 4, 3, 0.8),
    RFConfig(160, 6, 5, 0.8),
    RFConfig(200, 8, 3, 1.0),
)

def fit_rf(training: pd.DataFrame, horizon: int, config: RFConfig, seed: int = 42):
    """Использовать только обучающие пары, известные на дату отсечения."""
    if horizon not in (1, 5) or training.empty:
        raise ValueError("Нужен непустой train для горизонта 1 или 5")
    if not training.split.eq("train").all():
        raise ValueError("Обучать RF на test запрещено")
    x = training[list(FEATURES)].to_numpy(dtype=float)
    start_close = training.close.to_numpy(dtype=float)
    actual = training[f"target_close_next_{horizon}d"].to_numpy(dtype=float)
    if (not np.isfinite(x).all() or not np.isfinite(actual).all()
        or (start_close <= 0).any() or (actual <= 0).any()):
        raise ValueError("Обнаружены некорректные признаки или целевые значения")
    y = np.log(actual / start_close)
    model = RandomForestRegressor(
        n_estimators=config.n_estimators, max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf, max_features=config.max_features,
        n_jobs=1, random_state=seed,
    )
    started = perf_counter()
    model.fit(x, y)
    return model, perf_counter() - started

def predict_rf_fitted(model, test: pd.DataFrame) -> np.ndarray:
    """Цена t+h = текущая известная цена * exp(прогнозируемая доходность)."""
    x = test[list(FEATURES)].to_numpy(dtype=float)
    result = test.close.to_numpy(dtype=float) * np.exp(model.predict(x))
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Модель выдала некорректные цены")
    return result
