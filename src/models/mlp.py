"""Малая MLP: отдельные горизонты, стандартизация только по train.

Сеть предсказывает log(C[t+h]/C[t]); наружу возвращается цена в USD.
Входы — ровно 10 FEATURES из общего протокола.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.daily_features import FEATURES, model_table
from src.models.baseline import OUTPUT_COLUMNS


@dataclass(frozen=True)
class MLPConfig:
    hidden_layer_sizes: tuple[int, ...] = (8,)
    alpha: float = 1.0
    max_iter: int = 2000


CANDIDATES = tuple(MLPConfig(layers, alpha)
                   for layers in ((4,), (8,), (8, 4)) for alpha in (1.0, 10.0))


def _features(samples):
    x = samples.loc[:, list(FEATURES)].to_numpy(dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Нужны непустые конечные признаки")
    if (samples['close'].to_numpy(dtype=float) <= 0).any():
        raise ValueError("Цены должны быть положительными")
    return x


def fit_mlp(samples: pd.DataFrame, horizon: int, config=MLPConfig(), random_state=42):
    """Обучить один горизонт на явно переданном train; вернуть сеть и журнал."""
    if horizon not in (1, 5):
        raise ValueError("Горизонт должен быть 1 или 5")
    if 'split' not in samples or not samples['split'].eq('train').all():
        raise ValueError("Обучение разрешено только на строках train")
    x = _features(samples)
    close = samples['close'].to_numpy(dtype=float)
    target = samples[f'target_close_next_{horizon}d'].to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Нужны конечные положительные обучающие цели")
    y = np.log(target / close)
    estimator = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=config.hidden_layer_sizes, alpha=config.alpha,
            solver='lbfgs', activation='tanh', max_iter=config.max_iter,
            max_fun=50000, tol=1e-6, early_stopping=False,
            shuffle=False, random_state=random_state)),
        transformer=StandardScaler())
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(x, y)
    net = estimator.regressor_.named_steps['mlpregressor']
    info = dict(horizon=horizon, seed=random_state, train_rows=len(samples),
                train_first_date=str(samples.date.iloc[0]),
                train_last_origin=str(samples.date.iloc[-1]),
                fit_seconds=perf_counter()-start, n_iter=int(net.n_iter_),
                parameter_count=int(sum(a.size for a in net.coefs_ + net.intercepts_)),
                warnings=[str(w.message) for w in caught])
    return estimator, info


def predict_fitted(estimator, samples):
    """Только transform/predict: не читает цели и не меняет веса/масштабы."""
    values = samples['close'].to_numpy(dtype=float) * np.exp(estimator.predict(_features(samples)))
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Получен некорректный прогноз цены")
    return values


def forecast_frame(prices, samples, horizon, predicted):
    dates = pd.to_datetime(prices['date']).reset_index(drop=True)
    positions = pd.Series(np.arange(len(dates)), index=dates)
    origins = pd.to_datetime(samples['date'])
    targets = dates.iloc[origins.map(positions).to_numpy() + horizon]
    return pd.DataFrame(dict(origin_date=origins.dt.strftime('%Y-%m-%d').to_numpy(),
        target_date=targets.dt.strftime('%Y-%m-%d').to_numpy(), horizon=horizon,
        model='mlp', y_pred=predicted), columns=OUTPUT_COLUMNS)


def load_selected_configs(metadata_path: str | Path | None = None) -> dict[int, MLPConfig]:
    """Загрузить именно выбранные по валидации конфигурации из отчёта.

    Если метаданные отсутствуют или повреждены, запрещаем незаметно
    переходить на дефолтные параметры сети.
    """
    path = (Path(metadata_path) if metadata_path is not None else
            Path(__file__).resolve().parents[2] / 'results/mlp/metadata.json')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    selected = metadata.get('selected')
    if not isinstance(selected, dict) or set(selected) != {'1', '5'}:
        raise ValueError("В metadata.json должны быть выбранные параметры h=1 и h=5")
    configs = {}
    for horizon in (1, 5):
        raw = selected[str(horizon)]
        if not isinstance(raw, dict):
            raise ValueError(f"Некорректная конфигурация h={horizon}")
        layers = raw.get('hidden_layer_sizes')
        alpha = raw.get('alpha')
        max_iter = raw.get('max_iter')
        if (not isinstance(layers, (tuple, list)) or not layers
            or any(type(n) is not int or n <= 0 for n in layers)
            or type(alpha) not in (float, int) or not 0 < alpha < float('inf')
            or type(max_iter) is not int or max_iter <= 0):
            raise ValueError(f"Некорректные параметры MLP для h={horizon}")
        configs[horizon] = MLPConfig(tuple(layers), float(alpha), max_iter)
    return configs


def predict_mlp(prices, origins, *, last_known_index, configs=None, random_state=42):
    """Общий контракт: 5 стандартных столбцов, оба горизонта.

    Без configs загружает конфигурации из metadata.json, выбранные на CV,
    а НЕ использует параметры по умолчанию. Подбора по тесту здесь нет.
    """
    if not pd.to_datetime(prices.date).is_monotonic_increasing:
        raise ValueError("Дневные цены должны быть отсортированы")
    origins = tuple(origins)
    if not origins or min(origins) <= last_known_index:
        raise ValueError("Прогнозы должны начинаться после обучающей истории")
    table = model_table(prices, last_known_index, origins)
    test = table[table.split.eq('test')]
    if len(test) != len(origins):
        raise ValueError("Не для всех дат доступны признаки и целевые даты")
    if configs is None:
        configs = load_selected_configs()
    if set(configs) != {1, 5} or any(not isinstance(c, MLPConfig) for c in configs.values()):
        raise ValueError("Нужны MLPConfig для обоих горизонтов 1 и 5")
    result = []
    for h in (1, 5):
        estimator, _ = fit_mlp(table[table.split.eq('train')], h,
                              configs[h], random_state)
        result.append(forecast_frame(prices, test, h, predict_fitted(estimator, test)))
    return pd.concat(result, ignore_index=True)
