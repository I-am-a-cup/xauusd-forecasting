"""Инструменты временной валидации; разделение по торговым *наблюдениям*.

Важное условие: до вызова функции данные должны быть очищены и отсортированы
по окончательному календарю торговых сессий. Модуль не определяет этот календарь.

Используем строгую конвенцию: last_known_index — последний индекс, для которого
цена закрытия уже известна на момент обучения модели.
"""
from dataclasses import dataclass
from math import ceil
from numbers import Integral


@dataclass(frozen=True)
class TemporalSplit:
    """Граница обучения (включительно) и индексы дат формирования прогнозов."""
    last_known_index: int
    forecast_origins: tuple[int, ...]

    def eligible_train_origins(self, horizon: int, min_history: int = 1) -> tuple[int, ...]:
        """Допустимые обучающие примеры для конкретного горизонта.

        Для любого обучающего origin выполняется origin + horizon <= last_known_index.
        """
        return training_origins(self.last_known_index, horizon, min_history)


def _check_horizons(horizons) -> tuple[int, ...]:
    result = tuple(horizons)
    if not result or any(not isinstance(h, Integral) or isinstance(h, bool) or h < 1 for h in result):
        raise ValueError("Горизонты должны быть положительными целыми числами")
    if len(set(result)) != len(result):
        raise ValueError("Горизонты не должны повторяться")
    return result


def training_origins(last_known_index: int, horizon: int, min_history: int = 1) -> tuple[int, ...]:
    """Отдаёт индексы примеров, чья целевая цена доступна при обучении."""
    if not isinstance(last_known_index, Integral) or last_known_index < 0:
        raise ValueError("Последний известный индекс должен быть неотрицательным")
    _check_horizons((horizon,))
    if not isinstance(min_history, Integral) or min_history < 1:
        raise ValueError("Минимальная длина истории должна быть положительной")
    last_origin = last_known_index - horizon
    first_origin = min_history - 1
    if last_origin < first_origin:
        return ()
    return tuple(range(first_origin, last_origin + 1))


def holdout_split(
    n_observations: int,
    test_share: float = 0.20,
    horizons=(1, 5),
    min_train_observations: int = 90,
) -> TemporalSplit:
    """Фиксированный тестовый участок, общий по origin для всех горизонтов.

    Обучение заканчивается перед первым прогнозом. Последние max(h)
    наблюдений служат исключительно для проверки целевых цен, а не
    как даты формирования прогнозов.
    """
    hs = _check_horizons(horizons)
    if not isinstance(n_observations, Integral) or n_observations < 1:
        raise ValueError("Число наблюдений должно быть положительным")
    if not 0 < test_share < 1:
        raise ValueError("Доля теста должна находиться между 0 и 1")
    if not isinstance(min_train_observations, Integral) or min_train_observations < 1:
        raise ValueError("Минимальная длина обучения должна быть положительной")
    n_test = ceil(n_observations * test_share)
    test_start = n_observations - n_test
    if test_start < min_train_observations + max(hs):
        raise ValueError("Недостаточно данных для обучающих целевых значений")
    origins = tuple(range(test_start, n_observations - max(hs)))
    if not origins:
        raise ValueError("Слишком мало тестовых наблюдений для всех горизонтов")
    return TemporalSplit(last_known_index=test_start-1, forecast_origins=origins)


def expanding_folds(
    n_observations: int,
    initial_train_observations: int,
    validation_size: int,
    step_size: int,
    horizons=(1, 5),
    end_before: int | None = None,
) -> tuple[TemporalSplit, ...]:
    """Несколько расширяющихся обучающих окон.

    end_before — индекс *первого запрещённого наблюдения* (исключающая граница),
    например начало отложенного теста. Ни обучающие цели, ни проверочные цели
    не должны пересекать эту границу.
    """
    hs = _check_horizons(horizons)
    if any(not isinstance(v, Integral) or v < 1 for v in
           (n_observations, initial_train_observations, validation_size, step_size)):
        raise ValueError("Параметры размеров должны быть положительными целыми числами")
    if end_before is None:
        end_before = n_observations
    if not isinstance(end_before, Integral) or not 1 <= end_before <= n_observations:
        raise ValueError("Недопустимая граница данных для валидации")
    if validation_size <= max(hs):
        raise ValueError("Окно валидации должно быть длиннее максимального горизонта")

    folds = []
    for n_train in range(initial_train_observations, end_before, step_size):
        validation_end_exclusive = min(n_train + validation_size, end_before)
        # Целевая позиция origin + max(h) обязана оставаться внутри окна валидации.
        origins = tuple(range(n_train, validation_end_exclusive - max(hs)))
        if origins:
            folds.append(TemporalSplit(last_known_index=n_train-1, forecast_origins=origins))
    if not folds:
        raise ValueError("Недостаточно данных для временных окон валидации")
    return tuple(folds)
