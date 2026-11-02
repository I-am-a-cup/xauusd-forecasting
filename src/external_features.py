"""Подключение внешних признаков только по подтверждённому времени доступности."""
import numpy as np
import pandas as pd


RELEASE_FIELDS = {"feature", "value", "observed_at_utc", "available_at_utc"}


def attach_verified_features(
    sessions: pd.DataFrame,
    releases: pd.DataFrame,
    selected: tuple[str, ...],
    *,
    max_age_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Присоединить значения, опубликованные на момент закрытия сессии.

    Не принимает сырой дневной CSV: в нём нет available_at_utc.
    Возвращает NaN, если к дате прогноза нет подтверждённого значения.
    Заполнение этих NaN выполняется отдельно, только внутри train-фолдов.
    """
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Укажите неповторяющийся перечень проверенных признаков")
    if not {"date", "close_utc"}.issubset(sessions.columns):
        raise ValueError("Нужны date и close_utc из сессионного ряда Dukascopy")
    if not RELEASE_FIELDS.issubset(releases.columns):
        raise ValueError("Нет временных меток первой публикации внешних показателей")
    if any(s.startswith("xauusd_") or "target" in s.lower() or "future" in s.lower()
           for s in selected):
        raise ValueError("Другие котировки XAU/USD и будущие цели использовать нельзя")
    if not set(selected).issubset(set(releases["feature"])):
        raise ValueError("Нет подтверждённых публикаций для одного или нескольких признаков")
    out = sessions.copy()
    out["close_utc"] = pd.to_datetime(out["close_utc"], utc=True, errors="raise")
    if out["date"].duplicated().any() or not out["close_utc"].is_monotonic_increasing:
        raise ValueError("Даты закрытия сессий должны быть уникальными и возрастающими")
    releases = releases.copy()
    releases["available_at_utc"] = pd.to_datetime(releases["available_at_utc"], utc=True, errors="raise")
    releases["observed_at_utc"] = pd.to_datetime(releases["observed_at_utc"], utc=True, errors="raise")
    if releases[["available_at_utc", "observed_at_utc"]].isna().any().any():
        raise ValueError("Пустая временная метка недопустима")
    if (releases["available_at_utc"] < releases["observed_at_utc"]).any():
        raise ValueError("Публикация не может предшествовать наблюдению")
    if (releases["feature"].isna()).any():
        raise ValueError("Признак не указан")
    for col in selected:
        part = releases[releases["feature"] == col].copy()
        if part["available_at_utc"].duplicated().any():
            raise ValueError(f"Повторяющееся время публикации: {col}")
        part["value"] = pd.to_numeric(part["value"], errors="raise")
        if not np.isfinite(part["value"]).all():
            raise ValueError(f"Некорректное значение: {col}")
        part = part.sort_values("available_at_utc")
        before = out[["close_utc"]].sort_values("close_utc")
        match = pd.merge_asof(
            before, part[["available_at_utc", "value"]],
            left_on="close_utc", right_on="available_at_utc",
            direction="backward",
        )
        if max_age_days is not None and col in max_age_days:
            days = max_age_days[col]
            if days < 0:
                raise ValueError("Возраст публикации должен быть неотрицательным")
            age = match["close_utc"] - match["available_at_utc"]
            match.loc[age > pd.Timedelta(days=days), "value"] = np.nan
        # Из-за монотонности индекс совпадает с исходным индексом сессий.
        out[col] = match["value"].to_numpy()
    return out
