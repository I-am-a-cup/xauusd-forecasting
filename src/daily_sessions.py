"""Преобразование часовых свечей Dukascopy в дневные сессии золота.

XAU/USD: 18:00—17:00 по America/New_York; час с 17 до 18 — перерыв.
Временные метки H1 Dukascopy относятся к началу часовой свечи.
Данные не дополняются искусственными значениями. Сокращённые торговые
сессии сохраняются и явно отмечаются в отчёте для ручной проверки.
"""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd

NY = "America/New_York"
HOUR = pd.Timedelta(hours=1)
REQUIRED = {"timestamp", "open", "high", "low", "close", "volume"}


def build_daily_sessions(
    raw: pd.DataFrame,
    start: str = "2025-10-28",
    end_exclusive: str = "2026-10-28",
    min_session_hours: int = 19,
) -> tuple[pd.DataFrame, dict]:
    """Вернуть проверенные дневные свечи и отчёт о неполных сессиях.

    19–22 часа считаются потенциально сокращёнными сессиями, но не
    автоматически подтверждёнными официальным праздничным расписанием.
    """
    if not REQUIRED.issubset(raw.columns):
        raise ValueError(f"Не хватает столбцов H1: {sorted(REQUIRED-set(raw.columns))}")
    if not 1 <= min_session_hours <= 23:
        raise ValueError("Недопустимый минимальный размер торговой сессии")
    if raw.empty:
        raise ValueError("Исходная выгрузка пуста")

    d = raw.copy()
    for name in REQUIRED:
        d[name] = pd.to_numeric(d[name], errors="raise")
        if not np.isfinite(d[name].to_numpy(dtype=float)).all():
            raise ValueError(f"В столбце {name} есть NaN или бесконечности")
    d["bar_start_utc"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    if d["bar_start_utc"].duplicated().any():
        raise ValueError("Повторяющиеся часовые свечи")
    d = d.sort_values("bar_start_utc").reset_index(drop=True)
    prices = d[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (d["volume"] < 0).any():
        raise ValueError("Обнаружены отрицательные объёмы или неположительные цены")
    if ((d["low"] > d[["open", "close"]].min(axis=1)) |
        (d["high"] < d[["open", "close"]].max(axis=1))).any():
        raise ValueError("Неверные OHLC часовых свечей")

    d["bar_end_utc"] = d["bar_start_utc"] + HOUR
    local_end = d["bar_end_utc"].dt.tz_convert(NY)
    # Конец в 17:00 относится к завершившейся сегодня сессии.
    d["date"] = (local_end + pd.Timedelta(hours=7) -
                 pd.Timedelta(nanoseconds=1)).dt.date
    first_date, stop_date = date.fromisoformat(start), date.fromisoformat(end_exclusive)
    d = d[(d["date"] >= first_date) & (d["date"] < stop_date)]
    d = d[pd.to_datetime(d["date"].astype(str)).dt.dayofweek < 5].copy()
    if d.empty:
        raise ValueError("Не найдено сессий в выбранном периоде")

    grouped = d.groupby("date", sort=True).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        session_hours=("close", "size"),
        first_bar_end=("bar_end_utc", "min"),
        last_bar_end=("bar_end_utc", "max"),
        volume=("volume", "sum"),
    )
    included, excluded, shortened = [], [], []
    for session_day, row in grouped.iterrows():
        first_local = row["first_bar_end"].tz_convert(NY)
        last_local = row["last_bar_end"].tz_convert(NY)
        count = int(row["session_hours"])
        elapsed = (row["last_bar_end"] - row["first_bar_end"]) / HOUR
        contiguous = abs(float(elapsed) - (count - 1)) < 1e-9
        starts_at_open = first_local.strftime("%H:%M") == "19:00"
        ends_in_allowed_window = ("13:00" <= last_local.strftime("%H:%M") <= "17:00")
        reason = []
        if not contiguous:
            reason.append("Есть часовые пробелы внутри сессии")
        if not starts_at_open:
            reason.append(f"Нет начала сессии: первая закрытая свеча {first_local:%H:%M}")
        if not ends_in_allowed_window:
            reason.append(f"Неожиданный конец сессии: {last_local:%H:%M}")
        if count < min_session_hours:
            reason.append(f"Всего {count} часов, минимально {min_session_hours}")
        if count > 23:
            reason.append(f"Слишком много свечей: {count}")
        if reason:
            excluded.append({"date": str(session_day), "hours": count, "reason": "; ".join(reason)})
            continue
        short = count != 23 or last_local.strftime("%H:%M") != "17:00"
        record = {
            "date": str(session_day),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "session_hours": count,
            "short_session": int(short),
            "close_utc": row["last_bar_end"].isoformat(),
            "volume": float(row["volume"]),
        }
        included.append(record)
        if short:
            shortened.append({
                "date": str(session_day), "hours": count,
                "last_bar_end_ny": last_local.strftime("%H:%M"),
                "status": "Требует сверки с праздничным расписанием",
            })
    if not included:
        raise ValueError("Все сессии отклонены проверками")
    daily = pd.DataFrame(included)
    if daily["date"].duplicated().any():
        raise AssertionError("Повторяющаяся торговая дата")
    report = {
        "source": "Dukascopy XAU/USD H1 Bid, объём включён",
        "timezone": "America/New_York",
        "official_session": "18:00–17:00 по Нью-Йорку",
        "range_start": start, "range_end_exclusive": end_exclusive,
        "hourly_rows": int(len(raw)),
        "observed_sessions_in_range": int(len(grouped)),
        "accepted_sessions": int(len(daily)),
        "first_date": str(daily.iloc[0]["date"]),
        "last_date": str(daily.iloc[-1]["date"]),
        "short_sessions_for_review": shortened,
        "excluded_sessions": excluded,
        "note": "Сокращённые сессии включены при непрерывных котировках; праздничные часы сверить отдельно.",
    }
    return daily, report
