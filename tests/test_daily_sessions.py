"""Проверки формирования сессий золота и горизонтов без утечки."""
import pandas as pd
import pytest

from src.daily_sessions import build_daily_sessions
from src.daily_features import model_table, FEATURES
from src.validation import holdout_split


def hours_for_day(session_day: str, count: int = 23) -> pd.DataFrame:
    # Сессия даты 7 июля начинается 6 июля в 18:00 по Нью-Йорку.
    day = pd.Timestamp(session_day)
    begin = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    starts = pd.date_range(
        f"{begin} 18:00", periods=count, freq="h", tz="America/New_York"
    ).tz_convert("UTC")
    return pd.DataFrame({
        "timestamp": starts.asi8 // 1_000_000,
        "open": [100.0 + i for i in range(count)],
        "high": [101.0 + i for i in range(count)],
        "low": [99.0 + i for i in range(count)],
        "close": [100.5 + i for i in range(count)],
        "volume": [1.0] * count,
    })


def test_regular_session_has_23_bars():
    data, report = build_daily_sessions(hours_for_day("2026-07-07"),
        start="2026-07-07", end_exclusive="2026-07-08")
    assert len(data) == 1
    assert data.date.iloc[0] == "2026-07-07"
    assert data.session_hours.iloc[0] == 23
    assert data.short_session.iloc[0] == 0
    assert report["excluded_sessions"] == []


def test_shortened_session_kept_but_flagged():
    data, report = build_daily_sessions(hours_for_day("2026-07-07", 19),
        start="2026-07-07", end_exclusive="2026-07-08")
    assert data.short_session.iloc[0] == 1
    assert data.session_hours.iloc[0] == 19
    assert len(report["short_sessions_for_review"]) == 1


def test_internal_missing_hour_excludes_session():
    hours = hours_for_day("2026-07-07")
    with pytest.raises(ValueError, match="Все сессии отклонены"):
        build_daily_sessions(hours.drop(index=10),
           start="2026-07-07", end_exclusive="2026-07-08")


def test_missing_start_excludes_session():
    hours = hours_for_day("2026-07-07")
    with pytest.raises(ValueError, match="Все сессии отклонены"):
        build_daily_sessions(hours.iloc[2:],
           start="2026-07-07", end_exclusive="2026-07-08")


def test_daylight_saving_utc_close_changes():
    march_friday = hours_for_day("2026-03-06")
    march_monday = hours_for_day("2026-03-09")
    all_hours = pd.concat([march_friday, march_monday])
    data, _ = build_daily_sessions(all_hours,
        start="2026-03-06", end_exclusive="2026-03-10")
    assert data.date.tolist() == ["2026-03-06", "2026-03-09"]
    assert pd.to_datetime(data.close_utc, utc=True).dt.hour.tolist() == [22, 21]


def test_gold_only_features_are_historical_and_targets_are_aligned():
    prices = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=120).strftime("%Y-%m-%d"),
        "open": 1000.0 + pd.Series(range(120)),
        "high": 1002.0 + pd.Series(range(120)),
        "low": 999.0 + pd.Series(range(120)),
        "close": 1001.0 + pd.Series(range(120)),
    })
    split = holdout_split(120, test_share=0.20, min_train_observations=90)
    out = model_table(prices, split.last_known_index, split.forecast_origins)
    assert list(out.columns) == ["date", "split", *FEATURES,
                                  "target_close_next_1d", "target_close_next_5d"]
    assert not out.isna().any().any()
    first = out.iloc[0]
    i = prices.index[prices.date == first.date][0]
    assert first.target_close_next_5d == prices.close.iloc[i+5]
    assert first.target_close_next_1d == prices.close.iloc[i+1]
    assert prices.date.iloc[split.last_known_index] > out.loc[
        out.split.eq("train"), "date"].max()
    assert len(out[out.split.eq("test")]) == len(split.forecast_origins)
