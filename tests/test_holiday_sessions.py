"""Проверки девяти сокращённых сессий по опубликованному исходному H1."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "2025-11-27": (21, "15:00"),
    "2025-11-28": (21, "15:00"),
    "2025-12-24": (20, "14:00"),
    "2026-01-19": (21, "15:00"),
    "2026-02-16": (21, "15:00"),
    "2026-05-25": (21, "15:00"),
    "2026-06-19": (19, "13:00"),
    "2026-07-03": (19, "13:00"),
    "2026-09-07": (21, "15:00"),
}


def test_short_sessions_follow_raw_hours_without_internal_holes():
    raw = pd.read_csv(ROOT / "data/raw/xauusd-h1-bid-2025-10-27-2026-10-28.csv")
    audit = json.loads((ROOT / "data/processed/xauusd_daily_audit.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(ROOT / "data/processed/xauusd_daily_dukascopy.csv")
    assert set(EXPECTED) == {row["date"] for row in audit["short_sessions_for_review"]}
    assert set(EXPECTED) == set(daily.loc[daily.short_session.eq(1), "date"])
    hourly_end = pd.to_datetime(raw["timestamp"], unit="ms", utc=True) + pd.Timedelta(hours=1)
    local = hourly_end.dt.tz_convert("America/New_York")
    trade_date = (local + pd.Timedelta(hours=7) - pd.Timedelta(nanoseconds=1)).dt.strftime("%Y-%m-%d")
    for day, (expected_count, expected_time) in EXPECTED.items():
        selected = hourly_end[trade_date.eq(day)].sort_values().reset_index(drop=True)
        assert len(selected) == expected_count
        assert selected.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        assert selected.iloc[0].tz_convert("America/New_York").strftime("%H:%M") == "19:00"
        assert selected.iloc[-1].tz_convert("America/New_York").strftime("%H:%M") == expected_time
        row = daily.set_index("date").loc[day]
        assert int(row.session_hours) == expected_count
        assert pd.Timestamp(row.close_utc) == selected.iloc[-1]


def test_audit_document_lists_all_shortened_dates():
    doc = (ROOT / "HOLIDAY_SESSION_AUDIT.md").read_text(encoding="utf-8-sig")
    assert "не называем точное время" in doc.lower()
    for day in EXPECTED:
        parts = day.split("-")
        assert f"{parts[2]}.{parts[1]}.{parts[0]}" in doc
