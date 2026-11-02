"""Внешние данные нельзя использовать до их фактической публикации."""
import pandas as pd
import pytest
from src.external_features import attach_verified_features


def test_asof_respects_publication_time():
    sessions = pd.DataFrame({
        "date": ["2026-07-07", "2026-07-08"],
        "close_utc": ["2026-07-07T21:00:00Z", "2026-07-08T21:00:00Z"],
        "close": [100, 101],
    })
    releases = pd.DataFrame({
        "feature": ["index_close", "index_close"],
        "value": [10, 20],
        "observed_at_utc": ["2026-07-07T19:00:00Z", "2026-07-07T22:00:00Z"],
        "available_at_utc": ["2026-07-07T20:00:00Z", "2026-07-07T22:30:00Z"],
    })
    out = attach_verified_features(sessions, releases, ("index_close",))
    assert out["index_close"].tolist() == [10, 20]


def test_rejects_missing_publication_time():
    sessions = pd.DataFrame({"date": ["2026-07-07"], "close_utc": ["2026-07-07T21:00:00Z"]})
    raw = pd.DataFrame({"feature": ["index_close"], "value": [1], "trading_date": ["2026-07-07"]})
    with pytest.raises(ValueError, match="временных меток"):
        attach_verified_features(sessions, raw, ("index_close",))


def test_rejects_historical_revision_available_earlier_than_observation():
    sessions = pd.DataFrame({"date": ["2026-07-07"], "close_utc": ["2026-07-07T21:00:00Z"]})
    released = pd.DataFrame({
        "feature": ["macro"], "value": [1], "observed_at_utc": ["2026-07-08T00:00:00Z"],
        "available_at_utc": ["2026-07-07T00:00:00Z"],
    })
    with pytest.raises(ValueError, match="предшествовать"):
        attach_verified_features(sessions, released, ("macro",))


def test_rejects_alt_xauusd_prices():
    sessions = pd.DataFrame({"date": ["2026-07-07"], "close_utc": ["2026-07-07T21:00:00Z"]})
    release = pd.DataFrame({
        "feature": ["xauusd_close"], "value": [4000],
        "observed_at_utc": ["2026-07-07T19:00:00Z"],
        "available_at_utc": ["2026-07-07T20:00:00Z"],
    })
    with pytest.raises(ValueError, match="котировки"):
        attach_verified_features(sessions, release, ("xauusd_close",))
