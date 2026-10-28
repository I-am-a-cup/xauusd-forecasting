"""Проверки аудита CSV без применения произвольной очистки."""
import pandas as pd
import pytest

from src.data_audit import audit_quotes


def _sample(path, dates=("2026-10-01", "2026-10-02")):
    d = pd.DataFrame({
        "timestamp": pd.to_datetime(dates, utc=True).astype("int64") // 1_000_000,
        "open": [100.0, 101.0], "high": [102.0, 102.0],
        "low": [99.0, 100.0], "close": [101.0, 101.0],
        "volume": [1.0, 2.0],
    })
    d.to_csv(path, index=False)


def test_audit_valid_csv(tmp_path):
    p = tmp_path / "test.csv"
    _sample(p)
    report = audit_quotes(p)
    assert report["число_строк"] == 2
    assert report["плоские_свечи"] == 0
    assert report["дубликаты"] == 0


def test_audit_rejects_dates_after_issue(tmp_path):
    p = tmp_path / "future.csv"
    _sample(p, dates=("2026-10-01", "2026-10-28"))
    with pytest.raises(ValueError, match="на дату выдачи"):
        audit_quotes(p)


def test_audit_rejects_invalid_ohlc(tmp_path):
    p = tmp_path / "invalid.csv"
    _sample(p)
    data = pd.read_csv(p)
    data.loc[0, "high"] = 99.5
    data.to_csv(p, index=False)
    with pytest.raises(ValueError, match="OHLC"):
        audit_quotes(p)
