"""Создаёт реестр столбцов файла внешних признаков Алексея.

Не подготавливает обучающие признаки и не заполняет пропуски.
"""
from pathlib import Path

import pandas as pd


def catalog(source: Path, destination: Path) -> pd.DataFrame:
    raw = pd.read_csv(source, low_memory=False)
    if "trading_date" not in raw.columns:
        raise ValueError("Отсутствует trading_date")
    dates = pd.to_datetime(raw["trading_date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Даты должны идти по порядку без повторов")

    def status(name: str) -> tuple[str, str]:
        if name in {"trading_date", "datetime_utc"}:
            return "метка времени", "служебное поле, не признак"
        if name.startswith("target_") or "next_" in name or name.startswith("future_"):
            return "запрещено", "значения из будущего"
        if name.startswith("xauusd_"):
            return "не использовать", "основная цена золота берётся из Dukascopy"
        if name in {"weekday", "month", "day_of_year"}:
            return "пересчитать", "календарные признаки строятся по датам Dukascopy"
        return "не подтверждён", "не установлено время доступности значения"

    rows = []
    for col in raw.columns:
        state, reason = status(col)
        rows.append({
            "column": col,
            "dtype": str(raw[col].dtype),
            "missing_count": int(raw[col].isna().sum()),
            "present_count": int(raw[col].notna().sum()),
            "status": state,
            "reason": reason,
            "source": "сборка Алексея; уточнить по исходному коду",
            "available_at_verified": False,
        })
    out = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data/external/alexey_daily_features.csv"
    dest = repo / "data/external/alexey_feature_inventory.csv"
    result = catalog(src, dest)
    print(f"Сохранён реестр: {len(result)} столбцов")
    print(result.groupby("status").size().to_string())
