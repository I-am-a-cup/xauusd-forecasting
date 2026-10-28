"""Проверка исходной выгрузки Dukascopy без изменения котировок."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")
ASSIGNMENT_CUTOFF = pd.Timestamp("2026-10-28", tz="UTC")


def audit_quotes(path: str | Path) -> dict:
    """Возвращает наблюдаемые характеристики CSV; ничего не удаляет.

    Время исходных свечей — метка UTC в миллисекундах. Метка обозначает
    начало календарного интервала, а не гарантированное закрытие сессии.
    """
    data = pd.read_csv(path)
    missing = set(REQUIRED) - set(data.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {sorted(missing)}")
    if data.empty:
        raise ValueError("Файл котировок пуст")

    raw_numeric = data.loc[:, REQUIRED].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw_numeric.to_numpy(dtype=float)).all():
        raise ValueError("В котировках есть пропущенные или бесконечные значения")
    if not (raw_numeric["timestamp"] % 1000 == 0).all():
        raise ValueError("Неожиданная точность времени: ожидаются миллисекунды Unix")

    dates = pd.to_datetime(raw_numeric["timestamp"], unit="ms", utc=True)
    if dates.duplicated().any():
        raise ValueError("Повторяющиеся временные метки")
    if not dates.is_monotonic_increasing:
        raise ValueError("Временные метки не отсортированы")
    if dates.max() >= ASSIGNMENT_CUTOFF:
        raise ValueError("Обнаружена свеча на дату выдачи или после неё: проверьте завершённость")

    o, h, l, c = (raw_numeric[col] for col in ("open", "high", "low", "close"))
    if not ((l <= o) & (o <= h) & (l <= c) & (c <= h)).all():
        raise ValueError("Обнаружено нарушение соотношения OHLC")
    if (raw_numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Найдены неположительные цены золота")
    if (raw_numeric["volume"] < 0).any():
        raise ValueError("Обнаружен отрицательный объём")

    day_names = dates.dt.day_name()
    by_weekday = {name: int((day_names == name).sum()) for name in
                  ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")}
    flat = raw_numeric[["open", "high", "low", "close"]].nunique(axis=1) == 1
    volume_zero = raw_numeric["volume"] == 0
    return {
        "файл": Path(path).name,
        "число_строк": int(len(data)),
        "первая_метка_utc": dates.min().isoformat(),
        "последняя_метка_utc": dates.max().isoformat(),
        "дубликаты": int(dates.duplicated().sum()),
        "пустые_или_невалидные_цены": 0,
        "плоские_свечи": int(flat.sum()),
        "нулевой_объём": int(volume_zero.sum()),
        "субботние_метки": by_weekday["Saturday"],
        "воскресные_метки": by_weekday["Sunday"],
        "распределение_по_дням_недели": by_weekday,
        "предупреждение": (
            "UTC-свечи с активными котировками могут включать отдельную "
            "воскресную неполную сессию; границы торгового дня требуется согласовать. "
            "Эта проверка НЕ формирует очищенный ряд и не подтверждает календарь торгов."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить исходный CSV Dukascopy")
    parser.add_argument("csv", help="Путь к исходному CSV")
    args = parser.parse_args()
    print(json.dumps(audit_quotes(args.csv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
