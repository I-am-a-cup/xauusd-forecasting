"""Точка запуска: ключ --demo использует только искусственные данные."""
import argparse
import pandas as pd
from src.models.baseline import predict_naive
from src.metrics import attach_actuals, evaluate, assert_comparable_predictions


def demo() -> None:
    prices = pd.DataFrame({
        "date": pd.bdate_range("2026-09-01", periods=25),
        "close": [2500 + i * 2 + (i % 3) for i in range(25)],
    })
    origins = list(range(12, 20))
    preds = predict_naive(prices, origins)
    assert_comparable_predictions(preds)
    scored = attach_actuals(preds, prices)
    print("ПРОВЕРОЧНЫЙ ЗАПУСК НА ИСКУССТВЕННЫХ ДАННЫХ — это не результаты по XAU/USD")
    print(evaluate(scored).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Проверить работу на искусственных данных")
    args = parser.parse_args()
    if not args.demo:
        parser.error("Реальные данные и остальные модели пока не подключены. Используйте --demo.")
    demo()
