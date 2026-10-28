"""Starter entry point. --demo uses synthetic data only, not study results."""
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
    print("SYNTHETIC DEMO ONLY - not XAU/USD experimental results")
    print(evaluate(scored).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if not args.demo:
        parser.error("Real data and other models are not implemented yet. Use --demo.")
    demo()
