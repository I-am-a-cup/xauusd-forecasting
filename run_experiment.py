"""Starter entry point. --demo uses synthetic data ONLY, never study findings."""
import argparse
import pandas as pd
from src.models.baseline import predict_naive
from src.metrics import attach_actuals, evaluate


def demo() -> None:
    prices = pd.DataFrame({
        "date": pd.bdate_range("2026-09-01", periods=25),
        "close": [2500 + i * 2 + (i % 3) for i in range(25)],
    })
    origins = list(range(12, 20))  # h=5 available for each selected origin
    preds = predict_naive(prices, origins)
    scored = attach_actuals(preds, prices)
    print("SYNTHETIC DEMO ONLY — not XAU/USD experimental results")
    print(evaluate(scored).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run synthetic smoke test")
    args = parser.parse_args()
    if not args.demo:
        parser.error("Real data and other models are not implemented yet. Use --demo.")
    demo()
