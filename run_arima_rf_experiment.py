"""Корректный эксперимент: ARIMA и RF на том же Dukascopy и 47 датах.

Оригинальные файлы Дмитрия сохранены отдельно в archive/.
Запуск из корня: python run_arima_rf_experiment.py
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.daily_features import FEATURES, model_table
from src.metrics import assert_comparable_predictions, attach_actuals, evaluate
from src.models.arima import CANDIDATES as ARIMA_ORDERS, predict_arima
from src.models.random_forest import CANDIDATES as RF_CANDIDATES, fit_rf, predict_rf_fitted
from src.models.baseline import predict_naive
from src.validation import holdout_split, expanding_folds


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/processed/xauusd_daily_dukascopy.csv"
MLP = ROOT / "results/mlp/predictions.csv"


def rf_frame(prices, samples, horizon, predictions):
    dates = pd.to_datetime(prices.date).dt.strftime("%Y-%m-%d").to_numpy()
    indices = pd.Series(np.arange(len(dates)), index=pd.to_datetime(prices.date))
    current = pd.to_datetime(samples.date).map(indices).to_numpy(dtype=int)
    return pd.DataFrame({
        "origin_date": dates[current],
        "target_date": dates[current+horizon],
        "horizon": horizon,
        "model": "random_forest",
        "y_pred": predictions,
    })


def run():
    prices = pd.read_csv(SOURCE)
    split = holdout_split(len(prices))
    folds = expanding_folds(len(prices), 105, 25, 25,
                            end_before=split.last_known_index+1)
    final = model_table(prices, split.last_known_index, split.forecast_origins)
    published = pd.read_csv(ROOT / "data/processed/xauusd_daily_model_ready.csv")
    pd.testing.assert_frame_equal(final.reset_index(drop=True), published,
                                  check_dtype=False, rtol=1e-10, atol=1e-10)

    cv = []
    for candidate, order in enumerate(ARIMA_ORDERS):
        for fold_id, fold in enumerate(folds, 1):
            preds, _ = predict_arima(prices, fold.last_known_index,
                                     fold.forecast_origins, order)
            evaluated = attach_actuals(preds, prices)
            for horizon in (1, 5):
                g = evaluated[evaluated.horizon.eq(horizon)]
                cv.append(dict(model="arima", horizon=horizon, candidate=candidate,
                               fold=fold_id, mae=float(np.mean(np.abs(g.y_pred-g.y_true))),
                               order=str(order), n=len(g),
                               last_known_date=prices.date.iloc[fold.last_known_index]))
    for horizon in (1, 5):
        for candidate, config in enumerate(RF_CANDIDATES):
            for fold_id, fold in enumerate(folds, 1):
                table = model_table(prices, fold.last_known_index, fold.forecast_origins)
                train = table[table.split.eq("train")]
                valid = table[table.split.eq("test")]
                model, seconds = fit_rf(train, horizon, config)
                pred = predict_rf_fitted(model, valid)
                y = valid[f"target_close_next_{horizon}d"].to_numpy(dtype=float)
                cv.append(dict(model="random_forest", horizon=horizon,
                               candidate=candidate, fold=fold_id,
                               mae=float(np.mean(np.abs(pred-y))), n=len(y),
                               fit_seconds=seconds, config=str(asdict(config)),
                               last_known_date=prices.date.iloc[fold.last_known_index]))
    cv_frame = pd.DataFrame(cv)
    chosen = {}
    for model in ("arima", "random_forest"):
        chosen[model] = {}
        for h in (1, 5):
            scores = cv_frame[(cv_frame.model == model) & (cv_frame.horizon == h)].groupby("candidate").mae.mean()
            chosen[model][h] = int(scores.idxmin())

    all_predictions = []
    timings = []
    for horizon in (1, 5):
        order = ARIMA_ORDERS[chosen["arima"][horizon]]
        pred, info = predict_arima(prices, split.last_known_index,
                                   split.forecast_origins, order, horizons=(horizon,))
        all_predictions.append(pred)
        timings.append(dict(model="arima", horizon=horizon, **info))

        config = RF_CANDIDATES[chosen["random_forest"][horizon]]
        train = final[final.split.eq("train")]
        test = final[final.split.eq("test")]
        estimator, fit_seconds = fit_rf(train, horizon, config)
        before = perf_counter()
        y_pred = predict_rf_fitted(estimator, test)
        predict_seconds = perf_counter()-before
        all_predictions.append(rf_frame(prices, test, horizon, y_pred))
        timings.append(dict(model="random_forest", horizon=horizon,
                            config=asdict(config), train_rows=len(train),
                            fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                            refit_on_test=False))
    additions = pd.concat(all_predictions, ignore_index=True)
    naive = predict_naive(prices, split.forecast_origins)
    mlp = pd.read_csv(MLP)
    combined = pd.concat([naive, mlp, additions], ignore_index=True)
    assert_comparable_predictions(combined,
                                  expected_models=("naive", "mlp", "arima", "random_forest"))
    scored = attach_actuals(combined, prices)
    metrics = evaluate(scored)
    naive_mae = metrics[metrics.model.eq("naive")].set_index("horizon").mae
    metrics["mae_delta_vs_naive"] = metrics.mae - metrics.horizon.map(naive_mae)
    metrics["mae_change_pct_vs_naive"] = 100*metrics.mae_delta_vs_naive / metrics.horizon.map(naive_mae)

    blocks = []
    for num, indices in enumerate(np.array_split(np.array(split.forecast_origins), 3), 1):
        selected = pd.to_datetime(prices.date.iloc[indices])
        x = scored[scored.origin_date.isin(selected)].copy()
        b = evaluate(x)
        b["block"] = num
        b["first_origin"] = selected.iloc[0].strftime("%Y-%m-%d")
        b["last_origin"] = selected.iloc[-1].strftime("%Y-%m-%d")
        blocks.append(b)

    target = ROOT / "results/arima_rf"
    target.mkdir(parents=True, exist_ok=True)
    additions.to_csv(target / "predictions.csv", index=False, encoding="utf-8-sig")
    additions.to_csv(ROOT / "arima_rf_predictions.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(target / "all_four_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(target / "comparison.csv", index=False, encoding="utf-8-sig")
    cv_frame.to_csv(target / "validation.csv", index=False, encoding="utf-8-sig")
    pd.concat(blocks).to_csv(target / "temporal_blocks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timings).to_csv(target / "timings.csv", index=False, encoding="utf-8-sig")
    meta = {
        "source": str(SOURCE.relative_to(ROOT)),
        "test_last_known": prices.date.iloc[split.last_known_index],
        "first_test_origin": prices.date.iloc[split.forecast_origins[0]],
        "last_test_origin": prices.date.iloc[split.forecast_origins[-1]],
        "test_origins": len(split.forecast_origins),
        "validation_folds": len(folds),
        "features": list(FEATURES),
        "selected_arima": {str(h):list(ARIMA_ORDERS[i]) for h,i in chosen["arima"].items()},
        "selected_rf": {str(h):asdict(RF_CANDIDATES[i]) for h,i in chosen["random_forest"].items()},
        "python": platform.python_version(),
        "methodology": "CV on past, same test as Naive and MLP; frozen parameters; ARIMA state update without refit; RF log-return direct h",
        "original_dmitry_files": ["archive/dmitry_original_run_arima_rf_experiment.py", "archive/dmitry_original_arima_rf_predictions.csv"],
    }
    (target/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    scored["abs_error"]=(scored.y_true-scored.y_pred).abs()
    for i,h in enumerate((1,5)):
        for name, g in scored[scored.horizon.eq(h)].groupby("model"):
            axes[i,0].plot(g.origin_date,g.y_pred,label=name,linewidth=1)
            axes[i,1].plot(g.origin_date,g.abs_error,label=name,linewidth=1)
        ground=scored[(scored.horizon.eq(h)) & scored.model.eq("naive")]
        axes[i,0].plot(ground.origin_date,ground.y_true,color="black",label="actual",linewidth=1)
        for axis in axes[i]:
            axis.tick_params(axis="x",rotation=35)
            axis.grid(alpha=0.2)
            axis.legend()
        axes[i,0].set_title(f"h={h}: prediction and target")
        axes[i,1].set_title(f"h={h}: absolute error")
    fig.savefig(target/"comparison.png",dpi=140)
    plt.close(fig)
    print("Согласованные метрики четырёх подходов:")
    print(metrics.to_string(index=False))
    print("Выбрано по валидации:",meta["selected_arima"],meta["selected_rf"])
    return metrics


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
