"""Часть Андрея: python run_mlp_experiment.py (из корня проекта)."""
from dataclasses import asdict
from hashlib import sha256
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from src.daily_features import FEATURES, model_table
from src.metrics import assert_comparable_predictions, attach_actuals, evaluate
from src.models.baseline import predict_naive
from src.models.mlp import CANDIDATES, fit_mlp, predict_fitted, forecast_frame
from src.validation import expanding_folds, holdout_split

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 17, 73, 101, 137)


def run():
    started = perf_counter()
    source = ROOT / 'data/processed/xauusd_daily_dukascopy.csv'
    prices = pd.read_csv(source)
    if not pd.to_datetime(prices.date).is_monotonic_increasing:
        raise ValueError('Неупорядоченные торговые даты')
    split = holdout_split(len(prices))
    folds = expanding_folds(len(prices), 105, 25, 25,
                            end_before=split.last_known_index + 1)
    t = perf_counter()
    tables = [model_table(prices, f.last_known_index, f.forecast_origins) for f in folds]
    final = model_table(prices, split.last_known_index, split.forecast_origins)
    feature_seconds = perf_counter() - t
    published = pd.read_csv(ROOT / 'data/processed/xauusd_daily_model_ready.csv')
    pd.testing.assert_frame_equal(final.reset_index(drop=True), published,
                                  check_dtype=False, rtol=1e-10, atol=1e-10)
    cv, logs, chosen = [], [], {}
    cv_started = perf_counter()
    for h in (1, 5):
        for candidate, config in enumerate(CANDIDATES):
            for fold_id, (fold, table) in enumerate(zip(folds, tables), 1):
                train, valid = table[table.split.eq('train')], table[table.split.eq('test')]
                estimator, info = fit_mlp(train, h, config, 42)
                logs.append(dict(stage='cv', candidate=candidate, fold=fold_id, **info))
                pred = predict_fitted(estimator, valid)
                actual = valid[f'target_close_next_{h}d'].to_numpy()
                cv.append(dict(horizon=h, candidate=candidate, fold=fold_id,
                    n=len(valid), mae=float(np.mean(np.abs(pred-actual))),
                    naive_mae=float(np.mean(np.abs(valid.close.to_numpy()-actual))),
                    last_known_date=prices.date.iloc[fold.last_known_index],
                    train_last_origin=info['train_last_origin'],
                    first_validation_origin=valid.date.iloc[0],
                    last_validation_origin=valid.date.iloc[-1]))
        scores = pd.DataFrame(cv).query('horizon == @h').groupby('candidate').mae.mean()
        chosen[h] = int(scores.idxmin())
    cv_seconds = perf_counter() - cv_started
    train, test = final[final.split.eq('train')], final[final.split.eq('test')]
    predictions, seed_metrics = [], []
    for seed in SEEDS:
        frames = []
        for h in (1, 5):
            estimator, info = fit_mlp(train, h, CANDIDATES[chosen[h]], seed)
            t = perf_counter()
            predicted = predict_fitted(estimator, test)
            info['predict_seconds'] = perf_counter()-t
            logs.append(dict(stage='test', **info))
            frames.append(forecast_frame(prices, test, h, predicted))
        p = pd.concat(frames, ignore_index=True)
        p['seed'] = seed
        predictions.append(p)
        m = evaluate(attach_actuals(p, prices))
        m['seed'] = seed
        seed_metrics.append(m)
    all_predictions = pd.concat(predictions, ignore_index=True)
    main = all_predictions[all_predictions.seed.eq(42)].drop(columns='seed')
    t = perf_counter()
    naive = predict_naive(prices, split.forecast_origins)
    naive_seconds = perf_counter()-t
    combined = pd.concat([naive, main], ignore_index=True)
    assert_comparable_predictions(combined, expected_models=('naive', 'mlp'))
    scored = attach_actuals(combined, prices)
    metrics = evaluate(scored)
    base_mae = metrics[metrics.model.eq('naive')].set_index('horizon').mae
    metrics['mae_delta_vs_naive'] = metrics.mae - metrics.horizon.map(base_mae)
    metrics['mae_change_pct_vs_naive'] = 100*metrics.mae_delta_vs_naive / metrics.horizon.map(base_mae)
    blocks = []
    for number, dates in enumerate(np.array_split(np.array(split.forecast_origins), 3), 1):
        selected_dates = pd.to_datetime(prices.date.iloc[dates])
        part = scored[scored.origin_date.isin(selected_dates)]
        m = evaluate(part)
        m['block'] = number
        m['first_origin'] = str(selected_dates.iloc[0].date())
        m['last_origin'] = str(selected_dates.iloc[-1].date())
        blocks.append(m)
    scored['abs_error'] = (scored.y_true-scored.y_pred).abs()
    worst = scored.sort_values('abs_error', ascending=False).groupby(['model', 'horizon']).head(5)
    output = ROOT / 'results/mlp'
    output.mkdir(parents=True, exist_ok=True)
    outputs = dict(predictions=main, comparison=metrics, validation=pd.DataFrame(cv),
        seed_predictions=all_predictions, seed_metrics=pd.concat(seed_metrics),
        temporal_blocks=pd.concat(blocks), largest_errors=worst,
        timings=pd.DataFrame([{k:v for k,v in row.items() if k != 'warnings'} for row in logs]))
    for name, frame in outputs.items():
        frame.to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, h in enumerate((1, 5)):
        for model, group in scored[scored.horizon.eq(h)].groupby('model'):
            axes[row, 0].plot(group.origin_date, group.y_pred, label=model, linewidth=1.2)
            axes[row, 1].plot(group.origin_date, group.abs_error, label=model, linewidth=1.2)
        actual = scored[scored.horizon.eq(h) & scored.model.eq('mlp')]
        axes[row, 0].plot(actual.origin_date, actual.y_true, label='actual', color='black', linewidth=1)
        axes[row, 0].set_title(f'h={h}: target prices by forecast origin')
        axes[row, 1].set_title(f'h={h}: absolute error (USD)')
        for ax in axes[row]:
            ax.legend(); ax.grid(alpha=.2); ax.tick_params(axis='x', rotation=25)
    fig.suptitle('Dukascopy XAU/USD | frozen test | seed 42')
    fig.savefig(output / 'forecast_errors.png', dpi=150)
    plt.close(fig)
    metadata = dict(data_sha256=sha256(source.read_bytes()).hexdigest(),
        protocol_sha256=sha256((ROOT/'EXPERIMENT_PROTOCOL.md').read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scikit_learn=sklearn.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), processor=platform.processor(), threads=1,
        seed=42, sensitivity_seeds=SEEDS, features=list(FEATURES),
        target='log(C[t+h]/C[t]); price=C[t]*exp(predicted_log_return)',
        candidates=[asdict(c) for c in CANDIDATES],
        selected={h:asdict(CANDIDATES[i]) for h,i in chosen.items()},
        last_known_date=prices.date.iloc[split.last_known_index], train_rows=len(train),
        test_rows=len(test), fold_count=len(folds), feature_seconds=feature_seconds,
        cv_seconds=cv_seconds, naive_predict_seconds=naive_seconds,
        total_seconds=perf_counter()-started,
        training_policy='fixed weights and scalers; five-origin purge, known targets',
        limitations=['9 shortened sessions still require calendar confirmation',
                    'Only 47 origins; overlapping h=5 targets; no significance claim',
                    'Seed sensitivity does not select the final seed (always 42)'],
        fit_logs=logs)
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))
    print('Selected configs:', metadata['selected'])
    print('Saved:', output)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
