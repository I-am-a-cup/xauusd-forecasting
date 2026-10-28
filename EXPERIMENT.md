# Experiment protocol (draft, approve before training)

## Question
Can statistical, machine-learning or neural-network methods beat persistence when forecasting daily XAU/USD Close 1 and 5 trading sessions ahead?

## Instrument and observations
- Dukascopy XAU/USD, OHLC D1, twelve months prior to the **actual assignment date** (to be confirmed).
- Proposed Bid Close and UTC export are conventions to validate, **not facts about the downloaded data**.
- Record first/last complete observations, timezone/day cutoffs, weekends, abnormal gaps, missing bars, and any rejected records.
- Never invent prices for non-trading days.

## Forecast origin and targets
At the close of trading observation t, features may use observations <= t only. Targets are C[t+1], C[t+5]. A target is available for fitting only once that future observation has occurred.

## Candidate models
1. Persistence baseline: C-hat[t+h] = C[t].
2. ARIMA (statistical).
3. RandomForestRegressor (ML), two direct horizon-specific estimators.
4. Small MLP (neural), two direct horizon-specific estimators.
Optional extras only after the four approaches are reproducible.

## Validation — planned
- Reserve the final ~20% of the time-indexed observations for a single holdout; choose exact indices after inspecting data.
- On earlier data, validate on multiple expanding temporal folds.
- At each fit origin, exclude **all training examples whose target observation occurs later than the model-fit cutoff**. A gap is not a substitute for verifying target availability.
- Fit imputation, scalers, feature selection, and hyperparameters only on training data in each fold.
- Select the **same forecast-origin dates** for all models and both horizons; ensure C[t+5] is observed, or omit that origin from both horizons.
- Define model updating on holdout beforehand: either refit all per origin on only available data, or freeze all models at a common cutoff and use new observed covariates without refitting. Record deviations explicitly.
- Do not tune hyperparameters on the final holdout.

## Primary and secondary metrics
Primary: MAE, in USD per troy ounce. Also report RMSE, sMAPE, and direction accuracy.
- sMAPE = 100 * mean(2|forecast-actual| / (|actual| + |forecast|)); define the zero/zero term as 0.
- Direction accuracy = mean(sign(forecast-C[t]) == sign(actual-C[t])). For persistence the predicted sign is zero, so interpret this diagnostic carefully, particularly when actual changes are nonzero.
Compute these independently per horizon, also comparing absolute MAE/RMSE to persistence. Report fit and inference wall time, fixed random seeds, and stability across evaluation windows.

## Outputs
Prediction CSV schema: `origin_date,target_date,horizon,model,y_pred`.
The evaluator adds `y_true` and `origin_close` using the canonical price dataset.
Never evaluate a prediction on a target that is missing from data.

## Limitations and status
Small sample, price nonstationarity, price-level metrics, spot/CFD quote conventions, transaction costs excluded, no causal or trading-performance claims.
**Pending:** actual data audit, fold boundaries, model training, measured results, and approval of methodological choices.
