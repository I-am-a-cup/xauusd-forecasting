# XAU/USD Forecasting — Practices 8–9

Comparative forecasting of the daily closing price of gold (XAU/USD) for **1 and 5 trading observations ahead**.

> **Status:** Project scaffold. No real Dukascopy data have been downloaded or evaluated yet. All experiment results are pending.

## Assignment
- Twelve months of XAU/USD daily OHLC quotes from Dukascopy, anchored to the assignment date.
- Compare naive persistence, ARIMA, Random Forest, and a small MLP.
- Use chronological validation without future leakage; evaluate the same forecast origins for both horizons and every model.
- Report MAE, RMSE, sMAPE, directional accuracy, stability, horizon effects, and computational cost.
- Deliver a reproducible open repository, analysis, and a 10-minute presentation.

## Team ownership
| Area | Owner | Deliverables |
|---|---|---|
| Experimental design, baseline, integration | Captain | experiment protocol, baseline, metric calculations, conclusions |
| Data engineering and validation | Alexey | sourced CSV, EDA, preprocessing, folds and feature pipeline |
| Statistical + traditional ML | Dmitry | ARIMA and Random Forest with forecasts |
| Neural network | Andrey | MLP and computation/stability analysis |

Everyone prepares descriptions/plots for their part, reviews another component, and joins the presentation.

## Getting started (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python run_experiment.py --demo
```
The demo runs on **synthetic sample data only**. Do not report its results as experimental conclusions.

## Data convention
Processed CSV: `date,open,high,low,close`, dates in ascending order (date is the daily trading observation date; exact timestamp convention documented by data owner). Define whether Bid or Ask was used and UTC/day boundaries. Keep the actual raw/processed CSV in the repository once validated.

## Predictions contract
Every model emits `origin_date,target_date,horizon,model,y_pred`. The common evaluator joins the actual `y_true` from prices by `target_date`. Origins are trading-row positions after the current day's Close is known. Horizons are **observation steps**, not calendar days.

## Research and reproduction
See [EXPERIMENT.md](EXPERIMENT.md). Model code is a starter scaffold: ARIMA/RF/MLP and real temporal validation are **not implemented yet**. Do not claim fitted results before running them.

## Data provenance
Required source: [Dukascopy XAU/USD](https://www.dukascopy-node.app/instrument/xauusd). Proposed time window is 2025-10-28—2026-10-28 **only if** the assignment was issued 2026-10-28.
