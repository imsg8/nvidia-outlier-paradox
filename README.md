# The Outlier Paradox in NVIDIA Stock Prediction

Reproducibility package for the paper on the **outlier paradox** in NVIDIA (NVDA)
stock-price forecasting. It contains the full leakage-free pipeline (data
processing, statistical tests, and four model families), the raw experimental
outputs, and the 300-dpi figures used in the manuscript.

## The paradox

Models trained on **outlier-cleaned** data and then evaluated on the **full
(outlier-containing)** test set behave counter-intuitively: removing extreme
observations during training can degrade, not improve, out-of-sample accuracy,
because the outliers carry the regime-shift information the model must learn to
predict. The pipeline isolates this effect across three experimental modes and
three train/test splits.

## Repository structure

```
nvidia-outlier-paradox/
├── README.md
├── LICENSE                 Apache-2.0
├── requirements.txt        Python dependencies
├── .gitignore
├── src/                    Pipeline source code
│   ├── config.py           Paths, splits, seeds, hyperparameters
│   ├── data_pipeline.py    Cleaning, stationarity tests, outlier + break analysis, baseline
│   ├── plot_style.py       Shared matplotlib styling
│   ├── run_lr.py           Linear Regression (chronological + legacy baseline)
│   ├── run_arima.py        ARIMA (AIC grid search, walk-forward validation)
│   ├── run_dl_models.py    GRU + LSTM (5 seeds each)
│   └── run_all.py          Master runner
├── data/                   Input CSVs (NVDA + MAANG peers)
├── results/                Pre-computed experiment outputs (metrics, predictions, history)
└── figures/                300-dpi PNGs used in the paper
```

## Data

`data/NVDA.csv` covers daily NVDA OHLCV from 2015-01-02 to 2024-03-28. Peer
series (`AAPL`, `AMZN`, `GOOGL`, `META`, `NFLX`) support the cross-stock outlier
comparison; `NVDA_cleaned.csv` and `Diff_Normal_Wo-Outliers.csv` are
intermediate artifacts retained for transparency.

## Methodology highlights

- **No leakage:** chronological (not random) splits; IQR/z-score bounds and the
  scaler are fit on the training partition only; the rolling mean is causal
  (backward-looking).
- **Three modes:** `with_outliers`, `without_outliers`, and
  `clean_train_full_test` (the paradox condition).
- **Three splits:** 70:30, 80:20, 85:15.
- **Models:** Linear Regression, ARIMA, GRU, LSTM (deep models averaged over
  seeds 7, 42, 123, 456, 789).
- **Validation:** ADF/KPSS stationarity, structural breakpoint detection,
  persistence/naive baseline, and the Diebold-Mariano test.

## Reproducing the results

```bash
pip install -r requirements.txt

python src/run_all.py                 # full pipeline (pipeline + LR + ARIMA + DL)
python src/run_all.py --skip-dl       # fast: pipeline + ARIMA + LR only
python src/run_all.py --seeds 1       # quick smoke test (1 DL seed)
```

Outputs are written under `results/<MODEL>/<mode>/<split>/`. The committed
`results/` and `figures/` directories let readers verify every reported number
without re-running the models. Trained model weights (`*.h5`) are excluded by
`.gitignore` and are regenerated on each run.

## Results layout

Each run directory contains `metrics.json`, `test_predictions.csv` (and, for the
deep models, `training_history.csv`). Aggregate analyses live at the top of
`results/` (`stationarity_tests.csv`, `structural_breakpoints.csv`,
`persistence_baseline.csv`, `diebold_mariano_results.csv`,
`iqr_outlier_summary.csv`, `log_return_outlier_summary.csv`).

## License

Apache-2.0. See [LICENSE](https://github.com/imsg8/nvidia-outlier-paradox/blob/main/LICENSE).