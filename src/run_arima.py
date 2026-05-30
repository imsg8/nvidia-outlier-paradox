"""
NVIDIA Stock Prediction - ARIMA Model
=====================================
  - IQR on training data only for the "without outliers" mode
  - Full ARIMA parameter logging (p,d,q), AIC, stationarity tests, residual diagnostics
  - Exports forecast errors for the Diebold-Mariano test
  - clean_train_full_test mode (train on cleaned, test on full)
  - Plot styling with legends, annotations, metric boxes

Usage:
    python run_arima.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NVDA_CSV, RESULTS_DIR, TARGET_COL, DATE_COL, SPLIT_RATIOS,
    ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE,
)
from data_pipeline import remove_outliers_train_only, compute_metrics, compute_iqr_bounds
from plot_style import set_plot_style, COLORS, add_metric_box


def arima_grid_search(train, p_range, d_range, q_range):
    """Grid search for best ARIMA(p,d,q) using AIC."""
    best_aic = float("inf")
    best_order = None
    best_model = None
    search_log = []

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fit = model.fit()
                    aic = fit.aic
                    search_log.append({"p": p, "d": d, "q": q, "AIC": aic})
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fit
                except Exception:
                    continue

    return best_order, best_aic, best_model, pd.DataFrame(search_log)


def walk_forward_forecast(train, test, order):
    """Walk-forward one-step-ahead forecast on test set."""
    history = list(train)
    predictions = []
    forecast_errors = []

    for t in range(len(test)):
        model = ARIMA(history, order=order)
        fit = model.fit()
        yhat = fit.forecast()[0]
        predictions.append(yhat)
        forecast_errors.append(test.iloc[t] - yhat)
        history.append(test.iloc[t])

    return np.array(predictions), np.array(forecast_errors)


def residual_diagnostics(model_fit, save_dir, config_label):
    """Run and save residual diagnostic checks."""
    residuals = model_fit.resid

    results = {}

    # Ljung-Box test for residual autocorrelation
    try:
        lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        results["ljung_box"] = lb_result.to_dict()
        lb_result.to_csv(os.path.join(save_dir, "ljung_box_test.csv"))
    except Exception as e:
        results["ljung_box_error"] = str(e)

    # ADF test on residuals (should be stationary)
    try:
        adf = adfuller(residuals.dropna())
        results["residual_adf_statistic"] = adf[0]
        results["residual_adf_pvalue"] = adf[1]
        results["residual_adf_conclusion"] = (
            "Stationary" if adf[1] < 0.05 else "Non-stationary"
        )
    except Exception as e:
        results["residual_adf_error"] = str(e)

    # -- Residual Diagnostic Plots -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"ARIMA Residual Diagnostics - {config_label}",
                 fontsize=18, fontweight="bold", y=1.02)

    # Residual time series
    axes[0, 0].plot(residuals, color=COLORS["residual"], alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title("Residuals Over Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Residual Value")
    axes[0, 0].axhline(y=0, color=COLORS["zero_line"], linestyle="--", alpha=0.5,
                        label="Zero line")
    # Add ±2σ bands
    res_std = residuals.std()
    axes[0, 0].axhline(y=2*res_std, color="#999999", linestyle=":", alpha=0.4,
                        label=f"±2σ ({2*res_std:.2f})")
    axes[0, 0].axhline(y=-2*res_std, color="#999999", linestyle=":", alpha=0.4)
    axes[0, 0].legend(loc="upper right", fontsize=10)

    # Histogram
    axes[0, 1].hist(residuals, bins=40, edgecolor="black", alpha=0.7,
                     color=COLORS["fill"], label=f"n={len(residuals)}")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(x=0, color=COLORS["zero_line"], linestyle="--", alpha=0.5)
    # Add mean/std annotation
    add_metric_box(axes[0, 1], {
        "Mean": residuals.mean(),
        "Std": res_std,
        "Skew": residuals.skew(),
    }, loc="upper right")
    axes[0, 1].legend(loc="upper left", fontsize=10)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normal)")
    axes[1, 0].get_lines()[0].set_color(COLORS["train"])
    axes[1, 0].get_lines()[1].set_color(COLORS["zero_line"])

    # ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), ax=axes[1, 1], lags=30,
             color=COLORS["train"], vlines_kwargs={"color": COLORS["train"]})
    axes[1, 1].set_title("Residual ACF")
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_diagnostics.png"), dpi=300,
                bbox_inches="tight")
    plt.close()

    return results


def run_arima():
    """Run ARIMA for all splits and outlier modes (3-mode strategy)."""
    set_plot_style()

    df = pd.read_csv(NVDA_CSV, parse_dates=[DATE_COL])
    df = df[[DATE_COL, TARGET_COL]].dropna().sort_values(DATE_COL).reset_index(drop=True)
    df = df.set_index(DATE_COL)
    df = df.asfreq("B").ffill()
    prices = df[TARGET_COL]

    arima_dir = os.path.join(RESULTS_DIR, "ARIMA")
    os.makedirs(arima_dir, exist_ok=True)

    all_results = []

    # 3-mode outlier strategy
    for outlier_mode in ["with_outliers", "without_outliers", "clean_train_full_test"]:
        for ratio in SPLIT_RATIOS:
            label = f"{int(ratio*100)}_{100 - int(ratio*100)}"
            config_label = f"{outlier_mode}/{label}"
            print(f"\n{'='*60}")
            print(f"  ARIMA | {config_label}")
            print(f"{'='*60}")

            if outlier_mode in ("without_outliers", "clean_train_full_test"):
                # Reset index for outlier removal
                df_reset = df.reset_index()
                train_idx = int(len(df_reset) * ratio)
                df_clean, bounds, _ = remove_outliers_train_only(
                    df_reset, train_idx, column=TARGET_COL
                )
                df_clean = df_clean.set_index(DATE_COL)
                df_clean = df_clean.asfreq("B").ffill()
                clean_prices = df_clean[TARGET_COL]

                if outlier_mode == "clean_train_full_test":
                    # Train on cleaned data, test on FULL original test set
                    train_size = int(len(clean_prices) * ratio)
                    train = clean_prices[:train_size]
                    # Test set: original prices from the split point onward
                    orig_train_size = int(len(prices) * ratio)
                    test = prices[orig_train_size:]
                else:
                    # without_outliers: both train and test are from cleaned data
                    working_prices = clean_prices
                    train_size = int(len(working_prices) * ratio)
                    train = working_prices[:train_size]
                    test = working_prices[train_size:]
            else:
                working_prices = prices
                train_size = int(len(working_prices) * ratio)
                train = working_prices[:train_size]
                test = working_prices[train_size:]

            # ADF test on training data
            adf = adfuller(train.dropna())
            print(f"  ADF on training data: stat={adf[0]:.4f}, p={adf[1]:.4f} "
                  f"({'Stationary' if adf[1] < 0.05 else 'Non-stationary'})")

            # Grid search
            print("  Grid searching (p,d,q)...")
            best_order, best_aic, best_model, search_log = arima_grid_search(
                train, ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE
            )
            p, d, q = best_order
            print(f"Best: ARIMA({p},{d},{q}) | AIC={best_aic:.2f}")

            # Save grid search log
            config_dir = os.path.join(arima_dir, outlier_mode, label)
            os.makedirs(config_dir, exist_ok=True)
            search_log.to_csv(
                os.path.join(config_dir, "grid_search_log.csv"), index=False
            )

            # Residual diagnostics
            diag_results = residual_diagnostics(best_model, config_dir, config_label)

            # Walk-forward forecast
            print("  Walk-forward forecasting...")
            predictions, forecast_errors = walk_forward_forecast(
                train, test, best_order
            )

            # Training predictions
            train_pred = best_model.fittedvalues[d:]
            y_train_eval = train[d:]

            # Metrics
            train_metrics = compute_metrics(y_train_eval, train_pred)
            test_metrics = compute_metrics(test, predictions)

            print(f"  Train - RMSE={train_metrics['RMSE']:.3f}, "
                  f"MAE={train_metrics['MAE']:.3f}, R²={train_metrics['R2']:.4f}")
            print(f"  Test  - RMSE={test_metrics['RMSE']:.3f}, "
                  f"MAE={test_metrics['MAE']:.3f}, R²={test_metrics['R2']:.4f}")

            # Save predictions and errors
            pd.DataFrame({
                "Date": test.index,
                "Actual": test.values,
                "Predicted": predictions,
                "Error": forecast_errors,
            }).to_csv(os.path.join(config_dir, "predictions.csv"), index=False)

            # Save forecast errors for Diebold-Mariano test
            pd.DataFrame({
                "forecast_error": forecast_errors
            }).to_csv(
                os.path.join(config_dir, "forecast_errors.csv"), index=False
            )

            # ── Prediction Plot (with legends, annotations, styling) ──
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(train.index, train.values,
                    color=COLORS["train"], label="Train", alpha=0.7)
            ax.plot(test.index, test.values,
                    color=COLORS["test"], label="Test (Actual)", alpha=0.8,
                    linewidth=1.5)
            ax.plot(test.index, predictions,
                    color=COLORS["predicted"], label="Test (Predicted)",
                    linestyle="--", alpha=0.8, linewidth=1.5)

            # Train/test split line
            split_date = test.index[0]
            ax.axvline(x=split_date, color="#999999", linestyle=":",
                       alpha=0.6, label=f"Split ({label.replace('_', '/')})")

            ax.set_title(f"ARIMA({p},{d},{q}) - {config_label}", fontsize=18)
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price (USD)")
            ax.legend(loc="upper left", framealpha=0.9)

            # Metric annotation box
            add_metric_box(ax, {
                "Test R²": test_metrics["R2"],
                "Test RMSE": test_metrics["RMSE"],
                "Test MAE": test_metrics["MAE"],
                "AIC": best_aic,
            }, loc="lower right")

            plt.tight_layout()
            plt.savefig(os.path.join(config_dir, "predictions_plot.png"), dpi=300,
                        bbox_inches="tight")
            plt.close()

            # Collect results
            result = {
                "model": "ARIMA",
                "dataset": outlier_mode,
                "split": f"{int(ratio*100)}:{100 - int(ratio*100)}",
                "order_p": p, "order_d": d, "order_q": q,
                "AIC": best_aic,
                "train_adf_stat": adf[0],
                "train_adf_pvalue": adf[1],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            all_results.append(result)

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(arima_dir, "arima_summary.csv"), index=False)
    print(f"\nARIMA complete. Summary:")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_arima()
