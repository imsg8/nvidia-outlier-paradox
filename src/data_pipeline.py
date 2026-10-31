"""
NVIDIA Stock Prediction - Data Pipeline
=======================================
Leakage-free data preparation:
  - IQR bounds computed on training data only, per split
  - Rolling mean applied causally (no future data)
  - Scaler fit on training data only
  - Log-return based outlier analysis
  - Stationarity tests (ADF, KPSS)

Usage:
    python data_pipeline.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from plot_style import set_plot_style, COLORS, add_metric_box
import matplotlib.pyplot as plt

# Add parent to path for config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NVDA_CSV, DATA_DIR, RESULTS_DIR, TARGET_COL, DATE_COL,
    SPLIT_RATIOS, IQR_MULTIPLIER, DL_SMOOTHING_WINDOW, MAANG_TICKERS,
)


# =====================================================================
# IQR Outlier Detection (training-only bounds)
# =====================================================================
def compute_iqr_bounds(series, multiplier=IQR_MULTIPLIER):
    """Compute IQR-based outlier bounds from a given series."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return lower, upper, Q1, Q3, IQR


def remove_outliers_train_only(df, train_idx, column=TARGET_COL):
    """
    Remove outliers using IQR bounds computed ONLY from training data,
    which avoids leaking test-set statistics into the bounds.

    Returns:
        df_clean: DataFrame with outliers removed (from both train and test)
        bounds: dict with Q1, Q3, IQR, lower, upper
        outlier_mask: boolean mask indicating outlier positions
    """
    train_data = df.iloc[:train_idx]
    lower, upper, Q1, Q3, IQR = compute_iqr_bounds(train_data[column])

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    df_clean = df[~outlier_mask].copy()

    bounds = {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "lower": lower, "upper": upper,
        "n_outliers": outlier_mask.sum(),
        "pct_outliers": outlier_mask.mean() * 100,
    }
    return df_clean, bounds, outlier_mask


# =====================================================================
# Log-Return Outlier Analysis
# =====================================================================
def compute_log_returns(df, column=TARGET_COL):
    """Compute daily log returns: r_t = ln(P_t / P_{t-1})."""
    df = df.copy()
    df["log_return"] = np.log(df[column] / df[column].shift(1))
    return df.dropna(subset=["log_return"])


def iqr_outliers_on_returns(df, train_idx, column="log_return"):
    """Apply IQR outlier detection to log returns (training data only)."""
    train_data = df.iloc[:train_idx]
    lower, upper, Q1, Q3, IQR = compute_iqr_bounds(train_data[column])

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    bounds = {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "lower": lower, "upper": upper,
        "n_outliers": outlier_mask.sum(),
        "pct_outliers": outlier_mask.mean() * 100,
    }
    return outlier_mask, bounds


# =====================================================================
# Causal Rolling Mean
# =====================================================================
def causal_rolling_mean(series, window=DL_SMOOTHING_WINDOW):
    """
    Apply rolling mean using ONLY past values (no look-ahead), so no
    future information crosses the train-test boundary.

    Uses min_periods=1 to handle the start of the series.
    """
    return series.rolling(window=window, min_periods=1).mean()


# =====================================================================
# Train-Only Scaling
# =====================================================================
def fit_scaler_train_only(train_data):
    """
    Fit StandardScaler on training data ONLY, so test-set statistics
    are never used when scaling.
    """
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler


# =====================================================================
# DL Dataset Preparation (sliding window)
# =====================================================================
def create_sequences(data, window_size):
    """Create sliding-window sequences for DL models."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


# =====================================================================
# Stationarity Tests
# =====================================================================
def run_stationarity_tests(series, name="series"):
    """Run ADF and KPSS tests and return results dict."""
    from statsmodels.tsa.stattools import adfuller, kpss

    results = {"series": name}

    # ADF test (null: unit root / non-stationary)
    adf_result = adfuller(series.dropna(), autolag="AIC")
    results["adf_statistic"] = adf_result[0]
    results["adf_pvalue"] = adf_result[1]
    results["adf_conclusion"] = (
        "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
    )

    # KPSS test (null: stationary)
    try:
        kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
        results["kpss_statistic"] = kpss_result[0]
        results["kpss_pvalue"] = kpss_result[1]
        results["kpss_conclusion"] = (
            "Stationary" if kpss_result[1] > 0.05 else "Non-stationary"
        )
    except Exception as e:
        results["kpss_statistic"] = None
        results["kpss_pvalue"] = None
        results["kpss_conclusion"] = f"Error: {e}"

    return results


# =====================================================================
# Structural Break Test
# =====================================================================
def run_structural_break_test(series, n_bkps=5, model="l2"):
    """
    Apply Bai-Perron style structural break detection using ruptures.

    Uses BinSeg with a fixed number of breakpoints.

    Args:
        series: 1D numpy array of prices
        n_bkps: number of breakpoints to detect (default: 5)
        model: cost model ('l2', 'rbf', etc.)

    Returns:
        breakpoints: list of breakpoint indices
    """
    import ruptures as rpt

    algo = rpt.Binseg(model=model).fit(series.reshape(-1, 1))
    breakpoints = algo.predict(n_bkps=n_bkps)
    return breakpoints


# =====================================================================
# Naive/Persistence Baseline
# =====================================================================
def persistence_baseline(y_true):
    """
    Compute persistence baseline: y_hat(t+1) = y(t).

    Returns predictions aligned with y_true[1:].
    """
    return y_true[:-1]  # prediction for t+1 is value at t


# =====================================================================
# Metrics
# =====================================================================
def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2 for a pair of arrays."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# =====================================================================
# Full Pipeline: Prepare data for one split
# =====================================================================
def prepare_split(df, split_ratio, remove_outliers=False,
                  clean_train_full_test=False, for_dl=False,
                  window_size=DL_SMOOTHING_WINDOW, dl_time_step=100):
    """
    Prepare train/test data for one split ratio.

    Steps:
    1. Chronological split
    2. (Optional) IQR outlier removal using TRAINING bounds only
    3. (Optional) clean_train_full_test: remove outliers from training
       but keep the FULL (uncleaned) test set
    4. (For DL) Causal rolling mean
    5. (For DL) Scaler fit on training only
    6. (For DL) Sliding window sequences

    Args:
        df: DataFrame with DATE_COL and TARGET_COL
        split_ratio: float, fraction for training
        remove_outliers: if True, remove outliers from BOTH train and test
        clean_train_full_test: if True, remove outliers from training only,
                               keep the full original test set intact
        for_dl: if True, prepare sliding-window sequences for DL models
        window_size: rolling mean window size
        dl_time_step: sliding window length for DL sequences

    Returns a dict with all necessary data and metadata.
    """
    n = len(df)
    train_idx = int(n * split_ratio)

    result = {
        "split_ratio": split_ratio,
        "split_label": f"{int(split_ratio*100)}:{100 - int(split_ratio*100)}",
        "train_idx": train_idx,
        "n_total": n,
    }

    working_df = df.copy()

    # Step 1: Outlier removal - training-only bounds
    if clean_train_full_test:
        # Mode: Clean training data, but keep FULL original test set
        train_data = working_df.iloc[:train_idx].copy()
        test_data = working_df.iloc[train_idx:].copy()

        # Remove outliers from training only
        lower, upper, Q1, Q3, IQR_val = compute_iqr_bounds(train_data[TARGET_COL])
        train_mask = (train_data[TARGET_COL] >= lower) & (train_data[TARGET_COL] <= upper)
        train_clean = train_data[train_mask].copy()

        bounds = {
            "Q1": Q1, "Q3": Q3, "IQR": IQR_val,
            "lower": lower, "upper": upper,
            "n_outliers_removed_train": (~train_mask).sum(),
        }
        result["outlier_bounds"] = bounds

        # Recombine: cleaned train + full test
        working_df = pd.concat([train_clean, test_data], ignore_index=True)
        train_idx = len(train_clean)
        result["train_idx"] = train_idx

    elif remove_outliers:
        working_df, bounds, outlier_mask = remove_outliers_train_only(
            working_df, train_idx
        )
        result["outlier_bounds"] = bounds
        result["outlier_mask"] = outlier_mask
        # Recalculate train_idx after removal
        # Count how many training rows survived
        train_dates = df.iloc[:train_idx][DATE_COL]
        result["train_idx"] = working_df[
            working_df[DATE_COL].isin(train_dates)
        ].shape[0]
        train_idx = result["train_idx"]

    # Raw train/test split
    prices = working_df[TARGET_COL].values
    dates = working_df[DATE_COL].values

    result["train_prices"] = prices[:train_idx]
    result["test_prices"] = prices[train_idx:]
    result["train_dates"] = dates[:train_idx]
    result["test_dates"] = dates[train_idx:]

    if not for_dl:
        return result

    # Step 2: Causal rolling mean - applied AFTER split conceptually
    # We apply to the full working series but it's causal (backward-looking only)
    smoothed = causal_rolling_mean(
        pd.Series(prices), window=window_size
    ).values.reshape(-1, 1)

    # Step 3: Scaler - fit on TRAINING data only
    train_smoothed = smoothed[:train_idx]
    test_smoothed = smoothed[train_idx:]

    scaler = fit_scaler_train_only(train_smoothed)
    train_scaled = scaler.transform(train_smoothed)
    test_scaled = scaler.transform(test_smoothed)

    result["scaler"] = scaler
    result["train_scaled"] = train_scaled
    result["test_scaled"] = test_scaled

    # Step 4: Sliding window sequences
    # Gracefully handle case where test set is too small for the window
    effective_time_step = dl_time_step
    if len(test_scaled) < dl_time_step + 1:
        effective_time_step = max(10, len(test_scaled) // 3)
        result["adjusted_time_step"] = effective_time_step
        print(f"    Test set too small ({len(test_scaled)} pts) for "
              f"window={dl_time_step}. Reduced to {effective_time_step}.")

    X_train, y_train = create_sequences(train_scaled.flatten(), effective_time_step)
    X_test, y_test = create_sequences(test_scaled.flatten(), effective_time_step)

    # Check if we have enough data
    if len(X_test) == 0 or len(X_train) == 0:
        result["status"] = "SKIPPED"
        result["skip_reason"] = (
            f"Test set ({len(test_scaled)} pts) < window_size+1 "
            f"({effective_time_step+1}). Not enough data for sequences."
        )
        return result

    # Reshape for RNN: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Step 5: Create validation split from TRAINING data
    from config import DL_VAL_FRACTION
    val_size = int(len(X_train) * DL_VAL_FRACTION)
    result["X_val"] = X_train[-val_size:]
    result["y_val"] = y_train[-val_size:]
    result["X_train"] = X_train[:-val_size]
    result["y_train"] = y_train[:-val_size]
    result["X_test"] = X_test
    result["y_test"] = y_test

    return result


# =====================================================================
# MAIN - Run pipeline and save datasets + analyses
# =====================================================================
def main():
    set_plot_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(NVDA_CSV, parse_dates=[DATE_COL])
    df = df[[DATE_COL, TARGET_COL]].dropna().sort_values(DATE_COL).reset_index(drop=True)
    print(f"Loaded NVDA data: {len(df)} records")

    # -- Stationarity Tests --------------------------------------------
    print("\nRunning stationarity tests...")
    stationarity_results = []

    # Raw prices
    stationarity_results.append(
        run_stationarity_tests(df[TARGET_COL], "Raw Close Price")
    )

    # Log returns
    df_returns = compute_log_returns(df)
    stationarity_results.append(
        run_stationarity_tests(df_returns["log_return"], "Log Returns")
    )

    # Differenced prices
    diff_prices = df[TARGET_COL].diff().dropna()
    stationarity_results.append(
        run_stationarity_tests(diff_prices, "Differenced Prices")
    )

    stat_df = pd.DataFrame(stationarity_results)
    stat_df.to_csv(os.path.join(RESULTS_DIR, "stationarity_tests.csv"), index=False)
    print(stat_df.to_string(index=False))

    # -- IQR Outlier Analysis Per Split --------------------------------
    print("\nComputing IQR bounds per split...")
    outlier_summary = []
    for ratio in SPLIT_RATIOS:
        train_idx = int(len(df) * ratio)
        _, bounds, mask = remove_outliers_train_only(df, train_idx)
        label = f"{int(ratio*100)}:{100 - int(ratio*100)}"
        bounds["split"] = label
        outlier_summary.append(bounds)
        print(f"  Split {label}: {bounds['n_outliers']} outliers "
              f"({bounds['pct_outliers']:.2f}%) | "
              f"bounds=[{bounds['lower']:.2f}, {bounds['upper']:.2f}]")

    pd.DataFrame(outlier_summary).to_csv(
        os.path.join(RESULTS_DIR, "iqr_outlier_summary.csv"), index=False
    )

    # -- Log-Return Outlier Analysis -----------------------------------
    print("\nLog-return outlier analysis...")
    df_ret = compute_log_returns(df)
    return_outlier_summary = []
    for ratio in SPLIT_RATIOS:
        train_idx = int(len(df_ret) * ratio)
        mask, bounds = iqr_outliers_on_returns(df_ret, train_idx)
        label = f"{int(ratio*100)}:{100 - int(ratio*100)}"
        bounds["split"] = label
        return_outlier_summary.append(bounds)
        print(f"  Split {label}: {bounds['n_outliers']} return outliers "
              f"({bounds['pct_outliers']:.2f}%)")

    pd.DataFrame(return_outlier_summary).to_csv(
        os.path.join(RESULTS_DIR, "log_return_outlier_summary.csv"), index=False
    )

    # -- Structural Break Test -----------------------------------------
    print("\nStructural break detection (BinSeg, n_bkps=5)...")
    try:
        prices_array = df[TARGET_COL].values
        breakpoints = run_structural_break_test(prices_array, n_bkps=5)
        bp_dates = [df.iloc[min(bp, len(df)-1)][DATE_COL] for bp in breakpoints[:-1]]
        print(f"  Detected {len(bp_dates)} breakpoints at indices: {breakpoints[:-1]}")
        print(f"  Corresponding dates: {bp_dates}")

        bp_df = pd.DataFrame({
            "breakpoint_index": breakpoints[:-1],
            "breakpoint_date": bp_dates,
        })
        bp_df.to_csv(
            os.path.join(RESULTS_DIR, "structural_breakpoints.csv"), index=False
        )

        # Plot structural breaks
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df[DATE_COL], df[TARGET_COL],
                color=COLORS["train"], label="NVDA Close Price", alpha=0.8)
        bp_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#2ecc71", "#3498db",
                     "#f39c12", "#1abc9c"]  # distinct colors for each break
        for i, bp_date in enumerate(bp_dates):
            color = bp_colors[i % len(bp_colors)]
            ax.axvline(x=bp_date, color=color, linestyle="--", linewidth=2,
                       alpha=0.8, label=f"Break: {bp_date.strftime('%Y-%m-%d')}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        ax.set_title("NVIDIA Stock Price - Structural Breakpoints (BinSeg)")
        ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "structural_breaks.png"), dpi=300)
        plt.close()
        print("  Structural break plot saved.")
    except ImportError:
        print("Install 'ruptures' to run structural break tests: pip install ruptures")
    except Exception as e:
        print(f"Structural break test failed: {e}")

    # -- Naive/Persistence Baseline ------------------------------------
    print("\nComputing naive/persistence baseline...")
    baseline_results = []
    for ratio in SPLIT_RATIOS:
        for outlier_mode in ["with_outliers", "without_outliers", "clean_train_full_test"]:
            train_idx = int(len(df) * ratio)

            if outlier_mode == "without_outliers":
                working_df, _, _ = remove_outliers_train_only(df, train_idx)
                train_dates_set = set(df.iloc[:train_idx][DATE_COL])
                new_train_idx = working_df[
                    working_df[DATE_COL].isin(train_dates_set)
                ].shape[0]
                test_prices = working_df[TARGET_COL].values[new_train_idx:]
            elif outlier_mode == "clean_train_full_test":
                # Test set is the FULL original test set (no outlier removal)
                test_prices = df[TARGET_COL].values[train_idx:]
            else:
                test_prices = df[TARGET_COL].values[train_idx:]

            if len(test_prices) < 2:
                continue

            y_true = test_prices[1:]
            y_pred = persistence_baseline(test_prices)
            metrics = compute_metrics(y_true, y_pred)
            metrics["split"] = f"{int(ratio*100)}:{100 - int(ratio*100)}"
            metrics["dataset"] = outlier_mode
            baseline_results.append(metrics)

    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv(
        os.path.join(RESULTS_DIR, "persistence_baseline.csv"), index=False
    )
    print(baseline_df.to_string(index=False))

    print(f"\nAll pipeline outputs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
