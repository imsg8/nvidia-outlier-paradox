"""
NVIDIA Stock Prediction - Linear Regression Model
=================================================
Linear Regression baseline under the same pipeline used for
ARIMA / GRU / LSTM:

  - IQR bounds computed on TRAINING data only (per split),
    reusing data_pipeline.remove_outliers_train_only
  - Strictly CHRONOLOGICAL split (no shuffle)
  - clean_train_full_test mode (train cleaned, test full raw)

For comparison, a legacy variant using a random split is also reported,
which illustrates how a non-chronological split inflates accuracy via
data leakage.

Feature: ordinal time index (date as numeric) -> Close price, a
univariate linear-trend baseline evaluated leakage-free.

Usage:
    python run_lr.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NVDA_CSV, RESULTS_DIR, TARGET_COL, DATE_COL, SPLIT_RATIOS, DATA_DIR
from data_pipeline import remove_outliers_train_only, compute_metrics
from plot_style import set_plot_style, COLORS, add_metric_box


MODES = ["with_outliers", "without_outliers", "clean_train_full_test"]


def _prepare_prices(df, ratio, mode):
    """
    Return (train_dates, train_y, test_dates, test_y) for a given split
    ratio and outlier mode, using training-only IQR bounds (no leakage).
    """
    train_idx = int(len(df) * ratio)

    if mode == "with_outliers":
        work = df
        ti = train_idx
        train = work.iloc[:ti]
        test = work.iloc[ti:]

    elif mode == "without_outliers":
        # Remove outliers from BOTH train and test using TRAIN-only bounds
        work, _, _ = remove_outliers_train_only(df, train_idx, column=TARGET_COL)
        work = work.reset_index(drop=True)
        train_dates_set = set(df.iloc[:train_idx][DATE_COL])
        ti = work[work[DATE_COL].isin(train_dates_set)].shape[0]
        train = work.iloc[:ti]
        test = work.iloc[ti:]

    else:  # clean_train_full_test: clean training only, keep FULL raw test
        train_data = df.iloc[:train_idx]
        from data_pipeline import compute_iqr_bounds
        lower, upper, *_ = compute_iqr_bounds(train_data[TARGET_COL])
        mask = (train_data[TARGET_COL] >= lower) & (train_data[TARGET_COL] <= upper)
        train = train_data[mask]
        test = df.iloc[train_idx:]

    return train, test


def run_lr():
    """Run Linear Regression for all splits and outlier modes."""
    set_plot_style()

    df = pd.read_csv(NVDA_CSV, parse_dates=[DATE_COL])
    df = (df[[DATE_COL, TARGET_COL]].dropna()
          .sort_values(DATE_COL).reset_index(drop=True))
    print(f"Loaded NVDA data: {len(df)} records")

    lr_dir = os.path.join(RESULTS_DIR, "LR")
    os.makedirs(lr_dir, exist_ok=True)

    all_results = []
    # collect plots per mode for the combined paper figures
    plot_cache = {m: [] for m in MODES}

    for mode in MODES:
        for ratio in SPLIT_RATIOS:
            label = f"{int(ratio*100)}:{100 - int(ratio*100)}"
            label_us = f"{int(ratio*100)}_{100 - int(ratio*100)}"
            config_label = f"{mode}/{label}"
            print(f"\n{'='*60}\n  LR | {config_label}\n{'='*60}")

            train, test = _prepare_prices(df, ratio, mode)

            # ordinal time index as the single predictor (numeric date)
            t0 = train[DATE_COL].map(pd.Timestamp.toordinal).min()
            Xtr = (train[DATE_COL].map(pd.Timestamp.toordinal) - t0).values.reshape(-1, 1)
            Xte = (test[DATE_COL].map(pd.Timestamp.toordinal) - t0).values.reshape(-1, 1)
            ytr = train[TARGET_COL].values
            yte = test[TARGET_COL].values

            model = LinearRegression().fit(Xtr, ytr)
            train_pred = model.predict(Xtr)
            test_pred = model.predict(Xte)

            train_metrics = compute_metrics(ytr, train_pred)
            test_metrics = compute_metrics(yte, test_pred)
            print(f"  Train - RMSE={train_metrics['RMSE']:.3f}, "
                  f"MAE={train_metrics['MAE']:.3f}, R2={train_metrics['R2']:.4f}")
            print(f"  Test  - RMSE={test_metrics['RMSE']:.3f}, "
                  f"MAE={test_metrics['MAE']:.3f}, R2={test_metrics['R2']:.4f}")

            config_dir = os.path.join(lr_dir, mode, label_us)
            os.makedirs(config_dir, exist_ok=True)
            pd.DataFrame({
                "Date": test[DATE_COL].values,
                "Actual": yte,
                "Predicted": test_pred,
                "Error": yte - test_pred,
            }).to_csv(os.path.join(config_dir, "predictions.csv"), index=False)

            # Individual prediction plot
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(train[DATE_COL], ytr, color=COLORS["train"],
                    label="Train", alpha=0.7)
            ax.plot(test[DATE_COL], yte, color=COLORS["test"],
                    label="Test (Actual)", alpha=0.8, linewidth=1.5)
            ax.plot(test[DATE_COL], test_pred, color=COLORS["predicted"],
                    label="Test (Predicted)", linestyle="--", alpha=0.9,
                    linewidth=1.5)
            ax.axvline(x=test[DATE_COL].iloc[0], color="#999999", linestyle=":",
                       alpha=0.6, label=f"Split ({label})")
            ax.set_title(f"Linear Regression - {config_label}", fontsize=18)
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price (USD)")
            ax.legend(loc="upper left", framealpha=0.9)
            add_metric_box(ax, {
                "Test R2": test_metrics["R2"],
                "Test RMSE": test_metrics["RMSE"],
                "Test MAE": test_metrics["MAE"],
            }, loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(config_dir, "predictions_plot.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            plot_cache[mode].append({
                "label": label, "train": train, "test": test,
                "ytr": ytr, "yte": yte, "test_pred": test_pred,
                "metrics": test_metrics,
            })

            all_results.append({
                "model": "LR",
                "dataset": mode,
                "split": label,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })

    # Combined figures (one per outlier mode)
    fig_names = {
        "with_outliers": "linear_regression_with_outliers.png",
        "without_outliers": "linear_regression_without_outliers.png",
    }
    for mode, fname in fig_names.items():
        panels = plot_cache[mode]
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, p in zip(axes, panels):
            ax.plot(p["train"][DATE_COL], p["ytr"], color=COLORS["train"],
                    label="Train", alpha=0.6)
            ax.plot(p["test"][DATE_COL], p["yte"], color=COLORS["test"],
                    label="Actual", alpha=0.85, linewidth=1.4)
            ax.plot(p["test"][DATE_COL], p["test_pred"], color=COLORS["predicted"],
                    label="Predicted", linestyle="--", alpha=0.9, linewidth=1.4)
            ax.set_title(f"Split {p['label']}", fontsize=18)
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price (USD)")
            ax.legend(loc="upper left", fontsize=12)
            add_metric_box(ax, {
                "R2": p["metrics"]["R2"],
                "RMSE": p["metrics"]["RMSE"],
                "MAE": p["metrics"]["MAE"],
            }, loc="lower right")
        suffix = "With Outliers" if mode == "with_outliers" else "Without Outliers"
        fig.suptitle(f"Linear Regression - Prediction Trends ({suffix})",
                     fontsize=22, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(lr_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved combined figure: {fname}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(lr_dir, "lr_summary.csv"), index=False)
    print("\nLR complete. Summary:")
    print(summary_df.to_string(index=False))
    return summary_df


def run_lr_legacy():
    """
    Linear Regression with a random train/test split, reported as a
    leaky baseline for comparison against the chronological model:
      - feature = Date.timestamp()
      - random train_test_split(random_state=42)
      - "without outliers" = NVDA_cleaned.csv (global IQR)
    """
    lr_dir = os.path.join(RESULTS_DIR, "LR")
    os.makedirs(lr_dir, exist_ok=True)

    files = {
        "with_outliers": NVDA_CSV,
        "without_outliers": os.path.join(DATA_DIR, "NVDA_cleaned.csv"),
    }
    rows = []
    for mode, path in files.items():
        d = pd.read_csv(path)
        d[DATE_COL] = pd.to_datetime(d[DATE_COL])
        X = d[DATE_COL].apply(lambda x: x.timestamp()).values.reshape(-1, 1)
        y = d[TARGET_COL].values
        for ratio in SPLIT_RATIOS:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, train_size=ratio, random_state=42
            )
            m = LinearRegression().fit(Xtr, ytr)
            mets = compute_metrics(yte, m.predict(Xte))
            rows.append({
                "model": "LR", "method": "legacy_random_split",
                "dataset": mode,
                "split": f"{int(ratio*100)}:{100 - int(ratio*100)}",
                **{f"test_{k}": v for k, v in mets.items()},
            })
    legacy_df = pd.DataFrame(rows)
    legacy_df.to_csv(os.path.join(lr_dir, "lr_legacy_summary.csv"), index=False)
    print("\nLegacy (random-split) LR reproduced:")
    print(legacy_df.to_string(index=False))
    return legacy_df


def build_comparison(chrono_df, legacy_df):
    """Merge legacy vs chronological LR into one comparison table + figure."""
    lr_dir = os.path.join(RESULTS_DIR, "LR")
    keep = ["dataset", "split", "test_RMSE", "test_MAE", "test_R2"]
    leg = legacy_df[keep].rename(columns={
        "test_RMSE": "legacy_RMSE", "test_MAE": "legacy_MAE",
        "test_R2": "legacy_R2"})
    cor = chrono_df[chrono_df["dataset"].isin(["with_outliers",
          "without_outliers"])][keep].rename(columns={
        "test_RMSE": "chrono_RMSE", "test_MAE": "chrono_MAE",
        "test_R2": "chrono_R2"})
    comp = leg.merge(cor, on=["dataset", "split"])
    comp.to_csv(os.path.join(lr_dir, "lr_comparison.csv"), index=False)
    print("\nLegacy vs chronological comparison:")
    print(comp.to_string(index=False))

    # Comparison figure: RMSE bars (legacy vs chronological)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, mode, title in zip(
        axes, ["with_outliers", "without_outliers"],
        ["With Outliers", "Without Outliers"]):
        sub = comp[comp["dataset"] == mode]
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w/2, sub["legacy_RMSE"], w, color=COLORS["test"],
               label="Legacy (random split, leaky)")
        ax.bar(x + w/2, sub["chrono_RMSE"], w, color=COLORS["train"],
               label="Chronological")
        for i, (lv, cv) in enumerate(zip(sub["legacy_RMSE"], sub["chrono_RMSE"])):
            ax.text(i - w/2, lv, f"{lv:.1f}", ha="center", va="bottom", fontsize=11)
            ax.text(i + w/2, cv, f"{cv:.1f}", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(sub["split"])
        ax.set_title(f"LR Test RMSE - {title}", fontsize=18)
        ax.set_xlabel("Train:Test Split"); ax.set_ylabel("Test RMSE (USD)")
        ax.legend(loc="upper left", fontsize=12)
    fig.suptitle("Linear Regression: Effect of Data Leakage on Reported Accuracy",
                 fontsize=22, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(lr_dir, "lr_leakage_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved comparison figure: lr_leakage_comparison.png")
    return comp


if __name__ == "__main__":
    chrono = run_lr()
    legacy = run_lr_legacy()
    build_comparison(chrono, legacy)
