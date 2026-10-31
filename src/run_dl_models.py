"""
NVIDIA Stock Prediction - DL Models (GRU + LSTM)
================================================
  - IQR on training data only
  - Causal rolling mean
  - Scaler fit on training data only
  - Validation split from training data (not test set)
  - Multiple runs with different seeds for statistical validation
  - clean_train_full_test mode (train on cleaned, test on full)
  - Plot styling with legends, annotations, metric boxes
  - Graceful handling of small test sets after outlier removal

Usage:
    python run_dl_models.py [--model gru|lstm|both] [--seeds 5]
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NVDA_CSV, RESULTS_DIR, TARGET_COL, DATE_COL, SPLIT_RATIOS,
    DL_WINDOW_SIZE, DL_UNITS, DL_NUM_LAYERS, DL_DROPOUT, DL_BATCH_SIZE,
    DL_MAX_EPOCHS, DL_PATIENCE, DL_LOSS, DL_OPTIMIZER,
    DL_SMOOTHING_WINDOW, DL_SEEDS, DL_VAL_FRACTION,
)
from data_pipeline import prepare_split, compute_metrics
from plot_style import set_plot_style, COLORS, add_metric_box


# =====================================================================
# Model Builder
# =====================================================================
def build_model(model_type, time_step, units=DL_UNITS, n_layers=DL_NUM_LAYERS,
                dropout=DL_DROPOUT, loss=DL_LOSS, optimizer=DL_OPTIMIZER):
    """Build GRU or LSTM model."""
    from keras.models import Sequential
    from keras.layers import Dense, GRU, LSTM, Dropout

    LayerClass = GRU if model_type.upper() == "GRU" else LSTM

    model = Sequential()
    for i in range(n_layers):
        return_seq = i < (n_layers - 1)
        if i == 0:
            model.add(LayerClass(units, return_sequences=return_seq,
                                 input_shape=(time_step, 1)))
        else:
            model.add(LayerClass(units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    return model


# =====================================================================
# Single Training Run
# =====================================================================
def train_single_run(model_type, split_data, seed, run_dir):
    """Train one model run and save results."""
    import tensorflow as tf
    from keras.callbacks import EarlyStopping

    # Set seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Use adjusted time step if test set was too small
    time_step = split_data.get("adjusted_time_step", DL_WINDOW_SIZE)

    model = build_model(model_type, time_step)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=DL_PATIENCE, restore_best_weights=True
    )

    history = model.fit(
        split_data["X_train"], split_data["y_train"],
        validation_data=(split_data["X_val"], split_data["y_val"]),
        epochs=DL_MAX_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop],
    )

    # Predictions
    train_pred_scaled = model.predict(split_data["X_train"], verbose=0)
    val_pred_scaled = model.predict(split_data["X_val"], verbose=0)
    test_pred_scaled = model.predict(split_data["X_test"], verbose=0)

    # Inverse transform
    scaler = split_data["scaler"]
    train_pred = scaler.inverse_transform(train_pred_scaled)
    val_pred = scaler.inverse_transform(val_pred_scaled)
    test_pred = scaler.inverse_transform(test_pred_scaled)
    y_train_actual = scaler.inverse_transform(
        split_data["y_train"].reshape(-1, 1)
    )
    y_val_actual = scaler.inverse_transform(
        split_data["y_val"].reshape(-1, 1)
    )
    y_test_actual = scaler.inverse_transform(
        split_data["y_test"].reshape(-1, 1)
    )

    # Compute metrics
    train_metrics = compute_metrics(y_train_actual, train_pred)
    val_metrics = compute_metrics(y_val_actual, val_pred)
    test_metrics = compute_metrics(y_test_actual, test_pred)

    # Save results
    os.makedirs(run_dir, exist_ok=True)

    # Save model
    model.save(os.path.join(run_dir, "model.h5"))

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

    # Save predictions
    pd.DataFrame({
        "actual": y_test_actual.flatten(),
        "predicted": test_pred.flatten(),
    }).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

    # Save metrics
    metrics_dict = {
        "seed": seed,
        "train_RMSE": train_metrics["RMSE"],
        "train_MAE": train_metrics["MAE"],
        "train_R2": train_metrics["R2"],
        "val_RMSE": val_metrics["RMSE"],
        "val_MAE": val_metrics["MAE"],
        "val_R2": val_metrics["R2"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAE": test_metrics["MAE"],
        "test_R2": test_metrics["R2"],
        "epochs_trained": len(history.history["loss"]),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    return metrics_dict, history, test_pred, y_test_actual


# =====================================================================
# Run All Configurations
# =====================================================================
def run_model(model_type, seeds=None):
    """Run model for all splits x outlier modes x seeds (3-mode strategy)."""
    set_plot_style()

    if seeds is None:
        seeds = DL_SEEDS

    df = pd.read_csv(NVDA_CSV, parse_dates=[DATE_COL])
    df = df[[DATE_COL, TARGET_COL]].dropna().sort_values(DATE_COL).reset_index(drop=True)

    model_dir = os.path.join(RESULTS_DIR, model_type.upper())
    os.makedirs(model_dir, exist_ok=True)

    all_results = []

    # 3-mode outlier strategy
    for outlier_mode in ["with_outliers", "without_outliers", "clean_train_full_test"]:
        remove = outlier_mode == "without_outliers"
        clean_train = outlier_mode == "clean_train_full_test"

        for ratio in SPLIT_RATIOS:
            label = f"{int(ratio*100)}_{100 - int(ratio*100)}"
            config_label = f"{outlier_mode}/{label}"
            print(f"\n{'='*60}")
            print(f"  {model_type.upper()} | {config_label}")
            print(f"{'='*60}")

            # Prepare data
            split_data = prepare_split(
                df, ratio, remove_outliers=remove,
                clean_train_full_test=clean_train,
                for_dl=True, dl_time_step=DL_WINDOW_SIZE,
            )

            # Check if this config was skipped due to insufficient data
            if split_data.get("status") == "SKIPPED":
                skip_reason = split_data.get("skip_reason", "Unknown")
                print(f"  SKIPPED: {skip_reason}")
                summary = {
                    "model": model_type.upper(),
                    "dataset": outlier_mode,
                    "split": split_data["split_label"],
                    "status": "SKIPPED",
                    "skip_reason": skip_reason,
                }
                all_results.append(summary)

                # Save skip info
                config_dir = os.path.join(model_dir, outlier_mode, label)
                os.makedirs(config_dir, exist_ok=True)
                with open(os.path.join(config_dir, "skip_reason.txt"), "w") as f:
                    f.write(skip_reason)
                continue

            run_metrics = []
            for i, seed in enumerate(seeds):
                run_dir = os.path.join(
                    model_dir, outlier_mode, label, f"run_{i}_seed_{seed}"
                )
                print(f"  Run {i+1}/{len(seeds)} (seed={seed})...", end=" ")

                metrics, history, test_pred, y_test_actual = train_single_run(
                    model_type, split_data, seed, run_dir
                )
                run_metrics.append(metrics)
                print(f"Test RMSE={metrics['test_RMSE']:.3f}, "
                      f"MAE={metrics['test_MAE']:.3f}, "
                      f"R2={metrics['test_R2']:.3f}")

            # Aggregate multi-run results (mean ± std)
            run_df = pd.DataFrame(run_metrics)
            summary = {
                "model": model_type.upper(),
                "dataset": outlier_mode,
                "split": split_data["split_label"],
                "status": "OK",
            }
            for col in ["test_RMSE", "test_MAE", "test_R2",
                        "train_RMSE", "train_MAE", "train_R2"]:
                summary[f"{col}_mean"] = run_df[col].mean()
                summary[f"{col}_std"] = run_df[col].std()

            all_results.append(summary)

            # Save per-config summary
            config_dir = os.path.join(model_dir, outlier_mode, label)
            run_df.to_csv(
                os.path.join(config_dir, "all_runs_metrics.csv"), index=False
            )

            # ── Plot Best Run (with legends, annotations, styling) ────
            best_idx = run_df["test_RMSE"].idxmin()
            best_run_dir = os.path.join(
                config_dir, f"run_{best_idx}_seed_{seeds[best_idx]}"
            )
            best_hist = pd.read_csv(
                os.path.join(best_run_dir, "training_history.csv")
            )
            best_preds = pd.read_csv(
                os.path.join(best_run_dir, "test_predictions.csv")
            )
            best_metrics = run_df.iloc[best_idx]

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle(
                f"{model_type.upper()} - {config_label}  "
                f"(Best of {len(seeds)} runs, seed={seeds[best_idx]})",
                fontsize=16, fontweight="bold", y=1.02,
            )

            # ── Left: Loss Curve ──────────────────────────────────────
            axes[0].plot(best_hist["loss"],
                         color=COLORS["train"], label="Train Loss", linewidth=1.5)
            axes[0].plot(best_hist["val_loss"],
                         color=COLORS["val"], label="Val Loss", linewidth=1.5)
            axes[0].set_title(f"{model_type.upper()} Loss Curve")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss (Huber)")
            axes[0].legend(loc="upper right", framealpha=0.9)

            # Annotate best epoch
            best_epoch = best_hist["val_loss"].idxmin()
            best_val_loss = best_hist["val_loss"].min()
            axes[0].axvline(x=best_epoch, color="#999999", linestyle=":",
                            alpha=0.5)
            axes[0].annotate(
                f"Best epoch: {best_epoch}\nVal loss: {best_val_loss:.4f}",
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + len(best_hist)*0.1, best_val_loss * 1.5),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#666666"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85),
            )

            add_metric_box(axes[0], {
                "Epochs": len(best_hist),
                "Final Train": best_hist["loss"].iloc[-1],
                "Final Val": best_hist["val_loss"].iloc[-1],
            }, loc="lower left")

            # ── Right: Predictions ────────────────────────────────────
            axes[1].plot(best_preds["actual"],
                         color=COLORS["test"], label="Actual", alpha=0.8,
                         linewidth=1.5)
            axes[1].plot(best_preds["predicted"],
                         color=COLORS["predicted"], label="Predicted", alpha=0.8,
                         linewidth=1.5, linestyle="--")
            axes[1].set_title(f"{model_type.upper()} Predictions")
            axes[1].set_xlabel("Time Step")
            axes[1].set_ylabel("Price (USD)")
            axes[1].legend(loc="upper left", framealpha=0.9)

            # Metric annotation box
            add_metric_box(axes[1], {
                "R2": best_metrics["test_R2"],
                "RMSE": best_metrics["test_RMSE"],
                "MAE": best_metrics["test_MAE"],
            }, loc="lower right")

            plt.tight_layout()
            plt.savefig(
                os.path.join(config_dir, f"{model_type.lower()}_best_run.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close()

            # ── Multi-Run Comparison Plot ─────────────────────────────
            if len(seeds) > 1:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(
                    f"{model_type.upper()} Multi-Run Summary - {config_label}  "
                    f"({len(seeds)} seeds)",
                    fontsize=14, fontweight="bold", y=1.02,
                )

                for idx, (metric, title) in enumerate([
                    ("test_RMSE", "Test RMSE"),
                    ("test_MAE", "Test MAE"),
                    ("test_R2", "Test R2"),
                ]):
                    values = run_df[metric].values
                    axes[idx].bar(range(len(values)), values,
                                  color=COLORS["train"], alpha=0.7,
                                  edgecolor="black", linewidth=0.5)
                    axes[idx].axhline(y=values.mean(), color=COLORS["zero_line"],
                                      linestyle="--", alpha=0.7,
                                      label=f"Mean: {values.mean():.3f}")
                    axes[idx].set_title(title)
                    axes[idx].set_xlabel("Run")
                    axes[idx].set_ylabel(metric.split("_", 1)[1])
                    axes[idx].set_xticks(range(len(values)))
                    axes[idx].set_xticklabels(
                        [f"S{s}" for s in seeds], fontsize=9
                    )
                    axes[idx].legend(loc="best", fontsize=9)

                    # Add std annotation
                    add_metric_box(axes[idx], {
                        "Mean": values.mean(),
                        "Std": values.std(),
                    }, loc="upper right")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(config_dir,
                                 f"{model_type.lower()}_multi_run_summary.png"),
                    dpi=300, bbox_inches="tight",
                )
                plt.close()

    # Save overall summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        os.path.join(model_dir, f"{model_type.lower()}_summary.csv"), index=False
    )
    print(f"\n{model_type.upper()} complete. Summary saved to {model_dir}/")
    print(summary_df.to_string(index=False))

    return summary_df


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DL models for NVIDIA stock prediction"
    )
    parser.add_argument(
        "--model", choices=["gru", "lstm", "both"], default="both",
        help="Which model to run"
    )
    parser.add_argument(
        "--seeds", type=int, default=len(DL_SEEDS),
        help=f"Number of seeds to use (max {len(DL_SEEDS)})"
    )
    args = parser.parse_args()

    seeds = DL_SEEDS[:args.seeds]

    if args.model in ("gru", "both"):
        run_model("GRU", seeds=seeds)
    if args.model in ("lstm", "both"):
        run_model("LSTM", seeds=seeds)
