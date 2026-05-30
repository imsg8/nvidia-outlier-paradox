import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

splits = ["70_30", "80_20", "85_15"]
split_titles = ["70:30 Split", "80:20 Split", "85:15 Split"]

# 1. ARIMA plots (1x3)
for mode in ["with_outliers", "without_outliers"]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, split in enumerate(splits):
        df = pd.read_csv(os.path.join(BASE, "ARIMA", mode, split, "predictions.csv"))
        axes[i].plot(df["Date"], df["Actual"], label="Actual", color='blue', alpha=0.6)
        axes[i].plot(df["Date"], df["Predicted"], label="Predicted", color='red', alpha=0.8, linestyle='--')
        axes[i].set_title(f"ARIMA {split_titles[i]}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Price")
        axes[i].legend()
        axes[i].set_xticks([]) # Hide x ticks for cleaner look
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"arima_6in1_{mode}.png"))
    plt.close()

# 2. DL plots (2x3)
for model in ["GRU", "LSTM"]:
    for mode in ["with_outliers", "without_outliers"]:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for i, split in enumerate(splits):
            # Find run_0_seed_7 or any run folder
            run_dirs = glob.glob(os.path.join(BASE, model, mode, split, "run_*"))
            run_dirs.sort()
            if not run_dirs:
                continue
            run_dir = run_dirs[0] # Use first run for plotting
            
            # Top row: Predictions
            df_pred = pd.read_csv(os.path.join(run_dir, "test_predictions.csv"))
            axes[0, i].plot(df_pred["actual"], label="Actual", color='blue', alpha=0.6)
            axes[0, i].plot(df_pred["predicted"], label="Predicted", color='red', alpha=0.8, linestyle='--')
            axes[0, i].set_title(f"{model} Predictions ({split_titles[i]})")
            axes[0, i].set_xlabel("Time")
            axes[0, i].set_ylabel("Price")
            axes[0, i].legend()
            
            # Bottom row: Loss
            df_hist = pd.read_csv(os.path.join(run_dir, "training_history.csv"))
            axes[1, i].plot(df_hist.index, df_hist["loss"], label="Train Loss", color='blue')
            axes[1, i].plot(df_hist.index, df_hist["val_loss"], label="Val Loss", color='orange')
            axes[1, i].set_title(f"{model} Loss ({split_titles[i]})")
            axes[1, i].set_xlabel("Epoch")
            axes[1, i].set_ylabel("Huber Loss")
            axes[1, i].legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{model.lower()}_{mode}_6in1.png"))
        plt.close()

print("Plot generation complete.")
