"""
NVIDIA Stock Prediction - Pipeline Configuration
================================================
Central configuration for all experiments.

Auto-detects Kaggle environment and adjusts paths accordingly.
Works both locally and on Kaggle without modification.
"""
import os

# ─── Environment Detection ───────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")

# ─── Paths ────────────────────────────────────────────────────────────
if IS_KAGGLE:
    DATA_DIR = "/kaggle/input/datasets/shivanggulati/nvda-stocks"
    RESULTS_DIR = "/kaggle/working/results"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "stock_data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Data ─────────────────────────────────────────────────────────────
NVDA_CSV = os.path.join(DATA_DIR, "NVDA.csv")
TARGET_COL = "Close"
DATE_COL = "Date"

MAANG_TICKERS = {
    "NVDA": "NVIDIA",
    "META": "Meta",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "NFLX": "Netflix",
    "GOOGL": "Google",
}

# ─── Experiment ───────────────────────────────────────────────────────
SPLIT_RATIOS = [0.70, 0.80, 0.85]

# ─── IQR Outlier Detection ───────────────────────────────────────────
IQR_MULTIPLIER = 1.5

# ─── DL Hyperparameters ──────────────────────────────────────────────
DL_WINDOW_SIZE = 100          # Sliding window (time steps)
DL_UNITS = 100                # Units per recurrent layer
DL_NUM_LAYERS = 3             # Number of recurrent layers
DL_DROPOUT = 0.2              # Dropout between layers
DL_BATCH_SIZE = 256
DL_MAX_EPOCHS = 250
DL_PATIENCE = 25              # Early stopping patience
DL_LOSS = "huber"
DL_OPTIMIZER = "adam"
DL_SMOOTHING_WINDOW = 5       # Rolling mean window
DL_SEEDS = [7, 42, 123, 456, 789]  # Multi-run seeds
DL_VAL_FRACTION = 0.1         # Fraction of TRAINING data for validation

# ─── ARIMA ────────────────────────────────────────────────────────────
ARIMA_P_RANGE = range(0, 4)
ARIMA_D_RANGE = range(0, 3)
ARIMA_Q_RANGE = range(0, 4)
