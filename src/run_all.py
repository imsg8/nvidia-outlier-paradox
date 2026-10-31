"""
NVIDIA Stock Prediction - Master Runner
=======================================
Runs the complete pipeline in order:
  1. Data pipeline (stationarity tests, outlier analysis, structural breaks, baseline)
  2. ARIMA (all splits x outlier modes, with full parameter logging)
  3. DL models (GRU + LSTM, all splits x outlier modes x 5 seeds)

Usage:
    python run_all.py                  # Run everything
    python run_all.py --skip-dl        # Run pipeline + ARIMA only (fast)
    python run_all.py --seeds 1        # Only 1 seed per DL config (quick test)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip DL models (only pipeline + ARIMA)")
    parser.add_argument("--skip-arima", action="store_true",
                        help="Skip ARIMA (only pipeline + DL)")
    parser.add_argument("--skip-lr", action="store_true",
                        help="Skip Linear Regression baseline (chronological + legacy)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds for DL multi-run (default: 5)")
    parser.add_argument("--model", choices=["gru", "lstm", "both"], default="both",
                        help="Which DL model to run")
    args = parser.parse_args()

    # ── Step 1: Data Pipeline ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: DATA PIPELINE (stationarity, outliers, breaks, baseline)")
    print("=" * 70)
    from data_pipeline import main as pipeline_main
    pipeline_main()

    # -- Step 2: ARIMA ------------------------------------------------
    if not args.skip_arima:
        print("\n" + "=" * 70)
        print("  STEP 2: ARIMA (all splits x outlier modes)")
        print("=" * 70)
        from run_arima import run_arima
        run_arima()

    # -- Step 3: Linear Regression (chronological + legacy + comparison) --
    if not args.skip_lr:
        print("\n" + "=" * 70)
        print("  STEP 3: LINEAR REGRESSION (chronological, legacy, comparison)")
        print("=" * 70)
        from run_lr import run_lr, run_lr_legacy, build_comparison
        chrono = run_lr()
        legacy = run_lr_legacy()
        build_comparison(chrono, legacy)

    # -- Step 4: DL Models --------------------------------------------
    if not args.skip_dl:
        print("\n" + "=" * 70)
        print("  STEP 4: DL MODELS (GRU + LSTM)")
        print("=" * 70)
        from run_dl_models import run_model
        from config import DL_SEEDS

        seeds = DL_SEEDS[:args.seeds]

        if args.model in ("gru", "both"):
            run_model("GRU", seeds=seeds)
        if args.model in ("lstm", "both"):
            run_model("LSTM", seeds=seeds)

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"  Results saved in: results/")


if __name__ == "__main__":
    main()
