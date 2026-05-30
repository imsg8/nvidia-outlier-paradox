"""
Compute Diebold-Mariano test statistics from saved ARIMA forecast errors
vs persistence baseline forecast errors.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))

def dm_test(e1, e2, h=1):
    """
    Diebold-Mariano test.
    H0: E[d_t] = 0 (equal predictive accuracy)
    e1, e2: forecast errors from model 1 and model 2
    Returns: DM statistic, p-value
    """
    d = np.array(e1)**2 - np.array(e2)**2
    T = len(d)
    d_mean = np.mean(d)
    # Newey-West variance estimate (h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    if h > 1:
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_0 += 2 * (1 - k/h) * gamma_k
    dm_stat = d_mean / np.sqrt(gamma_0 / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

results = []

for mode in ["with_outliers", "without_outliers", "clean_train_full_test"]:
    for split in ["70_30", "80_20", "85_15"]:
        pred_file = os.path.join(BASE, "ARIMA", mode, split, "predictions.csv")
        if not os.path.exists(pred_file):
            continue
        
        df = pd.read_csv(pred_file)
        actual = df["Actual"].values
        arima_pred = df["Predicted"].values
        
        # Persistence baseline: predict tomorrow = today
        # For walk-forward: persistence prediction for day t = actual[t-1]
        # So we compare from index 1 onward
        actual_eval = actual[1:]
        arima_errors = actual_eval - arima_pred[1:]
        persist_errors = actual_eval - actual[:-1]
        
        dm_stat, p_val = dm_test(arima_errors, persist_errors)
        
        results.append({
            "mode": mode,
            "split": split.replace("_", ":"),
            "n_obs": len(actual_eval),
            "arima_rmse": np.sqrt(np.mean(arima_errors**2)),
            "persist_rmse": np.sqrt(np.mean(persist_errors**2)),
            "dm_statistic": dm_stat,
            "p_value": p_val,
            "significant_5pct": "Yes" if p_val < 0.05 else "No",
            "better_model": "ARIMA" if dm_stat < 0 else "Persistence"
        })

dm_df = pd.DataFrame(results)
dm_df.to_csv(os.path.join(BASE, "diebold_mariano_results.csv"), index=False)
print(dm_df.to_string(index=False))
