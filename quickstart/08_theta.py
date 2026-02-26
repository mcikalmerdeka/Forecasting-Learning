"""
=============================================================
08_theta.py  —  Theta Method Forecasting
=============================================================
WHEN TO USE:
  - Simple, robust competition baseline (won M3 competition!)
  - Small to medium datasets
  - When you need something better than Naive but simpler than ARIMA
  - Monthly or quarterly data with moderate seasonality

HOW IT WORKS:
  1. Seasonally adjust the series
  2. Create two Theta lines (θ=0 and θ=2)
     - θ=0: amplifies the long-run trend (removes noise)
     - θ=2: applies Simple Exponential Smoothing (local trend)
  3. Combine both: forecast = (forecast_θ0 + forecast_θ2) / 2
  4. Re-add seasonality

ASSUMPTIONS:
  1. Seasonality is stable (seasonal indices don't change much over time)
  2. Trend is approximately linear
  3. No sudden structural breaks

INSTALL:
  pip install statsmodels pandas matplotlib scikit-learn
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     Monthly data: trend + seasonal + small noise
# ─────────────────────────────────────────────
np.random.seed(8)
n = 96   # 8 years of monthly data
t = np.arange(n)

trend    = 200 + 2 * t
seasonal = 40 * np.sin(2 * np.pi * t / 12)
noise    = np.random.normal(0, 5, n)
values   = trend + seasonal + noise

dates  = pd.date_range(start="2016-01-01", periods=n, freq="MS")
series = pd.Series(values, index=dates, name="monthly_value")

print("=" * 55)
print("  Theta Method Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} monthly observations")
print(series.describe().round(2))

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING
# ─────────────────────────────────────────────
print("\n── Step 1: Seasonal Decomposition (additive) ──")
period = 12
decomp = seasonal_decompose(series, model="additive", period=period)

seasonal_indices = decomp.seasonal[:period].values
print(f"  Seasonal indices (12 months): {np.round(seasonal_indices, 2)}")

# Remove seasonality from training data
deseasoned = series - decomp.seasonal
print(f"  Seasonality removed. De-seasoned range: [{deseasoned.min():.1f}, {deseasoned.max():.1f}]")

plt.figure(figsize=(13, 6))
plt.subplot(2, 1, 1)
plt.plot(series, label="Original", color="steelblue")
plt.title("Original Series")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(deseasoned, label="De-seasoned", color="darkorange")
plt.title("De-seasoned Series")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "theta_deseasoned.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'theta_deseasoned.png'}")

# ─────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
h = 12  # forecast horizon
train_raw = series[:-h]
test_raw  = series[-h:]
train_des = deseasoned[:-h]

# ─────────────────────────────────────────────
# 4.  THETA METHOD IMPLEMENTATION
# ─────────────────────────────────────────────
print("\n── Step 2: Compute Theta Lines ──")

def theta_forecast(train_deseasoned, horizon, seasonal_idx):
    """
    Standard Theta method:
      - Theta line 0 (θ=0): linear regression on de-seasoned data → long-term trend
      - Theta line 2 (θ=2): Simple Exponential Smoothing → short-term level
      - Combine: (forecast_θ0 + forecast_θ2) / 2
      - Re-add seasonality
    """
    n = len(train_deseasoned)
    
    # ── Theta line 0: linear trend (OLS on de-seasoned data)
    x = np.arange(n)
    slope, intercept, _, _, _ = linregress(x, train_deseasoned.values)
    future_x = np.arange(n, n + horizon)
    forecast_t0 = intercept + slope * future_x

    print(f"  θ=0 linear trend: intercept={intercept:.2f}, slope={slope:.4f}")
    
    # ── Theta line 2: SES on de-seasoned data
    ses_model = SimpleExpSmoothing(train_deseasoned, initialization_method="estimated")
    ses_fit   = ses_model.fit(optimized=True)
    forecast_t2 = ses_fit.forecast(horizon).values

    alpha = ses_fit.params["smoothing_level"]
    print(f"  θ=2 SES alpha (smoothing level): {alpha:.4f}")
    
    # ── Combine
    combined = (forecast_t0 + forecast_t2) / 2
    print(f"  Combined (average of θ=0 and θ=2)")
    
    # ── Re-add seasonality
    seasonal_additions = np.tile(seasonal_idx, int(np.ceil(horizon / len(seasonal_idx))))[:horizon]
    final_forecast = combined + seasonal_additions
    
    return final_forecast, forecast_t0, forecast_t2, combined

forecast, fc_t0, fc_t2, fc_combined = theta_forecast(
    train_des, h, seasonal_indices
)

# ─────────────────────────────────────────────
# 5.  EVALUATE
# ─────────────────────────────────────────────
print("\n── Step 3: Evaluate ──")
forecast_series = pd.Series(forecast, index=test_raw.index)
mae  = mean_absolute_error(test_raw, forecast_series)
rmse = np.sqrt(mean_squared_error(test_raw, forecast_series))
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")

# Naive benchmark (last observed value repeated)
naive_fc = np.repeat(train_raw.iloc[-1], h)
mae_naive  = mean_absolute_error(test_raw, naive_fc)
rmse_naive = np.sqrt(mean_squared_error(test_raw, naive_fc))
print(f"\n  Naive MAE  : {mae_naive:.3f}  (baseline)")
print(f"  Theta MAE  : {mae:.3f}  ({'better ✓' if mae < mae_naive else 'worse'})")

# ─────────────────────────────────────────────
# 6.  PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

# De-seasoned component breakdown
axes[0].plot(train_des, color="steelblue", label="De-seasoned train")
future_index = test_raw.index
x_all = np.arange(len(train_des) + h)
slope_val, intercept_val, _, _, _ = linregress(np.arange(len(train_des)), train_des.values)
trend_all = intercept_val + slope_val * x_all
axes[0].plot(future_index, fc_t0, color="seagreen", linestyle="--", label="θ=0 (Trend)")
axes[0].plot(future_index, fc_t2, color="tomato", linestyle="--", label="θ=2 (SES)")
axes[0].plot(future_index, fc_combined, color="darkorchid", linestyle="-.", linewidth=2, label="Combined (de-seasoned)")
axes[0].set_title("Theta Lines — De-Seasoned Components")
axes[0].legend()

# Final forecast with original series
axes[1].plot(train_raw[-24:], color="steelblue", label="Train")
axes[1].plot(test_raw, color="black", linewidth=2, label="Actual")
axes[1].plot(forecast_series, color="darkorange", linestyle="--", linewidth=2, label="Theta Forecast")
axes[1].set_title("Theta Method — Final Forecast (re-seasonalized)")
axes[1].legend()

plt.suptitle("Theta Method Forecasting", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "theta_forecast.png", dpi=120)
plt.close()
print(f"\n  Saved: {output_dir / 'theta_forecast.png'}")

# ─────────────────────────────────────────────
# Note: statsmodels also has a built-in Theta implementation
# ─────────────────────────────────────────────
print("\n── Alternative: Using statsmodels built-in Theta ──")
try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    theta_sm = ThetaModel(train_raw, period=12)
    theta_res = theta_sm.fit()
    fc_sm = theta_res.forecast(h)
    mae_sm  = mean_absolute_error(test_raw, fc_sm)
    rmse_sm = np.sqrt(mean_squared_error(test_raw, fc_sm))
    print(f"  statsmodels ThetaModel — MAE: {mae_sm:.3f}  RMSE: {rmse_sm:.3f}")
except Exception as e:
    print(f"  (ThetaModel not available in this statsmodels version: {e})")

print("\n✅ Done.")
