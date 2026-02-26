"""
=============================================================
04_ets_holtwinters.py  —  Exponential Smoothing / Holt-Winters
=============================================================
WHEN TO USE:
  - Simple to moderate trend and seasonal patterns
  - Reliable and fast baseline method
  - Small to medium datasets
  - Retail inventory, demand forecasting

ASSUMPTIONS:
  1. Error structure is additive or multiplicative
  2. Trend is linear (additive) or exponential (multiplicative)
  3. Seasonality is stable in shape
  4. For multiplicative: all values must be positive

INSTALL:
  pip install statsmodels pandas matplotlib
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     Monthly retail sales: upward trend + yearly season
# ─────────────────────────────────────────────
np.random.seed(3)
n = 84   # 7 years of monthly data
t = np.arange(n)

trend    = 500 + 3 * t
seasonal = 80 * np.sin(2 * np.pi * t / 12) + 30 * np.sin(4 * np.pi * t / 12)
noise    = np.random.normal(0, 8, n)

values = trend + seasonal + noise
date_range = pd.date_range(start="2017-01-01", periods=n, freq="MS")
series = pd.Series(values, index=date_range, name="retail_sales")

print("=" * 55)
print("  ETS / Holt-Winters Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} monthly retail sales observations")

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING & ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n── Step 1: Visualise Raw Data ──")
plt.figure(figsize=(13, 4))
plt.plot(series, color="steelblue")
plt.title("Step 1: Monthly Retail Sales")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(output_dir / "ets_raw.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'ets_raw.png'}")

print("\n── Step 2: Choose Model Type ──")
print("  Look at the seasonal amplitude:")
cv_seasonal = series.std() / series.mean()
print(f"  Coefficient of variation: {cv_seasonal:.3f}")
if cv_seasonal < 0.3:
    model_type = "additive"
    print("  → Stable variance  →  Use ADDITIVE error & seasonality")
else:
    model_type = "multiplicative"
    print("  → Growing variance →  Use MULTIPLICATIVE error & seasonality")

print(f"\n  Positive values only: {(series > 0).all()} (required for multiplicative)")
print(f"  Contains NaN: {series.isna().any()}")

print("\n── Step 3: Define seasonality period ──")
period = 12
print(f"  Seasonal period m = {period} (monthly data, yearly pattern)")

# ─────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
test_n = 12
train = series[:-test_n]
test  = series[-test_n:]

# ─────────────────────────────────────────────
# 4.  FIT THREE VARIANTS
# ─────────────────────────────────────────────
print("\n── Step 4: Fit 3 ETS Variants ──")

models = {
    "Simple ES": ExponentialSmoothing(train, trend=None, seasonal=None),
    "Holt Linear": ExponentialSmoothing(train, trend="add", seasonal=None),
    "Holt-Winters": ExponentialSmoothing(train,
                                          trend="add",
                                          seasonal=model_type,
                                          seasonal_periods=period),
}

results  = {}
forecasts = {}
for name, mdl in models.items():
    fit = mdl.fit(optimized=True)
    fc  = fit.forecast(test_n)
    results[name]  = fit
    forecasts[name] = fc
    mae  = mean_absolute_error(test, fc)
    rmse = np.sqrt(mean_squared_error(test, fc))
    print(f"\n  [{name}]")
    print(f"    Alpha (level) : {fit.params.get('smoothing_level', 'N/A'):.4f}")
    if hasattr(fit, 'params') and 'smoothing_trend' in fit.params:
        print(f"    Beta  (trend) : {fit.params['smoothing_trend']:.4f}")
    if hasattr(fit, 'params') and 'smoothing_seasonal' in fit.params:
        print(f"    Gamma (season): {fit.params['smoothing_seasonal']:.4f}")
    print(f"    MAE  : {mae:.3f}")
    print(f"    RMSE : {rmse:.3f}")

# ─────────────────────────────────────────────
# 5.  PLOT COMPARISON
# ─────────────────────────────────────────────
print("\n── Step 5: Compare All Forecasts ──")
colors = ["tomato", "seagreen", "darkorange"]

plt.figure(figsize=(13, 6))
plt.plot(train[-24:], color="steelblue", label="Train")
plt.plot(test, color="black", linewidth=2, label="Actual")

for (name, fc), color in zip(forecasts.items(), colors):
    plt.plot(fc, linestyle="--", color=color, label=name)

plt.title("ETS Variants — 12-Month Forecast Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "ets_comparison.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'ets_comparison.png'}")

# Best model decomposed forecast
best_model = results["Holt-Winters"]
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
axes[0].plot(series, color="steelblue")
axes[0].set_title("Original Series")
axes[1].plot(best_model.level, color="darkorange")
axes[1].set_title("Level (smoothed)")
axes[2].plot(best_model.trend, color="seagreen")
axes[2].set_title("Trend Component")
plt.suptitle("Holt-Winters Decomposed Components", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "ets_components.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'ets_components.png'}")

print("\n✅ Done.")
