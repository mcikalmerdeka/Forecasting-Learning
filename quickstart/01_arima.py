"""
=============================================================
01_arima.py  —  ARIMA Forecasting
=============================================================
WHEN TO USE:
  - Univariate time series
  - Data can be made stationary through differencing
  - No strong seasonal pattern (use SARIMA for that)
  - Small to medium datasets

ASSUMPTIONS:
  1. Stationarity (after differencing)
  2. No autocorrelation in residuals
  3. Normally distributed residuals

INSTALL:
  pip install statsmodels pandas matplotlib
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     A simple trend + slight noise (no seasonality)
# ─────────────────────────────────────────────
np.random.seed(42)
n = 120  # 10 years of monthly data
t = np.arange(n)
# Trend + AR(1) noise
noise = np.zeros(n)
for i in range(1, n):
    noise[i] = 0.7 * noise[i - 1] + np.random.normal(0, 1.5)

values = 50 + 0.4 * t + noise   # upward trend + autocorrelated noise

date_range = pd.date_range(start="2014-01-01", periods=n, freq="MS")
series = pd.Series(values, index=date_range, name="value")

print("=" * 55)
print("  ARIMA Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} monthly observations (2014–2023)")
print(series.describe().round(2))

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING & ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n── Step 1: Check Stationarity (ADF Test) ──")

def adf_report(s, label="Series"):
    result = adfuller(s.dropna())
    print(f"\n  [{label}]")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Conclusion    : {'STATIONARY ✓' if result[1] < 0.05 else 'NON-STATIONARY — needs differencing'}")
    return result[1] < 0.05

stationary = adf_report(series, "Original series")

d = 0
s_diff = series.copy()
while not stationary and d < 3:
    d += 1
    s_diff = series.diff(d).dropna()
    stationary = adf_report(s_diff, f"After {d}-order differencing")

print(f"\n  ✔ Using d = {d} for ARIMA")

# ─────────────────────────────────────────────
print("\n── Step 2: Identify p and q from ACF / PACF ──")
print("  (Check the plots — PACF cuts off at lag p, ACF cuts off at lag q)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(s_diff, lags=30, ax=axes[0], title="ACF (differenced) — suggests q")
plot_pacf(s_diff, lags=30, ax=axes[1], title="PACF (differenced) — suggests p")
plt.suptitle("Step 2: ACF & PACF to choose p and q", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "arima_acf_pacf.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'arima_acf_pacf.png'}")

# ─────────────────────────────────────────────
# 3.  FIT ARIMA MODEL
# ─────────────────────────────────────────────
print("\n── Step 3: Fit ARIMA(2, 1, 1) ──")
train = series[:-12]
test  = series[-12:]

model  = ARIMA(train, order=(2, d, 1))
result = model.fit()
print(result.summary())

# ─────────────────────────────────────────────
# 4.  RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────────
print("\n── Step 4: Residual Diagnostics ──")
residuals = result.resid

lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\n  Ljung-Box test (p > 0.05 = residuals are white noise ✓):")
print(lb_test.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(residuals)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals over time")
axes[1].hist(residuals, bins=25, edgecolor="black")
axes[1].set_title("Residual distribution")
plt.suptitle("Step 4: Residual Diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "arima_residuals.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'arima_residuals.png'}")

# ─────────────────────────────────────────────
# 5.  FORECAST
# ─────────────────────────────────────────────
print("\n── Step 5: 12-Month Forecast ──")
forecast_obj = result.get_forecast(steps=12)
forecast_mean = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()

mae  = np.mean(np.abs(test.values - forecast_mean.values))
rmse = np.sqrt(np.mean((test.values - forecast_mean.values) ** 2))
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")

plt.figure(figsize=(13, 5))
plt.plot(train[-36:], label="Train (last 3 yrs)", color="steelblue")
plt.plot(test, label="Actual (test)", color="black")
plt.plot(forecast_mean, label="ARIMA Forecast", color="tomato", linestyle="--")
plt.fill_between(forecast_mean.index,
                 conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 alpha=0.25, color="tomato", label="95% CI")
plt.title("ARIMA(2,1,1) — 12-Month Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "arima_forecast.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'arima_forecast.png'}")

print("\n✅ Done. Check the PNG files for plots.")
