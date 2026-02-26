"""
=============================================================
02_sarima.py  —  SARIMA Forecasting
=============================================================
WHEN TO USE:
  - Univariate series WITH a clear seasonal pattern
  - Retail sales, energy, tourism, monthly/quarterly data
  - When ARIMA misses repeating cycles

ASSUMPTIONS:
  1. Stationarity (both seasonal and non-seasonal)
  2. Seasonality is fixed-period (e.g., every 12 months)
  3. No autocorrelation in residuals

INSTALL:
  pip install statsmodels pandas matplotlib
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     Trend + yearly seasonality + small noise
# ─────────────────────────────────────────────
np.random.seed(0)
n = 120
t = np.arange(n)

trend    = 100 + 0.5 * t
seasonal = 15 * np.sin(2 * np.pi * t / 12)   # 12-month cycle
noise    = np.random.normal(0, 2, n)

values = trend + seasonal + noise
date_range = pd.date_range(start="2014-01-01", periods=n, freq="MS")
series = pd.Series(values, index=date_range, name="sales")

print("=" * 55)
print("  SARIMA Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} monthly observations with yearly seasonality")

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING & ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n── Step 1: Visualise & Decompose the Series ──")

decomp = seasonal_decompose(series, model="additive", period=12)
fig, axes = plt.subplots(4, 1, figsize=(13, 10))
for ax, comp, lbl in zip(axes,
                          [series, decomp.trend, decomp.seasonal, decomp.resid],
                          ["Observed", "Trend", "Seasonal", "Residual"]):
    ax.plot(comp, color="steelblue")
    ax.set_title(lbl)
plt.suptitle("Step 1: Seasonal Decomposition", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "sarima_decompose.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'sarima_decompose.png'}")
print("  ✔ Seasonal period m = 12 confirmed visually")

print("\n── Step 2: Check Stationarity ──")

def adf_report(s, label=""):
    r = adfuller(s.dropna())
    stat = "STATIONARY ✓" if r[1] < 0.05 else "NON-STATIONARY"
    print(f"  [{label}]  p = {r[1]:.4f}  →  {stat}")
    return r[1] < 0.05

ok = adf_report(series, "Original")
if not ok:
    adf_report(series.diff(1).dropna(), "1st difference")
    adf_report(series.diff(1).diff(12).dropna(), "Seasonal + 1st diff")
    print("  ✔ Use d=1, D=1 (seasonal differencing with m=12)")
else:
    print("  ✔ Series already stationary, d=0, D=0")

print("\n── Step 3: ACF/PACF on differenced series ──")
diff_series = series.diff(1).diff(12).dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff_series, lags=40, ax=axes[0], title="ACF (differenced) — q, Q")
plot_pacf(diff_series, lags=40, ax=axes[1], title="PACF (differenced) — p, P")
plt.suptitle("Step 3: ACF & PACF to choose seasonal orders", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "sarima_acf_pacf.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'sarima_acf_pacf.png'}")
print("  Non-seasonal spikes at low lags → p=1 or 2, q=1")
print("  Seasonal spikes at lag 12 → P=1, Q=1")

# ─────────────────────────────────────────────
# 3.  FIT SARIMA MODEL
# ─────────────────────────────────────────────
print("\n── Step 4: Fit SARIMA(1,1,1)(1,1,1)[12] ──")
train = series[:-12]
test  = series[-12:]

model  = SARIMAX(train,
                 order=(1, 1, 1),
                 seasonal_order=(1, 1, 1, 12),
                 enforce_stationarity=False,
                 enforce_invertibility=False)
result = model.fit(disp=False)
print(result.summary())

# ─────────────────────────────────────────────
# 4.  RESIDUAL CHECK
# ─────────────────────────────────────────────
result.plot_diagnostics(figsize=(14, 8))
plt.suptitle("Step 4: SARIMA Residual Diagnostics", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "sarima_diagnostics.png", dpi=120)
plt.close()
print(f"\n  Saved: {output_dir / 'sarima_diagnostics.png'}")
print("  Check: residuals should look like white noise (top-left)")

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
plt.plot(test, label="Actual", color="black")
plt.plot(forecast_mean, label="SARIMA Forecast", color="darkorange", linestyle="--")
plt.fill_between(forecast_mean.index,
                 conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 alpha=0.25, color="darkorange", label="95% CI")
plt.title("SARIMA(1,1,1)(1,1,1)[12] — 12-Month Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "sarima_forecast.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'sarima_forecast.png'}")

print("\n✅ Done.")
