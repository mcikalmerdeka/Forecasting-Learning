"""
=============================================================
07_var.py  —  Vector AutoRegression (VAR)
=============================================================
WHEN TO USE:
  - Multiple related time series that influence each other
  - Macroeconomics: GDP, inflation, unemployment, interest rates
  - When you want to model INTERDEPENDENCIES between variables
  - All series must be stationary

ASSUMPTIONS:
  1. All series are stationary (ADF test each one)
  2. No cointegration between series (use VECM if cointegrated)
  3. Residuals are white noise (no autocorrelation)
  4. Lag order p is selected via AIC/BIC

INSTALL:
  pip install statsmodels pandas matplotlib
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY MULTIVARIATE DATA
#     Three macroeconomic-style series that influence each other
# ─────────────────────────────────────────────
np.random.seed(5)
n = 200

# GDP growth influences consumption and investment
gdp        = np.zeros(n)
consump    = np.zeros(n)
investment = np.zeros(n)

for i in range(2, n):
    gdp[i]        = 0.5 * gdp[i-1]        - 0.2 * gdp[i-2]     + np.random.normal(0, 1)
    consump[i]    = 0.3 * consump[i-1]    + 0.4 * gdp[i-1]      + np.random.normal(0, 0.8)
    investment[i] = 0.2 * investment[i-1] + 0.6 * gdp[i-1]      \
                  - 0.3 * consump[i-1]    + np.random.normal(0, 1.2)

dates = pd.date_range(start="2007-01-01", periods=n, freq="QS")
df = pd.DataFrame({
    "GDP_growth":  gdp,
    "Consumption": consump,
    "Investment":  investment,
}, index=dates)

print("=" * 55)
print("  VAR Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} quarterly observations, 3 series")
print(df.describe().round(3))

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING & ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n── Step 1: Visualise All Series ──")
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
for ax, col, color in zip(axes, df.columns, ["steelblue", "darkorange", "seagreen"]):
    ax.plot(df[col], color=color)
    ax.set_title(col)
    ax.axhline(0, color="gray", linestyle=":")
plt.suptitle("Step 1: Three Interrelated Time Series", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "var_series.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'var_series.png'}")

print("\n── Step 2: Stationarity Check (ADF for each series) ──")
print("  VAR REQUIRES all series to be stationary!")

def adf_report(s, label):
    r = adfuller(s.dropna())
    stat = "STATIONARY ✓" if r[1] < 0.05 else "NON-STATIONARY ✗"
    print(f"  [{label:<15}] p = {r[1]:.4f}  →  {stat}")
    return r[1] < 0.05

all_stationary = True
for col in df.columns:
    ok = adf_report(df[col], col)
    if not ok:
        all_stationary = False

if not all_stationary:
    print("\n  → Differencing non-stationary series...")
    df = df.diff().dropna()
    for col in df.columns:
        adf_report(df[col], col + " (diff)")

print("\n── Step 3: Check Granger Causality (optional) ──")
print("  Does GDP Granger-cause Consumption?")
gc = grangercausalitytests(df[["Consumption", "GDP_growth"]],
                            maxlag=4, verbose=False)
for lag, res in gc.items():
    pval = res[0]["ssr_ftest"][1]
    print(f"  Lag {lag}: p = {pval:.4f}  {'← Significant ✓' if pval < 0.05 else ''}")

# ─────────────────────────────────────────────
# 3.  SELECT LAG ORDER
# ─────────────────────────────────────────────
print("\n── Step 4: Select Optimal Lag Order via AIC/BIC ──")
train = df.iloc[:-8]
test  = df.iloc[-8:]

var_select = VAR(train)
lag_order_result = var_select.select_order(maxlags=10)
print(lag_order_result.summary())

best_lag = lag_order_result.aic
print(f"\n  Best lag by AIC: {best_lag}")

# ─────────────────────────────────────────────
# 4.  FIT VAR MODEL
# ─────────────────────────────────────────────
print("\n── Step 5: Fit VAR Model ──")
var_model = var_select.fit(best_lag)
print(var_model.summary())

# ─────────────────────────────────────────────
# 5.  RESIDUAL CHECK
# ─────────────────────────────────────────────
print("\n── Step 6: Residual Durbin-Watson Test ──")
dw = durbin_watson(var_model.resid)
for col, d in zip(df.columns, dw):
    print(f"  {col:<15}: DW = {d:.2f}  (ideal ≈ 2.0, < 2 = positive autocorr)")

# ─────────────────────────────────────────────
# 6.  FORECAST
# ─────────────────────────────────────────────
print("\n── Step 7: 8-Step Ahead Forecast ──")
forecast_input = train.values[-best_lag:]
fc_array = var_model.forecast(forecast_input, steps=8)
fc_df = pd.DataFrame(fc_array, index=test.index, columns=df.columns)

for col in df.columns:
    mae  = np.mean(np.abs(test[col].values - fc_df[col].values))
    rmse = np.sqrt(np.mean((test[col].values - fc_df[col].values) ** 2))
    print(f"  {col:<15} MAE={mae:.3f}  RMSE={rmse:.3f}")

# ─────────────────────────────────────────────
# 7.  PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 10))
colors = ["steelblue", "darkorange", "seagreen"]
for ax, col, color in zip(axes, df.columns, colors):
    ax.plot(train[col][-30:], color=color, label="Train")
    ax.plot(test[col], color="black", linewidth=2, label="Actual")
    ax.plot(fc_df[col], color=color, linestyle="--", linewidth=2, label="VAR Forecast")
    ax.set_title(col)
    ax.legend(fontsize=8)

plt.suptitle("VAR — 8-Quarter Ahead Forecast", fontsize=13)
plt.tight_layout()
plt.savefig(output_dir / "var_forecast.png", dpi=120)
plt.close()
print(f"\n  Saved: {output_dir / 'var_forecast.png'}")

# Impulse Response Function
irf = var_model.irf(20)
irf.plot(orth=False, figsize=(14, 10))
plt.suptitle("VAR Impulse Response Functions\n(shock to one variable → effect on others)", fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "var_irf.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'var_irf.png'}")

print("\n✅ Done.")
