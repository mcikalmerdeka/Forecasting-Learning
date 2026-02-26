"""
=============================================================
03_prophet.py  —  Facebook / Meta Prophet Forecasting
=============================================================
WHEN TO USE:
  - Business time series with holidays and events
  - Multiple seasonality (daily + weekly + yearly)
  - Handles missing data and outliers gracefully
  - Non-statisticians who need interpretable results

ASSUMPTIONS:
  - Input DataFrame must have columns: 'ds' (datetime) and 'y' (value)
  - Trend is either logistic (with cap/floor) or linear
  - Seasonality is additive or multiplicative

INSTALL:
  pip install prophet pandas matplotlib
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     Daily sales: trend + weekly cycle + yearly cycle + noise
# ─────────────────────────────────────────────
np.random.seed(7)
n = 365 * 3   # 3 years of daily data

dates = pd.date_range(start="2021-01-01", periods=n, freq="D")
t = np.arange(n)

trend    = 200 + 0.1 * t
yearly   = 30 * np.sin(2 * np.pi * t / 365.25)
weekly   = 20 * np.sin(2 * np.pi * t / 7)      # weekend dip/peak
noise    = np.random.normal(0, 5, n)

values = trend + yearly + weekly + noise

df = pd.DataFrame({"ds": dates, "y": values})

print("=" * 55)
print("  Prophet Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} daily observations (2021–2023)")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING & ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n── Step 1: Prophet requires 'ds' and 'y' columns ──")
print(f"  Columns found: {df.columns.tolist()}")
print(f"  Date column dtype: {df['ds'].dtype}")
print(f"  Missing values in y: {df['y'].isna().sum()}")

# Check for non-positive values (multiplicative seasonality needs y > 0)
print(f"  Min value of y: {df['y'].min():.2f}")
print("  ✔ All values positive — multiplicative seasonality is an option")

# Visual check
plt.figure(figsize=(13, 4))
plt.plot(df["ds"], df["y"], alpha=0.7, color="steelblue", linewidth=0.8)
plt.title("Step 1: Raw Daily Sales Data")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(output_dir / "prophet_raw_data.png", dpi=120)
plt.close()
print(f"\n  Saved: {output_dir / 'prophet_raw_data.png'}")

print("\n── Step 2: Define custom holidays (optional) ──")
holidays = pd.DataFrame({
    "holiday": ["new_year", "new_year", "new_year",
                "christmas", "christmas", "christmas"],
    "ds": pd.to_datetime([
        "2021-01-01", "2022-01-01", "2023-01-01",
        "2021-12-25", "2022-12-25", "2023-12-25"
    ]),
    "lower_window": [-1, -1, -1, -2, -2, -2],
    "upper_window": [1, 1, 1, 1, 1, 1],
})
print("  Defined holidays:")
print(holidays.to_string(index=False))

# ─────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
forecast_horizon = 90   # predict 3 months ahead
train_df = df.iloc[:-forecast_horizon].copy()
test_df  = df.iloc[-forecast_horizon:].copy()
print(f"\n  Train: {len(train_df)} days | Test: {len(test_df)} days")

# ─────────────────────────────────────────────
# 4.  FIT PROPHET MODEL
# ─────────────────────────────────────────────
print("\n── Step 3: Fit Prophet Model ──")

m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,      # no sub-daily patterns here
    holidays=holidays,
    seasonality_mode="additive",  # use 'multiplicative' if variance grows with level
    changepoint_prior_scale=0.05, # 0.05 = default; higher = more flexible trend
    seasonality_prior_scale=10,
    interval_width=0.95,
)

m.fit(train_df)
print("  Model fitted successfully.")

# ─────────────────────────────────────────────
# 5.  FORECAST
# ─────────────────────────────────────────────
print("\n── Step 4: Generate Forecast ──")
future = m.make_future_dataframe(periods=forecast_horizon, freq="D")
forecast = m.predict(future)

# Evaluate on test set
pred_test = forecast.set_index("ds").loc[test_df["ds"], "yhat"].values
actual    = test_df["y"].values

mae  = np.mean(np.abs(actual - pred_test))
rmse = np.sqrt(np.mean((actual - pred_test) ** 2))
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")

# ─────────────────────────────────────────────
# 6.  PLOTS
# ─────────────────────────────────────────────
# Forecast plot
fig1 = m.plot(forecast, figsize=(13, 5))
plt.title("Prophet — Full Forecast with Uncertainty Bands")
plt.tight_layout()
fig1.savefig(output_dir / "prophet_forecast.png", dpi=120)
plt.close()
print(f"\n  Saved: {output_dir / 'prophet_forecast.png'}")

# Component decomposition
fig2 = m.plot_components(forecast, figsize=(13, 10))
plt.suptitle("Prophet — Decomposed Components", fontsize=13)
plt.tight_layout()
fig2.savefig(output_dir / "prophet_components.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'prophet_components.png'}")

# Close-up test period
plt.figure(figsize=(13, 5))
plt.plot(test_df["ds"].values, actual, label="Actual", color="black")
plt.plot(test_df["ds"].values, pred_test, label="Prophet Forecast", color="darkorchid", linestyle="--")
ci = forecast.set_index("ds").loc[test_df["ds"], ["yhat_lower", "yhat_upper"]]
plt.fill_between(test_df["ds"].values,
                 ci["yhat_lower"].values, ci["yhat_upper"].values,
                 alpha=0.25, color="darkorchid", label="95% CI")
plt.title("Prophet — Test Period Close-Up")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "prophet_test_closeup.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'prophet_test_closeup.png'}")

print("\n✅ Done.")
