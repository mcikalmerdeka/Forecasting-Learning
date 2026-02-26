"""
=============================================================
06_xgboost_timeseries.py  —  XGBoost for Time Series
=============================================================
WHEN TO USE:
  - Feature-rich datasets with external variables
  - Complex non-linear relationships
  - Medium to large datasets
  - When you can engineer good temporal features

KEY INSIGHT:
  XGBoost does not natively understand time sequences.
  You must transform the series into a supervised ML problem
  by engineering lag features and calendar features.

ASSUMPTIONS:
  1. No data leakage — features ONLY use information from the past
  2. Train/test split must respect time order
  3. Lag features must cover the seasonal period
  4. Recursive vs. direct multi-step strategy required for multi-step ahead

INSTALL:
  pip install xgboost pandas matplotlib scikit-learn
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     Daily data: trend + weekly + yearly seasonality
# ─────────────────────────────────────────────
np.random.seed(22)
n = 730   # 2 years of daily data
dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
t = np.arange(n)

trend    = 100 + 0.08 * t
weekly   = 12 * np.sin(2 * np.pi * t / 7)
yearly   = 25 * np.sin(2 * np.pi * t / 365.25)
noise    = np.random.normal(0, 3, n)

values = trend + weekly + yearly + noise
df_raw = pd.DataFrame({"date": dates, "value": values})

print("=" * 55)
print("  XGBoost Time Series Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} daily data points")

# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING — Core Pre-Processing
# ─────────────────────────────────────────────
print("\n── Step 1: Feature Engineering ──")
print("  XGBoost needs explicit temporal features!")

def create_features(df, target_col="value"):
    df = df.copy()
    df = df.set_index("date")
    
    # ── Calendar features
    df["dayofweek"]  = df.index.dayofweek        # 0=Monday
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["dayofyear"]  = df.index.dayofyear
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # ── Lag features  (CRITICAL: only use past values!)
    for lag in [1, 2, 3, 7, 14, 21, 30, 60, 90, 365]:
        if lag < len(df):
            df[f"lag_{lag}"] = df[target_col].shift(lag)
    
    # ── Rolling statistics
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window=window).mean()
        df[f"rolling_std_{window}"]  = df[target_col].shift(1).rolling(window=window).std()
    
    return df

df = create_features(df_raw)
print(f"  Total features created: {df.shape[1] - 1}")
print(f"  Feature list:\n  {[c for c in df.columns if c != 'value']}")

# ─────────────────────────────────────────────
print("\n── Step 2: Drop NaN rows (from lag creation) ──")
n_before = len(df)
df = df.dropna()
n_after = len(df)
print(f"  Dropped {n_before - n_after} NaN rows (due to max lag = 365)")
print(f"  Dataset size after drop: {n_after}")

# ─────────────────────────────────────────────
print("\n── Step 3: Train/Test Split (time-based!) ──")
print("  ⚠️  Never shuffle — future data must NOT appear in training!")

split_date = "2023-07-01"
train = df[df.index < split_date]
test  = df[df.index >= split_date]
print(f"  Train: {len(train)} rows ({train.index.min().date()} → {train.index.max().date()})")
print(f"  Test : {len(test)}  rows ({test.index.min().date()} → {test.index.max().date()})")

feature_cols = [c for c in df.columns if c != "value"]
X_train, y_train = train[feature_cols], train["value"]
X_test,  y_test  = test[feature_cols],  test["value"]

# ─────────────────────────────────────────────
# 3.  FIT XGBOOST MODEL
# ─────────────────────────────────────────────
print("\n── Step 4: Fit XGBoost ──")
model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    eval_metric="rmse",
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)
print(f"  Best iteration: {model.best_iteration}")

# ─────────────────────────────────────────────
# 4.  EVALUATE
# ─────────────────────────────────────────────
print("\n── Step 5: Evaluate on Test Set ──")
pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")

# ─────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────
# Forecast plot
plt.figure(figsize=(13, 5))
plt.plot(train["value"][-60:], color="steelblue", label="Train")
plt.plot(test["value"], color="black", label="Actual")
plt.plot(test.index, pred, color="darkorange", linestyle="--", label="XGBoost Forecast")
plt.title("XGBoost — Test Set Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "xgb_forecast.png", dpi=120)
plt.close()
print(f"\n  Saved: {output_dir / 'xgb_forecast.png'}")

# Feature importance
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20, importance_type="gain", ax=plt.gca())
plt.title("XGBoost Feature Importance (top 20 by gain)")
plt.tight_layout()
plt.savefig(output_dir / "xgb_importance.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'xgb_importance.png'}")

# Residuals
residuals = y_test.values - pred
plt.figure(figsize=(13, 4))
plt.plot(test.index, residuals, color="gray")
plt.axhline(0, color="red", linestyle="--")
plt.title("XGBoost Forecast Residuals")
plt.tight_layout()
plt.savefig(output_dir / "xgb_residuals.png", dpi=120)
plt.close()
print(f"  Saved: {output_dir / 'xgb_residuals.png'}")

print("\n✅ Done.")
