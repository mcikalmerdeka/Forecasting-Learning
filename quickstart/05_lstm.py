"""
=============================================================
05_lstm.py  —  LSTM Neural Network Forecasting
=============================================================
WHEN TO USE:
  - Large datasets (1,000+ points) with complex patterns
  - Non-linear relationships between past and future
  - When traditional models underfit
  - Multivariate data with many interacting features
  - Accuracy > interpretability

ASSUMPTIONS:
  1. Large enough dataset (LSTMs overfit on small data)
  2. Features must be SCALED (MinMax or StandardScaler)
  3. Look-back window (sequence length) is a key hyperparameter
  4. Train/val/test split must respect time order (NO shuffling)

INSTALL:
  pip install tensorflow pandas matplotlib scikit-learn
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# Create the output directory if it doesn't exist
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Install with: pip install tensorflow")
    exit()

# ─────────────────────────────────────────────
# 1.  GENERATE DUMMY DATA
#     Non-linear series: damped sine with trend
# ─────────────────────────────────────────────
np.random.seed(17)
n = 1000
t = np.linspace(0, 20, n)

trend  = 0.3 * t
sine1  = 10 * np.sin(2 * np.pi * t / 4)
sine2  = 5  * np.sin(2 * np.pi * t / 1.5)
noise  = np.random.normal(0, 0.8, n)

values = trend + sine1 + sine2 + noise

dates = pd.date_range(start="2017-01-01", periods=n, freq="D")
series = pd.Series(values, index=dates, name="value")

print("=" * 55)
print("  LSTM Forecasting Walkthrough")
print("=" * 55)
print(f"\nDataset: {n} data points (complex non-linear pattern)")
print(series.describe().round(2))

plt.figure(figsize=(13, 4))
plt.plot(series.values, color="steelblue", linewidth=0.8)
plt.title("Raw Data: Complex Multi-Frequency Signal")
plt.tight_layout()
plt.savefig(output_dir / "lstm_raw.png", dpi=120)
plt.close()
print(f"\nSaved: {output_dir / 'lstm_raw.png'}")

# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING & ASSUMPTION CHECKS
# ─────────────────────────────────────────────
print("\n── Step 1: Scale Data to [0, 1] ──")
print("  CRITICAL: LSTM is sensitive to input scale.")
print("  MinMaxScaler → all values in [0, 1]")
print(f"  Data range before scaling: [{values.min():.2f}, {values.max():.2f}]")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values.reshape(-1, 1))

print(f"  Data range after scaling : [{scaled.min():.2f}, {scaled.max():.2f}]")

print("\n── Step 2: Create Supervised Sequences ──")
LOOK_BACK = 30   # use last 30 time steps to predict the next 1

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, LOOK_BACK)
print(f"  Look-back window : {LOOK_BACK} steps")
print(f"  Total sequences  : {len(X)}")
print(f"  X shape          : {X.shape}   [samples, timesteps]")
print(f"  y shape          : {y.shape}   [samples]")

print("\n── Step 3: Train/Validation/Test Split (time order!) ──")
print("  ⚠️  Never shuffle time-series data!")
train_size = int(len(X) * 0.70)
val_size   = int(len(X) * 0.15)

X_train, y_train = X[:train_size],              y[:train_size]
X_val,   y_val   = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test,  y_test  = X[train_size+val_size:],     y[train_size+val_size:]

print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# Reshape for LSTM: [samples, timesteps, features]
X_train = X_train.reshape(-1, LOOK_BACK, 1)
X_val   = X_val.reshape(-1, LOOK_BACK, 1)
X_test  = X_test.reshape(-1, LOOK_BACK, 1)

# ─────────────────────────────────────────────
# 3.  BUILD LSTM MODEL
# ─────────────────────────────────────────────
print("\n── Step 4: Build LSTM Architecture ──")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ─────────────────────────────────────────────
# 4.  TRAIN
# ─────────────────────────────────────────────
print("\n── Step 5: Train with Early Stopping ──")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1,
)

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("LSTM Training History")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "lstm_training.png", dpi=120)
plt.close()
print(f"Saved: {output_dir / 'lstm_training.png'}")

# ─────────────────────────────────────────────
# 5.  EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\n── Step 6: Evaluate on Test Set ──")
pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled).flatten()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")

# ─────────────────────────────────────────────
# 6.  PLOT
# ─────────────────────────────────────────────
plt.figure(figsize=(13, 5))
plt.plot(actual, label="Actual", color="black", linewidth=1.2)
plt.plot(pred,   label="LSTM Forecast", color="crimson", linestyle="--", linewidth=1.2)
plt.title("LSTM — Test Set Forecast")
plt.xlabel("Time Steps")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "lstm_forecast.png", dpi=120)
plt.close()
print(f"Saved: {output_dir / 'lstm_forecast.png'}")

print("\n✅ Done.")
