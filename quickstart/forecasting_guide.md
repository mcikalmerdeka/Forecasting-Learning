# Time Series Forecasting Methods: A Comprehensive Guide

## Overview

Time series forecasting is the process of predicting future values based on historical observations. Choosing the right method depends on your data's characteristics: trend, seasonality, stationarity, volume, and the need for interpretability.

---

## Quick Reference: When to Use What

| Method                      | Best For                    | Data Size    | Seasonality     | Non-linear | Interpretability |
| --------------------------- | --------------------------- | ------------ | --------------- | ---------- | ---------------- |
| ARIMA                       | Stationary/short series     | Small–Medium | No (use SARIMA) | No         | High             |
| SARIMA                      | Seasonal patterns           | Small–Medium | Yes             | No         | High             |
| Prophet                     | Business time series        | Medium–Large | Multi-seasonal  | Partial    | High             |
| Exponential Smoothing (ETS) | Smooth trends               | Small–Medium | Optional        | No         | High             |
| LSTM (RNN)                  | Complex sequential data     | Large        | Yes             | Yes        | Low              |
| XGBoost / ML                | Feature-rich, tabular       | Medium–Large | Engineered      | Yes        | Medium           |
| VAR                         | Multiple related series     | Medium       | No              | No         | Medium           |
| Theta                       | Simple competition baseline | Small–Medium | Yes             | No         | Medium           |

---

## 1. ARIMA (AutoRegressive Integrated Moving Average)

### When to Use

- Univariate time series (one variable)
- Data can be made **stationary** (constant mean/variance over time)
- No strong seasonal pattern (use SARIMA if seasonal)
- Short to medium datasets (< 10,000 points)
- You need statistical confidence intervals

### The Model

ARIMA(p, d, q) combines three components:

**AR(p) — AutoRegressive:**
$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t$$

**I(d) — Integrated (differencing to achieve stationarity):**
$$y'_t = y_t - y_{t-d}$$

**MA(q) — Moving Average:**
$$y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}$$

**Combined ARIMA(p,d,q):**
$$\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$$

Where $B$ is the backshift operator: $B y_t = y_{t-1}$

### Key Assumptions

1. **Stationarity** after differencing — check with ADF test
2. **No autocorrelation** in residuals — check with Ljung-Box test
3. Residuals are **normally distributed** with constant variance

### Parameters

- **p**: Number of lag observations (AR terms) — from PACF plot
- **d**: Degree of differencing — 1 or 2 usually sufficient
- **q**: Size of moving average window — from ACF plot

---

## 2. SARIMA (Seasonal ARIMA)

### When to Use

- Same as ARIMA but with **clear seasonal patterns** (weekly, monthly, yearly)
- Retail sales, energy consumption, tourism data

### The Model

SARIMA(p, d, q)(P, D, Q)[m]:

$$\Phi_P(B^m)\phi_p(B)(1-B^m)^D(1-B)^d y_t = \Theta_Q(B^m)\theta_q(B)\varepsilon_t$$

Where:

- (p, d, q) = non-seasonal orders
- (P, D, Q) = seasonal orders
- m = seasonal period (e.g., 12 for monthly data with yearly seasonality)

---

## 3. Prophet (Facebook/Meta Prophet)

### When to Use

- Business forecasting with **holidays and events**
- **Multiple seasonality** (daily + weekly + yearly)
- Handles **missing data** and outliers gracefully
- Non-statisticians who need interpretable results
- Large datasets with irregular intervals

### The Model

Prophet decomposes the series into trend + seasonality + holidays + noise:

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

**Trend** (logistic growth or linear):
$$g(t) = \frac{L}{1 + e^{-k(t - m)}}$$

**Seasonality** (Fourier series):
$$s(t) = \sum_{n=1}^{N} \left( a_n \cos\frac{2\pi n t}{P} + b_n \sin\frac{2\pi n t}{P} \right)$$

**Holiday effects:**
$$h(t) = Z(t) \cdot \kappa$$

### Key Advantages

- Automatically detects changepoints in trend
- Built-in uncertainty intervals
- Handles yearly, weekly, daily seasonality
- No need for stationary data

---

## 4. Exponential Smoothing (ETS)

### When to Use

- Simple to moderate trends and seasonal patterns
- When you want a **reliable, fast baseline**
- Small to medium datasets
- Retail inventory, demand forecasting

### Variants

**Simple Exponential Smoothing** — no trend, no seasonality:
$$\hat{y}_{t+1} = \alpha y_t + (1 - \alpha) \hat{y}_t, \quad \alpha \in (0, 1)$$

**Holt's Linear** — with trend:
$$\ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})$$
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$$
$$\hat{y}_{t+h} = \ell_t + h \cdot b_t$$

**Holt-Winters (Triple Exponential Smoothing)** — with trend + seasonality:
$$\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$$
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$$
$$s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m}$$
$$\hat{y}_{t+h} = \ell_t + h \cdot b_t + s_{t+h-m(k+1)}$$

Where:

- $\alpha$ = smoothing factor for level
- $\beta$ = smoothing factor for trend
- $\gamma$ = smoothing factor for seasonality
- $m$ = season length

### Key Assumptions

- Errors are additive or multiplicative (specify ETS model type)
- Trend is linear (additive) or exponential (multiplicative)

---

## 5. LSTM (Long Short-Term Memory Neural Network)

### When to Use

- **Large datasets** (thousands to millions of points)
- **Complex non-linear relationships**
- When traditional methods underfit
- Multivariate forecasting with many features
- When accuracy > interpretability

### The Model

LSTM is a type of Recurrent Neural Network (RNN) designed to learn long-range dependencies.

**LSTM Cell equations:**

**Forget Gate** — decides what to discard from cell state:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** — decides what new info to store:
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### Key Assumptions

- Large enough dataset to train deep networks
- Features are scaled (e.g., MinMax normalization)
- Sequence length (look-back window) is tuned
- Sufficient compute resources

### When NOT to Use

- Small datasets (< 500 points) — will overfit
- When interpretability is required
- When simpler models already perform well

---

## 6. XGBoost / Machine Learning for Time Series

### When to Use

- **Feature-rich** datasets with many external variables (e.g., price, weather, promotions)
- When relationships are **non-linear and complex**
- Medium to large datasets
- When you can engineer good lag features

### Approach

XGBoost doesn't natively understand time — you must convert the series into supervised learning by engineering features:

**Lag features:**
$$X_t = [y_{t-1}, y_{t-2}, \ldots, y_{t-k}]$$

**Rolling statistics:**
$$\text{rolling\_mean}_t = \frac{1}{w}\sum_{i=0}^{w-1} y_{t-i}$$

**Calendar features:** day of week, month, quarter, is_holiday, etc.

The model then learns:
$$\hat{y}_t = f(X_t)$$

Where $f$ is an ensemble of gradient-boosted decision trees.

### Key Assumptions

- No data leakage — features must only use past data
- Proper train/test split respecting time order
- Sufficient lag features to capture temporal patterns
- Features are scaled (for some ML models, not strictly for XGBoost)

---

## 7. VAR (Vector AutoRegression)

### When to Use

- **Multiple related time series** that influence each other
- Macroeconomics (GDP, inflation, interest rates)
- When you need to model **interdependencies** between variables
- All series must be stationary

### The Model

VAR(p) models k variables simultaneously:

$$\mathbf{y}_t = \mathbf{c} + A_1 \mathbf{y}_{t-1} + A_2 \mathbf{y}_{t-2} + \cdots + A_p \mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t$$

Where:

- $\mathbf{y}_t$ = vector of k time series at time t
- $A_i$ = coefficient matrix for lag i
- $\boldsymbol{\varepsilon}_t$ = white noise vector

### Key Assumptions

1. All series are **stationary** (difference if needed)
2. No **cointegration** (or use VECM instead)
3. Residuals are white noise
4. Lag order p selected via AIC/BIC

---

## 8. Theta Method

### When to Use

- Simple and robust **competition baseline**
- Works surprisingly well on M3/M4 forecasting competition data
- No feature engineering needed
- When you want something better than naive but simpler than ARIMA

### The Model

The Theta method decomposes a series into two "theta lines":

$$y_{\theta}(t) = (1-\theta) \cdot \text{LinearTrend}(t) + \theta \cdot y_t$$

- $\theta = 0$: pure linear trend (removes local curvature)
- $\theta = 2$: amplifies local curvature

The forecast combines the two lines:
$$\hat{y}_{t+h} = \frac{\hat{y}_{\theta=0}(t+h) + \hat{y}_{\theta=2}(t+h)}{2}$$

The $\theta=2$ line is forecasted using Simple Exponential Smoothing (SES).

---

## Evaluation Metrics

| Metric | Formula                                                        | Notes                         |
| ------ | -------------------------------------------------------------- | ----------------------------- | --- | ------------------------- |
| MAE    | $\frac{1}{n}\sum                                               | y_t - \hat{y}\_t              | $   | Interpretable, same units |
| RMSE   | $\sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2}$                   | Penalizes large errors        |
| MAPE   | $\frac{100}{n}\sum \left\|\frac{y_t - \hat{y}_t}{y_t}\right\|$ | Percentage, scale-independent |
| MASE   | $\frac{\text{MAE}}{\text{MAE}_{\text{naive}}}$                 | Relative to naive forecast    |

---

## Decision Flowchart

```
Do you have multiple related series? ──Yes──> VAR
        │
        No
        │
Is the dataset very large (>10k) with complex patterns? ──Yes──> LSTM or XGBoost
        │
        No
        │
Does the data have strong seasonality + holidays? ──Yes──> Prophet
        │
        No
        │
Does the data have trend + seasonality? ──Yes──> Holt-Winters ETS or SARIMA
        │
        No
        │
Is the series stationary (or can be made so)? ──Yes──> ARIMA
        │
        No
        │
Need a quick robust baseline? ──Yes──> Theta or Simple ETS
```

---

## Python Files in This Package

**The guide (`forecasting_guide.md`)** covers 8 methods with formulas, a decision flowchart, and a quick-reference table so you always know which method fits your situation.

**The 8 Python files**, each built as a mini-tutorial with clearly labeled steps:

| File                       | Method           | Dummy Data Type                                            |
| -------------------------- | ---------------- | ---------------------------------------------------------- |
| `01_arima.py`              | ARIMA            | Monthly trend + AR noise                                   |
| `02_sarima.py`             | SARIMA           | Monthly trend + yearly seasonality                         |
| `03_prophet.py`            | Prophet          | Daily sales with weekly + yearly cycles + holidays         |
| `04_ets_holtwinters.py`    | Holt-Winters ETS | Monthly retail with 3 model variants compared              |
| `05_lstm.py`               | LSTM             | Complex multi-frequency daily signal                       |
| `06_xgboost_timeseries.py` | XGBoost          | Daily data converted to supervised ML problem              |
| `07_var.py`                | VAR              | 3 interrelated macro series (GDP, consumption, investment) |
| `08_theta.py`              | Theta            | Monthly data with trend + seasonal decomposition           |

Each script walks you through the **pre-processing steps specific to that method** — stationarity tests (ADF), seasonal decomposition, feature engineering, scaling, sequence creation, etc. — and saves diagnostic plots alongside the forecast. A good order to go through them: ETS → ARIMA → SARIMA → Prophet → Theta → XGBoost → VAR → LSTM (roughly simple to complex).
