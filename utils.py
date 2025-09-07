import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# --- Data Loaders ---
def generate_random_stock(symbol="FAKE", start="2020-01-01", end="2023-01-01", seed=None):
    np.random.seed(seed)
    dates = pd.date_range(start=start, end=end, freq="B")
    returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))
    prices = 100 * (1 + returns).cumprod()
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df

def load_data(tickers, start="2010-01-01", end=None):
    data = {}
    for t in tickers:
        if t.startswith("FAKE"):
            data[t] = generate_random_stock(symbol=t, start=start, end=end)
        else:
            data[t] = yf.download(t, start=start, end=end, progress=False)
    return data

# --- Features ---
def add_features(df):
    df["Daily Return"] = df["Close"].pct_change()
    for ma in [10, 20, 50]:
        df[f"MA_{ma}"] = df["Close"].rolling(ma).mean()
    df = df.dropna()
    return df

# --- Metrics ---
def safe_mape(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))

# --- Plotting ---
def smooth_curve(values, window=5, poly=2):
    if len(values) < window: return values
    return savgol_filter(values, window_length=window, polyorder=poly)
