from scipy.signal import savgol_filter  # better smoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For reading stock data from Yahoo Finance
import yfinance as yf
import tensorflow as tf
import numpy as np

# Enable eager execution to avoid numpy() errors
#tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
# For timestamps
from numpy.typing import NDArray # type: ignore
import seaborn as sns
from keras.optimizers import Adam # type: ignore
from datetime import datetime, timedelta
import keras_tuner as kt # type: ignore
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import GRU
from keras.regularizers import l2
from tensorflow.keras import mixed_precision
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

mixed_precision.set_global_policy("mixed_float16")
Dense(32, activation="relu", kernel_regularizer=l2(1e-4))

pd.set_option("future.no_silent_downcasting", True)
pd.set_option('display.max_columns', None)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU completely
#import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')

# IMB Quantum api_key="7d88492b4694e966a3e759ca99d8b648ff7f7353a1af35f78d69f8498030e083f61a496531000ff1eb91b34df5abc16075833e1413af843ae40a66adfbb8c9e7"

# Set plotting styles
sns.set_style('whitegrid')
#plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (14, 8)

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)

# Initialize an empty dictionary to store stock data
company_dict = {}
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# --- Random stock generator ---
def generate_random_stock(symbol="FAKE", start="2020-01-01", end="2023-01-01", seed=None):
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start, end=end, freq="B")  # business days
    returns = np.random.normal(loc=0.0005, scale=0.02, size=len(dates))
    prices = 100 * (1 + returns).cumprod()

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
        "High": prices * (1 + np.random.normal(0.005, 0.002, len(dates))),
        "Low":  prices * (1 - np.random.normal(0.005, 0.002, len(dates))),
        "Close": prices,
        "Adj Close": prices,
        "Volume": np.random.randint(1e6, 5e6, len(dates))
    }, index=dates)
    df["Ticker"] = symbol
    return df

def add_volatility_indicators(df, close_col="Close", high_col="High", low_col="Low"):
    df["Return"] = df[close_col].pct_change()

    # rolling σ
    df["Volatility_20"] = df["Return"].rolling(20).std()

    # Bollinger Z with ε + clipping
    sma = df[close_col].rolling(20).mean()
    rolling_std = df[close_col].rolling(20).std()
    eps = 1e-8
    df["Boll_Z"] = ((df[close_col] - sma) / (rolling_std + eps)).clip(-5, 5)

    # ATR (14) normalized
    hl = df[high_col] - df[low_col]
    hc = (df[high_col] - df[close_col].shift()).abs()
    lc = (df[low_col]  - df[close_col].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR_14"]   = tr.rolling(14).mean()
    df["ATR_norm"] = df["ATR_14"] / (df[close_col] + eps)

    return df

# --- Loader: auto-detect FAKE vs real ---
def load_data(tickers, start="2010-01-01", end=None, seed=None):
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    company_dict = {}

    for i, ticker in enumerate(tickers):
        try:
            if ticker.upper().startswith("FAKE"):
                df = generate_random_stock(
                    ticker, start=start, end=end,
                    seed=None if seed is None else seed + i
                )
                print(f"🧪 Generated synthetic data for {ticker}")
            else:
                df = yf.download(ticker, start=start, end=end, progress=False)
                print(f"📈 Downloaded real data for {ticker}")

            if not df.empty:
                # --- Flatten MultiIndex if present ---
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]  # take first level only

                # --- Standardize column names ---
                df.columns = [c.title() for c in df.columns]  # Open, High, Low, Close, etc.

                # --- Rename Close column to ticker-specific ---
                # ✅ Keep Close universal
                if "Close" not in df.columns:
                    raise ValueError(f"No 'Close' column in data for {ticker}")


                company_dict[ticker] = df
            else:
                print(f"⚠️ No data for {ticker}")

        except Exception as e:
            print(f"❌ Error loading {ticker}: {e}")

    return company_dict

# --- Clean and augment company_dict ---
def preprocess_company_dict(company_dict):
    cleaned_dict = {}

    for symbol, df in company_dict.items():
        df = df.copy()

        # Ensure we always have a Close column
        if "Close" not in df.columns:
            raise ValueError(f"❌ No 'Close' column found for {symbol}")

        # Add Daily Return
        df["Daily Return"] = df["Close"].pct_change()

        # Add Moving Averages (only if enough data exists)
        for ma in [10, 20, 50]:
            if len(df) >= ma:
                df[f"MA_{ma}"] = df["Close"].rolling(ma).mean()
            else:
                df[f"MA_{ma}"] = np.nan

        # ✅ Add volatility indicators (ATR, Boll_Z, Volatility_20)
        df = add_volatility_indicators(df)

        df = df.dropna()
        cleaned_dict[symbol] = df

        print(f"✅ Preprocessed {symbol}: {df.shape[0]} rows")

    return cleaned_dict


def compute_bias_correction(y_true, y_pred, price_series):
    """
    Compute robust bias (median error) in price space, clamped to 10% of range.
    """
    err = y_true - y_pred
    bias = float(np.median(err))

    prange = float(price_series.max() - price_series.min())
    clamp = 0.10 * prange if prange > 0 else 1.0
    return float(np.clip(bias, -clamp, clamp))


def add_volatility_indicators(df, close_col="Close", high_col="High", low_col="Low"):
    """
    Add rolling stddev of returns, Bollinger Band z-score, and ATR to dataframe.
    """
    # --- Daily returns ---
    df["Return"] = df[close_col].pct_change()

    # --- Rolling StdDev of Returns (20-day) ---
    df["Volatility_20"] = df["Return"].rolling(window=20).std()

    # --- Bollinger Bands (20-day, 2σ) ---
    window = 20
    sma = df[close_col].rolling(window).mean()
    rolling_std = df[close_col].rolling(window).std()
    df["Boll_Upper"] = sma + 2 * rolling_std
    df["Boll_Lower"] = sma - 2 * rolling_std

    # Bollinger Z-score: normalized distance from SMA
    df["Boll_Z"] = (df[close_col] - sma) / rolling_std

    # --- ATR (Average True Range, 14-day) ---
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift())
    low_close = np.abs(df[low_col] - df[close_col].shift())
    tr = high_low.to_frame("HL")
    tr["HC"] = high_close
    tr["LC"] = low_close
    tr["TR"] = tr.max(axis=1)  # true range
    df["ATR_14"] = tr["TR"].rolling(14).mean()

    # Normalize ATR by price to keep scale consistent
    df["ATR_norm"] = df["ATR_14"] / df[close_col]

    return df

def safe_mape(y_true, y_pred, eps=1e-6):
    """
    Mean Absolute Percentage Error (MAPE) with epsilon 
    to prevent division by zero when y_true ≈ 0.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))


# --- Sequence preparation ---
def make_sequences(df, features, time_steps=60, test_size=0.2):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i-time_steps:i])
        y.append(scaled[i, 0])  # Close = first feature
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        print(f"⚠️ Not enough data for time_steps={time_steps}. Reduce time_steps or extend date range.")
        return None, None, None, None, None

    split = int(len(X) * (1 - test_size))
    x_train, x_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return scaler, x_train, x_test, y_train, y_test


# --- Preprocessing function (multi-ticker wrapper) ---
# --- Preprocessing function (safe feature builder) ---
def prepare_features(
    company_dict,
    features=["Close", "Daily Return", "MA_10", "MA_20", "MA_50"],
    time_steps=60,
    test_size=0.2
):
    if not company_dict:
        raise ValueError("company_dict is empty. Please load stock data first.")

    # Use first ticker for now (can extend to loop later)
    first_ticker = list(company_dict.keys())[0]
    df = company_dict[first_ticker].copy()

    # --- Ensure required columns exist ---
    if "Close" not in df.columns:
        raise KeyError(f"'Close' column missing in {first_ticker} dataset")

    # Daily Return
    if "Daily Return" not in df.columns:
        df["Daily Return"] = df["Close"].pct_change()

    # Moving averages
    for ma in [10, 20, 50]:
        col = f"MA_{ma}"
        if col not in df.columns:
            df[col] = df["Close"].rolling(ma).mean()

    # Drop missing rows from rolling windows
    df = df.dropna()

    # Check all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features in {first_ticker} dataset: {missing}")

    # --- Create sequences ---
    scaler, x_train, x_test, y_train, y_test = make_sequences(
        df, features, time_steps=time_steps, test_size=test_size
    )

    return df[features], scaler, x_train, x_test, y_train, y_test

# ==============================
tickers = ['CGNX', 'OPEN', 'FAKE1', 'FAKE2']

company_dict = load_data(
    tickers,
    start="2020-01-01",
    end="2023-01-01",
    seed=None
)

# ✅ Preprocess to add MAs + volatility features
company_dict = preprocess_company_dict(company_dict)

# Now prepare sequences
feature_df, scaler, x_train, x_test, y_train, y_test = prepare_features(
    company_dict,
    features=[
        "Close", "Daily Return", "MA_10", "MA_20", "MA_50",
        "Volatility_20", "Boll_Z", "ATR_norm"
    ],
    time_steps=60,
    test_size=0.2
)


feature_df, scaler, x_train, x_test, y_train, y_test = prepare_features(
    company_dict,
    features=[
        "Close", "Daily Return", "MA_10", "MA_20", "MA_50",
        "Volatility_20", "Boll_Z", "ATR_norm"
    ],
    time_steps=60,
    test_size=0.2
)

print(feature_df.tail())

print("Feature DataFrame shape:", feature_df.shape)
print("x_train:", None if x_train is None else x_train.shape)
print("x_test:", None if x_test is None else x_test.shape)

"""

# Plot historical view of the closing prices for SFAKE1 and TFAKE1
plt.figure(figsize=(15, 10))
plt.title('Historical Closing Prices')

for company in ['SFAKE1', 'TFAKE1']:
    if company in multiIndex_df.columns:  # Check if the company is in the columns
        plt.plot(multiIndex_df.index,  multiIndex_df[company]['Close'], label=company, color=colors[company])
    else:
        print(f"Company {company} not found in MultiIndex DataFrame columns.")

# Define and customize the legend
plt.legend(title='Tech Stocks', loc='best', fontsize='large', title_fontsize='13', shadow=True, fancybox=True)
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot historical view of the closing prices for FAKE2 and FAKE3
plt.figure(figsize=(15, 10))
plt.title('Historical Closing Prices')

for company in [ 'FAKE2', 'FAKE3' ]:
    if company in tech_list:  # Check if the company is in the columns
        plt.plot(df.index, df[company]['Close'], label=company, color=colors[company])
    else:
        print(f"Company {company} not found in MultiIndex DataFrame columns.")

# Define and customize the legend
plt.legend(title='Tech Stocks', loc='best', fontsize='large', title_fontsize='13', shadow=True, fancybox=True)
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Create a figure with subplots for each company's volume
plt.figure(figsize=(15, 10))
plt.suptitle('Total Volume of Stock Being Traded Each Day', fontsize=16)

for i, (company_name, company_df) in enumerate(company_dict.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(company_df.index, company_df['Volume'], label=f'{company_name} Volume', color=colors[company_name])
    plt.ylabel('Volume')
    plt.xlabel('Date')
    plt.title(f'{company_name}')
    plt.grid(True)
    plt.legend()

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the main title
plt.show()



# Create a figure with subplots for each company's daily returns
# Create subplots for Daily Returns
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
fig.suptitle('Daily Returns')

# Plot daily returns with custom colors
for ax, (company, company_df) in zip(axes.flatten(), company_dict.items()):
    # Check if 'Daily Return' exists in the DataFrame
    if 'Daily Return' in company_df.columns:
        ax.plot(company_df.index, company_df['Daily Return'], label=f'{company} Daily Return', color=colors[company])
        ax.set_title(f'{company}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Return')
        ax.grid(True)
        ax.legend()
    else:
        print(f"Error: 'Daily Return' column not found for {company}.")

# Adjust layout
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Create subplots for Moving Averages
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
fig.suptitle('Moving Averages')

# Plot moving averages with custom colors
for ax, (company_name, company_df) in zip(axes.flatten(), company_dict.items()):
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        # Check if the moving average column exists
        if column_name in company_df.columns:
            ax.plot(company_df.index, company_df[column_name], label=f'{ma}-Day MA', linestyle='--')
        else:
            print(f"Error: '{column_name}' column not found for {company_name}.")
    ax.set_title(f'{company_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()

# Adjust layout
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Plot histograms of Daily Returns for each company
plt.figure(figsize=(15, 10))
plt.suptitle('Histogram of Daily Returns for Each Company', fontsize=16)

for i, (company_name, company_df) in enumerate(company_dict.items(), 1):
    plt.subplot(2, 2, i)
    company_df['Daily Return'].hist(bins=50, color=colors[company_name], edgecolor='black')
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(company_name)
    plt.grid(True)

# Adjust layout to ensure everything fits well
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

"""


"""
# --- Forecast Function (with recomputed features) ---
def make_forecast(model, df, scaler, features, time_steps, steps, symbol):
    # Generate multi-step forecast with feature consistency.
    df_future = df.copy()
    preds = []

    last_window = scaler.transform(df_future[features].iloc[-time_steps:].values)

    for _ in range(steps):
        input_seq = last_window.reshape(1, time_steps, len(features))
        next_pred = model.predict(input_seq, verbose=0)[0, 0]
        preds.append(next_pred)

        # construct next row
        new_row = df_future[features].iloc[-1:].copy()
        new_row["Close"] = next_pred
        new_row["Daily Return"] = new_row["Close"].pct_change().ffill()
        for ma in [10, 20, 50]:
            new_row[f"MA_{ma}"] = new_row["Close"].rolling(ma).mean().ffill()

        # scale new_row properly
        new_row_scaled = scaler.transform(new_row[features])
        last_window = np.vstack([last_window[1:], new_row_scaled])

        df_future = pd.concat([df_future, new_row], axis=0)

    preds_unscaled = scaler.inverse_transform(
        pd.DataFrame(
            np.concatenate([np.array(preds).reshape(-1,1), np.zeros((steps, len(features)-1))], axis=1),
            columns=features
        )
    )[:, 0]

    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=steps, freq="B")
    return pd.DataFrame(preds_unscaled, index=future_dates, columns=[symbol])
________________

# Comparing returns with seaborn's jointplot
sns.jointplot(x='SFAKE1', y='TFAKE1', data=rets, kind='scatter', color='seagreen')
sns.jointplot(x='SFAKE1', y='FAKE2', data=rets, kind='scatter')
plt.show()

# Pairplot for automatic visual analysis
sns.pairplot(rets, kind='reg')
plt.show()

# Set up our figure by naming it returns_fig, call PairGrid on the DataFrame
return_fig = sns.PairGrid(rets.dropna())

# Customize the PairGrid
return_fig.map_upper(plt.scatter, color='purple')
return_fig.map_lower(sns.kdeplot, cmap='cool_d')
return_fig.map_FAKE3g(plt.hist, bins=30)
plt.show()

# Create correlation heatmaps
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.heatmap(rets.corr(), annot=True, cmap='summer')
plt.title('Correlation of Stock Returns')

plt.subplot(2, 2, 2)
sns.heatmap(pd.DataFrame({company: data['Close'] for company, data in company_dict.items()}).corr(), annot=True, cmap='summer')
plt.title('Correlation of Stock Closing Prices')
# Adjust layout to make space for the main title and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

fig.tight_layout()

# Plot daily returns for each company
# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
fig.suptitle('Daily Returns')

# Plot daily returns with custom colors
for ax, (company, company_df) in zip(axes.flatten(), company_dict.items()):
    ax.plot(company_df.index, company_df['Daily Return'], label=f'{company} Daily Return', color=colors[company])
    ax.set_title(f'{company}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return')
    ax.grid(True)
    ax.legend()

# Adjust layout
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

for ax, (company_name, company_df) in zip(axes.flatten(), company_dict.items()):
    company_df['Daily Return'].plot(ax=ax, legend=True, linestyle='--', marker='o')
    ax.set_title(company_name)

fig.tight_layout()

# Plot histogram of Daily Returns
plt.figure(figsize=(15, 10))

for i, (company_name, company_df) in enumerate(company_dict.items(), 1):
    plt.subplot(2, 2, i)
    company_df['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(company_name)

plt.tight_layout()

# Comparing returns with seaborn's jointplot
sns.jointplot(x='SFAKE1', y='TFAKE1', data=rets, kind='scatter', color='seagreen')
sns.jointplot(x='SFAKE1', y='FAKE2', data=rets, kind='scatter')

# Pairplot for automatic visual analysis
sns.pairplot(rets, kind='reg')

# Set up our figure by naming it returns_fig, call PairPlot on the DataFrame
return_fig = sns.PairGrid(rets.dropna())

# Customize the PairGrid
return_fig.map_upper(plt.scatter, color='purple')
return_fig.map_lower(sns.kdeplot, cmap='cool_d')
return_fig.map_FAKE3g(plt.hist, bins=30)

# Create correlation heatmaps
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(rets.corr(), annot=True, cmap='summer')
plt.title('Correlation of Stock Returns')

plt.subplot(2, 2, 2)
sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
plt.title('Correlation of Stock Closing Prices')

# Risk vs. Return scatter plot
rets = rets.dropna()
area = np.pi * 20
plt.title('Risk vs. Return Scatter Plot')
plt.figure(figsize=(10, 8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

# Annotate the plot
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, (x, y), xytext=(10, 0), textcoords='offset points',
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
"""
# ==============================
def smooth_curve(values, window=5, poly=2):
    """Smooth a curve with Savitzky-Golay filter or fallback to rolling mean."""
    if len(values) < window:
        return values  # not enough data
    try:
        return savgol_filter(values, window_length=window, polyorder=poly)
    except ValueError:
        # fallback: simple rolling mean
        return pd.Series(values).rolling(window, min_periods=1, center=True).mean().values

def plot_results(history, adj_close_prices, training_data_len,
                 prediction_dates, predictions_unscaled, symbol,
                 mae, rmse, r2, mape, direction_acc,
                 y_test, future_predictions_df, six_month_df, feature_df,
                 bias_correction=0):

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # --- Panel 1: Actual + Predictions + Forecasts ---
    axs[0].plot(adj_close_prices.index, adj_close_prices, label="Actual Prices", color="blue")
    axs[0].plot(prediction_dates, predictions_unscaled, label="Predicted Prices", color="orange")

    if bias_correction != 0:
        predictions_corrected = predictions_unscaled + bias_correction
        axs[0].plot(prediction_dates, predictions_corrected,
                    label="Bias-Corrected", color="green", linestyle=":")

    axs[0].axvline(adj_close_prices.index[training_data_len], color="gray",
                   linestyle="--", label="Train/Test Split")

    axs[0].plot(future_predictions_df.index, future_predictions_df[symbol],
                label="60-Day Forecast", linestyle="--", color="green")
    axs[0].plot(six_month_df.index, six_month_df[symbol],
                label="6-Month Forecast", linestyle="--", color="red")

    axs[0].set_title(f"{symbol} Actual + Predictions + Forecasts")
    axs[0].legend()

    # Metrics overlay
    metrics_text = (
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"R²: {r2:.4f}\n"
        f"MAPE: {mape*100:.2f}%\n"
        f"Dir. Acc: {direction_acc:.2f}%"
    )
    axs[0].text(
        0.02, 0.98, metrics_text,
        transform=axs[0].transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    # --- Panel 2: Residuals ---
    residuals = y_test - predictions_unscaled
    axs[1].plot(prediction_dates, residuals, color="purple", label="Residuals")
    axs[1].axhline(0, linestyle="--", color="black")
    axs[1].set_title(f"{symbol} Residuals (Actual - Predicted)")
    axs[1].set_ylabel("Residuals")
    axs[1].legend()

    # --- Panel 3: Loss curves ---
    train_loss = history.history["loss"]
    val_loss = history.history.get("val_loss", [])

    axs[2].plot(np.arange(len(train_loss)), smooth_curve(train_loss),
                label="Training Loss (Smoothed)", color="blue")
    if len(val_loss) > 0:
        axs[2].plot(np.arange(len(val_loss)), smooth_curve(val_loss),
                    label="Validation Loss (Smoothed)", color="orange")

    axs[2].set_title(f"{symbol} --- LOSS CURVES ---")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Loss (MSE)")
    axs[2].legend()

    # --- Panel 4: Volatility indicators ---
    vol = feature_df["Close"].pct_change().rolling(20).std()
    bb_width = (feature_df["Close"].rolling(20).max() -
                feature_df["Close"].rolling(20).min()) / feature_df["Close"]

    axs[3].plot(vol.index, vol, label="20-Day Volatility", color="navy")
    axs[3].plot(bb_width.index, bb_width, label="BB Width (20-Day)", color="red", linestyle="--")
    axs[3].set_title(f"{symbol} Volatility Indicators")
    axs[3].legend()

    # --- Save + Show main 4-panel figure ---
    plt.tight_layout()
    plt.savefig(f"{symbol}_results_with_residuals_and_loss.png", dpi=300)
    plt.show()

    # --- Console metrics ---
    print(f"\n📊 {symbol} Evaluation Metrics:")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    if mape is not None:
        print(f"MAPE  : {mape*100:.2f}%")
    if direction_acc is not None:
        print(f"Directional Accuracy (Up/Down): {direction_acc:.2f}%")
    print(f"✅ Saved → {symbol}_results_with_residuals_and_loss.png")

    # --- Extra Forecast Plots ---
    if future_predictions_df is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(future_predictions_df.index, future_predictions_df[symbol],
                 label='60-Day Forecast', color='green', linestyle='--')
        plt.title(f"{symbol} 60-Day Forecast (Daily)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{symbol}_60d_forecast.png", dpi=300)

    if six_month_df is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(six_month_df.index, six_month_df[symbol],
                 label='6-Month Forecast', color='red', linestyle='--')
        plt.title(f"{symbol} 6-Month Forecast (Daily)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{symbol}_6m_forecast.png", dpi=300)
        print(f"Saved forecasts → {symbol}_60d_forecast.png and {symbol}_6m_forecast.png")

    # --- Extra Residuals Plot ---
    if residuals is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(prediction_dates, residuals, color="purple", alpha=0.7)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.title(f"{symbol} Residuals (Actual - Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{symbol}_residuals.png", dpi=300)

    # --- Loss Curve Overlay (Smoothed) ---
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_curve(train_loss), label="Training Loss (Smoothed)", color="blue")
    if len(val_loss) > 0:
        plt.plot(smooth_curve(val_loss), label="Validation Loss (Smoothed)", color="orange")
    plt.title(f"{symbol} Loss Curves (Smoothed)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.savefig(f"{symbol}_loss_curve_overlay.png", dpi=300)

    print(f"Saved diagnostics → {symbol}_loss_curve_overlay.png and {symbol}_residuals.png")
# =======================
# ==============================
# Model Preparation
# ==============================
# --- Clean up features in company_dict ---
for symbol, df in company_dict.items():
    df = df.copy()

    # flatten MultiIndex if present (FAKE1, Close, FAKE1) → Close
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # add Daily Return if missing
    if "Daily Return" not in df.columns:
        df["Daily Return"] = df["Close"].pct_change()

    # add moving averages with consistent names
    for ma in [10, 20, 50]:
        col_name = f"MA_{ma}"
        if col_name not in df.columns:
            df[col_name] = df["Close"].rolling(window=ma).mean()

    # drop NaNs caused by rolling returns/MAs
    df = df.dropna()

    # replace back into company_dict
    company_dict[symbol] = df

print("✅ Standardized features in company_dict")
print(company_dict[f"{symbol}"].head())

# --- Model Builder ---
def build_model(time_steps, n_features):
    from keras.layers import GRU
    model = Sequential([
        GRU(64, return_sequences=False, input_shape=(time_steps, n_features)),
        Dropout(0.2),
        Dense(52, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.000055, clipnorm=1.0)  # smaller LR + gradient clipping
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model

"""
    model = Sequential([
        Input(shape=(time_steps, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])
"""
# ==============================
# Training + Forecast Loop
# ==============================
time_steps = 60
features = ["Close", "Daily Return", "MA_10", "MA_20", "MA_50"]

for symbol, df in company_dict.items():
    print(f"\nProcessing {symbol}...")

    df = df.dropna().copy()
    scaler = MinMaxScaler(feature_range=(0, 1))

    # --- Training data ---
    scaled_data = pd.DataFrame(
        scaler.fit_transform(df[features]),
        index=df.index,
        columns=features
    )
    adj_close_prices = df["Close"]
    n_features = len(features)
    # --- Training data ---
    scaled_data = pd.DataFrame(
        scaler.fit_transform(df[features]),
        index=df.index,
        columns=features
    )

    # --- Sequences ---
    def make_sequences(data: pd.DataFrame):
        X, y = [], []
        values = data.to_numpy()  # convert once inside
        for i in range(time_steps, len(values)):
            X.append(values[i - time_steps:i])
            y.append(values[i, 0])   # "Close" is col 0
        return np.array(X), np.array(y)

    training_data_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data.iloc[:training_data_len]
    test_data  = scaled_data.iloc[training_data_len - time_steps:]

    x_train, y_train = make_sequences(train_data)
    x_test, y_test   = make_sequences(test_data)


    # --- Load or build model ---
    checkpoint_path = f"checkpoints/{symbol}_best_model.keras"
    try:
        model = load_model(checkpoint_path, compile=False)
        print(f"✅ Loaded existing model for {symbol}")
        model.compile(optimizer=Adam(learning_rate=0.00005, clipnorm=1.0),
                      loss="mean_squared_error")
    except:
        print(f"⚠️ No saved model found for {symbol}, creating new...")
        model = build_model(time_steps, n_features)

    # --- Callbacks ---
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True,
                                    monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                  min_lr=1e-6, verbose=1)

    # --- Train ---
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_cb, early_stop, reduce_lr],
        verbose=2
    )

    # --- Predictions (scaled + unscaled) ---
    predictions = model.predict(x_test)

    predictions_unscaled = scaler.inverse_transform(
        np.concatenate([predictions,
                        np.zeros((len(predictions), n_features - 1))], axis=1)
    )[:, 0]

    y_test_unscaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1),
                        np.zeros((len(y_test), n_features - 1))], axis=1)
    )[:, 0]

    # --- Metrics (scaled) ---
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = safe_mape(y_test_unscaled, predictions_unscaled)


    # --- Directional accuracy ---
    direction_true = np.sign(np.diff(y_test.flatten()))
    direction_pred = np.sign(np.diff(predictions.flatten()))
    direction_acc = np.mean(direction_true == direction_pred) * 100

    # --- Bias correction (price space) ---
    bias_correction = np.mean(y_test_unscaled - predictions_unscaled)
    print(f"\n📊 {symbol} Evaluation Metrics:")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"MAPE  : {mape*100:.2f}%")
    print(f"Directional Accuracy: {direction_acc:.2f}%")
    print(f"⚖️ Bias correction applied for {symbol}: {bias_correction:.4f}")

    # --- Forecast function ---
    def make_forecast(model, df, scaler, features, time_steps, horizon, symbol, bias_correction=0):
        window = df[features].tail(time_steps).copy()
        preds = []
        for _ in range(horizon):
            window_scaled = scaler.transform(window[features])
            x_input = window_scaled.reshape(1, time_steps, len(features))
            pred_scaled = model.predict(x_input, verbose=0)[0][0]

            # inverse-transform Close only
            close_only = np.zeros((1, len(features)))
            close_only[0, 0] = pred_scaled
            pred_unscaled = scaler.inverse_transform(close_only)[0, 0]

            if np.isfinite(bias_correction):
                pred_unscaled += bias_correction
            preds.append(pred_unscaled)

            # rebuild new row with recomputed features
            new_row = {
                "Close": pred_unscaled,
                "Daily Return": (pred_unscaled / window["Close"].iloc[-1]) - 1,
                "MA_10": pd.Series([*window["Close"].iloc[-9:], pred_unscaled]).mean(),
                "MA_20": pd.Series([*window["Close"].iloc[-19:], pred_unscaled]).mean(),
                "MA_50": pd.Series([*window["Close"].iloc[-49:], pred_unscaled]).mean(),
            }
            window = pd.concat([window, pd.DataFrame([new_row])]).tail(time_steps)

        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                       periods=horizon, freq="B")
        return pd.DataFrame({symbol: preds}, index=forecast_dates)

    # --- Forecasts ---
    future_predictions_df_60 = make_forecast(model, df, scaler, features, time_steps, 60, symbol, bias_correction)
    future_predictions_df_6m = make_forecast(model, df, scaler, features, time_steps, 126, symbol, bias_correction)

    # --- Save CSVs ---
    results_df = pd.DataFrame({
        f"{symbol}_Actual": y_test_unscaled,
        f"{symbol}_Predicted": predictions_unscaled
    }, index=adj_close_prices.index[-len(y_test_unscaled):])
    results_df.to_csv(f"{symbol}_predictions.csv")

    future_predictions_df_60.to_csv(f"future_predictions_{symbol}_60d.csv")
    future_predictions_df_6m.to_csv(f"future_predictions_{symbol}_6m.csv")

    # --- Plots ---
    plot_results(
        history=history,
        adj_close_prices=adj_close_prices,
        training_data_len=training_data_len,
        prediction_dates=adj_close_prices.index[training_data_len:
                                               training_data_len + len(predictions_unscaled)],
        predictions_unscaled=predictions_unscaled,
        symbol=symbol,
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        direction_acc=direction_acc,
        y_test=y_test_unscaled,
        future_predictions_df=future_predictions_df_60,
        six_month_df=future_predictions_df_6m,
        feature_df=df[features]
    )

    # --- Save final model ---
    model.save(f"checkpoints/{symbol}_final_model.keras")
    print("--------------------------------------------------")

"""
# =======================
# STEP 0: Setup
# =======================
data = closing_df.dropna()
print(f"\nClosing DF:\n{data}")

symbols = ["FAKE1", "FAKE2", "FAKE3"]  # update as needed
all_predictions_df = pd.DataFrame()   # combined across symbols

# =======================
# STEP 1: Data Prep
# =======================
def extract_window_data(series, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(series) - window_len):
        tmp = series[idx:(idx + window_len)].copy()
        if zero_base:
            tmp = tmp / tmp[0] - 1
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(data, aim="Close", window_len=10, zero_base=True, test_size=0.2):
    nrows = len(data)
    split_index = int(nrows * (1 - test_size))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = extract_window_data(train_data[aim], window_len, zero_base)
    X_test = extract_window_data(test_data[aim], window_len, zero_base)

    y_train = train_data[aim].values[window_len:]
    y_test = test_data[aim].values[window_len:]

    if zero_base:
        y_train = y_train / train_data[aim].values[:-window_len] - 1
        y_test = y_test / test_data[aim].values[:-window_len] - 1

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    return X_train, X_test, y_train, y_test

# =======================
# STEP 2: Model Builder
# =======================
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('lstm_units', 32, 128, step=16),
        return_sequences=False,
        input_shape=(X_train.shape[1], 1)
    ))
    model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.05)))
    model.add(Dense(hp.Int('dense_units', 8, 64, step=8), activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
        loss='mse',
        metrics=['mae']
    )
    return model

# =======================
# STEP 3: Loop over symbols
# =======================
for symbol in symbols:
    print(f"\n🚀 Processing stock: {symbol}")
    adj_close_prices = data[symbol].squeeze().to_frame(name="close")

    # Data split
    X_train, X_test, y_train, y_test = prepare_data(
        data=adj_close_prices,
        aim="close",
        window_len=10,
        zero_base=True,
        test_size=0.2
    )

    # Hyperparameter tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='tuner_results',
        project_name=f'{symbol}_lstm_tuning'
    )
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    tuner.search(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[stop_early],
        verbose=0
    )

    # Train best model
    best_hps = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        X_train, y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[stop_early],
        verbose=0
    )

    # Predictions
    preds = best_model.predict(X_test).squeeze()

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    print(f"\n📊 {symbol} Evaluation:")
    print(f"MAE={mae:.4f}  MSE={mse:.6f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Per-symbol results DataFrame
    results_df = pd.DataFrame({
        f"{symbol}_Actual": y_test,
        f"{symbol}_Predicted": preds
    }, index=adj_close_prices.index[-len(y_test):])
    results_df.to_csv(f"{symbol}_predictions.csv")

    # Merge into combined DataFrame
    all_predictions_df = pd.concat([all_predictions_df, results_df], axis=1)

    # Training vs Validation Loss plot
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title(f"{symbol} - LSTM Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"{symbol}_loss_plot.png")
    plt.close()

# =======================
# STEP 4: Save Combined
# =======================
all_predictions_df.to_csv("all_symbols_predictions.csv")
print("\n✅ Combined predictions saved to all_symbols_predictions.csv")
"""