from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from keras.optimizers import Adam
import keras_tuner as kt
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from tensorflow.keras import mixed_precision
from datetime import datetime, timedelta

mixed_precision.set_global_policy("mixed_float16")
Dense(32, activation="relu", kernel_regularizer=l2(1e-4))

pd.set_option("future.no_silent_downcasting", True)
pd.set_option('display.max_columns', None)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU completely
#import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')


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

    # Bias corrected line (optional overlay)
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
    axs[2].plot(np.arange(len(train_loss)), train_loss, label="Training Loss", color="blue")
    if val_loss:
        axs[2].plot(np.arange(len(val_loss)), val_loss, label="Validation Loss", color="orange")
    axs[2].set_title(f"{symbol} --- LOSS CURVES --- (Smoothed)")
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

    plt.tight_layout()
    plt.show()

    # --- Print Metrics ---
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

    # --- Extra residuals plot if available ---
    if residuals is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(prediction_dates, residuals, color="purple", alpha=0.7)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.title(f"{symbol} Residuals (Actual - Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{symbol}_residuals.png", dpi=300)

    # --- Loss Curve Overlay ---
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss, label="Validation Loss", color="orange")
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
    mape = mean_absolute_percentage_error(y_test, predictions)

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
