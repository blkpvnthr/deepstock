# DeepStock📊 
![Training Forecast GIF](images/example2.gif)
An end-to-end time-series forecasting pipeline that trains a **GRU-based neural network** on historical OHLCV stock data (via **Yahoo Finance**) and optionally on **synthetic “FAKE” tickers**, then evaluates performance, generates diagnostics, and produces **60-business-day** and **~6-month (126 business-day)** ---

## Example Output
> Actual + Predictions + Forecasts
> Residuals & Loss Curves
> Volatility Indicators

### 📊 Example Metrics
``` bash
CGNX Evaluation Metrics:
MAE   : 0.0621
RMSE  : 0.0717
R²    : -1.1457
MAPE  : 48.67%
Directional Accuracy: 46.81%
Bias Correction: +1.235
```
### Actual + Predictions + Forecasts
![Forecast](images/example1.png)

---



___
## Installation

Clone the repo:
```bash
git clone https://github.com/blkpvnthr/deepstock.git
cd deepstock
```
Create a virtual environment:<br>
> macOS/Linux
``` bash
python -m venv .venv
source .venv/bin/activate
```
> Windows
``` bash
.venv\Scripts\activate
```

Install dependencies:
``` bash
pip install -r requirements.txt
```
Run the script:
```bash
python prediction.py
```
---

## Features

- **Real + Synthetic data support**
  - Downloads real market data with `yfinance`
  - Generates realistic-ish OHLCV for tickers like `FAKE1`, `FAKE2` for testing

- **Feature engineering**
  - Daily returns
  - Moving averages: `MA_10`, `MA_20`, `MA_50`
  - Volatility indicators:
    - `Volatility_20` (rolling std of returns)
    - `Boll_Z` (Bollinger z-score)
    - `ATR_norm` (normalized Average True Range)

- **Sequence modeling**
  - Builds rolling sequences (default: **60 timesteps**) for supervised learning
  - Predicts **next-day Close** from the prior 60 business days of engineered features

- **Training workflow**
  - Per-ticker GRU model training
  - Checkpointing: loads best saved model if present, otherwise trains from scratch
  - Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
  - Mixed precision enabled (`mixed_float16`) for faster training on supported GPUs

- **Evaluation + diagnostics**
  - MAE, RMSE, R², MAPE
  - Directional accuracy (up/down correctness)
  - Residual plots and loss curves

- **Forecasting**
  - Autoregressive multi-step forecasts:
    - **60 business days**
    - **126 business days** (~6 months)
  - Optional bias correction applied in price space

- **Outputs**
  - CSVs: predictions + future forecasts
  - PNGs: price charts, residuals, loss curves, forecast plots
  - Saved models: `.keras` files per ticker

---
## 📂  Project Structure
> You can organize the repo like this (recommended):

```
deepstock/
│
├── data/                  # input stock data (CSV or yfinance)
│
├── checkpoints/           # saved models (.keras)
│
├── results/               # predictions, plots, forecasts
│   ├── predictions.csv
│   ├── future_predictions_60d.csv
│   ├── future_predictions_*_6m.csv
│   └── plots/
│
├── images/                # example plots & animations
│   ├── example1.png
│   └── example2.gif
│
├── prediction.py          # main training + forecasting loop
├── requirements.txt       # dependencies
└── README.md              # project docs
```

---

## Usage

1. Run training + forecasting
``` bash
   python prediction.py
```

This will:
- Train or load an existing model per ticker.
- Save checkpoints to checkpoints/.
- Generate predictions and forecasts (60-day & 6-month).
- Save plots into results/plots/.


## 🛠 Requirements

> Python 3.10+
> pandas
> numpy
> scikit-learn
> matplotlib
> tensorflow / keras

Install them with:
``` bash
pip install -r requirements.txt
```

---

## 🔮 Roadmap

> Add hyperparameter tuning<br>
> Integrate additional technical indicators<br>
> Support transformer-based time-series models<br>

---
## 📜 License

MIT License.
Feel free to fork and adapt for your own trading experiments 🚀

---

## ✨ Acknowledgements

Inspired by real-world quantitative finance research and experimentation with deep learning on market data.
