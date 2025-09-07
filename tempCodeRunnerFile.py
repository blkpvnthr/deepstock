        "Close": pred_unscaled,
            "Daily Return": (pred_unscaled / last_close - 1) if last_close != 0 else 0,
            "MA_10": pd.Series([*window["Close"].iloc[-9:], pred_unscaled]).mean(),
            "MA_20": pd.Series([*window["Close"].iloc[-19:], pred_unscaled]).mean(),
            "MA_50": pd.Series([*window["Close"].iloc[-49:], pred_unscaled]).mean(),
        }