"""
features.py

Feature engineering for ETF weekly trend modelling.
Includes EMA calculation and MACD histogram computation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to sys.path to enable script-level imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_data import load_and_clean_data

def manual_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Calculate EMA manually to match spreadsheet logic.

    Parameters:
        prices (pd.Series): Series of price values
        span (int): EMA span (e.g., 5, 12, 26)

    Returns:
        pd.Series: EMA values with NaNs for initial periods
    """
    alpha = 2 / (span + 1)
    ema = [np.nan] * (span - 1)
    initial_avg = prices.iloc[:span].mean()
    ema.append(initial_avg)
    for i in range(span, len(prices)):
        ema_today = prices.iloc[i] * alpha + ema[-1] * (1 - alpha)
        ema.append(ema_today)
    return pd.Series(ema, index=prices.index)

def add_ema_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA (5, 12, 26) and MACD histogram features to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'close' column and 'date'

    Returns:
        pd.DataFrame: Modified DataFrame with new EMA and MACD features
    """
    # Compute EMAs
    df["ema_5"] = manual_ema(df["close"], 5)
    df["ema_12"] = manual_ema(df["close"], 12)
    df["ema_26"] = manual_ema(df["close"], 26)

    # Compute MACD components
    df["fast_line"] = df["ema_12"]
    df["slow_line"] = df["ema_26"]
    df["macd_h"] = (df["fast_line"] - df["slow_line"]).round(4)

    # Round selected features for clarity
    df[["ema_5", "ema_12", "ema_26", "fast_line", "slow_line", "macd_h"]] = df[[
        "ema_5", "ema_12", "ema_26", "fast_line", "slow_line", "macd_h"
    ]].round(4)

    return df

if __name__ == "__main__":
    # Example usage (for testing only)
    df = load_and_clean_data("data/SGLN Historical Data.csv")
    df = add_ema_macd_features(df)

    # Print MACD features for visual verification at a known date
    target_date = datetime.strptime("22/01/2023", "%d/%m/%Y").date()
    start_index = df[df["date"] == target_date].index[0]
    print("MACD features:")
    print(df.loc[start_index:start_index + 5, ["date", "fast_line", "slow_line", "macd_h"]])
    print("-------------------")
