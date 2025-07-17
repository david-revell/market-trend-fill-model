"""
prepare_data.py

Load and clean the SGLN weekly price dataset.
Removes unused columns, formats data, and computes derived price movement columns.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean weekly price data for SGLN.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame.
    """
    # Load CSV
    df_raw = pd.read_csv(filepath)

    # Drop unused volume column
    df = df_raw.drop(columns=["Vol."])

    # Reverse ordering to ensure chronological order (oldest first)
    df = df.iloc[::-1].reset_index(drop=True)

    # Format dates
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.date

    # Clean numeric columns (remove commas and convert to float)
    df["Close"] = df["Price"].astype(str).str.replace(",", "").astype(float)
    df["Low"] = df["Low"].astype(str).str.replace(",", "").astype(float)
    df["High"] = df["High"].astype(str).str.replace(",", "").astype(float)
    df["Open"] = df["Open"].astype(str).str.replace(",", "").astype(float)

    # Clean percentage column
    df["change_pct"] = df["Change %"].str.replace("%", "").astype(float)

    # Drop no-longer-needed columns
    df = df.drop(columns=["Price", "Change %"])

    # Calculate weekly price movement from previous close
    df["close_prev"] = df["Close"].shift(1)
    df["change_to_low"] = ((df["Low"] - df["close_prev"]) / df["close_prev"] * 100).round(2)
    df["change_to_high"] = ((df["High"] - df["close_prev"]) / df["close_prev"] * 100).round(2)

    # Drop rows with missing values from shift
    df = df.dropna(subset=["close_prev", "change_to_low", "change_to_high"])

    # Keep only the most recent 149 rows
    df = df.iloc[-149:].reset_index(drop=True)

    # Rename columns to lower case
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close"
    })

    return df

if __name__ == "__main__":
    # Example usage (for testing)
    test_path = "data/SGLN Historical Data.csv"  # Update with your actual filename
    df = load_and_clean_data(test_path)
    print(df.head())