"""
model_fill.py

Fill label construction for ETF order execution modelling.
Determines if buy/sell orders would have filled based on high/low vs previous close.
Includes exploratory analysis and predictive modelling for buy and sell fill.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to sys.path to enable script-level imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_data import load_and_clean_data
from scripts.features import add_ema_macd_features

def create_fill_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["buy_filled"] = df["low"] < df["close_prev"]
    df["sell_filled"] = df["high"] > df["close_prev"]
    return df

def explore_buy_fill(df: pd.DataFrame):
    print("\nBuy fill counts:")
    print(df["buy_filled"].value_counts())

    print("\nBuy Fill Descriptive Stats:")
    buy_filled_stats = df.groupby("buy_filled")[
        ["change_to_low", "change_pct", "macd_h", "ema_5", "ema_12", "ema_26"]
    ].describe().round(2)
    print(buy_filled_stats)

    features_to_plot = ["change_to_low", "change_pct", "macd_h"]
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(features_to_plot):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x="buy_filled", y=col, data=df)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    df_corr = df.copy()
    df_corr["buy_filled_int"] = df_corr["buy_filled"].astype(int)
    features_model_1 = [
        "change_pct", "change_to_low", "change_to_high",
        "ema_5", "ema_12", "ema_26",
        "fast_line", "slow_line", "macd_h"
    ]
    features_plus_target = features_model_1 + ["buy_filled_int"]
    corr_buy = df_corr[features_plus_target].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_buy, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (Buy Filled)")
    plt.tight_layout()
    plt.show()

def explore_sell_fill(df: pd.DataFrame):
    print("\nSell fill counts:")
    print(df["sell_filled"].value_counts())

    print("\nSell Fill Descriptive Stats:")
    sell_filled_stats = df.groupby("sell_filled")[
        ["change_to_high", "change_pct", "macd_h", "ema_5", "ema_12", "ema_26"]
    ].describe().round(2)
    print(sell_filled_stats)

    features_to_plot = ["change_to_high", "change_pct", "macd_h"]
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(features_to_plot):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x="sell_filled", y=col, data=df)
        plt.title(col)
    plt.tight_layout()
    plt.show()

    df_corr = df.copy()
    df_corr["sell_filled_int"] = df_corr["sell_filled"].astype(int)
    features_model_1 = [
        "change_pct", "change_to_low", "change_to_high",
        "ema_5", "ema_12", "ema_26",
        "fast_line", "slow_line", "macd_h"
    ]
    features_plus_target = features_model_1 + ["sell_filled_int"]
    corr_sell = df_corr[features_plus_target].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_sell, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix (Sell Filled)")
    plt.tight_layout()
    plt.show()

def predict_buy_fill(df: pd.DataFrame):
    features = [
        "change_pct", "change_to_low", "change_to_high",
        "ema_5", "ema_12", "ema_26",
        "fast_line", "slow_line", "macd_h"
    ]
    df_clean = df.dropna(subset=features + ["buy_filled"])
    X = df_clean[features]
    y = df_clean["buy_filled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nBuy Fill Model Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def predict_sell_fill(df: pd.DataFrame):
    features = [
        "change_pct", "change_to_low", "change_to_high",
        "ema_5", "ema_12", "ema_26",
        "fast_line", "slow_line", "macd_h"
    ]
    df_clean = df.dropna(subset=features + ["sell_filled"])
    X = df_clean[features]
    y = df_clean["sell_filled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nSell Fill Model Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    df = load_and_clean_data("data/SGLN Historical Data.csv")
    df = add_ema_macd_features(df)
    df = create_fill_labels(df)
    explore_buy_fill(df)
    explore_sell_fill(df)
    predict_buy_fill(df)
    predict_sell_fill(df)
