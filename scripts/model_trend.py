"""
model_trend.py

Trend label creation for ETF weekly trend modelling.
Classifies weekly trends as Buy, Sell, or Sideways based on forward return.
Also prepares input features and trains Model 1.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add parent directory to sys.path to enable script-level imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_data import load_and_clean_data
from scripts.features import add_ema_macd_features

def create_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["next_close"] = df["close"].shift(-1)
    df["pct_change_next"] = ((df["next_close"] - df["close"]) / df["close"] * 100).round(2)

    def classify_trend(pct):
        if pd.isna(pct):
            return np.nan
        if pct >= 1.0:
            return "Buy"
        elif pct <= -1.0:
            return "Sell"
        else:
            return "Sideways"

    df["trend_label"] = df["pct_change_next"].apply(classify_trend)
    return df

def prepare_model_1_inputs(df: pd.DataFrame):
    features_model_1 = [
        "change_pct", "change_to_low", "change_to_high",
        "ema_5", "ema_12", "ema_26",
        "fast_line", "slow_line", "macd_h"
    ]
    X = df[features_model_1].copy()
    y = df["trend_label"].copy()
    return X, y

if __name__ == "__main__":
    df = load_and_clean_data("data/SGLN Historical Data.csv")
    df = add_ema_macd_features(df)
    df = create_trend_labels(df)

    df = df.dropna(subset=[
        "change_pct", "change_to_low", "change_to_high",
        "ema_5", "ema_12", "ema_26",
        "fast_line", "slow_line", "macd_h",
        "trend_label"
    ])

    X, y = prepare_model_1_inputs(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel 1 Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Trend label distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, order=y.value_counts().index)
    plt.title("Trend Label Distribution")
    plt.xlabel("Trend")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Feature importances
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx], orient='h')
    plt.title("Feature Importances (Decision Tree)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Decision tree plot
    plt.figure(figsize=(16, 8))
    plot_tree(model, feature_names=feature_names, class_names=model.classes_,
              filled=True, max_depth=2, fontsize=8)
    plt.title("Decision Tree Structure (Depth ≤ 2)")
    plt.show()

    # PCA projection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["label"] = y.values

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", alpha=0.7,
                    hue_order=["Buy", "Sell", "Sideways"])
    plt.title("PCA Projection of Model Features")
    plt.legend(title="Trend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
