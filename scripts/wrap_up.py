'''
wrap_up.py

Final project summary, model interpretation, and export of visuals/statistics.
Intended to support portfolio showcasing and project packaging.
'''

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Add parent directory to sys.path to allow script imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.prepare_data import load_and_clean_data
from scripts.features import add_ema_macd_features

# Trend classification helper
def add_trend_labels(df):
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

# Define features used for Model 1
features_model_1 = [
    "change_pct", "change_to_low", "change_to_high",
    "ema_5", "ema_12", "ema_26",
    "fast_line", "slow_line", "macd_h"
]

def prepare_model_inputs(df):
    X = df[features_model_1].copy()
    y = df["trend_label"].copy()
    return X, y

def train_model(X, y):
    df_clean = pd.concat([X, y], axis=1).dropna()
    X_clean = df_clean[X.columns]
    y_clean = df_clean[y.name]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("\nModel 1 Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))

    print("\nClassification Report:")
    report = classification_report(y_test, model.predict(X_test))
    print(report)

    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/classification_report.txt", "w") as f:
        f.write(report)
    print("Saved: outputs/reports/classification_report.txt")

    return model

def save_plot(fig, name):
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_path}")

def wrap_up_summary(df, model, X, y):
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/data", exist_ok=True)

    # Drop rows with any missing values in the input matrix
    X = X.dropna()
    y = y[X.index]  # Keep aligned labels

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, order=y.value_counts().index)
    plt.title("Trend Label Distribution")
    plt.xlabel("Trend")
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(plt.gcf(), "trend_label_distribution")
    plt.close()

    # Feature importances
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx], orient="h")
    plt.title("Feature Importances (Decision Tree)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    save_plot(plt.gcf(), "feature_importances")
    plt.close()

    # Decision tree (depth=2)
    plt.figure(figsize=(16, 8))
    plot_tree(model, feature_names=feature_names, class_names=model.classes_,
              filled=True, max_depth=2, fontsize=8)
    plt.title("Decision Tree Structure (Depth ≤ 2)")
    save_plot(plt.gcf(), "decision_tree_depth2")
    plt.close()

    # PCA projection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["label"] = y.values
    df_pca.to_csv("outputs/data/pca_projection.csv", index=False)
    print("Saved: outputs/data/pca_projection.csv")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", alpha=0.7,
                    hue_order=["Buy", "Sell", "Sideways"])
    plt.title("PCA Projection of Model Features")
    plt.legend(title="Trend", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    save_plot(plt.gcf(), "pca_projection")
    plt.close()

    print("Wrap-up visuals and data saved.")

if __name__ == "__main__":
    df = load_and_clean_data("data/SGLN Historical Data.csv")
    df = add_ema_macd_features(df)
    df = add_trend_labels(df)
    X_model_1, y_model_1 = prepare_model_inputs(df)
    model = train_model(X_model_1, y_model_1)
    wrap_up_summary(df, model, X_model_1, y_model_1)
