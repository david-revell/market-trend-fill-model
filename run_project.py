import os
import sys

# Add parent directory to sys.path to allow script imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))

from prepare_data import load_and_clean_data
from features import add_ema_macd_features
from wrap_up import add_trend_labels, features_model_1, prepare_model_inputs, train_model, wrap_up_summary


def main():
    # Load and prepare data
    df = load_and_clean_data("data/SGLN Historical Data.csv")
    df = add_ema_macd_features(df)
    df = add_trend_labels(df)

    # Prepare model inputs and train model 1
    X_model_1, y_model_1 = prepare_model_inputs(df)
    model = train_model(X_model_1, y_model_1)

    # Generate summary plots and exports (includes classification report)
    wrap_up_summary(df, model, X_model_1, y_model_1)


if __name__ == "__main__":
    main()
