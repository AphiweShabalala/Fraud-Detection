import json
import joblib
import pandas as pd

from src.features.engineering import create_time_features


def load_model(path: str):
    return joblib.load(path)


def load_threshold(path: str) -> float:
    with open(path, "r") as file:
        threshold_config = json.load(file)

    return threshold_config["threshold"]


def predict_transaction(
    df: pd.DataFrame,
    preprocessor,
    model,
    threshold: float
) -> dict:

    # Create the same features used during training
    df_engineered = create_time_features(df)

    # Transform using the fitted training preprocessor
    X_processed = preprocessor.transform(df_engineered)

    # Get fraud probabilities
    fraud_probability = model.predict_proba(X_processed)[:, 1]

    # Apply the production threshold
    predictions = (fraud_probability >= threshold).astype(int)

    return {
        "fraud_probability": float(fraud_probability[0]),
        "prediction": int(predictions[0])
    }