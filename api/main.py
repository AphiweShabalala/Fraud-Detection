
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import TransactionRequest


# --------------------------------------------------
# 1. Paths and artifact loading
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

PIPELINE_PATH = MODEL_DIR / "xgboost_fraud_pipeline.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"


def load_artifacts():
    """Load the trained pipeline, preprocessor, and threshold."""

    required_files = [
        PIPELINE_PATH,
        PREPROCESSOR_PATH,
        THRESHOLD_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Required model artifacts are missing: {missing_files}"
        )

    pipeline = joblib.load(PIPELINE_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)

    threshold = float(threshold_info["threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The classification threshold must be between 0 and 1.")

    return pipeline, preprocessor, threshold


pipeline, preprocessor, threshold = load_artifacts()


# --------------------------------------------------
# 2. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "An API that estimates the probability of credit-card "
        "transaction fraud using a trained XGBoost model."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 3. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Report whether the model artifacts loaded successfully."""

    return {
        "status": "healthy",
        "model": "XGBoost",
        "threshold": threshold,
    }


# --------------------------------------------------
# 4. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_transaction(transaction: TransactionRequest):
    """Predict whether a transaction is fraudulent."""

    try:
        # Convert the validated request into a dictionary.
        features = transaction.model_dump()

        # Reproduce the hour feature engineering used in training.
        hour = int((features["Time"] // 3600) % 24)

        features["Hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["Hour_cos"] = math.cos(2 * math.pi * hour / 24)

        # Build a DataFrame in the exact order expected by the
        # fitted preprocessor.
        input_df = pd.DataFrame(
            [features],
            columns=preprocessor.feature_names_in_,
        )

        # Apply the saved training-time transformations.
        transformed_features = preprocessor.transform(input_df)

        # Predict probability. SMOTE is skipped during inference
        # by the imbalanced-learn pipeline.
        fraud_probability = float(
            pipeline.predict_proba(transformed_features)[0, 1]
        )

        # Apply the saved classification threshold.
        prediction = int(fraud_probability >= threshold)

        return {
            "fraud_probability": fraud_probability,
            "prediction": prediction,
            "classification": (
                "fraud" if prediction == 1 else "legitimate"
            ),
            "threshold": threshold,
        }

    except Exception as exc:
        # Log the exception server-side in a later logging step.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc