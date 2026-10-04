
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def create_preprocessor() -> ColumnTransformer:
   amount_transformer = Pipeline([
    (
        "log",
        FunctionTransformer(
            np.log1p,
            feature_names_out="one-to-one"
        )
    ),
    ("scaler", StandardScaler()) 
    ])
   
   numeric_features = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Hour_sin",
        "Hour_cos"
    ]

   preprocessor = ColumnTransformer(
        transformers=[
            (
                "amount",
                amount_transformer,
                ["Amount"]
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        ]
    )

   return preprocessor


def fit_preprocessor(X_train: pd.DataFrame):
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(df: pd.DataFrame, preprocessor):
    return preprocessor.transform(df)


def save_preprocessor(preprocessor, path: str) -> None:
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str):
    return joblib.load(path)