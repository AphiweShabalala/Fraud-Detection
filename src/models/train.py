from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.config import load_config


def create_xgboost_pipeline(
    config:dict
) -> Pipeline:
    """
    Create the SMOTE + XGBoost training pipeline.

    Parameters
    ----------
    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    Pipeline
        Configured imbalanced-learn pipeline.
    """

    model_config = config["model"]
    smote_config = config["smote"]
     
    model = XGBClassifier(
        **model_config["parameters"],
        random_state=model_config['random_state']
    )

    pipeline = Pipeline(
        steps=[
            ("smote", SMOTE(**smote_config)),
            ("model", model)
        ]
    )

    return pipeline


def train_model(
    X_train,
    y_train,
   config:dict
) -> Pipeline:
    """
    Train the SMOTE + XGBoost pipeline.

    Parameters
    ----------
    X_train : array-like
        Preprocessed training features.
    y_train : array-like
        Training target.
    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    Pipeline
        Fitted SMOTE + XGBoost pipeline.
    """

    pipeline = create_xgboost_pipeline(
        config
    )

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline

import joblib

def save_model(
    model,
    path: str
) -> None:
    """
    Save a fitted model to disk.

    Parameters
    ----------
    model : object
        Fitted model or pipeline.
    path : str
        Destination path for the model artifact.
    """

    joblib.dump(
        model,
        path
    )