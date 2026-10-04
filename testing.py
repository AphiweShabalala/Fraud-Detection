import joblib
import pandas as pd

from src.features.engineering import create_time_features
from src.preprocessing.preprocessing import (
    load_preprocessor,
    transform_features
)

preprocessor = load_preprocessor(
    "models/preprocessor.joblib"
)

from sklearn.model_selection import train_test_split


df_clean = pd.read_csv(
    "Data/Processed/creditcard_clean.csv"
)

print("\nCleaned dataset shape:")
print(df_clean.shape)


df_clean_features = create_time_features(df_clean)

X_clean = df_clean_features.drop(columns=["Class"])
y_clean = df_clean_features["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X_clean,
    y_clean,
    test_size=0.20,
    stratify=y_clean,
    random_state=42
)

print("\nRecreated training shape:")
print(X_train.shape)

print("\nRecreated test shape:")
print(X_test.shape)

X_train_processed_new = transform_features(
    X_train,
    preprocessor
)
print("X_train_processed_new shape:", X_train_processed_new.shape)
print("y_train shape:", y_train.shape)
print("Class distribution:")
print(y_train.value_counts())
print("\nProduction-transformed training shape:")
print(X_train_processed_new.shape)

X_train_processed_saved = pd.read_csv(
    "Data/Processed/X_train_processed.csv"
).values

import numpy as np

print("\nSaved training shape:")
print(X_train_processed_saved.shape)

print("\nProduction training shape:")
print(X_train_processed_new.shape)

print("\nArrays equal:")
print(np.allclose(
    X_train_processed_saved,
    X_train_processed_new
))

max_difference = np.max(
    np.abs(
        X_train_processed_saved -
        X_train_processed_new
    )
)

print("\nMaximum numerical difference:")
print(max_difference)

#////////////////////////////////////////////////////

from src.data.validation import validate_required_columns

required_columns = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class"
]

validate_required_columns(
    df_clean,
    required_columns
)

print("\nSchema validation:")
print("PASSED")


df_invalid = df_clean.drop(columns=["Amount"])

try:
    validate_required_columns(
        df_invalid,
        required_columns
    )
except ValueError as e:
    print("\nInvalid schema test:")
    print("PASSED")
    print(e)

##////////////////////////////////////////////////////////

from src.data.validation import (
    validate_required_columns,
    validate_no_missing_values,
    validate_target,
    validate_numeric_columns,
    validate_model_input
)


validate_no_missing_values(df_clean)

print("\nMissing-value validation:")
print("PASSED")


validate_target(df_clean)

print("\nTarget validation:")
print("PASSED")


df_missing = df_clean.copy()

df_missing.loc[0, "Amount"] = None

try:
    validate_no_missing_values(df_missing)
except ValueError as e:
    print("\nInvalid missing-value test:")
    print("PASSED")
    print(e)

df_invalid_target = df_clean.copy()

df_invalid_target.loc[0, "Class"] = 2

try:
    validate_target(df_invalid_target)
except ValueError as e:
    print("\nInvalid target test:")
    print("PASSED")
    print(e)


feature_columns = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Hour_sin",
    "Hour_cos"
]
validate_numeric_columns(
    df_clean_features,
    feature_columns
)

print("\nNumeric validation:")
print("PASSED")

df_invalid_type = df_clean_features.copy()

df_invalid_type["Amount"] = df_invalid_type["Amount"].astype(str)

try:
    validate_numeric_columns(
        df_invalid_type,
        feature_columns
    )
except ValueError as e:
    print("\nInvalid numeric-type test:")
    print("PASSED")
    print(e)


feature_columns = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Hour_sin",
    "Hour_cos"
]

X_model_input = df_clean_features[feature_columns]

validate_model_input(
    X_model_input,
    feature_columns
)

print("\nModel-input validation:")
print("PASSED")

df_missing_feature = X_model_input.drop(
    columns=["Hour_cos"]
)

try:
    validate_model_input(
        df_missing_feature,
        feature_columns
    )
except ValueError as e:
    print("\nMissing model-input feature test:")
    print("PASSED")
    print(e)

df_extra_feature = X_model_input.copy()

df_extra_feature["ExtraFeature"] = 0

try:
    validate_model_input(
        df_extra_feature,
        feature_columns
    )
except ValueError as e:
    print("\nUnexpected model-input feature test:")
    print("PASSED")
    print(e)

##/////////////////////////////////////////////////////////

from src.models.train import create_xgboost_pipeline
from src.config import load_config
config = load_config()
#pipeline = create_xgboost_pipeline(config)

#print("\nTraining pipeline:")
#print(pipeline)

#print("\nPipeline steps:")
#print(pipeline.named_steps)

#print("\nXGBoost parameters:")

#model = pipeline.named_steps["model"]

#print("n_estimators:", model.n_estimators)
#print("max_depth:", model.max_depth)
#print("learning_rate:", model.learning_rate)
#print("subsample:", model.subsample)
#print("colsample_bytree:", model.colsample_bytree)
#print("random_state:", model.random_state)


from src.models.train import (
    create_xgboost_pipeline,
    train_model
)

from sklearn.model_selection import train_test_split

# Small training subset for testing
X_train_sample = X_train_processed_new[:5000]
y_train_sample = y_train.iloc[:5000]

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train_processed_new,
    y_train,
    train_size=5000,
    stratify=y_train,
    random_state=42
)

print("\nTraining small test model...")

test_model = train_model(
    X_train_sample,
    y_train_sample,
    config
)

print("Training completed.")
print("Returned model type:")
print(type(test_model))

y_train.iloc[:5000]


print("\nFitted pipeline steps:")
print(test_model.named_steps.keys())

print("\nSMOTE:")
print(test_model.named_steps["smote"])

print("\nXGBoost:")
print(test_model.named_steps["model"])

print("\nXGBoost fitted estimators:")
print(test_model.named_steps["model"].n_estimators)

sample_predictions = test_model.predict(
    X_train_sample[:10]
)

sample_probabilities = test_model.predict_proba(
    X_train_sample[:10]
)[:, 1]

print("\nSample predictions:")
print(sample_predictions)

print("\nSample fraud probabilities:")
print(sample_probabilities)

###//////////////////////////////////////////////////////

from src.models.train import (
    create_xgboost_pipeline,
    train_model,
    save_model
)

test_model_path = "models/test_xgboost_pipeline.joblib"

save_model(
    test_model,
    test_model_path
)

print("\nModel saved to:")
print(test_model_path)

import os

print("\nFile exists:")
print(os.path.exists(test_model_path))

loaded_test_model = joblib.load(
    test_model_path
)

original_predictions = test_model.predict(
    X_train_sample[:10]
)

loaded_predictions = loaded_test_model.predict(
    X_train_sample[:10]
)

print("\nPredictions match:")
print(
    np.array_equal(
        original_predictions,
        loaded_predictions
    )
)

original_probabilities = test_model.predict_proba(
    X_train_sample[:10]
)[:, 1]

loaded_probabilities = loaded_test_model.predict_proba(
    X_train_sample[:10]
)[:, 1]

print("\nProbabilities match:")
print(
    np.allclose(
        original_probabilities,
        loaded_probabilities
    )
)

from src.config import load_config

config = load_config()

print(config)

print(config["model"]["parameters"])

print(config["smote"])

from src.config import load_config
from src.models.train import create_xgboost_pipeline

config = load_config()

pipeline = create_xgboost_pipeline(config)

print(pipeline)


model = pipeline.named_steps["model"]
smote = pipeline.named_steps["smote"]

print("XGBoost parameters:")
print(model.get_params()["n_estimators"])
print(model.get_params()["max_depth"])
print(model.get_params()["learning_rate"])
print(model.get_params()["subsample"])
print(model.get_params()["colsample_bytree"])
print(model.get_params()["random_state"])

print("\nSMOTE random_state:")
print(smote.get_params()["random_state"])

config = load_config()

production_model = train_model(
    X_train_processed_new,
    y_train,
    config
)

#///////////////////////////////////////////////

print("\nTraining production model...")

config = load_config()

production_model = train_model(
    X_train_processed_new,
    y_train,
    config
)

print("Production training completed.")

print("\nProduction model type:")
print(type(production_model))

print("\nPipeline steps:")
print(production_model.named_steps.keys())

print("\nSMOTE:")
print(production_model.named_steps["smote"])

print("\nXGBoost:")
print(production_model.named_steps["model"])

print("\nFitted estimators:")
print(production_model.named_steps["model"].n_estimators)

model_path = config["paths"]["model"]

save_model(
    production_model,
    model_path
)

print("\nProduction model saved to:")
print(model_path)

import os

print("\nFile exists:")
print(os.path.exists(model_path))

print("File size (bytes):")
print(os.path.getsize(model_path))

import joblib
import numpy as np

loaded_model = joblib.load(model_path)

original_probabilities = production_model.predict_proba(
    X_train_processed_new[:100]
)

loaded_probabilities = loaded_model.predict_proba(
    X_train_processed_new[:100]
)

print("\nPredictions match:")
print(
    np.allclose(
        original_probabilities,
        loaded_probabilities
    )
)

print("\nMaximum probability difference:")
print(
    np.max(
        np.abs(
            original_probabilities - loaded_probabilities
        )
    )
)

from src.data.ingestion import load_data
from src.data.validation import (
    validate_required_columns,
    validate_no_missing_values
)

df = load_data(config["paths"]["raw_data"])

required_columns = [
    "Time",
    "Amount",
    "Class"
] + [f"V{i}" for i in range(1, 29)]

validate_required_columns(df, required_columns)
validate_no_missing_values(df)

print("Data ingestion and validation passed.")
print("Shape:", df.shape)

