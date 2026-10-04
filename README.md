# Fraud Detection API

An end-to-end machine learning system for detecting fraudulent credit card transactions using **XGBoost**, **SMOTE**, and **FastAPI**.

The project covers the complete machine learning lifecycle: data cleaning, preprocessing, class-imbalance handling, model training and evaluation, probability-threshold optimization, model serialization, REST API development, testing, and cloud deployment.

**Live API:** https://fraud-detection-ujad.onrender.com  
**Interactive API Documentation:** https://fraud-detection-ujad.onrender.com/docs

---

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem. In the dataset used for this project, fraudulent transactions represent only a small fraction of all transactions.

The objective was to build a model that could distinguish fraudulent transactions from legitimate ones while prioritizing **precision and recall**, rather than relying solely on accuracy.

The final system uses an **XGBoost classifier** with **SMOTE** oversampling and a decision threshold of **0.8** selected using out-of-fold F1-score maximization.

The trained model is exposed through a FastAPI REST API and deployed as a live cloud service.

---

## Key Features

- Data cleaning and duplicate analysis
- Class-imbalance analysis
- SMOTE oversampling
- XGBoost classification
- Hyperparameter optimization using `RandomizedSearchCV`
- Probability-threshold optimization using out-of-fold predictions
- Precision, recall, F1-score, ROC-AUC and PR-AUC evaluation
- Serialized production model and preprocessing pipeline
- FastAPI REST API
- Automated API tests
- Docker support for local/containerized deployment
- Cloud deployment using Render
- Interactive Swagger/OpenAPI documentation

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset containing transactions made by European cardholders.

The original dataset contains:

- **284,807 transactions**
- **31 columns**
- **492 fraudulent transactions**
- **28 anonymized PCA-transformed features (`V1`–`V28`)**
- `Time`
- `Amount`
- `Class` — target variable

Because fraud is extremely rare, accuracy alone is not an appropriate primary evaluation metric.

### Data Cleaning

The dataset contained duplicate transactions.

The analysis identified:

- **1,081 duplicate rows**
- **1,854 duplicated observations including first occurrences**
- **773 duplicate groups**
- **0 duplicate groups with conflicting fraud labels**

Duplicate observations were removed before model development.

---

## Feature Engineering

The original `Time` feature was transformed into an hourly representation.

The project extracts the hour of day and represents it using cyclical encoding:

- `Hour_sin`
- `Hour_cos`

This allows the model to represent the circular nature of time, where hour 23 and hour 0 are adjacent.

The final feature set therefore contains:

```text
V1–V28
Amount
Hour_sin
Hour_cos
```

The original `Time` and intermediate time-binning features are not used directly by the final model.

---

## Machine Learning Pipeline

The final classification pipeline consists of:

```text
Raw Transaction
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
SMOTE
       │
       ▼
XGBoost Classifier
       │
       ▼
Fraud Probability
       │
       ▼
Threshold = 0.8
       │
       ▼
Fraud / Legitimate
```

### Why SMOTE?

Fraudulent transactions are heavily underrepresented in the dataset.

**SMOTE (Synthetic Minority Over-sampling Technique)** was used during model training to increase representation of the minority class.

SMOTE was applied inside the machine-learning pipeline so that oversampling was performed only on the relevant training data and did not contaminate validation or test data.

### Why XGBoost?

XGBoost was selected because gradient-boosted decision trees perform well on structured/tabular data and can model nonlinear relationships between transaction features.

---

## Model Selection and Evaluation

Hyperparameter optimization was performed using `RandomizedSearchCV` with **average precision (PR-AUC)** as the optimization metric.

The selected XGBoost configuration was:

```text
n_estimators      = 300
max_depth         = 5
learning_rate     = 0.1
subsample         = 0.8
colsample_bytree  = 0.8
random_state      = 42
```

### Threshold Optimization

The default classification threshold of `0.5` was not used.

Instead, the decision threshold was selected using **5-fold stratified cross-validation** and out-of-fold predictions.

The threshold that maximized F1-score was:

```text
Threshold = 0.8
```

This threshold is stored separately in:

```text
models/threshold.json
```

---

## Final Test Performance

At the selected threshold of **0.8**, the final model achieved:

| Metric | Score |
|---|---:|
| Precision | 0.9359 |
| Recall | 0.7684 |
| F1-score | 0.8439 |
| ROC-AUC | 0.9721 |
| PR-AUC | 0.8135 |

### Confusion Matrix

```text
                 Predicted
                 Legit   Fraud
Actual Legit     56646      5
Actual Fraud        22     73
```

The model therefore identified **73 of the 95 fraudulent transactions** in the held-out test set while producing **5 false positives**.

---

## Project Structure

```text
Fraud-Detection/
│
├── api/
│   └── main.py
│
├── Data/
│   └── Raw/
│       └── creditcard.csv
│
├── models/
│   ├── xgboost_fraud_pipeline.joblib
│   ├── threshold.json
│   └── model_metadata.json
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── ...
│
├── tests/
│   └── ...
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

> The exact notebook filenames may differ depending on the final repository structure.

---

# FastAPI

The trained model is exposed through a REST API built with **FastAPI**.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "XGBoost",
  "threshold": 0.8
}
```

### Prediction

```http
POST /predict
```

The API accepts transaction features and returns the predicted fraud probability, binary prediction, classification, and threshold used.

Example response:

```json
{
  "fraud_probability": 0.00033160377643071115,
  "prediction": 0,
  "classification": "legitimate",
  "threshold": 0.8
}
```

A prediction of:

```text
0 = legitimate
1 = fraud
```

The API compares the predicted probability against the stored threshold of `0.8`.

---

## API Documentation

Interactive Swagger documentation is available at:

https://fraud-detection-ujad.onrender.com/docs

The deployed API can be used to inspect the available endpoints and submit prediction requests directly from the browser.

---

## Testing

The API was tested locally before deployment.

The test suite verified the core API functionality, including:

- Application startup
- Health endpoint
- Prediction endpoint
- Model loading
- Prediction response structure
- Classification behavior

The test suite passed successfully before deployment.

---

## Deployment

The API is deployed on **Render** without requiring a Docker image build.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

The deployment uses the serialized model artifacts stored in the `models/` directory.

### Production Health Check

The deployed service responds successfully to:

```http
GET https://fraud-detection-ujad.onrender.com/health
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/AphiweShabalala/Fraud-Detection.git
cd Fraud-Detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

## Technologies

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- imbalanced-learn / SMOTE
- Joblib

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Testing

- pytest

### Deployment

- Render
- Docker support

### Development

- Jupyter Notebook
- Git
- GitHub

---

## Key Engineering Decisions

### Precision and Recall over Accuracy

Because fraud is a rare event, a model that predicts every transaction as legitimate could achieve very high accuracy while being useless for fraud detection.

Therefore, the project focuses on:

- Precision
- Recall
- F1-score
- PR-AUC

alongside ROC-AUC.

### Threshold Optimization

The model outputs a probability rather than directly deciding whether a transaction is fraudulent.

The threshold was therefore treated as a model-deployment decision rather than automatically using `0.5`.

The selected threshold of `0.8` was obtained using out-of-fold predictions and F1-score maximization.

### Reproducibility

The final model, preprocessing pipeline, threshold and metadata are serialized so that the deployed API uses the same artifacts produced during model development.

---

## Limitations

This project is intended as a machine-learning portfolio and demonstration system rather than a production banking fraud platform.

Important limitations include:

- The dataset contains anonymized PCA-transformed features, limiting business interpretability.
- The dataset represents a historical transaction distribution and may not reflect current fraud patterns.
- Fraud detection is subject to concept drift.
- The selected threshold optimizes F1-score rather than a financial cost function.
- The API does not include authentication or authorization.
- The project does not implement real-time streaming ingestion.
- Production fraud systems would typically require monitoring, alerting, model retraining, drift detection, and stronger infrastructure controls.

---

## Future Improvements

Potential extensions include:

- Cost-sensitive threshold optimization
- Model calibration
- Probability calibration and reliability analysis
- Feature drift monitoring
- Model performance monitoring in production
- Automated model retraining
- Experiment tracking
- CI/CD
- API authentication
- Rate limiting
- Structured logging
- Database-backed transaction storage
- Real-time event-stream processing
- Dockerized production deployment
- Cloud monitoring and alerting

---

## Author

**Aphiwe Shabalala**

GitHub:  
https://github.com/AphiweShabalala

---

## License

This project is intended for educational and portfolio purposes.