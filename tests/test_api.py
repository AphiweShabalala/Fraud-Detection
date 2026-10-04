import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "XGBoost"
    assert data["threshold"] == 0.8

@pytest.fixture
def valid_transaction():
    return {
        "Time": 61290.0,
        "V1": 1.2288211502,
        "V2": -0.0634077165,
        "V3": 0.2741451422,
        "V4": 0.6474650218,
        "V5": -0.0481345612,
        "V6": 0.3720730286,
        "V7": -0.2242305874,
        "V8": 0.0799390492,
        "V9": 0.6407588171,
        "V10": -0.2730537022,
        "V11": -1.2527279388,
        "V12": 0.4650787707,
        "V13": 0.4005021153,
        "V14": -0.2928418606,
        "V15": -0.101774016,
        "V16": -0.3998358978,
        "V17": 0.0343356568,
        "V18": -0.7835502549,
        "V19": 0.1413449004,
        "V20": -0.0965659024,
        "V21": -0.1295544481,
        "V22": -0.0837793282,
        "V23": -0.1516614739,
        "V24": -0.7003715973,
        "V25": 0.5985501645,
        "V26": 0.4914090706,
        "V27": 0.0029892597,
        "V28": 0.0017822861,
        "Amount": 11.5,
    }

def test_predict_valid_transaction(valid_transaction):
    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "prediction" in data
    assert "classification" in data
    assert "threshold" in data

    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["threshold"] == 0.8

def test_prediction_matches_notebook_reference(valid_transaction):
    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 200

    data = response.json()

    expected_probability = 2.2671161787002347e-05

    assert data["fraud_probability"] == pytest.approx(
        expected_probability,
        rel=1e-9,
        abs=1e-12,
    )

    assert data["prediction"] == 0
    assert data["classification"] == "legitimate"

def test_negative_amount_rejected(valid_transaction):
    valid_transaction["Amount"] = -10

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 422

def test_extra_field_rejected(valid_transaction):
    valid_transaction["Class"] = 0

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 422
