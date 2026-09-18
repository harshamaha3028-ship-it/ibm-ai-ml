import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

def test_api_predict_valid():
    payload = {
        "cylinders": 4,
        "displacement": 120.0,
        "horsepower": 80.0,
        "weight": 2200.0,
        "acceleration": 17.0,
        "model_year": 78,
        "origin": "USA"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_mpg" in data
    assert data["predicted_mpg"] > 10.0
    assert "efficiency_category" in data
    assert len(data["feature_contributions"]) > 0

def test_api_predict_validation_error():
    # Invalid cylinders <= 0 and negative displacement
    payload = {
        "cylinders": 0,
        "displacement": -50.0,
        "horsepower": 80.0,
        "weight": 2200.0,
        "acceleration": 17.0,
        "model_year": 78,
        "origin": "USA"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "Validation Error" in data["error"]

def test_api_analyze_and_history():
    payload = {
        "cylinders": 6,
        "displacement": 230.0,
        "horsepower": 105.0,
        "weight": 3200.0,
        "acceleration": 15.5,
        "model_year": 77,
        "origin": "USA",
        "car_name": "API Test Midsize"
    }
    res = client.post("/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_mpg" in data
    assert "vehicle_profile" in data
    assert "cluster_id" in data
    assert "key_characteristics" in data

    # Verify recorded in /predictions
    h_res = client.get("/predictions?limit=1")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["total"] >= 1
    assert len(h_data["items"]) >= 1

def test_api_clusters():
    res = client.get("/clusters")
    assert res.status_code == 200
    data = res.json()
    assert "clusters" in data
    assert len(data["clusters"]) >= 2
    assert "features_used" in data

def test_api_statistics():
    res = client.get("/statistics")
    assert res.status_code == 200
    data = res.json()
    assert "dataset_summary" in data
    assert "model_summary" in data
    assert "clustering_summary" in data

def test_api_model_info():
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "best_model" in data
    assert "models_comparison" in data
    assert "feature_importances" in data
