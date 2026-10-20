from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_available" in response.json()

def test_prediction_input_validation():
    response = client.post("/predict", json={"study_hours": -1, "attendance": 75})
    assert response.status_code == 422
