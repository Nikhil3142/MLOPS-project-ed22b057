"""
test_api.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for FastAPI endpoints.
Uses TestClient to test without running the server.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock model setup ──────────────────────────────────────────────────────────
@pytest.fixture
def mock_model():
    """Mock sklearn model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([12345.67])
    return model


@pytest.fixture
def client(mock_model):
    """Create test client with mocked model."""
    with patch("backend.model_loader._model", mock_model), \
         patch("backend.model_loader._model_name", "insurance_cost_model"), \
         patch("backend.model_loader._model_version", "1"):
        from backend.main import app
        with TestClient(app) as c:
            yield c


# ── Health endpoint tests ─────────────────────────────────────────────────────
class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        """Health endpoint returns ok status."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_returns_service_name(self, client):
        """Health endpoint returns service name."""
        response = client.get("/health")
        assert "service" in response.json()


# ── Ready endpoint tests ──────────────────────────────────────────────────────
class TestReadyEndpoint:
    """Tests for /ready endpoint."""

    def test_ready_returns_200(self, client):
        """Ready endpoint returns 200."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_when_model_loaded(self, client):
        """Ready endpoint returns ready when model is loaded."""
        response = client.get("/ready")
        assert response.json()["model_loaded"] is True

    def test_ready_returns_model_info(self, client):
        """Ready endpoint returns model name and version."""
        response = client.get("/ready")
        data = response.json()
        assert data["model_name"] == "insurance_cost_model"
        assert data["model_version"] is not None


# ── Predict endpoint tests ────────────────────────────────────────────────────
class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    @pytest.fixture
    def valid_payload(self):
        return {
            "age": 35,
            "sex": "male",
            "bmi": 28.5,
            "children": 2,
            "smoker": "no",
            "region": "northeast"
        }

    def test_predict_returns_200(self, client, valid_payload):
        """Predict endpoint returns 200 for valid input."""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_returns_charge(self, client, valid_payload):
        """Predict endpoint returns predicted charge."""
        response = client.post("/predict", json=valid_payload)
        assert "predicted_charge" in response.json()
        assert isinstance(response.json()["predicted_charge"], float)

    def test_predict_returns_feature_importance(self, client, valid_payload):
        """Predict endpoint returns SHAP feature importance."""
        response = client.post("/predict", json=valid_payload)
        assert "feature_importance" in response.json()

    def test_predict_returns_latency(self, client, valid_payload):
        """Predict endpoint returns latency."""
        response = client.post("/predict", json=valid_payload)
        assert "latency_ms" in response.json()

    def test_predict_invalid_sex(self, client):
        """Predict endpoint returns 422 for invalid sex."""
        payload = {
            "age": 35, "sex": "invalid", "bmi": 28.5,
            "children": 2, "smoker": "no", "region": "northeast"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_region(self, client):
        """Predict endpoint returns 422 for invalid region."""
        payload = {
            "age": 35, "sex": "male", "bmi": 28.5,
            "children": 2, "smoker": "no", "region": "invalid"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_age_out_of_range(self, client):
        """Predict endpoint returns 422 for age out of range."""
        payload = {
            "age": 5, "sex": "male", "bmi": 28.5,
            "children": 2, "smoker": "no", "region": "northeast"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_smoker_yes(self, client, valid_payload):
        """Predict endpoint works for smoker=yes."""
        valid_payload["smoker"] = "yes"
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_all_regions(self, client, valid_payload):
        """Predict endpoint works for all regions."""
        for region in ["northeast", "northwest", "southeast", "southwest"]:
            valid_payload["region"] = region
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 200


# ── Feedback endpoint tests ───────────────────────────────────────────────────
class TestFeedbackEndpoint:
    """Tests for /feedback endpoint."""

    @pytest.fixture
    def valid_feedback(self):
        return {
            "prediction_id": "test123",
            "actual_charge": 12000.0,
            "predicted_charge": 11500.0,
            "input_features": {
                "age": 35, "sex": "male", "bmi": 28.5,
                "children": 2, "smoker": "no", "region": "northeast"
            }
        }

    def test_feedback_returns_200(self, client, valid_feedback):
        """Feedback endpoint returns 200."""
        response = client.post("/feedback", json=valid_feedback)
        assert response.status_code == 200

    def test_feedback_returns_response(self, client, valid_feedback):
        """Feedback endpoint returns a response with status field."""
        response = client.post("/feedback", json=valid_feedback)
        assert "status" in response.json()
        assert "message" in response.json()


# ── Input validation tests ────────────────────────────────────────────────────
class TestInputValidation:
    """Tests for input validation."""

    def test_missing_required_field(self, client):
        """Returns 422 when required field is missing."""
        payload = {"age": 35, "sex": "male"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_data_type(self, client):
        """Returns 422 when wrong data type is provided."""
        payload = {
            "age": "thirty-five", "sex": "male", "bmi": 28.5,
            "children": 2, "smoker": "no", "region": "northeast"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422