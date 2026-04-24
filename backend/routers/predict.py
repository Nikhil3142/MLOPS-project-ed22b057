"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Prediction endpoint.
POST /predict — takes user inputs, returns predicted cost + SHAP values
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
import uuid
import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, HTTPException
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.model_loader import get_model, get_model_info, is_model_loaded
from backend.metrics import (
    prediction_requests_total,
    prediction_errors_total,
    prediction_latency_seconds,
    predicted_cost_value,
)

logger = logging.getLogger(__name__)
router = APIRouter()

FEATURE_COLUMNS = [
    "age", "sex", "bmi", "children", "smoker",
    "region_northwest", "region_southeast", "region_southwest"
]


def preprocess_input(request: PredictionRequest) -> pd.DataFrame:
    """
    Convert PredictionRequest to a feature DataFrame
    matching the format used during training.
    """
    # Binary encoding
    sex = 1 if request.sex == "female" else 0
    smoker = 1 if request.smoker == "yes" else 0

    # One-hot encode region (drop northeast as reference)
    region_northwest = 1 if request.region == "northwest" else 0
    region_southeast = 1 if request.region == "southeast" else 0
    region_southwest = 1 if request.region == "southwest" else 0

    features = {
        "age": request.age,
        "sex": sex,
        "bmi": request.bmi,
        "children": request.children,
        "smoker": smoker,
        "region_northwest": region_northwest,
        "region_southeast": region_southeast,
        "region_southwest": region_southwest,
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def compute_shap_importance(model, X: pd.DataFrame) -> dict:
    """
    Compute SHAP feature importance for a single prediction.
    Returns a dict of feature name -> SHAP value.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        importance = dict(zip(FEATURE_COLUMNS, shap_values[0].tolist()))
    except Exception:
        try:
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X, nsamples=50)
            importance = dict(zip(FEATURE_COLUMNS, shap_values[0].tolist()))
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            importance = {col: 0.0 for col in FEATURE_COLUMNS}
    return importance


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict annual insurance charges for an individual.
    Returns predicted cost and SHAP feature importance values.
    """
    prediction_requests_total.inc()

    if not is_model_loaded():
        prediction_errors_total.inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for the service to be ready."
        )

    start_time = time.perf_counter()

    try:
        model = get_model()
        model_name, model_version = get_model_info()

        # Preprocess input
        X = preprocess_input(request)

        # Predict
        prediction = model.predict(X)[0]
        predicted_charge = float(prediction)

        # Compute latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Compute SHAP values
        feature_importance = compute_shap_importance(model, X)

        # Update Prometheus metrics
        prediction_latency_seconds.observe(latency_ms / 1000)
        predicted_cost_value.observe(predicted_charge)

        logger.info(
            f"Prediction: ${predicted_charge:.2f} | "
            f"Latency: {latency_ms:.2f}ms"
        )

        return PredictionResponse(
            predicted_charge=predicted_charge,
            feature_importance=feature_importance,
            model_name=model_name or "unknown",
            model_version=str(model_version) if model_version else "unknown",
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        prediction_errors_total.inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))