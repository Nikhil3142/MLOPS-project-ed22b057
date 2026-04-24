"""
model_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads the Production model from MLflow Model Registry on startup.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
_model = None
_model_name = None
_model_version = None


def load_model():
    """
    Load the Production model from MLflow Registry.
    Called once on FastAPI startup.
    """
    global _model, _model_name, _model_version

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    model_name = os.getenv("MODEL_NAME", "insurance_cost_model")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    logger.info(f"Loading model '{model_name}' from MLflow at {tracking_uri}")

    try:
        # Try loading by alias 'Production' first (new MLflow way)
        model_uri = f"models:/{model_name}@production"
        _model = mlflow.sklearn.load_model(model_uri)
        
        # Get version info
        version_info = client.get_model_version_by_alias(model_name, "production")
        _model_version = version_info.version
        _model_name = model_name

        logger.info(f"Model loaded: {model_name} v{_model_version}")

    except Exception as e:
        logger.warning(f"Could not load by alias: {e}")
        logger.warning("No Production model found. Service will start but /predict will fail.")
        _model = None
        _model_name = None
        _model_version = None


def get_model():
    """Return the loaded model."""
    return _model


def get_model_info():
    """Return model name and version."""
    return _model_name, _model_version


def is_model_loaded():
    """Return True if model is loaded and ready."""
    return _model is not None