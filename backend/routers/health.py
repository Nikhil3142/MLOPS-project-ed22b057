"""
health.py
─────────────────────────────────────────────────────────────────────────────
Health and readiness endpoints.
/health — liveness check (is the service running?)
/ready  — readiness check (is the model loaded?)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from fastapi import APIRouter
from backend.schemas.prediction import HealthResponse, ReadyResponse
from backend.model_loader import is_model_loaded, get_model_info

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Liveness check — returns 200 if the service is running.
    Used by Docker health checks and load balancers.
    """
    return HealthResponse(status="ok", service="insurance-mlops-backend")


@router.get("/ready", response_model=ReadyResponse, tags=["Health"])
def ready():
    """
    Readiness check — returns 200 if model is loaded and ready to serve.
    Returns 503 if model is not yet loaded.
    """
    model_loaded = is_model_loaded()
    model_name, model_version = get_model_info()

    return ReadyResponse(
        status="ready" if model_loaded else "not ready",
        model_loaded=model_loaded,
        model_name=model_name,
        model_version=str(model_version) if model_version else None,
    )