"""
main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entrypoint.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from backend.routers import predict, health, feedback
from backend.model_loader import load_model
from backend.schemas.prediction import ModelInfoResponse
from backend.model_loader import get_model_info, is_model_loaded
import mlflow
from mlflow.tracking import MlflowClient

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Insurance Cost Prediction API",
    description="MLOps pipeline for predicting annual medical insurance charges",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(feedback.router)


# ── Startup event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load the Production model from MLflow on startup."""
    logger.info("Starting Insurance Cost Prediction API")
    load_model()
    logger.info("API startup complete")


# ── Model info endpoint ───────────────────────────────────────────────────────
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return current Production model name, version and RMSE."""
    model_name, model_version = get_model_info()

    if not is_model_loaded():
        return ModelInfoResponse(
            model_name="none",
            model_version="none",
            model_stage="not loaded",
            rmse=None,
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        version_info = client.get_model_version_by_alias(model_name, "production")
        run = client.get_run(version_info.run_id)
        rmse = run.data.metrics.get("rmse", None)
    except Exception:
        rmse = None

    return ModelInfoResponse(
        model_name=model_name or "unknown",
        model_version=str(model_version) if model_version else "unknown",
        model_stage="Production",
        rmse=rmse,
    )

@app.post("/reload", tags=["Model"])
def reload_model():
    """Reload the Production model from MLflow Registry without restarting."""
    logger.info("Manual model reload requested")
    load_model()
    model_name, model_version = get_model_info()
    if is_model_loaded():
        return {"status": "reloaded", "model_name": model_name, "model_version": model_version}
    else:
        raise HTTPException(status_code=503, detail="Model reload failed")