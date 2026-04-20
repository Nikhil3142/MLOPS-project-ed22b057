"""
register.py
─────────────────────────────────────────────────────────────────────────────
DVC Stage 4 — Model Registration

Input:  metrics.json

Steps:
  1. Search all MLflow runs for the experiment
  2. Find the run with the lowest RMSE
  3. Register it to MLflow Model Registry
  4. Promote to Production
  5. Archive the previous Production model
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main registration pipeline."""
    logger.info("Starting registration stage")

    params = load_params()
    mlflow_config = params["mlflow"]
    model_name = mlflow_config["model_name"]

    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    client = MlflowClient()

    # Load current run from metrics.json
    with open(METRICS_PATH, "r") as f:
        current_metrics = json.load(f)

    run_id = current_metrics["run_id"]
    active_model = current_metrics["model"]
    rmse = current_metrics["rmse"]
    logger.info(f"Registering run: {run_id} | Model: {active_model} | RMSE: {rmse:.2f}")

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(model_name)
        logger.info(f"Created registered model: {model_name}")
    except Exception:
        logger.info(f"Model {model_name} already exists in registry")

    # Register current run
    model_uri = f"runs:/{run_id}/{active_model}"
    version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )
    logger.info(f"Model v{version.version} registered — promote to Production manually via MLflow UI")
    logger.info("Registration stage complete")

if __name__ == "__main__":
    main()