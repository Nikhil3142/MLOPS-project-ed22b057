"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
DVC Stage 3 — Model Evaluation

Input:  data/processed/test.csv
        models/artifacts/model_info.json  (run_id from train stage)
Output: models/artifacts/shap_values.pkl
        metrics.json

Logs RMSE, MAE, R², inference latency, SHAP values,
drift baselines, and git hash to MLflow.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import time
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "artifacts", "model_info.json")
SHAP_PATH = os.path.join(BASE_DIR, "models", "artifacts", "shap_values.pkl")
BASELINES_PATH = os.path.join(BASE_DIR, "data", "drift_baselines", "baselines.json")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def load_test_data(path: str):
    """Load and separate features/target from test CSV."""
    df = pd.read_csv(path)
    X = df.drop("charges", axis=1)
    y = df["charges"]
    logger.info(f"Test data: {len(df)} rows")
    return X, y


def measure_latency(model, X: pd.DataFrame, n_samples: int = 100) -> float:
    """
    Measure average inference latency in milliseconds
    by running n_samples single-row predictions.
    """
    latencies = []
    for i in range(min(n_samples, len(X))):
        row = X.iloc[[i]]
        start = time.perf_counter()
        model.predict(row)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    return float(np.mean(latencies))


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """
    Compute SHAP values for explainability.
    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        logger.info("SHAP TreeExplainer used")
    except Exception:
        logger.info("Falling back to SHAP KernelExplainer")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50))
        shap_values = explainer(X)
    return shap_values


def main():
    """Main evaluation pipeline."""
    logger.info("Starting evaluation stage")

    params = load_params()
    mlflow_config = params["mlflow"]
    latency_threshold = params["thresholds"]["max_inference_latency_ms"]

    # Load model info from train stage
    with open(MODEL_INFO_PATH, "r") as f:
        model_info = json.load(f)

    run_id = model_info["run_id"]
    active_model = model_info["active_model"]
    logger.info(f"Evaluating run: {run_id} | Model: {active_model}")

    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    # Load model from MLflow
    model_uri = f"runs:/{run_id}/{active_model}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Model loaded from MLflow: {model_uri}")

    # Load test data
    X_test, y_test = load_test_data(TEST_PATH)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    latency_ms = measure_latency(model, X_test)

    logger.info(f"RMSE:    {rmse:.2f}")
    logger.info(f"MAE:     {mae:.2f}")
    logger.info(f"R²:      {r2:.4f}")
    logger.info(f"Latency: {latency_ms:.2f}ms (threshold: {latency_threshold}ms)")

    if latency_ms > latency_threshold:
        logger.warning(f"Latency {latency_ms:.2f}ms exceeds threshold {latency_threshold}ms")

    # Compute SHAP values
    shap_values = compute_shap_values(model, X_test)
    joblib.dump(shap_values, SHAP_PATH)
    logger.info(f"SHAP values saved to: {SHAP_PATH}")

    # Log everything to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("inference_latency_ms", latency_ms)

        # Log SHAP as artifact
        mlflow.log_artifact(SHAP_PATH, artifact_path="explainability")

        # Log drift baselines as artifact if they exist
        if os.path.exists(BASELINES_PATH):
            mlflow.log_artifact(BASELINES_PATH, artifact_path="drift")
            logger.info("Drift baselines logged to MLflow")

    # Save metrics.json for DVC
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "inference_latency_ms": latency_ms,
        "model": active_model,
        "run_id": run_id,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to: {METRICS_PATH}")
    logger.info("Evaluation stage complete")


if __name__ == "__main__":
    main()