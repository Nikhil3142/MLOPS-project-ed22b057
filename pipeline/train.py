"""
train.py
─────────────────────────────────────────────────────────────────────────────
DVC Stage 2 — Model Training

Input:  data/processed/train.csv
        params.yaml (active_model + hyperparameters)
Output: models/artifacts/model_info.json  (run_id for evaluate stage)

Dynamically imports the active model from pipeline/models/<active_model>.py
Logs params, git hash, dataset hash, and model artifact to MLflow.
─────────────────────────────────────────────────────────────────────────────
"""

import hashlib
import importlib
import json
import logging
import os
import subprocess
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "artifacts", "model_info.json")
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def get_git_hash() -> str:
    """Get current git commit hash for reproducibility tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_dataset_hash(path: str) -> str:
    """Compute MD5 hash of dataset file to track which data was used."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_training_data(path: str):
    """Load and separate features/target from training CSV."""
    df = pd.read_csv(path)
    X = df.drop("charges", axis=1)
    y = df["charges"]
    logger.info(f"Training data: {len(df)} rows, {len(X.columns)} features")
    return X, y


def main():
    """Main training pipeline."""
    logger.info("Starting training stage")

    # Load params
    params = load_params()
    active_model = params["active_model"]
    hyperparams = params["hyperparameters"][active_model]
    mlflow_config = params["mlflow"]

    logger.info(f"Active model: {active_model}")
    logger.info(f"Hyperparameters: {hyperparams}")

    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Load training data
    X_train, y_train = load_training_data(TRAIN_PATH)

    # Dynamically import the active model
    module = importlib.import_module(f"pipeline.models.{active_model}")
    model = module.get_model(hyperparams)
    logger.info(f"Model loaded: {model.__class__.__name__}")

    # Train and log to MLflow
    with mlflow.start_run(run_name=active_model) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # Train
        model.fit(X_train, y_train)
        logger.info("Model training complete")

        # Log hyperparameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("active_model", active_model)

        # Log custom tags for reproducibility
        mlflow.set_tag("git_commit", get_git_hash())
        mlflow.set_tag("dataset_hash", get_dataset_hash(TRAIN_PATH))
        mlflow.set_tag("model_class", model.__class__.__name__)

        # Log model artifact
        mlflow.sklearn.log_model(model, active_model)
        logger.info("Model artifact logged to MLflow")

    # Save run_id for evaluate stage
    os.makedirs(os.path.dirname(MODEL_INFO_PATH), exist_ok=True)
    model_info = {
        "run_id": run_id,
        "active_model": active_model,
        "hyperparameters": hyperparams,
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Model info saved to: {MODEL_INFO_PATH}")
    logger.info("Training stage complete")


if __name__ == "__main__":
    main()