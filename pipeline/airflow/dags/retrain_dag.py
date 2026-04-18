"""
retrain_dag.py
─────────────────────────────────────────────────────────────────────────────
Airflow DAG — Drift Detection & Retraining Pipeline

Tasks:
  1. ingest_holdout     — load holdout_data.csv, apply same feature engineering
  2. detect_drift       — compare holdout distributions to saved baselines
                          using PSI (Population Stability Index)
  3. evaluate_production— run production model on holdout, compute RMSE
  4. trigger_retrain    — if drift or RMSE degradation detected,
                          combine initial + holdout and trigger dvc repro

Triggers: scheduled (simulates new data arriving)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
HOLDOUT_RAW_PATH = "/opt/airflow/data/raw/holdout_data.csv"
HOLDOUT_PROCESSED_PATH = "/opt/airflow/data/processed/holdout_data.csv"
INITIAL_PROCESSED_PATH = "/opt/airflow/data/processed/initial_data.csv"
COMBINED_PATH = "/opt/airflow/data/processed/combined_data.csv"
BASELINES_PATH = "/opt/airflow/data/drift_baselines/baselines.json"

# ── Thresholds ────────────────────────────────────────────────────────────────
PSI_THRESHOLD = 0.2       # PSI > 0.2 = significant drift
RMSE_DEGRADATION_PCT = 10  # retrain if RMSE degrades > 10%

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — PSI Calculation
# ─────────────────────────────────────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.
    PSI < 0.1  = no significant change
    PSI < 0.2  = moderate change
    PSI >= 0.2 = significant drift
    """
    expected_perc, bin_edges = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bin_edges)

    # Avoid division by zero
    expected_perc = np.where(expected_perc == 0, 0.001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.001, actual_perc)

    expected_perc = expected_perc / expected_perc.sum()
    actual_perc = actual_perc / actual_perc.sum()

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Ingest Holdout Data
# ─────────────────────────────────────────────────────────────────────────────
def ingest_holdout(**context):
    """
    Load holdout_data.csv and apply the same feature engineering
    as the ingestion DAG (binary encode + one-hot region).
    """
    logger.info(f"Loading holdout data from: {HOLDOUT_RAW_PATH}")

    if not os.path.exists(HOLDOUT_RAW_PATH):
        raise FileNotFoundError(f"Holdout file not found: {HOLDOUT_RAW_PATH}")

    df = pd.read_csv(HOLDOUT_RAW_PATH)
    logger.info(f"Loaded {len(df)} holdout rows")

    # Apply same encoding as ingestion DAG
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"] = df["sex"].map({"female": 1, "male": 0})
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df.drop("region", axis=1), region_dummies], axis=1)

    os.makedirs(os.path.dirname(HOLDOUT_PROCESSED_PATH), exist_ok=True)
    df.to_csv(HOLDOUT_PROCESSED_PATH, index=False)
    logger.info(f"Holdout processed and saved to: {HOLDOUT_PROCESSED_PATH}")

    context["ti"].xcom_push(key="holdout_rows", value=len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Detect Drift
# ─────────────────────────────────────────────────────────────────────────────
def detect_drift(**context):
    """
    Compare holdout feature distributions to saved baselines using PSI.
    Pushes drift_detected=True/False to XCom.
    """
    logger.info("Starting drift detection")

    with open(BASELINES_PATH, "r") as f:
        baselines = json.load(f)

    holdout_df = pd.read_csv(HOLDOUT_PROCESSED_PATH)
    initial_df = pd.read_csv(INITIAL_PROCESSED_PATH)

    drift_scores = {}
    drift_detected = False

    feature_cols = [c for c in holdout_df.columns if c != "charges" and not c.startswith("_")]

    for col in feature_cols:
        if col not in initial_df.columns:
            continue

        psi = compute_psi(
            initial_df[col].values,
            holdout_df[col].values,
        )
        drift_scores[col] = psi

        if psi >= PSI_THRESHOLD:
            logger.warning(f"DRIFT DETECTED in '{col}': PSI={psi:.4f}")
            drift_detected = True
        else:
            logger.info(f"  {col}: PSI={psi:.4f} (stable)")

    # Save drift scores
    drift_report = {
        "computed_at": datetime.utcnow().isoformat(),
        "drift_detected": drift_detected,
        "psi_threshold": PSI_THRESHOLD,
        "scores": drift_scores,
    }
    drift_report_path = "/opt/airflow/data/drift_baselines/drift_report.json"
    with open(drift_report_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    logger.info(f"Drift detection complete. Drift detected: {drift_detected}")
    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    context["ti"].xcom_push(key="drift_scores", value=drift_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Evaluate Production Model on Holdout
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_production(**context):
    """
    Load the current Production model from MLflow Registry,
    run inference on holdout data, and compute RMSE.
    Pushes rmse_degraded=True/False to XCom.
    """
    import mlflow
    from sklearn.metrics import mean_squared_error

    logger.info("Evaluating production model on holdout data")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        model = mlflow.sklearn.load_model("models:/insurance_cost_model/Production")
    except Exception as e:
        logger.warning(f"Could not load production model: {e}. Skipping RMSE check.")
        context["ti"].xcom_push(key="rmse_degraded", value=False)
        return

    holdout_df = pd.read_csv(HOLDOUT_PROCESSED_PATH)
    feature_cols = [c for c in holdout_df.columns if c != "charges"]

    X = holdout_df[feature_cols]
    y = holdout_df["charges"]

    predictions = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    logger.info(f"Holdout RMSE: {rmse:.2f}")

    # Get baseline RMSE from MLflow
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("insurance_cost_model", stages=["Production"])
    baseline_rmse = None

    if versions:
        run = client.get_run(versions[0].run_id)
        baseline_rmse = float(run.data.metrics.get("rmse", rmse))
        logger.info(f"Production baseline RMSE: {baseline_rmse:.2f}")

    rmse_degraded = False
    if baseline_rmse:
        degradation_pct = ((rmse - baseline_rmse) / baseline_rmse) * 100
        logger.info(f"RMSE degradation: {degradation_pct:.1f}%")
        if degradation_pct > RMSE_DEGRADATION_PCT:
            logger.warning(f"RMSE degraded by {degradation_pct:.1f}% — retraining needed")
            rmse_degraded = True

    context["ti"].xcom_push(key="rmse_degraded", value=rmse_degraded)
    context["ti"].xcom_push(key="holdout_rmse", value=rmse)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Branch: retrain or skip
# ─────────────────────────────────────────────────────────────────────────────
def should_retrain(**context):
    """Branch operator — decides whether to retrain or skip."""
    ti = context["ti"]
    drift_detected = ti.xcom_pull(task_ids="detect_drift", key="drift_detected")
    rmse_degraded = ti.xcom_pull(task_ids="evaluate_production", key="rmse_degraded")

    if drift_detected or rmse_degraded:
        logger.info("Retraining triggered.")
        return "trigger_retrain"
    else:
        logger.info("No retraining needed.")
        return "skip_retrain"


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Trigger Retraining
# ─────────────────────────────────────────────────────────────────────────────
def trigger_retrain(**context):
    """
    Combine initial + holdout data and trigger dvc repro
    to retrain on the full dataset.
    """
    logger.info("Combining initial + holdout data for retraining")

    initial_df = pd.read_csv(INITIAL_PROCESSED_PATH)
    holdout_df = pd.read_csv(HOLDOUT_PROCESSED_PATH)

    combined_df = pd.concat([initial_df, holdout_df], ignore_index=True)
    combined_df.to_csv(COMBINED_PATH, index=False)
    logger.info(f"Combined dataset: {len(combined_df)} rows → {COMBINED_PATH}")

    # Replace initial_data.csv with combined for dvc repro
    combined_df.to_csv(INITIAL_PROCESSED_PATH, index=False)
    logger.info("Replaced initial_data.csv with combined data")

    # Trigger DVC pipeline
    logger.info("Triggering dvc repro...")
    result = subprocess.run(
        ["dvc", "repro"],
        capture_output=True,
        text=True,
        cwd="/opt/airflow",
    )
    if result.returncode != 0:
        logger.error(f"dvc repro failed:\n{result.stderr}")
        raise RuntimeError(f"dvc repro failed: {result.stderr}")

    logger.info(f"dvc repro completed:\n{result.stdout}")


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_dag",
    description="Drift detection and model retraining pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # simulates new data arriving weekly
    catchup=False,
    tags=["mlops", "retraining", "drift"],
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_holdout",
        python_callable=ingest_holdout,
        provide_context=True,
    )

    task_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
        provide_context=True,
    )

    task_evaluate = PythonOperator(
        task_id="evaluate_production",
        python_callable=evaluate_production,
        provide_context=True,
    )

    task_branch = BranchPythonOperator(
        task_id="should_retrain",
        python_callable=should_retrain,
        provide_context=True,
    )

    task_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=trigger_retrain,
        provide_context=True,
    )

    task_skip = DummyOperator(task_id="skip_retrain")

    # Pipeline order
    task_ingest >> task_drift >> task_evaluate >> task_branch
    task_branch >> task_retrain
    task_branch >> task_skip