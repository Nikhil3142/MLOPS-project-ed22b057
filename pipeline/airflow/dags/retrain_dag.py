"""
retrain_dag.py
─────────────────────────────────────────────────────────────────────────────
Airflow DAG — Drift Detection & Retraining Pipeline

Tasks:
  1. ingest_holdout      — load holdout_data.csv, apply same feature engineering
  2. detect_drift        — compare holdout distributions to saved baselines
                           using PSI (Population Stability Index)
  3. evaluate_production — run production model on holdout, compute RMSE
  4. should_retrain      — branch: retrain or skip
  5. trigger_retrain     — combine initial + holdout into initial_data.csv
                           and write retrain_triggered.txt as signal
  6. skip_retrain        — no-op if no drift/degradation detected

Triggers: manually (simulates new data arriving)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
HOLDOUT_RAW_PATH        = "/opt/airflow/data/raw/holdout_data.csv"
HOLDOUT_PROCESSED_PATH  = "/opt/airflow/data/processed/holdout_data.csv"
INITIAL_PROCESSED_PATH  = "/opt/airflow/data/processed/initial_data.csv"
BASELINES_PATH          = "/opt/airflow/data/drift_baselines/baselines.json"
DRIFT_REPORT_PATH       = "/opt/airflow/data/drift_baselines/drift_report.json"
TRIGGER_FILE_PATH       = "/opt/airflow/data/retrain_triggered.txt"

# ── Thresholds ────────────────────────────────────────────────────────────────
PSI_THRESHOLD        = 0.01   # lowered for demo — use 0.2 in production
RMSE_DEGRADATION_PCT = 10     # retrain if RMSE degrades > 10%

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
    actual_perc   = np.where(actual_perc   == 0, 0.001, actual_perc)

    expected_perc = expected_perc / expected_perc.sum()
    actual_perc   = actual_perc   / actual_perc.sum()

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
    df["sex"]    = df["sex"].map({"female": 1, "male": 0})

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
    Saves drift_report.json and pushes drift_detected to XCom.
    """
    logger.info("Starting drift detection")

    if not os.path.exists(BASELINES_PATH):
        raise FileNotFoundError(
            f"Baselines not found at {BASELINES_PATH}. "
            "Run the ingestion_dag first."
        )

    with open(BASELINES_PATH, "r") as f:
        baselines = json.load(f)

    holdout_df = pd.read_csv(HOLDOUT_PROCESSED_PATH)
    initial_df = pd.read_csv(INITIAL_PROCESSED_PATH)

    drift_scores  = {}
    drift_detected = False

    feature_cols = [
        c for c in holdout_df.columns
        if c != "charges" and not c.startswith("_")
    ]

    for col in feature_cols:
        if col not in initial_df.columns:
            continue

        psi = compute_psi(initial_df[col].values, holdout_df[col].values)
        drift_scores[col] = round(psi, 6)

        if psi >= PSI_THRESHOLD:
            logger.warning(f"DRIFT DETECTED in '{col}': PSI={psi:.4f}")
            drift_detected = True
        else:
            logger.info(f"  {col}: PSI={psi:.4f} (stable)")

    drift_report = {
        "computed_at":   datetime.utcnow().isoformat(),
        "drift_detected": drift_detected,
        "psi_threshold":  PSI_THRESHOLD,
        "scores":         drift_scores,
    }

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(drift_report, f, indent=2)

    logger.info(f"Drift detection complete. Drift detected: {drift_detected}")
    context["ti"].xcom_push(key="drift_detected",  value=drift_detected)
    context["ti"].xcom_push(key="drift_scores",    value=drift_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Evaluate Production Model on Holdout
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_production(**context):
    """
    Load the Production model directly from MLflow Registry,
    run batch inference on holdout data, and compute RMSE.
    Pushes rmse_degraded=True/False to XCom.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import mean_squared_error

    logger.info("Evaluating production model on holdout data")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Load model via alias
    try:
        model = mlflow.sklearn.load_model("models:/insurance_cost_model@production")
        logger.info("Production model loaded via alias 'production'")
    except Exception as e:
        logger.warning(f"Could not load production model: {e}. Skipping RMSE check.")
        context["ti"].xcom_push(key="rmse_degraded", value=False)
        context["ti"].xcom_push(key="holdout_rmse",  value=None)
        return

    # Load holdout data
    holdout_df   = pd.read_csv(HOLDOUT_PROCESSED_PATH)
    feature_cols = [c for c in holdout_df.columns if c != "charges"]
    X = holdout_df[feature_cols]
    y = holdout_df["charges"]

    # Batch inference
    predictions = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    logger.info(f"Holdout RMSE: {rmse:.2f}")

    # Get baseline RMSE from MLflow via alias
    baseline_rmse = None
    try:
        version_info  = client.get_model_version_by_alias("insurance_cost_model", "production")
        run           = client.get_run(version_info.run_id)
        baseline_rmse = float(run.data.metrics.get("rmse", rmse))
        logger.info(f"Production baseline RMSE: {baseline_rmse:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch baseline RMSE: {e}")

    # Check degradation
    rmse_degraded = False
    if baseline_rmse is not None:
        degradation_pct = ((rmse - baseline_rmse) / baseline_rmse) * 100
        logger.info(f"RMSE degradation: {degradation_pct:.1f}%")
        if degradation_pct > RMSE_DEGRADATION_PCT:
            logger.warning(
                f"RMSE degraded by {degradation_pct:.1f}% — retraining needed"
            )
            rmse_degraded = True

    context["ti"].xcom_push(key="rmse_degraded", value=rmse_degraded)
    context["ti"].xcom_push(key="holdout_rmse",  value=rmse)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Branch: retrain or skip
# ─────────────────────────────────────────────────────────────────────────────
def should_retrain(**context):
    """Branch operator — decides whether to retrain or skip."""
    ti = context["ti"]
    drift_detected = ti.xcom_pull(task_ids="detect_drift",        key="drift_detected")
    rmse_degraded  = ti.xcom_pull(task_ids="evaluate_production", key="rmse_degraded")

    if drift_detected or rmse_degraded:
        logger.info(
            f"Retraining triggered — drift={drift_detected}, "
            f"rmse_degraded={rmse_degraded}"
        )
        return "trigger_retrain"
    else:
        logger.info("No retraining needed — distributions stable, RMSE acceptable.")
        return "skip_retrain"


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Trigger Retraining
# ─────────────────────────────────────────────────────────────────────────────
def trigger_retrain(**context):
    """
    Combine initial + holdout data and overwrite initial_data.csv
    so that the next dvc repro run on the host retrains on the full dataset.
    Writes retrain_triggered.txt as a signal file.
    """
    ti             = context["ti"]
    drift_scores   = ti.xcom_pull(task_ids="detect_drift",        key="drift_scores")
    holdout_rmse   = ti.xcom_pull(task_ids="evaluate_production", key="holdout_rmse")
    drift_detected = ti.xcom_pull(task_ids="detect_drift",        key="drift_detected")
    rmse_degraded  = ti.xcom_pull(task_ids="evaluate_production", key="rmse_degraded")

    logger.info("Combining initial + holdout data for retraining")

    initial_df = pd.read_csv(INITIAL_PROCESSED_PATH)
    holdout_df = pd.read_csv(HOLDOUT_PROCESSED_PATH)

    before_rows = len(initial_df)
    combined_df = pd.concat([initial_df, holdout_df], ignore_index=True)

    # Overwrite initial_data.csv — this is what dvc repro will use
    combined_df.to_csv(INITIAL_PROCESSED_PATH, index=False)
    logger.info(
        f"initial_data.csv updated: {before_rows} → {len(combined_df)} rows"
    )

    # Write trigger file as signal for host to run dvc repro
    trigger_info = {
        "triggered_at":    datetime.utcnow().isoformat(),
        "reason":          {
            "drift_detected": drift_detected,
            "rmse_degraded":  rmse_degraded,
        },
        "drift_scores":    drift_scores,
        "holdout_rmse":    holdout_rmse,
        "rows_before":     before_rows,
        "rows_after":      len(combined_df),
        "action_required": "Run 'dvc repro' on host to retrain the model.",
    }

    with open(TRIGGER_FILE_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RETRAIN TRIGGERED\n")
        f.write("=" * 60 + "\n")
        f.write(f"Triggered at : {trigger_info['triggered_at']}\n")
        f.write(f"Drift detected: {drift_detected}\n")
        f.write(f"RMSE degraded : {rmse_degraded}\n")
        f.write(f"Holdout RMSE  : {holdout_rmse}\n")
        f.write(f"Rows before   : {before_rows}\n")
        f.write(f"Rows after    : {len(combined_df)}\n")
        f.write("\nDrift scores (PSI):\n")
        if drift_scores:
            for feat, score in drift_scores.items():
                f.write(f"  {feat}: {score}\n")
        f.write("\nACTION REQUIRED: Run 'dvc repro' on host to retrain.\n")

    logger.info(f"Trigger file written to: {TRIGGER_FILE_PATH}")
    logger.info("Data preparation complete. Run 'dvc repro' on host to retrain.")


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_dag",
    description="Drift detection and model retraining pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
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