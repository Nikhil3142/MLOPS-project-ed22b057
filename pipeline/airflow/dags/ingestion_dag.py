"""
ingestion_dag.py
─────────────────────────────────────────────────────────────────────────────
Airflow DAG — Data Ingestion Pipeline

Tasks:
  1. validate_data     — schema check, null check, value range check
  2. clean_data        — encode categoricals, handle outliers
  3. feature_engineer  — create encoded features, scale prep
  4. compute_baselines — compute mean/variance/distribution per feature
                         and save to data/drift_baselines/baselines.json

Triggers: manually or on schedule
Input:  data/raw/initial_data.csv
Output: data/processed/initial_data.csv + data/drift_baselines/baselines.json
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = "/opt/airflow/data/raw/initial_data.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/initial_data.csv"
BASELINES_PATH = "/opt/airflow/data/drift_baselines/baselines.json"

# ── Schema definition ─────────────────────────────────────────────────────────
EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
NUMERIC_COLUMNS = ["age", "bmi", "children", "charges"]
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]

VALUE_RANGES = {
    "age": (18, 100),
    "bmi": (10.0, 60.0),
    "children": (0, 10),
    "charges": (0, 200000),
}

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Validate Data
# ─────────────────────────────────────────────────────────────────────────────
def validate_data(**context):
    """
    Validate schema, null values, and value ranges of the raw input file.
    Raises ValueError and stops the pipeline if validation fails.
    """
    logger.info(f"Loading raw data from: {RAW_PATH}")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    logger.info("Schema check passed")

    # Null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")
    logger.info("Null check passed")

    # Value range check
    for col, (min_val, max_val) in VALUE_RANGES.items():
        out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
        if len(out_of_range) > 0:
            logger.warning(
                f"Column '{col}': {len(out_of_range)} rows outside "
                f"expected range [{min_val}, {max_val}]"
            )

    # Categorical value check
    valid_sex = {"male", "female"}
    valid_smoker = {"yes", "no"}
    valid_region = {"northeast", "northwest", "southeast", "southwest"}

    if not set(df["sex"].unique()).issubset(valid_sex):
        raise ValueError(f"Unexpected values in 'sex': {set(df['sex'].unique())}")
    if not set(df["smoker"].unique()).issubset(valid_smoker):
        raise ValueError(f"Unexpected values in 'smoker': {set(df['smoker'].unique())}")
    if not set(df["region"].unique()).issubset(valid_region):
        raise ValueError(f"Unexpected values in 'region': {set(df['region'].unique())}")

    logger.info("Categorical value check passed")
    logger.info(f"Validation complete — {len(df)} rows passed all checks")

    # Push row count to XCom for downstream tasks
    context["ti"].xcom_push(key="row_count", value=len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Clean Data
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(**context):
    """
    Handle outliers using IQR capping on numeric columns.
    Saves cleaned data back to a temp location via XCom pass-through.
    """
    logger.info("Starting data cleaning")
    df = pd.read_csv(RAW_PATH)

    for col in ["bmi", "charges"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            logger.info(f"Capping {len(outliers)} outliers in '{col}'")
            df[col] = df[col].clip(lower=lower, upper=upper)

    logger.info(f"Cleaning complete — {len(df)} rows retained")

    # Save cleaned data temporarily
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH + ".cleaned.tmp", index=False)
    context["ti"].xcom_push(key="cleaned_rows", value=len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineer(**context):
    """
    Encode categorical features:
      - smoker: yes/no → 1/0
      - sex: female/male → 1/0
      - region: one-hot encoded (drop first to avoid multicollinearity)
    """
    logger.info("Starting feature engineering")
    df = pd.read_csv(PROCESSED_PATH + ".cleaned.tmp")

    # Binary encoding
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"] = df["sex"].map({"female": 1, "male": 0})

    # One-hot encode region (drop first = northeast as reference)
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df.drop("region", axis=1), region_dummies], axis=1)

    logger.info(f"Feature engineering complete — columns: {list(df.columns)}")

    # Save processed data
    df.to_csv(PROCESSED_PATH, index=False)

    # Clean up temp file
    os.remove(PROCESSED_PATH + ".cleaned.tmp")

    context["ti"].xcom_push(key="final_columns", value=list(df.columns))
    logger.info(f"Saved processed data to: {PROCESSED_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Compute Drift Baselines
# ─────────────────────────────────────────────────────────────────────────────
def compute_baselines(**context):
    """
    Compute statistical baselines (mean, variance, min, max, percentiles,
    and value distribution) for each feature.
    Saved to data/drift_baselines/baselines.json.
    Used later by Airflow retrain DAG to detect data drift.
    """
    logger.info("Computing drift baselines")
    df = pd.read_csv(PROCESSED_PATH)

    baselines = {}
    feature_cols = [c for c in df.columns if c != "charges"]

    for col in feature_cols:
        col_data = df[col].astype(float)
        baselines[col] = {
            "mean": float(col_data.mean()),
            "variance": float(col_data.var()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "p25": float(col_data.quantile(0.25)),
            "p50": float(col_data.quantile(0.50)),
            "p75": float(col_data.quantile(0.75)),
            "count": int(col_data.count()),
        }
        logger.info(
            f"  {col}: mean={baselines[col]['mean']:.4f}, "
            f"std={baselines[col]['std']:.4f}"
        )

    # Add metadata
    baselines["_meta"] = {
        "computed_at": datetime.utcnow().isoformat(),
        "row_count": len(df),
        "source": RAW_PATH,
    }

    os.makedirs(os.path.dirname(BASELINES_PATH), exist_ok=True)
    with open(BASELINES_PATH, "w") as f:
        json.dump(baselines, f, indent=2)

    logger.info(f"Baselines saved to: {BASELINES_PATH}")


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
    dag_id="ingestion_dag",
    description="Data ingestion, validation, cleaning, feature engineering, and drift baseline computation",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # triggered manually
    catchup=False,
    tags=["mlops", "ingestion", "phase-1"],
) as dag:

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        provide_context=True,
    )

    task_clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
        provide_context=True,
    )

    task_features = PythonOperator(
        task_id="feature_engineer",
        python_callable=feature_engineer,
        provide_context=True,
    )

    task_baselines = PythonOperator(
        task_id="compute_baselines",
        python_callable=compute_baselines,
        provide_context=True,
    )

    # Pipeline order
    task_validate >> task_clean >> task_features >> task_baselines