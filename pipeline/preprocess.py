"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
DVC Stage 1 — Preprocessing

Input:  data/processed/initial_data.csv  (output of Airflow ingestion DAG)
Output: data/processed/train.csv
        data/processed/test.csv
        models/artifacts/scaler.pkl

Steps:
  1. Load Airflow-processed data
  2. Split into train/test
  3. Fit StandardScaler on train, apply to both
  4. Save splits + scaler
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "initial_data.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "artifacts", "scaler.pkl")
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)
    return params


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from the given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def split_features_target(df: pd.DataFrame):
    """Separate features and target column."""
    X = df.drop("charges", axis=1)
    y = df["charges"]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit StandardScaler on training data only.
    Apply the fitted scaler to both train and test.
    Returns scaled arrays and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def save_splits(
    X_train, y_train, X_test, y_test, feature_cols: list
) -> None:
    """Save train and test sets as CSV files."""
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["charges"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["charges"] = y_test.reset_index(drop=True)

    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    logger.info(f"Train set: {len(train_df)} rows → {TRAIN_PATH}")
    logger.info(f"Test set:  {len(test_df)} rows → {TEST_PATH}")


def main():
    """Main preprocessing pipeline."""
    logger.info("Starting preprocessing stage")

    # Load params
    params = load_params()
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    logger.info(f"Params: test_size={test_size}, random_state={random_state}")

    # Load data
    df = load_data(INPUT_PATH)

    # Split features and target
    X, y = split_features_target(df)
    feature_cols = list(X.columns)
    logger.info(f"Features: {feature_cols}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    logger.info("StandardScaler fitted and applied")

    # Save splits
    save_splits(X_train_scaled, y_train, X_test_scaled, y_test, feature_cols)

    # Save scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to: {SCALER_PATH}")

    logger.info("Preprocessing stage complete")


if __name__ == "__main__":
    main()