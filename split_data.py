"""
split_data.py
─────────────────────────────────────────────────────────────────────────────
One-time script to split the raw insurance.csv into:
  - data/raw/initial_data.csv  (80%) → given to Airflow for the main pipeline
  - data/raw/holdout_data.csv  (20%) → locked away, simulates future data


    python split_data.py


─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
SOURCE_FILE = os.path.join(RAW_DIR, "insurance.csv")
INITIAL_FILE = os.path.join(RAW_DIR, "initial_data.csv")
HOLDOUT_FILE = os.path.join(RAW_DIR, "holdout_data.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
HOLDOUT_SIZE = 0.2
RANDOM_STATE = 42

EXPECTED_COLUMNS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}


def validate_source(df: pd.DataFrame) -> None:
    """Basic validation of the raw source file before splitting."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Source file is missing columns: {missing}")
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    logger.info(f"Source validation passed — {len(df)} rows, {len(df.columns)} columns")


def split_and_save(df: pd.DataFrame) -> None:
    """Split into initial + holdout and save both CSVs."""
    initial, holdout = train_test_split(
        df,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
    )
    initial.to_csv(INITIAL_FILE, index=False)
    holdout.to_csv(HOLDOUT_FILE, index=False)

    logger.info(f"initial_data.csv → {len(initial)} rows  ({100 - HOLDOUT_SIZE*100:.0f}%)")
    logger.info(f"holdout_data.csv → {len(holdout)} rows  ({HOLDOUT_SIZE*100:.0f}%)")
    logger.info(f"Saved to: {RAW_DIR}")


def main() -> None:
    # Guard: don't overwrite if already split
    if os.path.exists(INITIAL_FILE) or os.path.exists(HOLDOUT_FILE):
        logger.warning(
            "initial_data.csv or holdout_data.csv already exists. "
            "Delete them first if you want to re-split."
        )
        return

    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}\n"
            "Download insurance.csv from Kaggle and place it in data/raw/"
        )

    logger.info(f"Loading source file: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE)

    validate_source(df)
    split_and_save(df)

    logger.info("Done. Run the Airflow DAG next to start the pipeline.")


if __name__ == "__main__":
    main()
