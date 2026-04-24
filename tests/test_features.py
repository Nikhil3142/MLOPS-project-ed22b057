"""
test_features.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for feature engineering in the Airflow ingestion DAG.
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    """Sample raw data before feature engineering."""
    return pd.DataFrame({
        "age": [25, 35, 45, 55],
        "sex": ["male", "female", "male", "female"],
        "bmi": [22.5, 28.0, 31.5, 25.0],
        "children": [0, 2, 1, 3],
        "smoker": ["no", "yes", "no", "no"],
        "region": ["northeast", "northwest", "southeast", "southwest"],
        "charges": [3000.0, 15000.0, 8000.0, 5000.0],
    })


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same encoding as Airflow ingestion DAG."""
    df = df.copy()
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["sex"] = df["sex"].map({"female": 1, "male": 0})
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df.drop("region", axis=1), region_dummies], axis=1)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestSmokerEncoding:
    """Tests for smoker binary encoding."""

    def test_yes_encoded_as_1(self, raw_df):
        """Smoker 'yes' is encoded as 1."""
        df = encode_features(raw_df)
        assert df.loc[raw_df["smoker"] == "yes", "smoker"].iloc[0] == 1

    def test_no_encoded_as_0(self, raw_df):
        """Smoker 'no' is encoded as 0."""
        df = encode_features(raw_df)
        assert df.loc[raw_df["smoker"] == "no", "smoker"].iloc[0] == 0

    def test_no_null_values(self, raw_df):
        """No null values after encoding."""
        df = encode_features(raw_df)
        assert df["smoker"].isnull().sum() == 0


class TestSexEncoding:
    """Tests for sex binary encoding."""

    def test_female_encoded_as_1(self, raw_df):
        """Sex 'female' is encoded as 1."""
        df = encode_features(raw_df)
        assert df.loc[raw_df["sex"] == "female", "sex"].iloc[0] == 1

    def test_male_encoded_as_0(self, raw_df):
        """Sex 'male' is encoded as 0."""
        df = encode_features(raw_df)
        assert df.loc[raw_df["sex"] == "male", "sex"].iloc[0] == 0


class TestRegionEncoding:
    """Tests for region one-hot encoding."""

    def test_northeast_is_reference(self, raw_df):
        """Northeast is the reference category (dropped)."""
        df = encode_features(raw_df)
        assert "region_northeast" not in df.columns

    def test_three_region_columns_created(self, raw_df):
        """Three region dummy columns are created."""
        df = encode_features(raw_df)
        region_cols = [c for c in df.columns if c.startswith("region_")]
        assert len(region_cols) == 3

    def test_correct_region_columns(self, raw_df):
        """Correct region columns are created."""
        df = encode_features(raw_df)
        assert "region_northwest" in df.columns
        assert "region_southeast" in df.columns
        assert "region_southwest" in df.columns

    def test_region_column_removed(self, raw_df):
        """Original region column is removed."""
        df = encode_features(raw_df)
        assert "region" not in df.columns

    def test_binary_values_only(self, raw_df):
        """Region columns contain only 0 and 1."""
        df = encode_features(raw_df)
        for col in ["region_northwest", "region_southeast", "region_southwest"]:
            assert set(df[col].unique()).issubset({0, 1, True, False})


class TestDataIntegrity:
    """Tests for overall data integrity after feature engineering."""

    def test_row_count_preserved(self, raw_df):
        """Row count is preserved after encoding."""
        df = encode_features(raw_df)
        assert len(df) == len(raw_df)

    def test_charges_column_preserved(self, raw_df):
        """Target column charges is preserved."""
        df = encode_features(raw_df)
        assert "charges" in df.columns
        assert df["charges"].equals(raw_df["charges"])

    def test_no_null_values(self, raw_df):
        """No null values after encoding."""
        df = encode_features(raw_df)
        assert df.isnull().sum().sum() == 0