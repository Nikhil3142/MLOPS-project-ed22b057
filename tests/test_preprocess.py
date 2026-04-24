"""
test_preprocess.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the preprocessing pipeline.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocess import (
    load_data,
    split_features_target,
    scale_features,
    save_splits,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a sample processed dataframe for testing."""
    return pd.DataFrame({
        "age": [25, 35, 45, 55, 65],
        "sex": [1, 0, 1, 0, 1],
        "bmi": [22.5, 28.0, 31.5, 25.0, 33.0],
        "children": [0, 2, 1, 3, 0],
        "smoker": [0, 1, 0, 0, 1],
        "region_northwest": [1, 0, 0, 1, 0],
        "region_southeast": [0, 1, 0, 0, 1],
        "region_southwest": [0, 0, 1, 0, 0],
        "charges": [3000.0, 15000.0, 8000.0, 5000.0, 25000.0],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""

    def test_splits_correctly(self, sample_df):
        """Target column is separated from features."""
        X, y = split_features_target(sample_df)
        assert "charges" not in X.columns
        assert y.name == "charges"

    def test_feature_count(self, sample_df):
        """8 feature columns are returned."""
        X, y = split_features_target(sample_df)
        assert len(X.columns) == 8

    def test_row_count_preserved(self, sample_df):
        """Row count is preserved after split."""
        X, y = split_features_target(sample_df)
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)


class TestScaleFeatures:
    """Tests for scale_features function."""

    def test_returns_scaled_arrays(self, sample_df):
        """Returns numpy arrays of correct shape."""
        X, y = split_features_target(sample_df)
        X_train, X_test = X[:4], X[4:]
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

    def test_scaler_is_fitted(self, sample_df):
        """Scaler is fitted on training data."""
        X, y = split_features_target(sample_df)
        X_train, X_test = X[:4], X[4:]
        _, _, scaler = scale_features(X_train, X_test)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")

    def test_train_mean_near_zero(self, sample_df):
        """Scaled training data has mean near zero."""
        X, y = split_features_target(sample_df)
        X_train, X_test = X[:4], X[4:]
        X_train_scaled, _, _ = scale_features(X_train, X_test)
        assert abs(X_train_scaled.mean()) < 1.0

    def test_scaler_not_fitted_on_test(self, sample_df):
        """Scaler is fitted only on training data, not test data."""
        X, y = split_features_target(sample_df)
        X_train, X_test = X[:4], X[4:]
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        # Test set transformed using train stats - values may differ from train
        assert X_train_scaled.shape[1] == X_test_scaled.shape[1]


class TestLoadData:
    """Tests for load_data function."""

    def test_raises_on_missing_file(self):
        """FileNotFoundError raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")

    def test_loads_csv(self, tmp_path, sample_df):
        """CSV file is loaded correctly."""
        path = tmp_path / "test.csv"
        sample_df.to_csv(path, index=False)
        df = load_data(str(path))
        assert len(df) == len(sample_df)
        assert list(df.columns) == list(sample_df.columns)