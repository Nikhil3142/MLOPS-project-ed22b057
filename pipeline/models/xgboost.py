"""XGBoost Regression model definition."""
from xgboost import XGBRegressor


def get_model(params: dict):
    """Return an XGBRegressor instance."""
    return XGBRegressor(**params)