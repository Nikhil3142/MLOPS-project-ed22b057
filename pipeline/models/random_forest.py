"""Random Forest Regression model definition."""
from sklearn.ensemble import RandomForestRegressor


def get_model(params: dict):
    """Return a RandomForestRegressor instance."""
    return RandomForestRegressor(**params)