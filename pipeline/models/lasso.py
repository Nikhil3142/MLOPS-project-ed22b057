"""Lasso Regression model definition."""
from sklearn.linear_model import Lasso


def get_model(params: dict):
    """Return a Lasso regression instance."""
    return Lasso(**params)