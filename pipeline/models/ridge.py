"""Ridge Regression model definition."""
from sklearn.linear_model import Ridge


def get_model(params: dict):
    """Return a Ridge regression instance."""
    return Ridge(**params)