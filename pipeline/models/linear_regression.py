"""Linear Regression model definition."""
from sklearn.linear_model import LinearRegression


def get_model(params: dict):
    """Return a LinearRegression instance."""
    return LinearRegression(**params)