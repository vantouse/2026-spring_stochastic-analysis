import numpy as np


def distribution_error(
    p: np.ndarray,
    p_true: np.ndarray
):
    """
    MSE for comparison of empirical and theoretical distributions.
    """
    return np.sum((p - p_true) ** 2)