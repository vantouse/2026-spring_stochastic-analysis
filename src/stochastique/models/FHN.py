import numpy as np


def model_FHN(
    time: float,
    state: np.ndarray,
    delta: float = 0.1,
    a: float = 1,
):
    """
    FitzHugh-Nagumo neural model.
    Accepts `time` argument for interface consistency, but does not use it.
    
    Parameter restrictions:
    - delta << 1 (delta = 0.1)
    - a > 0
    """
    x, y = state
    return np.array([
        (x - (x ** 3) / 3 - y) / delta,
        x + a
    ])


def equilibria_FHN(
    delta: float = 0.1,
    a: float = 1,
):
    """
    Return all the real equilibrium values of the FHN model at given parameter values.
    """
    return [
        np.array([
            -a,
            a**3 / 3 - a
        ])
    ]
