import numpy as np


def model_SN(
    time: float,
    state: np.ndarray,
    a: float = 1,
    b: float = 2,
):
    """
    Saltzman-Nicolis climate model.
    Accepts `time` argument for interface consistency, but does not use it.
    
    Parameter restrictions:
    - a > 0
    - b = 2
    """
    x, y = state
    return np.array([
        y - x,
        -a * x + b * y - (x ** 2) * y
    ])


def equilibria_FHN(
    a: float = 1,
    b: float = 2,
):
    """
    Return all the real equilibrium values of the FHN model at given parameter values.
    """
    return [
        np.array([
            np.sqrt(b - a),
            np.sqrt(b - a)
        ]),
        np.array([
            0,
            0
        ]),
        np.array([
            -np.sqrt(b - a),
            -np.sqrt(b - a)
        ]),
    ]
