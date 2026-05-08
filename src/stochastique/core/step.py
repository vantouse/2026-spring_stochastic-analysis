import numpy as np


def step_stochastic_em(
    model_func: callable,
    params: dict,
    state: np.ndarray,
    dt: float,
    epsilon: float,
    noise_mask: np.ndarray = np.array([1, 1]),
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Euler-Maruyama step for stochastic system given by `model_func` with `params`.

    Args:
        dt: integration step
        epsilon: noise intensity (when set to 0, the system is considered deterministic)
        noise_mask: denotes the equations to which noise should be added (noise vector is
            multiplied by the noise mask element-wise)
    
    Returns:
        system state after the randomized step
    """
    drift = model_func(0, state, **params)
    
    dW = rng.normal(0., 1., size=2) * np.sqrt(dt)
    noise = epsilon * dW * noise_mask
    
    return state + drift * dt + noise


def step_stochastic_rk4(
    model_func: callable,
    params: dict,
    state: np.ndarray,
    dt: float,
    epsilon: float,
    noise_mask: np.ndarray,
    rng: np.random.Generator,
):
    """
    Runge-Kutta 4-th order step for stochastic system given by `model_func` with `params`.

    First, the deterministic RK4 step is performed, and then the computed state is augmented with
    the noise.

    Args:
        dt: integration step
        epsilon: noise intensity (when set to 0, the system is considered deterministic)
        noise_mask: denotes the equations to which noise should be added (noise vector is
            multiplied by the noise mask element-wise)
    
    Returns:
        system state after the randomized step
    """
    k1 = dt * model_func(0, state, **params)
    k2 = dt * model_func(0, state + k1/2, **params)
    k3 = dt * model_func(0, state + k2/2, **params)
    k4 = dt * model_func(0, state + k3, **params)
    
    state_rk = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    dW = rng.normal(0., 1., size=2) * np.sqrt(dt)
    noise = epsilon * dW * noise_mask

    return state_rk + noise
