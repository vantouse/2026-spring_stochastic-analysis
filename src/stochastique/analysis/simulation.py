import numpy as np
import matplotlib.pyplot as plt

from stochastique.core.step import step_stochastic_em, step_stochastic_rk4


def simulate_trajectory(
    model_func: callable,
    params: dict,
    state_init: np.ndarray,
    time_span: np.ndarray,
    epsilon: float,
    noise_mask: np.ndarray,
    rng: np.random.Generator,
    method: str = 'rk4',   # 'em' for Euler-Maruyama step
):
    n = len(time_span)
    dt = time_span[1] - time_span[0]
    
    trajectory = np.zeros((n, 2))
    trajectory[0] = state_init
    
    if method == 'rk4':
        step_func = step_stochastic_rk4
    elif method == 'em':
        step_func = step_stochastic_em
    else:
        raise ValueError(f'Unknown step method: {method}!')

    for i in range(n - 1):
        trajectory[i + 1] = step_func(
            model_func=model_func,
            params=params,
            state=trajectory[i],
            dt=dt,
            epsilon=epsilon,
            noise_mask=noise_mask,
            rng=rng,
        )
    
    return trajectory


def simulate_stochastic_cloud(
    ax: plt.Axes,
    model_func: callable,
    params: dict,
    equilibrium: np.ndarray,
    time_span: np.ndarray,
    epsilon: float,
    noise_mask: np.ndarray,
    n_trajectories: int = 1,
    spread_init: float = 0.05,
):
    rng = np.random.default_rng()
    trajectories = []
    
    for _ in range(n_trajectories):
        # start near the equilirium
        state_init = equilibrium + rng.normal(0, spread_init, size=2)
        
        trajectory = simulate_trajectory(
            model_func=model_func,
            params=params,
            state_init=state_init,
            time_span=time_span,
            epsilon=epsilon,
            noise_mask=noise_mask,
            rng=rng,
        )
        trajectories.append(trajectory)

        if ax is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3)
    
    if ax is not None:
        ax.scatter(*equilibrium, color='red', s=50, zorder=100, label='equilibrium')
        ax.set_title(f"Noise cloud ({epsilon=})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
        ax.grid(True)

    return trajectories
