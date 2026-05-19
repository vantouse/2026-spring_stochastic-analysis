import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from kneed import find_shape, KneeLocator

from stochastique.analysis.simulation import simulate_trajectory


def localize_stochastic_excitement_zones(
    model_func: callable,
    params: dict,
    state_init: np.ndarray,
    time_span: np.ndarray,
    noise_mask: np.ndarray,
    eps_values: np.ndarray,
    num_simulations: int = 100,
    rng: np.random.Generator = np.random.default_rng(),
    ax: plt.Axes = None,
):
    """
    Empirically estimate epsilon-zones of stochastic excitement for a stochastic system given by
    `model_func` and `params`.

    Returns:
        epsilon-excitement zone bounds (min, max)
    """
    excitement_points = []

    for _ in tqdm(range(num_simulations), desc='Run multiple simulations'):
        # x_values = np.empty_like(eps_values)

        x_max_values = np.empty_like(eps_values)
        x_min_values = np.empty_like(eps_values)
        x_amplitude_values = np.empty_like(eps_values)

        for idx in range(len(eps_values)):
            trajectory = simulate_trajectory(
                model_func=model_func,
                params=params,
                state_init=state_init,
                time_span=time_span,
                epsilon=eps_values[idx],
                noise_mask=noise_mask,
                rng=rng,
            )
            trajectory = trajectory[100:]   # clip transient
            trajectory_x = trajectory[:, 0]
            
            # Check if the simulation failed/exploded
            # if np.any(np.isnan(trajectory_x)) or np.any(np.isinf(trajectory_x)):
            #     x_values[idx] = x_values[idx - 1] if idx > 0 else 0
            # else:
            #     x_values[idx] = np.max(trajectory_x)    # TODO: использовать амплитуду вместо максимума!

            valid_data = trajectory_x[np.isfinite(trajectory_x)]
            
            if valid_data.size > 0:
                x_max_values[idx] = np.max(valid_data)
                x_min_values[idx] = np.min(valid_data)
            else:
                x_max_values[idx] = x_max_values[idx - 1] if idx > 0 else 0
                x_min_values[idx] = x_min_values[idx - 1] if idx > 0 else 0
            x_amplitude_values[idx] = x_max_values[idx] - x_min_values[idx]

        x_diff_abs = np.abs(np.diff(x_amplitude_values, prepend=x_amplitude_values[0]))
        excitement_idx = np.argmax(x_diff_abs)
        excitement = eps_values[excitement_idx]
        excitement_points.append(excitement)

        # excitement_idx_top5 = np.argsort(x_diff_abs)[-2:][::-1]
        # excitements = eps_values[excitement_idx_top5].tolist()
        # excitement_points.extend(excitements)
        
        if ax is not None:
            ax.vlines(eps_values, x_min_values, x_max_values, alpha=0.1, color='blue')
            # ax.plot(eps_values, x_values, alpha=0.3)
    
    if ax is not None:
        ax.axvspan(min(excitement_points), max(excitement_points), color='green', alpha=0.3, label='stochastic excitement zone')

    return min(excitement_points), max(excitement_points)