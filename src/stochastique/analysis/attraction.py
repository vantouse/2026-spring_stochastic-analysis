import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def localize_basins_of_attraction(
    system,
    equilibria: list[np.ndarray],
    time_span: np.ndarray,
    ax: plt.Axes,
    bounds: tuple = (-3, 3),
    grid_size: int = 150,
    clip_ratio: float = 0.25,
    atol: float = 1e-2,
):
    """
    Localize attraction basins for equilibria of a DynamicSystem2D.

    Args:
        system: DynamicSystem2D instance.
        equilibria: list of equilibrium points.
        time_span: integration grid.
        ax: matplotlib axis.
        bounds: phase space bounds.
        grid_size: number of grid points per axis.
        clip_ratio: transient clipping ratio.
        atol: equilibrium matching tolerance.
    """

    xmin, xmax = bounds
    ymin, ymax = bounds

    x_vals = np.linspace(xmin, xmax, grid_size)
    y_vals = np.linspace(ymin, ymax, grid_size)

    basin_map = -np.ones((grid_size, grid_size))

    for i, x0 in enumerate(tqdm(x_vals, desc='Localizing basins')):
        for j, y0 in enumerate(y_vals):

            state_init = np.array([x0, y0])

            try:
                solution = system.solve(
                    state_init=state_init,
                    time_span=time_span
                )

            except Exception:
                continue

            if not np.all(np.isfinite(solution)):
                continue

            # discard transient
            clip = int(len(solution) * clip_ratio)
            solution_ss = solution[clip:]

            endpoint = np.mean(solution_ss, axis=0)

            matched = False

            for k, eq in enumerate(equilibria):
                if np.linalg.norm(endpoint - eq) < atol:
                    basin_map[j, i] = k
                    matched = True
                    break

            # unresolved attractor
            if not matched:
                basin_map[j, i] = -1

    cmap = plt.cm.get_cmap('Pastel1', len(equilibria))

    ax.imshow(
        basin_map,
        extent=[xmin, xmax, ymin, ymax],
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        alpha=0.8,
        aspect='auto'
    )

    ax.set_title('Basins of attraction')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
