import numpy as np
import matplotlib.pyplot as plt

from stochastique.analysis.simulation import simulate_stochastic_cloud
from stochastique.core.numerical import compute_covariance_matrix_2d, stochastic_sensitivity_matrix_2d


def plot_eigenvalues_per_param_value(
    ax: plt.Axes,
    model_func: callable,
    params: dict,
    equilibrium_expr: callable,
    time_span: np.ndarray,
    param_name: str,
    param_values: np.ndarray,
    stable_range: tuple[float, float],
    epsilon: float = 0.1,
    noise_mask: np.ndarray = np.array([1, 1]),
    covariance: bool = True,
    n_trajectories: int = 1,
):
    """
    Plot eigenvalues of the covariance matrix (or analytically derived stochastic sensitivity matrix) at
    different values `param_values` of a given parameter `param_name`.

    Args:
        equilibrium_expr: function, which returns equilibrium point at given model parameter values
        stable_range: parameter range, where given equilibrium remains stable
        epsilon: noise intensity (when set to 0, the system is considered deterministic)
        noise_mask: denotes the equations to which noise should be added (noise vector is multiplied by the noise mask element-wise)
        covariance: whether to use covariance matrix on a simulation instead of analytically derived stochastic sensitivity matrix
        n_trajectories: number of simulations (unused if `covariance` is False, i.e. analytical method is used)
    """
    lambda1_values = []
    lambda2_values = []

    for val in param_values:
        params[param_name] = val
        equilibrium = equilibrium_expr(val)

        if covariance:
            trajectories = simulate_stochastic_cloud(
                ax=None,
                model_func=model_func,
                params=params,
                equilibrium=equilibrium,
                time_span=time_span,
                epsilon=epsilon,
                noise_mask=noise_mask,
                n_trajectories=n_trajectories
            )
            matrix = compute_covariance_matrix_2d(trajectories)
            eigenvalues = np.linalg.eigvals(matrix)
        else:
            matrix = stochastic_sensitivity_matrix_2d(  # TODO: defined later in the notebook, and not visible here
                model_func=model_func,
                params=params,
                equilibrium=equilibrium,
                noise_mask=noise_mask,
            )
            eigenvalues = np.linalg.eigvals(matrix) * epsilon ** 2
        
        lambda1_values.append(eigenvalues[0])
        lambda2_values.append(eigenvalues[1])

    ax.axvspan(*stable_range, color='green', alpha=0.3, label='stable equilibria')
    ax.plot(param_values, lambda1_values, label=rf'$\lambda_1({param_name})$')
    ax.plot(param_values, lambda2_values, label=rf'$\lambda_2({param_name})$')

    ax.set_xlabel(param_name)
    ax.set_ylabel('eigenvalues')
    ax.legend(loc='upper right')
    ax.grid(True)


def plot_confidence_ellipse_2d(
    ax: plt.Axes,
    equilibrium: np.ndarray,
    W: np.ndarray,
    epsilon: float,
    confidence: float = 0.95,
    n_points: int = 100,
    major_axes: bool = False,
):
    """
    Plot confidence ellipse for a 2D stochastic system based on the
    stochastic sensitivity matrix W.

    Vectorized and numerically correct version.
    
    Args:
        ax: Matplotlib axes to plot on
        equilibrium : equilibrium point (x, y) of the deterministic system
        W: 2x2 stochastic sensitivity matrix
        epsilon: noise intensity
        confidence: confidence probability P (0 < P < 1)
        n_points: number of points to generate the ellipse using the parameterized space
    """
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sorted]
    eigenvectors = eigenvectors[:, idx_sorted]
    
    lambda1, lambda2 = eigenvalues
    
    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError("W must be positive definite")
    
    phi = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack((np.cos(phi), np.sin(phi)))
    
    k = np.sqrt(-np.log(1 - confidence))
    axis_scales = np.array([
        np.sqrt(2 * lambda1),
        np.sqrt(2 * lambda2)
    ]) * epsilon * k
    
    # Translate ellipse points into a basis formed by the eigenvectors of W
    ellipse_local = axis_scales[:, None] * circle
    ellipse = eigenvectors @ ellipse_local
    ellipse += equilibrium.reshape(2, 1)
    
    ax.plot(ellipse[0], ellipse[1], 'b--', label='confidence ellipse')
    
    if major_axes:
        for i in range(2):
            axis_length = axis_scales[i]
            v = eigenvectors[:, i]
            
            p1 = equilibrium - axis_length * v
            p2 = equilibrium + axis_length * v
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c--')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True)
