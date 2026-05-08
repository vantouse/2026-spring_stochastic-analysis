import numpy as np
import scipy as sp

from stochastique.random.generators import RNG_DEFAULT


def estimate_integral_mc(
    func: callable,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    sample_size: int,
    rng: np.random.Generator = RNG_DEFAULT,
):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    x_values = rng.uniform(x_min, x_max, size=sample_size)
    y_values = rng.uniform(y_min, y_max, size=sample_size)

    points_inside = np.sum(y_values < func(x_values))
    prob_est = points_inside / sample_size

    integral_est = prob_est * y_max * (x_max - x_min)
    return integral_est


def find_jacobian(
    func: callable,
    t: float,
    state: np.ndarray,
    params: dict,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Numerically compute Jacobian matrix of a 2D dynamic system.
    """
    n = len(state)
    jacobian = np.zeros((n, n))

    for i in range(n):
        perturb = np.zeros(n)
        perturb[i] = eps
        
        f1 = func(t, state + perturb, **params)
        f2 = func(t, state - perturb, **params)
        jacobian[:, i] = (f1 - f2) / (2 * eps)
    
    return jacobian


def find_equilibrium(
    func: callable,
    guess: np.ndarray,
    params: dict,
) -> np.ndarray:    
    def root_fn(state):
        return func(0, state, **params)
    
    solution, info, ier, _ = sp.optimize.fsolve(root_fn, guess, full_output=True)
    
    if ier == 1:
        return solution
    return None


def classify_equilibrium(
    eigenvalues: np.ndarray,
    eps: float = 1e-6
) -> str:
    """
    Determine equillibrium type by the eigenvalues of the Jacobian matrix computed at this
    equilibrium.
    """
    real = np.real(eigenvalues)
    imag = np.imag(eigenvalues)

    if np.any(real > eps) and np.any(real < -eps):
        return 'saddle'

    if np.all(np.abs(real) < eps):
        if np.any(np.abs(imag) > eps):
            return 'center'
        else:
            return 'degenerate'

    if np.any(np.abs(imag) > eps):
        if np.all(real < 0):
            return 'stable focus'
        elif np.all(real > 0):
            return 'unstable focus'

    if np.all(real < 0):
        return 'stable node'
    elif np.all(real > 0):
        return 'unstable node'

    return 'unknown'


def compute_covariance_matrix_2d(
    trajectories: list[np.ndarray],
    clip_ratio: float = 0.5,
) -> np.ndarray:
    """
    Compute 2x2 covariance matrix for a stochastic cloud.

    Args:
        trajectories: list of trajectories of shape `(n_steps, 2)`
    
    Returns:
        cov_matrix: covariance matrix
        points_concat: stochastic cloud after clipping the transient
    """
    points = []
    
    for trajectory in trajectories:
        # clip transient (consider only the asymptotic behavior of the system)
        clip = int(len(trajectory) * clip_ratio)
        traj_stationary = trajectory[clip:]
        points.append(traj_stationary)
    
    points_concat = np.vstack(points)  # shape (N_total, 2)
    cov_matrix = np.cov(points_concat.T)
    
    return cov_matrix


def solve_lyapunov_symmetric_2d(F: np.ndarray, Q: np.ndarray):
    """
    Solve Lyapunov equation $F W + W F^T + Q = 0$ for 2D symmetric matrix W (stochastic sensitivity
    matrix) and noise matrix Q ($Q = S S^T$).

    The equation is represented in a matrix form with respect to variables w_1, w_2, w_3.

    Notes:
        F = [[f'_x(x, y), f'_y(x, y)],
            [g'_x(x, y), g'_y(x, y)]]

        W = [[w_1, w_2],
            [w_2, w_3]]
    """
    a, b = F[0, 0], F[0, 1]
    c, d = F[1, 0], F[1, 1]

    q11 = Q[0, 0]
    q12 = Q[0, 1]
    q22 = Q[1, 1]

    A = np.array([
        [2*a, 2*b, 0],
        [c, a+d, b],
        [0, 2*c, 2*d],
    ], dtype=float)

    rhs = -np.array([q11, q12, q22], dtype=float)

    w1, w2, w3 = np.linalg.solve(A, rhs)

    W = np.array([
        [w1, w2],
        [w2, w3]
    ])

    return W


def stochastic_sensitivity_matrix_2d(
    model_func: callable,
    params: dict,
    equilibrium: np.ndarray,
    noise_mask: np.ndarray = np.array([1, 1]),
):
    """
    Find stochastic sensitivity matrix `W` near the `equilibrium` point for a dynamic system given
    by `model_func` with parameter values `params`.
    """
    F = find_jacobian(func=model_func, t=0, state=equilibrium, params=params)
    eigenvalues = np.linalg.eigvals(F)

    if np.any(np.real(eigenvalues) >= 0):
        raise ValueError(f"Equilibrium is not asymptotically stable at {params=}. Eigenvalues: {eigenvalues}")

    S = np.array([
        [noise_mask[0], 0.],
        [0., noise_mask[1]]
    ])
    Q = S @ S.T
    W = solve_lyapunov_symmetric_2d(F, Q)

    return W
