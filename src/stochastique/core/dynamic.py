"""
This file contains the main version of DynamicSystem2D with all the functionality used in lab 3.

The `dynamic_system.py` is secondary, as it was created in attempt to decompose
the functionality of DynamicSystem2D into narrower subclasses. It must be refactored ASAP.
"""

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from stochastique.core.numerical import find_jacobian, find_equilibrium, classify_equilibrium


class DynamicSystem2D:

    def __init__(
        self,
        model_func: callable,
        params: dict = None,
    ) -> None:
        """
        Args:
            model_func: function returning a vector of (dx/dt, dy/dt)
            params: dictionary of parameters
        """
        self.model_func = model_func
        self.params = params if params is not None else {}

    def solve(
        self,
        state_init: np.ndarray,
        time_span: np.ndarray,
    ) -> np.ndarray:
        """
        Find numeric solution for the system using 4-th order Runge-Kutta method.
        
        Args:
            state_init: initial state (x_0, y_0) of the dynamic system
            time_span: uniform time grid
        
        Returns:
            solution: array system states per time point throughout the time span
        """
        n = len(time_span)
        h = time_span[1] - time_span[0]
        
        solution = np.zeros((n, len(state_init)))
        solution[0] = state_init.copy()

        for i in range(n - 1):
            t = time_span[i]
            state_curr = solution[i]
            
            k1 = h * self.model_func(t, state_curr, **self.params)
            k2 = h * self.model_func(t + h/2, state_curr + k1/2, **self.params)
            k3 = h * self.model_func(t + h/2, state_curr + k2/2, **self.params)
            k4 = h * self.model_func(t + h, state_curr + k3, **self.params)
            
            solution[i + 1] = state_curr + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            
        return solution

    def plot_bifurcation_diagram(
        self,
        param_name: str,
        param_values: np.ndarray,
        state_init: np.ndarray,
        ax: plt.Axes,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Plot bifurcation diagram for a dynamic system.

        Iterate over the dense grid of parameter values, and detect eigenvalue sign changes
        for Jacobians of equilibrium states, marking them as bifurcation points.

        Real part of the eigenvalues of the Jacobian indicates the system stability/instability
        per parameter value (<0 ==> stable, >0 ==> unstable).

        Returns:
            bifurcation_points: array of bifurcation.
            equilibria: array of equilibrium point coordinates.
        """
        equilibria = []
        stability_mask = []
        types = []
        param_used = []
        bifurcation_points = []

        eigenvalues_last = None

        for val in param_values:
            self.params[param_name] = val

            guess = self._resolve_state_init(
                state_init=state_init,
                val=val,
                prev_equilibrium=equilibria[-1] if len(equilibria) > 0 else None
            )

            equilibrium = find_equilibrium(
                func=self.model_func,
                guess=guess,
                params=self.params
            )

            if equilibrium is None:
                continue

            jacobian = find_jacobian(
                func=self.model_func,
                t=0,
                state=equilibrium,
                params=self.params
            )
            eigenvalues = np.linalg.eigvals(jacobian)

            # equillibrium classification
            eq_type = classify_equilibrium(eigenvalues)
            types.append(eq_type)

            is_stable = np.all(np.real(eigenvalues) < 0)
            stability_mask.append(is_stable)

            # bifurcation check
            if eigenvalues_last is not None:
                if np.any(np.real(eigenvalues_last) * np.real(eigenvalues) < 0):
                    bifurcation_points.append(val)
                    ax.axvline(val, linestyle='--', label='bifurcation point')

            equilibria.append(equilibrium)
            param_used.append(val)

            eigenvalues_last = eigenvalues
            state_init = equilibrium

        equilibria = np.array(equilibria)
        stability_mask = np.array(stability_mask)
        param_used = np.array(param_used)
        bifurcation_points = np.array(bifurcation_points)

        x_eq = equilibria[:, 0]

        type_to_color = {
            'stable node': 'blue',
            'unstable node': 'red',
            'saddle': 'black',
            'stable focus': 'green',
            'unstable focus': 'orange',
            'center': 'purple',
            'degenerate': 'gray',
        }

        for t in set(types):
            mask = np.array([tt == t for tt in types])
            ax.scatter(param_used[mask], x_eq[mask], label=t, color=type_to_color.get(t, 'gray'), s=20)

        ax.set_xlabel(param_name)
        ax.set_ylabel('x')
        ax.set_title('Bifurcation diagram (with equilibrium types)')
        ax.legend()
        ax.grid(True)

        return bifurcation_points, equilibria
    
    def plot_trajectory(
        self,
        state_init: np.ndarray,
        time_span: np.ndarray,
        ax: plt.Axes,
        label: str,
    ) -> None:
        """
        Plot trajectory for given initial state.
        """
        solution = self.solve(state_init, time_span)
        ax.plot(solution[:, 0], solution[:, 1], label=label)
        ax.set_title('Phase portrait')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right')
        ax.grid(True)
    
    def extract_limit_cycle(
        self,
        solution: np.ndarray,
        clip_ratio: float = 0.5,
        eps: float = 1e-4,
    ) -> np.ndarray | None:
        """
        Extract limit cycle from a trajectory if it exists.
        
        Returns:
            solution_ss: limit cycle (if found) or None
        """
        x = solution[:, 0]

        # clip transient (consider only the asymptotic behavior of the system)
        clip = int(len(x) * clip_ratio)
        solution_ss = solution[clip:]
        x_ss = solution_ss[:, 0]

        # check: is solution bounded
        if not np.all(np.isfinite(solution_ss)):
            return None
        
        # check: not an equilibrium (consider only cycles, not points)
        if np.var(x_ss) < eps:
            return None

        return solution_ss
    
    def plot_limit_cycle(
        self,
        ax: plt.Axes,
        time_span: np.ndarray,
        state_init: np.ndarray = None,
        bounds: tuple = (-0.1, 0.1),
        n_attempts: int = 20,
        unstable: bool = False,
    ) -> bool:
        """
        Try to detect and plot limit cycle.
        """
        if unstable:
            time_span = time_span[::-1]
        for _ in range(n_attempts):
            if state_init is None:
                state_init = np.random.uniform(bounds[0], bounds[1], size=2)
            solution = self.solve(
                state_init,
                time_span
            )
            cycle = self.extract_limit_cycle(solution)

            if cycle is not None:
                color = 'lime' if not unstable else 'cyan'
                label = 'stable cycle' if not unstable else 'unstable cycle'
                ax.plot(cycle[:,0], cycle[:,1], color=color, linewidth=2, label=label)

                return True
        return False
    
    def plot_phase_portrait_dense(
        self,
        ax: plt.Axes,
        time_span: np.ndarray,
        bounds: tuple = (-2, 2),
        grid_size: int = 10,
        seacrh_limit_cycles: bool = True,
        limit_cycle_per_state: bool = True,
        equilibrium: np.ndarray = None
    ) -> None:
        """
        Plot phase portrait based on a grid of initial states.
        """
        xmin, xmax = bounds
        ymin, ymax = bounds

        # initialize grid
        x_vals = np.linspace(xmin, xmax, grid_size)
        y_vals = np.linspace(ymin, ymax, grid_size)

        # test trajectories for all initial states on the grid
        progress_bar = tqdm(product(x_vals, y_vals), total=len(x_vals) * len(y_vals), desc='Test trajectories and search cycles')
        for x0, y0 in progress_bar:
            state_init = np.array([x0, y0])

            solution = self.solve(state_init, time_span)
            if np.all(np.isfinite(solution)):
                ax.plot(solution[:, 0], solution[:, 1], color='blue', alpha=0.5)

            if seacrh_limit_cycles and limit_cycle_per_state:
                found_stable = self.plot_limit_cycle(ax=ax, time_span=time_span, state_init=state_init)
                found_unstable = self.plot_limit_cycle(ax=ax, time_span=time_span, state_init=state_init, unstable=True)
                if found_stable:
                    progress_bar.set_description("Stable limit cycle found!")
                if found_unstable:
                    progress_bar.set_description("Unstable limit cycle found!")

        # plot vector field
        X, Y = np.meshgrid(np.linspace(xmin, xmax, 25), np.linspace(ymin, ymax, 25))
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                dx, dy = self.model_func( 0, np.array([X[i, j], Y[i, j]]), **self.params)
                U[i, j] = dx
                V[i, j] = dy

        # normalize vector field
        norm = np.sqrt(U**2 + V**2)
        U /= (norm + 1e-8)
        V /= (norm + 1e-8)

        ax.streamplot(X, Y, U, V, density=1.2, color='black', linewidth=0.5)

        if seacrh_limit_cycles and not limit_cycle_per_state:
            found_stable = self.plot_limit_cycle(ax=ax, time_span=time_span, state_init=state_init)
            found_unstable = self.plot_limit_cycle(ax=ax, time_span=time_span, state_init=state_init, unstable=True)
            if found_stable:
                progress_bar.set_description("Stable limit cycle found!")
            if found_unstable:
                progress_bar.set_description("Unstable limit cycle found!")
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
    
    def _resolve_state_init(
        self,
        state_init,
        val: float,
        prev_equilibrium: np.ndarray | None,
    ) -> np.ndarray:
        """
        Resolve initial guess for equilibrium search.

        Priority:
        1. continuation (previous equilibrium)
        2. callable(`state_init(val)`)
        3. constant `np.ndarray` point (`state_init`)
        """

        # continuation
        if prev_equilibrium is not None:
            return prev_equilibrium

        if callable(state_init):
            return np.asarray(state_init(val))

        return np.asarray(state_init)
