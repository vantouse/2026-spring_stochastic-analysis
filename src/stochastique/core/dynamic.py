from dataclasses import dataclass

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from stochastique.core.numerical import find_jacobian, find_equilibrium, classify_equilibrium


def cycles_are_close(
    cycle1: LimitCycle,
    cycle2: LimitCycle,
    tol_period: float = 1e-2,
    tol_shape: float = 1e-1,
):
    """Сравнивает два предельных цикла по периоду и средней "амплитуде" (радиусу)."""
    if abs(cycle1.period - cycle2.period) > tol_period:
        return False

    c1 = np.mean(cycle1.trajectory, axis=0)
    c2 = np.mean(cycle2.trajectory, axis=0)

    r1 = np.mean(np.linalg.norm(cycle1.trajectory - c1, axis=1))
    r2 = np.mean(np.linalg.norm(cycle2.trajectory - c2, axis=1))

    if abs(r1 - r2) > tol_shape:
        return False

    # Дополнительная проверка расстояния между центрами (важно для внутренних/внешнего циклов)
    if np.linalg.norm(c1 - c2) > tol_shape * 2:
        return False

    return True


@dataclass
class LimitCycle:
    initial_point: np.ndarray
    period: float
    trajectory: np.ndarray
    stable: bool
    floquet_multiplier: float


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
        time_span: np.ndarray
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

    # ... (plot_bifurcation_diagram, plot_trajectory, extract_limit_cycle — оставлены без изменений, 
    # кроме небольшой доработки extract_limit_cycle)

    def extract_limit_cycle(
        self,
        solution: np.ndarray,
        clip_ratio: float = 0.5,
        eps: float = 1e-5,          # чуть строже
    ) -> np.ndarray | None:
        """
        Extract limit cycle from a trajectory if it exists.
        
        Returns:
            solution_ss: limit cycle (if found) or None
        """
        x = solution[:, 0]

        # clip transient
        clip = int(len(x) * clip_ratio)
        solution_ss = solution[clip:]
        x_ss = solution_ss[:, 0]

        if not np.all(np.isfinite(solution_ss)):
            return None
        
        # not an equilibrium
        if np.var(x_ss) < eps:
            return None

        # дополнительная проверка замкнутости (последняя точка близка к первой)
        if np.linalg.norm(solution_ss[-1] - solution_ss[0]) > 0.1 * np.ptp(solution_ss, axis=0).max():
            return None

        return solution_ss

    # plot_limit_cycle, plot_limit_cycle_near_equilibrium — оставлены (используются в dense portrait)

    def plot_phase_portrait_dense(
        self,
        ax: plt.Axes,
        equilibria: list[np.ndarray],
        time_span: np.ndarray,
        bounds: tuple = (-2, 2),
        grid_size: int = 10,
        show_limit_cycle: bool = True,
        perturbation_radius: float = 0.1,
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
        for x0 in x_vals:
            for y0 in y_vals:
                state_init = np.array([x0, y0])

                solution = self.solve(state_init, time_span)
                if np.all(np.isfinite(solution)):
                    ax.plot(solution[:, 0], solution[:, 1], color='blue', alpha=0.5)

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

        # try to find and plot the limit cycle
        if show_limit_cycle:
            self.plot_limit_cycles(ax=ax)

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

    def find_poincare_crossings(
        self,
        solution: np.ndarray,
        time_span: np.ndarray,
        x_section: float = 0.0,
    ):
        xs = solution[:, 0]

        crossings = []
        crossing_times = []

        for i in range(len(xs) - 1):
            x1 = xs[i] - x_section
            x2 = xs[i + 1] - x_section

            if x1 < 0 and x2 >= 0:
                dx = solution[i + 1, 0] - solution[i, 0]

                if dx <= 0:
                    continue

                alpha = -x1 / (x2 - x1)

                point = solution[i] + alpha * (solution[i + 1] - solution[i])
                t = time_span[i] + alpha * (time_span[i + 1] - time_span[i])

                crossings.append(point)
                crossing_times.append(t)

        return np.array(crossings), np.array(crossing_times)
    
    def shooting_residual(
        self,
        state_init: np.ndarray,
        period: float,
        n_steps: int = 4000,
    ):
        time_span = np.linspace(0, period, n_steps)

        solution = self.solve(state_init, time_span)

        return solution[-1] - state_init
    
    def find_limit_cycle_shooting(
        self,
        guess_state: np.ndarray,
        guess_period: float,
    ):
        def objective(z):
            x0 = z[:2]
            T = z[2]

            residual = self.shooting_residual(x0, T)

            phase_condition = np.dot(
                self.model_func(0, x0, **self.params),
                residual
            )

            return np.concatenate([
                residual,
                [phase_condition]
            ])

        z0 = np.concatenate([
            guess_state,
            [guess_period]
        ])

        result = sp.optimize.root(objective, z0)

        if not result.success:
            return None

        x0 = result.x[:2]
        T = result.x[2]

        time_span = np.linspace(0, T, 4000)
        trajectory = self.solve(x0, time_span)

        return x0, T, trajectory

    def compute_floquet_multiplier(
        self,
        cycle: np.ndarray,
        period: float,
        eps: float = 1e-5,
    ):
        """Улучшенная аппроксимация Floquet (несколько направлений)."""
        x0 = cycle[0]
        f = self.model_func(0, x0, **self.params)
        normal = np.array([-f[1], f[0]])
        normal /= np.linalg.norm(normal)

        time_span = np.linspace(0, period, 4000)
        sol_ref = self.solve(x0, time_span)

        mus = []
        for phi in [0, np.pi/2]:
            pert = eps * (np.cos(phi) * normal + np.sin(phi) * np.array([f[0], f[1]]) / np.linalg.norm(f))
            sol_pert = self.solve(x0 + pert, time_span)
            delta = sol_pert[-1] - sol_ref[-1]
            mus.append(np.linalg.norm(delta) / eps)

        return np.max(mus)  # консервативно — наибольший множитель

    def find_all_limit_cycles(
        self,
        bounds=(-3, 3),
        grid_size=12,           # чуть плотнее
        time_horizon=500,       # длиннее для SN
        equilibria: list[np.ndarray] | None = None,   # для targeted поиска
    ):
        """Улучшенный поиск **всех** предельных циклов (внешний + внутренние около eq)."""
        cycles = []

        # 1. Глобальный поиск (Poincare + shooting)
        xs = np.linspace(bounds[0], bounds[1], grid_size)
        ys = np.linspace(bounds[0], bounds[1], grid_size)

        for x in xs:
            for y in ys:
                state_init = np.array([x, y])
                time_span = np.linspace(0, time_horizon, 20000)  # больше точек

                solution = self.solve(state_init, time_span)
                crossings, crossing_times = self.find_poincare_crossings(solution, time_span)

                if len(crossings) < 8:   # строже
                    continue
                
                distances = np.linalg.norm(crossings[1:] - crossings[:-1], axis=1)
                if np.std(distances[-6:]) > 5e-4 or np.mean(distances[-5:]) > 5e-3:
                    continue

                point = crossings[-1]
                period = crossing_times[-1] - crossing_times[-2]

                result = self.find_limit_cycle_shooting(point, period)
                if result is None:
                    continue

                x0, T, trajectory = result
                mu = self.compute_floquet_multiplier(trajectory, T)

                cycle = LimitCycle(
                    initial_point=x0,
                    period=T,
                    trajectory=trajectory,
                    stable=(abs(mu) < 1.0),
                    floquet_multiplier=mu,
                )

                if not any(cycles_are_close(existing, cycle) for existing in cycles):
                    cycles.append(cycle)

        # 2. Targeted поиск внутренних циклов около устойчивых равновесий (критично для SN при a<1)
        if equilibria is not None:
            for eq in equilibria:
                # проверяем устойчивость
                J = find_jacobian(self.model_func, 0, eq, self.params)
                ev = np.linalg.eigvals(J)
                if np.all(np.real(ev) < 0):   # устойчивое — ищем неустойчивый цикл вокруг него
                    for r in [0.3, 0.6, 1.0]:   # разные радиусы
                        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                            state_init = eq + r * np.array([np.cos(angle), np.sin(angle)])
                            solution = self.solve(state_init, np.linspace(0, time_horizon, 15000))
                            cycle_traj = self.extract_limit_cycle(solution, clip_ratio=0.6)
                            if cycle_traj is not None:
                                # shooting для точности
                                crossings, ctimes = self.find_poincare_crossings(cycle_traj, np.linspace(0, time_horizon, len(cycle_traj)))
                                if len(crossings) >= 4:
                                    pt = crossings[-1]
                                    per = ctimes[-1] - ctimes[-2] if len(ctimes)>1 else 10.0
                                    res = self.find_limit_cycle_shooting(pt, per)
                                    if res:
                                        x0, T, traj = res
                                        mu = self.compute_floquet_multiplier(traj, T)
                                        cyc = LimitCycle(x0, T, traj, abs(mu)<1, mu)
                                        if not any(cycles_are_close(existing, cyc) for existing in cycles):
                                            cycles.append(cyc)
                                        break

        return cycles

    def plot_limit_cycles(
        self,
        ax: plt.Axes,
        cycles: list = None,
    ):
        """Plot all found limit cycles."""
        if cycles is None:
            # Для SN-модели передавайте equilibria=equilibria_SN(**self.params)
            cycles = self.find_all_limit_cycles()
        for i, cycle in enumerate(cycles):
            color = 'magenta' if cycle.stable else 'red'
            linestyle = '-' if cycle.stable else '--'
            ax.plot(
                cycle.trajectory[:, 0],
                cycle.trajectory[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                label=f'cycle {i} ({"stable" if cycle.stable else "unstable"}, μ≈{cycle.floquet_multiplier:.3f})'
            )
        if cycles:
            ax.legend()
