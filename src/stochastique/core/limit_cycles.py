# stochastique/core/limit_cycles.py
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Optional

from .solvers import TrajectorySolver, extract_limit_cycle
from .numerical import find_jacobian


@dataclass
class LimitCycle:
    initial_point: np.ndarray
    period: float
    trajectory: np.ndarray
    stable: bool
    floquet_multiplier: float


def cycles_are_close(
    cycle1: LimitCycle,
    cycle2: LimitCycle,
    tol_period: float = 1e-2,
    tol_shape: float = 1e-1,
) -> bool:
    """Сравнение двух предельных циклов."""
    if abs(cycle1.period - cycle2.period) > tol_period:
        return False

    c1 = np.mean(cycle1.trajectory, axis=0)
    c2 = np.mean(cycle2.trajectory, axis=0)

    if np.linalg.norm(c1 - c2) > tol_shape * 2:
        return False

    r1 = np.mean(np.linalg.norm(cycle1.trajectory - c1, axis=1))
    r2 = np.mean(np.linalg.norm(cycle2.trajectory - c2, axis=1))

    return abs(r1 - r2) <= tol_shape


class LimitCycleAnalyzer:
    """Поиск и анализ предельных циклов."""

    def __init__(self, model_func: Callable):
        self.model_func = model_func
        self.solver = TrajectorySolver(model_func)

    def solve(self, state_init: np.ndarray, time_span: np.ndarray, params: dict) -> np.ndarray:
        """Удобный shortcut."""
        return self.solver.solve(state_init, time_span, params)

    def find_poincare_crossings(
        self,
        solution: np.ndarray,
        time_span: np.ndarray,
        x_section: float = 0.0,
    ):
        """Поиск пересечений с секущей плоскостью x = const."""
        xs = solution[:, 0]
        crossings = []
        crossing_times = []

        for i in range(len(xs) - 1):
            x1 = xs[i] - x_section
            x2 = xs[i + 1] - x_section

            if x1 < 0 and x2 >= 0:
                dx = xs[i + 1] - xs[i]
                if dx <= 0:
                    continue
                alpha = -x1 / (x2 - x1)
                point = solution[i] + alpha * (solution[i + 1] - solution[i])
                t = time_span[i] + alpha * (time_span[i + 1] - time_span[i])

                crossings.append(point)
                crossing_times.append(t)

        return np.array(crossings), np.array(crossing_times)

    def shooting_residual(self, state_init: np.ndarray, period: float, params: dict, n_steps: int = 4000):
        time_span = np.linspace(0, period, n_steps)
        solution = self.solve(state_init, time_span, params)
        return solution[-1] - state_init

    def find_limit_cycle_shooting(self, guess_state: np.ndarray, guess_period: float, params: dict):
        """Метод стрельбы."""
        def objective(z):
            x0 = z[:2]
            T = z[2]
            residual = self.shooting_residual(x0, T, params)
            phase_cond = np.dot(self.model_func(0, x0, **params), residual)
            return np.concatenate([residual, [phase_cond]])

        z0 = np.concatenate([guess_state, [guess_period]])
        result = sp.optimize.root(objective, z0, method='hybr')

        if not result.success:
            return None

        x0 = result.x[:2]
        T = result.x[2]
        time_span = np.linspace(0, T, 4000)
        trajectory = self.solve(x0, time_span, params)

        return x0, T, trajectory

    def compute_floquet_multiplier(self, cycle: np.ndarray, period: float, params: dict, eps: float = 1e-5):
        """Приближённый расчёт мультипликатора Флоке."""
        x0 = cycle[0]
        f = self.model_func(0, x0, **params)
        normal = np.array([-f[1], f[0]])
        normal /= np.linalg.norm(normal)

        time_span = np.linspace(0, period, 4000)
        sol_ref = self.solve(x0, time_span, params)

        mus = []
        for phi in [0.0, np.pi/2]:
            pert_dir = np.cos(phi) * normal + np.sin(phi) * f / (np.linalg.norm(f) + 1e-12)
            pert = eps * pert_dir
            sol_pert = self.solve(x0 + pert, time_span, params)
            delta = sol_pert[-1] - sol_ref[-1]
            mus.append(np.linalg.norm(delta) / eps)

        return max(mus)

    def find_all_limit_cycles(
        self,
        params: dict,
        bounds: tuple = (-3.0, 3.0),
        grid_size: int = 12,
        time_horizon: float = 500.0,
        equilibria: Optional[List[np.ndarray]] = None,
    ) -> List[LimitCycle]:
        """Поиск всех предельных циклов (глобальный + targeted)."""
        cycles: List[LimitCycle] = []

        # ====================== ГЛОБАЛЬНЫЙ ПОИСК ======================
        xs = np.linspace(bounds[0], bounds[1], grid_size)
        ys = np.linspace(bounds[0], bounds[1], grid_size)

        for x in xs:
            for y in ys:
                state_init = np.array([x, y])
                time_span = np.linspace(0, time_horizon, 20000)

                solution = self.solve(state_init, time_span, params)
                crossings, crossing_times = self.find_poincare_crossings(solution, time_span)

                if len(crossings) < 8:
                    continue

                distances = np.linalg.norm(crossings[1:] - crossings[:-1], axis=1)
                if np.mean(distances[-5:]) > 5e-3 or np.std(distances[-6:]) > 5e-4:
                    continue

                point = crossings[-1]
                period = crossing_times[-1] - crossing_times[-2]

                result = self.find_limit_cycle_shooting(point, period, params)
                if result is None:
                    continue

                x0, T, traj = result
                mu = self.compute_floquet_multiplier(traj, T, params)

                cycle = LimitCycle(x0, T, traj, abs(mu) < 1.0, mu)

                if not any(cycles_are_close(cycle, existing) for existing in cycles):
                    cycles.append(cycle)

        # ====================== TARGETED ПОИСК (для SN, FHN и т.д.) ======================
        if equilibria:
            for eq in equilibria:
                for radius in [0.4, 0.8, 1.3]:
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        state_init = eq + radius * np.array([np.cos(angle), np.sin(angle)])
                        time_span = np.linspace(0, time_horizon, 15000)

                        solution = self.solve(state_init, time_span, params)
                        cycle_traj = extract_limit_cycle(solution)

                        if cycle_traj is not None:
                            crossings, ctimes = self.find_poincare_crossings(cycle_traj, 
                                                                           np.linspace(0, time_horizon, len(cycle_traj)))
                            if len(crossings) >= 4:
                                pt = crossings[-1]
                                per = ctimes[-1] - ctimes[-2] if len(ctimes) > 1 else 10.0
                                res = self.find_limit_cycle_shooting(pt, per, params)
                                if res:
                                    x0, T, traj = res
                                    mu = self.compute_floquet_multiplier(traj, T, params)
                                    cyc = LimitCycle(x0, T, traj, abs(mu) < 1.0, mu)
                                    if not any(cycles_are_close(cyc, existing) for existing in cycles):
                                        cycles.append(cyc)
                                    break
        return cycles

    def plot_limit_cycles(self, ax: plt.Axes, cycles: List[LimitCycle]) -> None:
        """Визуализация найденных циклов."""
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
