# stochastique/core/dynamic_system.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional

from .solvers import TrajectorySolver
from .bifurcation import BifurcationAnalyzer
from .limit_cycles import LimitCycleAnalyzer, LimitCycle


class DynamicSystem2D:
    """Основной фасад динамической системы (композиция)."""

    def __init__(self, model_func: Callable, params: dict = None):
        self.model_func = model_func
        self.params = params or {}

        self.solver = TrajectorySolver(model_func)
        self.bifurcation = BifurcationAnalyzer(model_func)
        self.limit_cycles = LimitCycleAnalyzer(model_func)

    def solve(self, state_init: np.ndarray, time_span: np.ndarray) -> np.ndarray:
        return self.solver.solve(state_init, time_span, self.params)

    def plot_bifurcation_diagram(
        self,
        param_name: str,
        param_values: np.ndarray,
        state_init: np.ndarray,
        ax: plt.Axes
    ):
        return self.bifurcation.plot_bifurcation_diagram(
            param_name, param_values, state_init, ax, self.params
        )

    def find_all_limit_cycles(
        self,
        bounds: tuple = (-3.0, 3.0),
        grid_size: int = 12,
        time_horizon: float = 500.0,
        equilibria: Optional[List[np.ndarray]] = None,
    ) -> List[LimitCycle]:
        return self.limit_cycles.find_all_limit_cycles(
            params=self.params,
            bounds=bounds,
            grid_size=grid_size,
            time_horizon=time_horizon,
            equilibria=equilibria
        )

    def plot_limit_cycles(self, ax: plt.Axes, cycles: Optional[List[LimitCycle]] = None):
        if cycles is None:
            cycles = self.find_all_limit_cycles()
        self.limit_cycles.plot_limit_cycles(ax, cycles)

    # Дополнительные удобные методы
    def extract_limit_cycle(self, solution: np.ndarray, **kwargs):
        from .solvers import extract_limit_cycle
        return extract_limit_cycle(solution, **kwargs)
