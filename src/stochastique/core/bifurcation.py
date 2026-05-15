# stochastique/core/bifurcation.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

from .numerical import find_jacobian, find_equilibrium, classify_equilibrium


class BifurcationAnalyzer:
    """Анализ бифуркаций и равновесий."""

    def __init__(self, model_func: Callable):
        self.model_func = model_func

    def plot_bifurcation_diagram(
        self,
        param_name: str,
        param_values: np.ndarray,
        state_init: Callable,
        ax: plt.Axes,
        params: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Построение бифуркационной диаграммы."""
        equilibria = []
        types_list = []
        param_used = []
        bifurcation_points = []

        eigenvalues_last = None
        guess = state_init(**params)

        for val in param_values:
            params[param_name] = val

            eq = find_equilibrium(self.model_func, guess, params)
            if eq is None:
                continue

            J = find_jacobian(self.model_func, 0, eq, params)
            evals = np.linalg.eigvals(J)
            eq_type = classify_equilibrium(evals)

            if eigenvalues_last is not None:
                if np.any(np.real(eigenvalues_last) * np.real(evals) < 0):
                    bifurcation_points.append(val)
                    ax.axvline(val, linestyle='--', color='red', alpha=0.7)

            equilibria.append(eq)
            types_list.append(eq_type)
            param_used.append(val)
            eigenvalues_last = evals
            guess = eq

        equilibria = np.array(equilibria)
        param_used = np.array(param_used)

        type_to_color = {
            'stable node': 'blue', 'unstable node': 'red', 'saddle': 'black',
            'stable focus': 'green', 'unstable focus': 'orange',
            'center': 'purple', 'degenerate': 'gray'
        }

        for t in set(types_list):
            mask = np.array([tt == t for tt in types_list])
            ax.scatter(param_used[mask], equilibria[mask, 0],
                       label=t, color=type_to_color.get(t, 'gray'), s=20)

        ax.set_xlabel(param_name)
        ax.set_ylabel('x')
        ax.set_title('Bifurcation Diagram')
        ax.legend()
        ax.grid(True)

        return np.array(bifurcation_points), equilibria
