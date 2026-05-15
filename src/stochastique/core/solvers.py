# stochastique/core/solvers.py
import numpy as np
from typing import Callable


class TrajectorySolver:
    """Отвечает только за численное решение ОДУ (RK4)."""

    def __init__(self, model_func: Callable):
        self.model_func = model_func

    def solve(
        self,
        state_init: np.ndarray,
        time_span: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """Runge-Kutta 4th order."""
        n = len(time_span)
        h = time_span[1] - time_span[0]

        solution = np.zeros((n, len(state_init)))
        solution[0] = state_init.copy()

        for i in range(n - 1):
            t = time_span[i]
            y = solution[i]

            k1 = h * self.model_func(t, y, **params)
            k2 = h * self.model_func(t + h/2, y + k1/2, **params)
            k3 = h * self.model_func(t + h/2, y + k2/2, **params)
            k4 = h * self.model_func(t + h, y + k3, **params)

            solution[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

        return solution


def extract_limit_cycle(
    solution: np.ndarray,
    clip_ratio: float = 0.5,
    eps: float = 1e-5,
) -> np.ndarray | None:
    """Извлекает предельный цикл из траектории."""
    clip = int(len(solution) * clip_ratio)
    ss = solution[clip:]

    if not np.all(np.isfinite(ss)):
        return None
    if np.var(ss[:, 0]) < eps:
        return None
    # Проверка замкнутости
    if np.linalg.norm(ss[-1] - ss[0]) > 0.15 * np.ptp(ss, axis=0).max():
        return None

    return ss
