import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


def find_limit_cycle(model_func: callable, params: dict):
    """
    Find limit cycle in a 2D dynamic system using Poincare section.
    """
    
    # 1. Находим особую точку (x*, y*)
    x_star = -params
    y_star = x_star - (x_star**3)/3
    
    # 2. Определяем отображение Пуанкаре
    # Стартуем с линии x = x_star, y > y_star
    def poincare_map(y0):
        state0 = [x_star, y0]
        # Событие для остановки при пересечении сечения
        def event(t, state): return state[0] - x_star
        event.terminal = True
        event.direction = 1 # Считаем только пересечение в одну сторону
        
        sol = solve_ivp(
            fun=model_func,
            t_span=[0, 100],
            y0=state0, 
            args=params.values(),
            events=event,
            rtol=1e-9
        )
        
        if len(sol.y_events[0]) > 0:
            return sol.y_events[0][0][1] - y0 # P(y) - y = 0
        return 1e6 # Если цикл не найден

    # 3. Ищем корень (неподвижную точку)
    res = root(poincare_map, x0=y_star + 1.0) # Начальное приближение
    return res.x # Координата y на цикле
