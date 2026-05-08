import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


RNG_DEFAULT = np.random.default_rng()


def random_normal_method_of_12(
    size: int = 1,
    rng: np.random.Generator = RNG_DEFAULT,
):
    n = 12
    uniform = rng.uniform(0., 1., size=(n, size))
    return np.sum(uniform - 0.5, axis=0) * np.sqrt(12) / np.sqrt(n)


def random_normal_box_muller(
    size: int = 1,
    rng: np.random.Generator = RNG_DEFAULT,
):
    a, b = rng.uniform(0., 1., size=(2, size))
    xi_1 = np.sqrt(-2 * np.log(a)) * np.cos(2 * np.pi * b)
    # xi_2 = np.sqrt(-2 * np.log(a)) * np.sin(2 * np.pi * b)
    return xi_1
