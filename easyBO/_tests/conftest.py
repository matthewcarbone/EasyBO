import pytest

import numpy as np


@pytest.fixture
def case1():
    np.random.seed(124)
    idx = np.random.choice([xx for xx in range(1000)], 10, replace=False)
    idx.sort()
    grid = np.linspace(-20, 50, 1000)
    X = grid[idx]  # Feature data
    alpha = np.linspace(-2, 2, 1000)[idx] ** 2 + 0.1  # Noise/uncertainty

    def truth(x):
        return x + np.sin(x) * 2.345  # Linear upwards trend

    return X, truth(X), alpha, truth
