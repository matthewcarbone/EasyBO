import numpy as np


def get_1d_features(seed=127, N=1000, Nsmall=15, xmin=-20.0, xmax=50.0):
    """Gets an example 1d grid and a randomly down-sampled version of that grid
    for fitting a GP to.

    Parameters
    ----------
    seed : int, optional
        Seed for the numpy random number generator.
    N : int, optional
        The total number of points on the full grid.
    Nsmall : int, optional
        The total number of randomly selected points on the downsampled grid.
    xmin : TYPE, optional
        The minimum value of the grid.
    xmax : float, optional
        The maximum value of the grid.

    Returns
    -------
    dict
        A dictionary containing the keys ``"features"`` and ``"full_grid"``.
        The features are the downsampled grid, and the full_grid is
        self-explanatory.
    """

    np.random.seed(seed)
    idx = np.random.choice([xx for xx in range(N)], Nsmall, replace=False)
    idx.sort()
    grid = np.linspace(xmin, xmax, N)
    X = grid[idx]
    X[0] = xmin
    X[-1] = xmax
    return {"features": X, "full_grid": grid}


def test_function_1(x):
    """Gets an example curve and noise for that curve. The example curve is

    .. math::

        f(x) = 0.2 x + 2.345 \\sin x

    and the noise term is always a given by::

        (np.linspace(-2, 2, len(x)))**2 + 0.1

    Parameters
    ----------
    x : numpy.ndarray
        The input x-grid.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The signal and noise.
    """

    y = 0.2 * x + np.sin(x) * 2.345
    alpha = np.ones_like(y) + np.random.normal(size=y.shape)
    return y, np.abs(alpha)
