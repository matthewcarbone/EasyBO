from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize


def optimize_01(f, bounds, num_restarts=10):
    """Wrapper for the scipy.optimize.minimize class. This operates under the
    assumption that the input data is scaled to minimum 0 and maximum 1 at all
    times.

    Parameters
    ----------
    f : callable
        Function to minimize.
    bounds : list of tuple, optional
        A list of tuple or lists that contains the minimum and maximum bound
        for those dimensions. The length of bounds should be equal to the
        number of input features of the function. E.g., for three features
        where each is between 0 and 1, bounds=[(0, 1), (0, 1), (0, 1)].
    num_restarts : int, optional
        The number of times to randomly try the fitting procedure.

    Returns
    -------
    float
        The value corresponding to the located optimum.

    Raises
    ------
    ValueError
        If the optimization was not successful.
    """

    min_f = np.inf
    min_x = None

    for n in range(num_restarts):
        x_0 = np.random.uniform(
            low=[b[0] for b in bounds], high=[b[1] for b in bounds]
        )
        res = minimize(f, x_0, bounds=bounds)
        if res.fun < min_f:
            min_x = res.x
            min_f = res.fun

    if min_x is None:
        raise ValueError("Optimization unsuccessful")

    return min_x


class BasePolicy(ABC):
    """Base Policy. Initializes the target and ybest values to None by
    default. These must be manually set in later instances."""

    @property
    def target(self):
        return self._target

    @property
    def ybest(self):
        return self._ybest

    def __init__(self):
        """Initializes the target and ybest objects to None by default."""

        self._target = None
        self._ybest = None

    @abstractmethod
    def acquisition(self):
        pass

    def suggest(self, gp, n_restarts=10):
        """Suggests a new point based on maximizing the defined acquisition
        function. Assumes that the gaussian process acts on normalized x data,
        on the support 0 to 1 for each feature.

        Parameters
        ----------
        gp : GaussianProcessRegressor
            The Gaussian Process object.
        n_restarts : int, optional
            The number of times to restart the optimization procedure.

        Returns
        -------
        float
            The suggested next x point.
        """

        def aq(x):
            return -self.acquisition(x, gp)

        return optimize_01(aq, gp.bounds, n_restarts)

    def objective(self, x):
        """Objective function to maximize. This is just the L2 norm by default,
        but could in principle be overridden by other objective functions in
        derived classes.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray
        """

        return -((self.target - x) ** 2)


class MaxVariancePolicy(BasePolicy):
    """Defines an acquisition function :math:`A(x) = \\mathrm{Var}[r(x)]`. Used
    essentially for active learning, by sampling areas where the variance is
    highest."""

    def objective(self):
        raise NotImplementedError

    def acquisition(self, x, gp):
        __, sd = gp.predict(x, return_std=True)
        return sd**2


class MaxVarianceTargetPolicy(BasePolicy):
    """Defines an acquisition function :mat:`A(x) = \\mathrm{Var}[J(r(X))]`.
    Requires the target to be defined."""

    def acquisition(self, x, gp):
        r_samples = gp.sample_y(x)
        J_samples = self.objective(r_samples)
        return np.var(J_samples)


class RequiresTarget:
    """Helper class which defines the explicit setter for the target."""

    def set_target(self, target):
        if not isinstance(target, (float, int)):
            raise ValueError(f"Invalid target: {target}")
        self._target = target


class RequiresYbest:
    """Helper class which defines the explicit setter for the best y value so
    far."""

    def set_ybest(self, ybest):
        if not isinstance(ybest, (float, int)):
            raise ValueError(f"Invalid ybest: {ybest}")
        self._ybest = ybest


class ExploitationTargetPolicy(BasePolicy, RequiresTarget):
    """Defines an acquisition function :math:`A(x) = J(E[r(x)])`."""

    def acquisition(self, x, gp):
        mu, _ = gp.predict(x, return_std=True)
        return self.objective(mu)


class ExpectedImprovementPolicy(BasePolicy, RequiresTarget, RequiresYbest):
    """Defines an acquisition function
    :math:`A(x) = E[J(r(x)) - y_\\mathrm{best}]^+`."""

    def __init__(self, *args, n_samples=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_samples = n_samples

    def acquisition(self, x, gp):
        r_samples = gp.sample_y(x, n_samples=self._n_samples)
        J_samples = self.objective(r_samples) - self.ybest
        J_samples[J_samples < 0] = 0
        return np.mean(J_samples)


class TargetPerformance:
    """A special helper standalone class for measuring the target
    performance."""

    @property
    def target(self):
        return self._target

    def set_target(self, target):
        self._policy.set_target(target)

    def __init__(self):
        self._policy = ExploitationTargetPolicy()
        self._target = None

    def __call__(self, gp, truth, n_restarts=10):
        """Finds the target performance.

        Parameters
        ----------
        gp : GaussianProcessRegressor
        truth : GPSampler
            The ground truth function. A single instance of a Gaussian Process,
            used for the campaigning.
        n_features : int
        n_restarts : int, optional

        Returns
        -------
        float
        """

        estimated = self._policy.suggest(gp, n_restarts=n_restarts)
        gt = truth(estimated)
        return -self._policy.objective(gt)


class MultiTargetPolicy:
    """Summary"""

    pass
