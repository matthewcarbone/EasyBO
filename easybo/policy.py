"""Policies, and their helper classes and functions, are defined here. All
documentation follows the same conventions, namely:

* The surrogate model (usually a Gaussian Process, specifically the
  :class:`easygp.gp.EasyGP`) is always defined as :math:`f`. Note that the
  surrogate should always provide an uncertainty estimate.
* The objective function is (traditionally) denoted as :math:`J`.
* The acquisition function is given by :math:`A`.
* The expectation symbol is :math:`\\mathop{\\mathbb{E}}[\\cdot]`, and
  indicates an average over the estimators. In the case of a Gaussian Process,
  this is simply the predicted mean.
* The variance symbol is :math:`\\mathrm{Var}[\\cdot]` and indicates the
  variance over the estimators.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

from easygp import logger


def optimize(f, bounds, n_restarts=10):
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
    n_restarts : int, optional
        The number of times to randomly try the fitting procedure.

    Returns
    -------
    float
        The value corresponding to the located optimum.

    Raises
    ------
    RuntimeError
        If the optimization was unsuccessful.
    """

    min_f = np.inf
    min_x = None

    for n in range(n_restarts):
        x_0 = np.random.uniform(
            low=[b[0] for b in bounds], high=[b[1] for b in bounds]
        )
        res = minimize(f, x_0, bounds=bounds)
        if res.fun < min_f:
            min_x = res.x
            min_f = res.fun

    if min_x is None:
        logger.critical("Optimization unsuccessful")
        raise RuntimeError

    return min_x.reshape(-1, len(bounds))


class BasePolicy(ABC):
    """Base Policy. Initializes the target and ybest values to None by
    default. These must be manually set in later instances."""

    def __init__(self):
        """Initializes the target and ybest objects to None by default."""

        self._target = None
        self._ybest = None

    @abstractmethod
    def acquisition(self):
        pass

    def suggest(self, gp, bounds, n_restarts=10):
        """Suggests a new point based on maximizing the defined acquisition
        function.

        Parameters
        ----------
        gp : easygp.gp.EasyGP
            The Gaussian Process object.
        bounds : list
            A list of tuple. The lower and upper bounds for each dimension.
            Should be of length of the number of features in the input data.
        n_restarts : int, optional
            The number of times to restart (re-attempt) the optimization
            procedure.

        Returns
        -------
        numpy.ndarray
            The suggested next input point.
        """

        def aq(x):
            return -self.acquisition(x, gp)

        return optimize(aq, bounds, n_restarts)

    def objective(self, x):
        """Objective function to maximize. This is just the :math:`L_2` norm by
        default, but could in principle be overridden by other objective
        functions in derived classes.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        float
        """

        return -((self._target - x) ** 2)


class MaxVariancePolicy(BasePolicy):
    """Utilizes the acquisition function

    .. math::

        A(x) = \\mathrm{Var}[f(x)]

    Used for active learning, by sampling areas where the variance is highest.
    """

    def objective(self):
        raise NotImplementedError

    def acquisition(self, x, gp):
        __, sd = gp.predict(x, return_std=True)
        return (sd**2).sum().item()


class MaxVarianceTargetPolicy(BasePolicy):
    """Utilizes the acquisition function

    .. math::

        A(x) = \\mathrm{Var}[J(f(x))]

    Instead of sampling areas where the variance is highest, this method
    samples areas where the variance in the acquisition function, :math:`J`,
    is highest. Note that this requires the ``target`` to be defined.

    Parameters
    ----------
    n_samples : int
        The number of samples to take when calculating the expectations.
    """

    def acquisition(self, x, gp):
        r_samples = gp.sample_y(x)
        J_samples = self.objective(r_samples)
        return np.var(J_samples).item()


class _RequiresTarget:
    """Helper class which defines the explicit setter for the target."""

    def set_target(self, x):
        logger.debug(f"Set target to {x}")
        self._target = x


class _RequiresYbest:
    """Helper class which defines the explicit setter for the best y value so
    far."""

    def set_ybest(self, x):
        logger.debug(f"Set ybest to {x}")
        self._ybest = x


class ExploitationTargetPolicy(BasePolicy, _RequiresTarget):
    """Utilizes the acquisition function

    .. math::

        A(x) = J(\\mathop{\\mathbb{E}}[f(x)])

    This is a pure exploitation policy. It is extremely dependent on the data
    used during the initial fitting, since there is no exploratory part of
    this policy.
    """

    def acquisition(self, x, gp):
        mu, _ = gp.predict(x, return_std=True)
        return self.objective(mu).item()


class UpperConfidenceBoundPolicy(BasePolicy, _RequiresTarget):
    """Utilizes the acquisition function

    .. math::

        A(x) = J(\\mathop{\\mathbb{E}}[f(x)]) + k \\sqrt{\\mathrm{Var}[f(x)]}

    This method explicitly balances the purely exploitative policy
    (:class:`.ExploitationTargetPolicy`) and the purely exploratory policy
    (:class:`.MaxVariancePolicy`) with a weight factor given by
    :math:`k=\\sqrt{2}` by default.

    Parameters
    ----------
    k : float
        The weighting term given to the standard deviation component of the
        acquisition function.
    """

    def __init__(self, *args, k=np.sqrt(2.0), **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k

    def acquisition(self, x, gp):
        # Trying to maximize the acquisition function
        # Large objective is good - we want points close to that
        # Large sd is good if we've already found close points
        mu, sd = gp.predict(x, return_std=True)
        mu_obj = self.objective(mu)
        return (mu_obj + self._k * sd).item()


class ExpectedImprovementPolicy(BasePolicy, _RequiresTarget, _RequiresYbest):
    """Defines an acquisition function

    .. math::

        A(x) = \\mathop{\\mathbb{E}}[J(f(x)) - y_\\mathrm{best}]^+

    The expected improvement policy accounts for the magnitude of the
    improvement a newly sampled point will provide, and chooses the point that
    is thought to maximize that improvement.

    Parameters
    ----------
    n_samples : int
        The number of samples to take when calculating the expectations.
    """

    def __init__(self, *args, n_samples=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_samples = n_samples

    def acquisition(self, x, gp):
        r_samples = gp.sample_y(x, n_samples=self._n_samples)
        J_samples = self.objective(r_samples) - self._ybest
        J_samples[J_samples < 0] = 0
        return np.mean(J_samples).item()


class _TargetPerformance:
    """A special helper standalone class for measuring the target
    performance."""

    def set_target(self, target):
        self._policy.set_target(target)

    def __init__(self):
        self._policy = ExploitationTargetPolicy()

    def __call__(self, gp, truth, bounds, n_restarts=10):
        """Finds the target performance.

        Parameters
        ----------
        gp : easygp.gp.EasyGP
        truth : easygp.campaign.GPSampler
            The ground truth function. A single instance of a Gaussian Process,
            used for the campaigning.
        n_features : int
        n_restarts : int, optional

        Returns
        -------
        float
        """

        estimated = gp.suggest(self._policy, bounds, n_restarts=n_restarts)
        gt = truth(estimated)
        return -self._policy.objective(gt)
