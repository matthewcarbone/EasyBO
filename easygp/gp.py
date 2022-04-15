from contextlib import contextmanager
from copy import deepcopy
from time import time
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_gp
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from easygp import logger
from easygp.policy import TargetPerformance, RequiresYbest, RequiresTarget


class AutoscalingGaussianProcessRegressor:
    """A lightweight wrapper for the sklearn Gaussian Process Regressor which
    takes an extra input, n_features, and basically ensures that all
    predictions reshape the inputs properly so that they're compatible with the
    base GP. It also automatically scales all inputs to the 0 -> 1 support, and
    executes a StandardScaler on the targets, scaling the noise appropriately
    as well."""

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, x):
        assert x > 0
        self._n_features = x

    @property
    def gp(self):
        return self._gp

    @property
    def bounds(self):
        return self._bounds

    def __init__(
        self,
        *,
        bounds,
        gp_kwargs={"kernel": RBF(length_scale=1.0), "n_restarts_optimizer": 10},
    ):
        self._n_features = len(bounds)
        self._bounds = bounds
        self._Xscaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._yscaler = StandardScaler()
        self._gp = None
        self._gp_kwargs = gp_kwargs
        self._scale_X = True

    @contextmanager
    def disable_scale_X(self):
        self._scale_X = False
        try:
            yield None
        finally:
            self._scale_X = True

    def fit(self, X, y, alpha=1e-10, **kwargs):
        """Summary

        Parameters
        ----------
        X : TYPE
            Description
        y : TYPE
            Description
        alpha : float, optional
            Description
        **kwargs
            Description
        """

        X = X.reshape(-1, self._n_features)
        if self._scale_X:
            X = self._Xscaler.fit_transform(X)
        y = self._yscaler.fit_transform(y.reshape(-1, 1))
        y = y.squeeze()
        alpha = alpha / self._yscaler.scale_
        self._gp = sklearn_gp(alpha=(alpha**2).squeeze(), **self._gp_kwargs)
        self._gp.fit(X, y)

    def predict(self, X, return_std=True):
        X = X.reshape(-1, self._n_features)
        if self._scale_X:
            X = self._Xscaler.transform(X)
        pred, std_or_cov = self._gp.predict(X, return_std, not return_std)
        pred = self._yscaler.inverse_transform(pred)
        std_or_cov = std_or_cov * self._yscaler.scale_
        return pred, std_or_cov

    def sample_y(self, X, n_samples=10, randomstate=0):
        np.random.seed(randomstate)
        y_mean, y_cov = self.predict(X, return_std=False)
        return np.random.multivariate_normal(
            y_mean, y_cov, n_samples
        ).T.squeeze()


class GPSampler:
    """A method for deterministically sampling from a single instance of a
    Gaussian Process."""

    def __init__(self, gp, randomstate=None):
        """Initializes the class with a Gaussian Process and random state
        variable, which should be set to not None to ensure deterministic
        calculations.

        Parameters
        ----------
        gp : AutoscalingGaussianProcessRegressor
            The Gaussian Process regressor to use.
        randomstate : int, optional
            The random state used for seeding the numpy random number
            generator.
        """

        # Train nearest neighbor regressor on samples over dense grid
        self._gp = gp
        self.n_features = gp.n_features
        self.randomstate = randomstate

    def __call__(self, x):
        """Samples the specific instance of the Gaussian Process.

        Parameters
        ----------
        x : np.ndarray
            The input grid to sample on.

        Returns
        -------
        np.ndarray
            The samples.
        """

        x = x.reshape(-1, self.n_features)
        return np.array(
            [
                self._gp.sample_y(xx, n_samples=1, randomstate=self.randomstate)
                for xx in x
            ]
        )


FIT_WARNING = "Decreasing the bound and calling fit again may find a better"


class Campaign:
    @property
    def gp(self):
        return self._gp

    @gp.setter
    def gp(self, x):
        raise RuntimeError("Do not try and set the GP yourself!")

    @property
    def truth(self):
        return self._truth

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def alpha(self):
        return self._alpha

    @property
    def bounds(self):
        return self._bounds

    @property
    def target(self):
        return self._target

    def __init__(
        self,
        X,
        y,
        alpha,
        bounds,
        policy,
        target,
        randomstate=0,
        gp_kwargs={"kernel": RBF(length_scale=1.0), "n_restarts_optimizer": 10},
    ):
        """Initializes the campaign.

        Parameters
        ----------
        X : np.ndarray
            Initial feature data. Should be of shape (n x n_features).
        y : np.ndarray
            Initial target data. Should be of shape (n x 1).
        alpha : np.ndarray
            Initial target noise (standard deviation). Should be of shape
            (n x 1).
        bounds : list of tuple
            The lower and upper bounds for each dimension. Should be of length
            of the number of features in the input data.
        policy : easygp.policy.BasePolicy
            The policy for running the campaign. This defines the procedure by
            which new points are sampled.
        target : float
            The campaign's target. Necessary to quantify performance. For now,
            the target is a single value, the goal of which is to find this
            value as quickly as possible.
        randomstate : int, optional
            The randomstate for the underlying Gaussian Process instance that
            is treated as the ground truth.
        gp_kwargs : dict, optional
            Keyword arguments passed to the Gaussian Process.
        """

        for key, value in locals().items():
            if key != "self":
                setattr(self, f"_{key}", value)

        self._iteration = -1

        # For campaigning we must always have some target value, but the policy
        # itself may not require one.
        if isinstance(self._policy, RequiresTarget):
            self._policy.set_target(target)

        self.fit()

        self._performance_func = TargetPerformance()
        self._performance_func.set_target(self._target)

        # Set the truth function based on the GP sampler
        self._truth = GPSampler(deepcopy(self._gp), self._randomstate)

    def fit(self):
        """Fits the internal Gaussian Process using the current stored data."""

        t0 = time()
        self._gp = AutoscalingGaussianProcessRegressor(
            bounds=self._bounds, gp_kwargs=self._gp_kwargs
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            self._gp.fit(self._X, self._y, self._alpha)
        self._iteration += 1
        dt = time() - t0
        for warn in caught_warnings:
            if FIT_WARNING in str(warn.message):
                logger.warning(
                    f"Model bad fit: {self.gp._gp.kernel_} fit in {dt:.01} s"
                )
                return
        logger.info(f"\tModel {self.gp._gp.kernel_} fit in {dt:.01} s")

    def update(self, X, y, alpha):
        """Updates the data with new X, y and alpha values. The data is always
        assumed to be unscaled.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        alpha : np.ndarray
        """

        X = X.reshape(-1, self._gp.n_features)
        self._X = np.concatenate([self._X, X], axis=0)
        self._y = np.concatenate([self._y, y], axis=0)
        self._alpha = np.concatenate([self._alpha, alpha], axis=0)

    def run(self, n=10, n_restarts=10, ignore_criterion=1e-5):
        """Runs the campaign.

        Parameters
        ----------
        n : int, optional
            The number of experiments to run.
        """

        performance = []
        counter = 0

        logger.info(f"Beginning campaign (n={n}) with target {self.target}")
        logger.info(f"Policy is {self._policy.__class__.__name__}")

        while counter < n:

            logger.info(f"Beginning iteration {counter:03}")

            # Start with setting ybest if needed
            if isinstance(self._policy, RequiresYbest):
                to_max = self._policy.objective(self._y)
                y_best = self._y[np.argmax(to_max)].item()
                self._policy.set_ybest(y_best)
                logger.info(f"\ty-best set to {y_best:.02f}")

            # Suggest a new point
            new_X = self._policy.suggest(self._gp, n_restarts)
            new_X = new_X.reshape(-1, self._gp.n_features)

            diff = np.abs(self.X - new_X).sum(axis=1) < ignore_criterion
            if diff.sum() > 0:
                logger.warning(
                    f"\tProposed point {new_X} too close to existing data. "
                    "Ignoring."
                )
                p = self._performance_func(self._gp, self._truth, n_restarts)
                performance.append(p.item())
                counter += 1
                continue

            # Get the performance given this new point
            p = self._performance_func(self._gp, self._truth, n_restarts)
            performance.append(p.item())

            # Get the new truth result for the suggested X value
            new_y = self._truth(new_X)
            logger.info(f"\ty-value of proposed points {new_y}")

            # As for noise, use the average in the dataset plus/minus one
            # standard deviation
            avg_noise = np.array([np.mean(self._alpha)]) + np.random.normal(
                scale=self._alpha.std()
            )

            # Update the datasets stored in _X, _y, and _alpha
            self.update(new_X, new_y, avg_noise)

            # Refit on the new data
            self.fit()

            # Increment the counter
            counter += 1

        return performance
