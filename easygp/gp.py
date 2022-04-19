from contextlib import contextmanager
from copy import deepcopy
from tqdm import tqdm
from time import time
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_gp
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from easygp import logger, _DEBUG
from easygp.policy import TargetPerformance, RequiresYbest


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
    def n_targets(self):
        return self._n_targets

    @n_targets.setter
    def n_targets(self, x):
        assert x > 0
        self._n_targets = x

    @property
    def gps(self):
        return self._gps

    @property
    def bounds(self):
        return self._bounds

    @property
    def kernels(self):
        return [xx.kernel_ for xx in self._gps]

    def __init__(
        self,
        *,
        bounds,
        n_targets=1,
        gp_kwargs={"kernel": RBF(length_scale=1.0), "n_restarts_optimizer": 10},
    ):
        self._n_features = len(bounds)
        self._n_targets = n_targets
        self._bounds = bounds
        self._Xscaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._yscaler = StandardScaler()
        self._gps = None
        self._gp_kwargs = gp_kwargs
        self._scale_X = True

    @contextmanager
    def disable_scale_X(self):
        """Context manager that disables the scaling of the input variable
        X."""

        self._scale_X = False
        try:
            yield None
        finally:
            self._scale_X = True

    def fit(self, X, y, alpha=1e-5):
        """Fits independent Gaussian Process(es). Will raise a warning if
        the correlation coefficients^2 between any pair of targets is > 0.95.
        This would imply highly correlated targets and, simply put, it's not
        productive to train two GP's when one will probably suffice.

        Parameters
        ----------
        X : np.ndarray
            Features to fit on. Of shape N x N_features.
        y : np.ndarray
            Targets to fit multiple, independent GPs on. Of shape
            N x N_targets.
        alpha : float or np.ndarray, optional
            Noise (standard deviation), of shape N x N_targets or is a float
            (same noise for every target). Default is 1e-5 (to prevent
            numerical instability during the GP fitting process).
        """

        if y.shape[1] != self._n_targets:
            logger.error(
                f"Target array shape {y.shape} incompatible with provided "
                f"number of targets: {self._n_targets}"
            )
            return

        self._gps = []

        X = X.reshape(-1, self._n_features)
        if self._scale_X:
            X = self._Xscaler.fit_transform(X)
        y = self._yscaler.fit_transform(y.reshape(-1, self._n_targets))
        alpha = alpha / self._yscaler.scale_

        if y.shape[1] > 1:
            corr = pd.DataFrame(y).corr().to_numpy() ** 2 - np.eye(y.shape[1])
            if np.any(corr > 0.95):
                logger.warning(
                    "Correlation coefficient between at least one pair of "
                    "targets is greater than 0.95."
                )

        for target_index in range(y.shape[1]):
            _y = y[:, target_index].squeeze()
            if isinstance(alpha, (float, int)):
                _alpha = alpha
            else:
                _alpha = alpha.copy().reshape(-1, self._n_targets)
                _alpha = _alpha[:, target_index].squeeze()
            gp = sklearn_gp(alpha=_alpha**2, **self._gp_kwargs)
            gp.fit(X, _y)
            self._gps.append(gp)

    def predict(self, X, return_std=True):
        """Runs the predict operation on the Gaussian Processes. Two items are
        always returned, the mean and either the standard deviation or
        covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Input feature array of shape N x N_features.
        return_std : bool, optional
            If True, returns the standard deviation of the predictor ensemble.
            Else, returns the covariance matrix.

        Returns
        -------
        tuple
            The mean prediction and either the standard deviation or covariance
            matrix.
        """

        X = X.reshape(-1, self._n_features)
        if self._scale_X:
            X = self._Xscaler.transform(X)

        pred = []
        std_or_cov = []

        for gp in self._gps:
            _pred, _std_or_cov = gp.predict(X, return_std, not return_std)
            pred.append(_pred)
            std_or_cov.append(_std_or_cov)

        pred = np.array(pred).T

        if return_std:
            std_or_cov = np.array(std_or_cov).T
            std_or_cov = std_or_cov * self._yscaler.scale_

        # Return a list of covariance matrices in that case
        else:
            std_or_cov = [
                xx * self._yscaler.scale_[ii]
                for ii, xx in enumerate(std_or_cov)
            ]

        pred = self._yscaler.inverse_transform(pred)

        return pred, std_or_cov

    def sample_y(self, X, n_samples=1, randomstate=0):
        """Samples a single instance of the Gaussian Processes.

        Parameters
        ----------
        X : np.ndarray
            Input feature array of shape N x N_features.
        n_samples : int, optional
            The number of random samples to take from the Gaussian Processes.
            Default is 1.
        randomstate : int, optional
            The random state which ensures reproducibility. Each unique number
            will produce a different samlple. Default is 0.

        Returns
        -------
        np.ndarray
            The resultant samples. Will be of shape
            len(X) x num_targets x num_samples if num_samples > 1, else
            just len(X) x num_targets.
        """

        y_mean, y_cov = self.predict(X, return_std=False)
        samples = []
        for ii in range(len(self._gps)):
            np.random.seed(randomstate)
            samples.append(
                np.random.multivariate_normal(
                    y_mean[:, ii], y_cov[ii], n_samples
                ).T.squeeze()
            )
        samples = np.array(samples)
        if n_samples > 1:
            return samples.swapaxes(0, 1)
        return samples.T

    def sample_y_reproducibly(self, X, n_samples=1, randomstate=0):
        """There is a subtlety when sampling from the Gaussian Process
        posterior that must be accounted for when using different sampling
        grids.

        .. warning::

            Even for the same random state, different input grids (X) will
            produce different samples from the posterior if using the sklearn
            `sample_y` method. This is likely due to how the random sampling
            works under the hood in sklearn.

        Here, we fix the issue by explicitly setting the random state to the
        same value every time a new point x is sampled. This is slightly less
        efficient, but does fix the problem.

        Parameters
        ----------
        X : np.ndarray
            Input feature array of shape N x N_features.
        n_samples : int, optional
            The number of random samples to take from the Gaussian Processes.
            Default is 1.
        randomstate : int, optional
            The random state which ensures reproducibility. Each unique number
            will produce a different samlple. Default is 0.

        Returns
        -------
        np.ndarray
            The resultant samples. Will be of shape
            len(X) x num_targets x num_samples if num_samples > 1, else
            just len(X) x num_targets.
        """

        arr = np.array(
            [self.sample_y(xx, n_samples, randomstate).squeeze() for xx in X]
        )
        if n_samples > 1:
            return arr.swapaxes(1, 2)
        return arr


class GPSampler:
    """A method for deterministically sampling from a single instance of a
    Gaussian Process."""

    @property
    def gp(self):
        return self._gp

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
        self.n_targets = gp.n_targets
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

        return self._gp.sample_y_reproducibly(
            x.reshape(-1, self.n_features),
            n_samples=1,
            randomstate=self.randomstate,
        )


FIT_WARNING = "Decreasing the bound and calling fit again may find a better"


class Campaign:
    """Used for running optimization campaigns given some data. While choosing
    a policy is ultimately up to the user, it has been shown that running
    campaigns on minimal data can be useful in helping to choose an optimal
    policy given some objective. This class allows the user to rapidly test
    different policies (from easygp.policy) and evaluate their effectiveness
    given some initial dataset."""

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

    def __init__(
        self,
        X,
        y,
        alpha,
        bounds,
        policy,
        randomstate=0,
        gp_kwargs={"kernel": RBF(length_scale=1.0), "n_restarts_optimizer": 10},
    ):
        """Initializes the campaign.

        Parameters
        ----------
        X : np.ndarray
            Initial feature data. Should be of shape (n x n_features).
        y : np.ndarray
            Initial target data. Should be of shape (n x n_targets).
        alpha : np.ndarray
            Initial target noise (standard deviation). Should be of shape
            (n x n_targets), or a float.
        bounds : list of tuple
            The lower and upper bounds for each dimension. Should be of length
            of the number of features in the input data.
        policy : easygp.policy.BasePolicy
            The policy for running the campaign. This defines the procedure by
            which new points are sampled.
        randomstate : int, optional
            The randomstate for the underlying Gaussian Process instance that
            is treated as the ground truth.
        gp_kwargs : dict, optional
            Keyword arguments passed to the
            AutoscalingGaussianProcessRegressor.
        """

        # Set every input as a private attribute. Public attributes are handled
        # via properties
        for key, value in locals().items():
            if key != "self":
                setattr(self, f"_{key}", value)

        self._iteration = -1

        self.fit()

        self._performance_func = TargetPerformance()
        if self._policy._target is None:
            logger.warning("Policy has no target: assuming target is 0")
            self._performance_func.set_target(0.0)
        else:
            self._performance_func.set_target(self._policy._target)
        self._performance_func.set_weight(self._policy._weight)

        # Set the truth function based on the GP sampler
        self._truth = GPSampler(deepcopy(self._gp), self._randomstate)

    def fit(self):
        """Fits the internal Gaussian Process using the current stored data.

        Returns
        -------
        tuple
            Returns a message and boolean value indicating whether or not the
            fitting procedure finished with a warning.
        """

        t0 = time()
        self._gp = AutoscalingGaussianProcessRegressor(
            bounds=self._bounds,
            gp_kwargs=self._gp_kwargs,
            n_targets=self._y.shape[1],
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            self._gp.fit(self._X, self._y, self._alpha)
        self._iteration += 1
        dt = time() - t0
        for warn in caught_warnings:
            if FIT_WARNING in str(warn.message):
                msg = f"Model (bad fit) {self.gp.kernels} fit in {dt:.01} s"
                return msg, True
        return f"\tModel {self.gp.kernels} fit in {dt:.01} s", False

    def _update(self, X, y, alpha):
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
        n_restarts : int, optional
            The number of times the optimizer should be restarted in a new
            location while maximizing the acquisition function.
        ignore_criterion : float, optional
            This number determines when to ignore a suggested data point. If
            the mean absolute difference between any value currently in the
            campaign.X dataset, and the suggested value is less than this
            number, the suggested point will not be added to the campaign data.

        Returns
        -------
        list of float
            A list of the performance values acquired during the campaign.
        """

        performance = []
        fit_info = []
        fit_warnings = []

        logger.info(f"Beginning campaign (n={n})")
        logger.info(f"Policy is {self._policy.__class__.__name__}")

        points_too_close = 0

        for counter in tqdm(range(n), disable=_DEBUG):

            logger.debug(f"Beginning iteration {counter:03}")

            # Start with setting ybest if needed
            if isinstance(self._policy, RequiresYbest):
                to_max = self._policy.objective(self._y)
                y_best = self._y[np.argmax(to_max, axis=0)].item()
                self._policy.set_ybest(y_best)
                logger.debug(f"\ty-best set to {y_best}")

            # Suggest a new point
            new_X = self._policy.suggest(self._gp, n_restarts)
            new_X = new_X.reshape(-1, self._gp.n_features)

            diff = np.abs(self.X - new_X).sum(axis=1) < ignore_criterion
            if diff.sum() > 0:
                logger.debug(
                    f"\tProposed point {new_X} too close to existing data. "
                    "Ignoring."
                )
                points_too_close += 1
                p = self._performance_func(self._gp, self._truth, n_restarts)
                performance.append(p)
                continue

            # Get the performance given this new point
            p = self._performance_func(self._gp, self._truth, n_restarts)
            performance.append(p)

            # Get the new truth result for the suggested X value
            new_y = self._truth(new_X)
            logger.debug(f"\ty-value of proposed points {new_y}")

            # As for noise, use the average in the dataset plus/minus one
            # standard deviation
            avg_noise = np.array([np.mean(self._alpha)]) + np.random.normal(
                scale=self._alpha.std()
            )

            # Update the datasets stored in _X, _y, and _alpha
            self._update(new_X, new_y, avg_noise)

            # Refit on the new data
            msg, warning = self.fit()
            if warning:
                fit_warnings.append(msg)
            else:
                fit_info.append(msg)

        if points_too_close > 0:
            logger.warning(
                f"{points_too_close} too close to previous points in the "
                "dataset - those points were ignored"
            )

        for msg in fit_warnings:
            logger.warning(msg)

        return {
            "performance": performance,
            "info": fit_info,
            "warnings": fit_warnings,
        }
