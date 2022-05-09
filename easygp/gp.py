from contextlib import contextmanager
from copy import copy, deepcopy
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import getpid
from tqdm import tqdm
from time import time
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_gp
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from easygp import logger, disable_logger
from easygp.policy import TargetPerformance


class AutoscalingGaussianProcessRegressor:
    """A lightweight wrapper for the sklearn Gaussian Process Regressor which
    takes an extra input, n_features, and basically ensures that all
    predictions reshape the inputs properly so that they're compatible with the
    base GP. It also automatically scales all inputs to the 0 -> 1 support, and
    executes a StandardScaler on the targets, scaling the noise appropriately
    as well."""

    @property
    def n_features(self):
        """The number of features that the GP is constrained to train on.

        Returns
        -------
        int
        """

        return len(self._bounds)

    @property
    def gps(self):
        """A list of the
        :class:`sklearn.gaussian_process.GaussianProcessRegressor` objects.

        Returns
        -------
        list of sklearn.gaussian_process.GaussianProcessRegressor
        """

        return self._gp

    @property
    def bounds(self):
        """The boundaries of the input features. Bounds has the form::

            [(d1_min, d1_max), (d2_min, d2_max), ...]

        where each entry in the list is a tuple containing the minimum and
        maximum allowed value for that dimension.

        Returns
        -------
        list of tuple
        """

        return self._bounds

    @property
    def kernel(self):
        """A list of the Gaussian Process Regressor's string representations of
        the kernels

        Returns
        -------
        list of str
        """

        return self._gp.kernel_

    def __init__(
        self,
        *,
        bounds,
        gp_kwargs={
            "kernel": RBF(length_scale=1.0),
            "n_restarts_optimizer": 10,
        },
    ):
        self._bounds = bounds
        self._gp_kwargs = gp_kwargs
        self._Xscaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._yscaler = StandardScaler()
        self._gp = None

        self._scale_X = True
        self._scale_y = True

    @contextmanager
    def disable_Xscaler(self):
        """Context manager that disables the forward scaling of the input
        variable X."""

        self._scale_X = False
        try:
            yield None
        finally:
            self._scale_X = True

    @contextmanager
    def disable_yscaler(self):
        """Context manager that disables the inverse scaling of the output
        variable y as well as the noise term."""

        self._scale_y = False
        try:
            yield None
        finally:
            self._scale_y = True

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
            Targets to fit multiple, independent GPs on. Of shape N x 1.
        alpha : float or np.ndarray, optional
            Noise (standard deviation), of shape N x 1 or is a float
            (same noise for every target). Default is 1e-5 (to prevent
            numerical instability during the GP fitting process).
        """

        if len(y.shape) > 1:
            assert y.shape[1] == 1
        y = y.reshape(-1, 1)

        X = X.reshape(-1, self.n_features)

        if self._scale_X:
            X = self._Xscaler.fit_transform(X)
        else:
            logger.warning("Disabling Xscaler for fitting is not recommended")

        if self._scale_y:
            y = self._yscaler.fit_transform(y.reshape(-1, 1))
            alpha = alpha / self._yscaler.scale_
        else:
            logger.warning("Disabling yscaler for fitting is not recommended")

        self._gp = sklearn_gp(alpha=alpha.squeeze() ** 2, **self._gp_kwargs)
        self._gp.fit(X, y)

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

        X = X.reshape(-1, self.n_features)

        if self._scale_X:
            X = self._Xscaler.transform(X)

        pred, std_or_cov = self._gp.predict(X, return_std, not return_std)

        if return_std and self._scale_y:
            std_or_cov *= self._yscaler.scale_
            std_or_cov = std_or_cov.reshape(-1, 1)

        # Return a list of covariance matrices in that case
        elif self._scale_y:
            std_or_cov *= self._yscaler.scale_**2

        if self._scale_y:
            pred = self._yscaler.inverse_transform(pred)

        return pred, std_or_cov

    def sample_y(self, X, n_samples=1, random_state=0):
        """Samples a single instance of the Gaussian Processes.

        Parameters
        ----------
        X : np.ndarray
            Input feature array of shape N x N_features.
        n_samples : int, optional
            The number of random samples to take from the Gaussian Processes.
            Default is 1.
        random_state : int, optional
            The random state which ensures reproducibility. Each unique number
            will produce a different samlple. Default is 0.

        Returns
        -------
        np.ndarray
            The resultant samples. Will be of shape
            len(X) x num_targets x num_samples if num_samples > 1, else
            just len(X) x num_targets.
        """

        X = X.reshape(-1, self.n_features)

        if self._scale_X:
            X = self._Xscaler.transform(X)
        y = self._gp.sample_y(
            X, n_samples=n_samples, random_state=random_state
        ).reshape(-1, 1)
        if self._scale_y:
            y = self._yscaler.inverse_transform(y)
        return y

    def sample_y_reproducibly(self, X, n_samples=1, random_state=0):
        """There is a subtlety when sampling from the Gaussian Process
        posterior that must be accounted for when using different sampling
        grids.

        .. warning::

            Even for the same random state, different input grids (X) will
            produce different samples from the posterior if using the sklearn
            `sample_y` method. This is likely due to how the random sampling
            works under the hood in sklearn. While it is not a bug, we need a
            different way of doing the sampling for the campaigning.

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
        random_state : int, optional
            The random state which ensures reproducibility. Each unique number
            will produce a different samlple. Default is 0.

        Returns
        -------
        np.ndarray
            The resultant samples. Will be of shape
            len(X) x num_targets x num_samples if num_samples > 1, else
            just len(X) x num_targets.
        """

        X = X.reshape(-1, self.n_features)

        return np.array(
            [
                self.sample_y(
                    xx.reshape(1, -1), n_samples, random_state
                ).squeeze()
                for xx in X
            ]
        )


class GPSampler:
    """A method for deterministically sampling from a single instance of a
    Gaussian Process."""

    @property
    def gp(self):
        return self._gp

    def __init__(self, gp, random_state=None):
        """Initializes the class with a Gaussian Process and random state
        variable, which should be set to not None to ensure deterministic
        calculations.

        Parameters
        ----------
        gp : AutoscalingGaussianProcessRegressor
            The Gaussian Process regressor to use.
        random_state : int, optional
            The random state used for seeding the numpy random number
            generator.
        """

        # Train nearest neighbor regressor on samples over dense grid
        self._gp = gp
        self.random_state = random_state

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
            x,
            n_samples=1,
            random_state=self.random_state,
        )


FIT_WARNING = "Decreasing the bound and calling fit again may find a better"


class _TargetMethod:
    def __init__(self, gps, target_function, target_function_uncertainty):
        self._gps = gps
        self._target_function = target_function
        self._target_function_uncertainty = target_function_uncertainty

    def predict(self, X):
        preds = []
        errors = []
        for gp in self._gps:
            mu, std = gp.predict(X)
            preds.append(mu)
            errors.append(std)
        preds = np.array(preds).reshape(-1, self._n_targets)
        errors = np.array(errors).reshape(-1, self._n_targets)
        mu = self._target_function(preds)
        sd = self._target_function_uncertainty(preds, errors)
        return mu, sd

    def __call__(self, X):
        preds = []
        for gp in self._gps:
            mu, std = gp.sample_y_reproducibly(
                X, n_samples=1, random_state=self.random_state
            )
            preds.append(mu)
        preds = np.array(preds).reshape(-1, self._n_targets)
        return self._target_function(preds)


class Campaign:
    """Used for running optimization campaigns given some data. While choosing
    a policy is ultimately up to the user, it has been shown that running
    campaigns on minimal data can be useful in helping to choose an optimal
    policy given some objective. This class allows the user to rapidly test
    different policies (from easygp.policy) and evaluate their effectiveness
    given some initial dataset."""

    @property
    def gps(self):
        return self._gps

    @gps.setter
    def gps(self, x):
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
    def random_state(self):
        return self._random_state

    def __init__(
        self,
        initial_X,
        initial_y,
        initial_alpha,
        bounds,
        policy,
        random_state=0,
        gp_kwargs={
            "kernel": RBF(length_scale=1.0),
            "n_restarts_optimizer": 10,
        },
        iteration=-1,
        target_function=None,
        target_function_uncertainty=None,
    ):
        """Initializes the campaign.

        Parameters
        ----------
        initial_X : numpy.ndarray
            Initial feature data. Should be of shape ``n`` x ``n_features``.
        initial_y : numpy.ndarray
            Initial target data. Should be of shape ``n`` x ``n_targets``.
        initial_alpha : numpy.ndarray
            Initial target noise (standard deviation). Should be of shape
            ``n`` x ``n_targets``, or a float.
        bounds : list of tuple
            The lower and upper bounds for each dimension. Should be of length
            of the number of features in the input data.
        policy : easygp.policy.BasePolicy
            The policy for running the campaign. This defines the procedure by
            which new points are sampled.
        random_state : int, optional
            The random_state for the underlying Gaussian Process instance that
            is treated as the ground truth.
        gp_kwargs : dict, optional
            Keyword arguments passed to the
            AutoscalingGaussianProcessRegressor.
        iteration : int, optional
            The number of times that ``fit`` has been called.
        target_function : callable, optional
            If the number of targets is more than one, this is required. The
            ``target_function`` should take a single array as input, of shape
            ``N`` x ``n_features``. It is used as the overall output of the
            models for campaigning. If None, and the number of targets is 1,
            then the outputs of the GP will be used.
        target_function_uncertainty : callable, optional
            Similar to ``target_function``, but for the uncertainties. This
            should be calculated using standard propagation of errors on the
            ``target_function``. This function takes two inputs: one for the
            values, one for the errors.
        """

        # Set every input as a private attribute. Public attributes are handled
        # via properties
        self._X = initial_X.copy()
        self._y = initial_y.copy()
        self._n_targets = self._y.shape[1]
        self._n_features = self._X.shape[1]
        self._alpha = initial_alpha.copy()
        self._bounds = copy(bounds)
        self._policy = deepcopy(policy)
        self._random_state = random_state
        self._gp_kwargs = copy(gp_kwargs)
        self._iteration = iteration
        self._gps = None
        self.fit()
        self._performance_func = TargetPerformance()
        if self._policy._target is None:
            logger.warning(
                "Policy has no target- Saving performance function target to 0"
            )
            self._performance_func.set_target(0.0)
        else:
            self._performance_func.set_target(self._policy._target)

        # Set the target functions and their uncertainties
        self._target_function = target_function
        self._target_function_uncertainty = target_function_uncertainty

        if self._target_function is None:
            if self._n_targets == 1:
                klass = self._gps[0]
            else:
                logger.critical("Invalid option for target_function")
        else:
            klass = _TargetMethod(
                self._gps,
                self._target_function,
                self._target_function_uncertainty,
            )

        # Set the truth function based on the GP sampler
        self._truth = GPSampler(deepcopy(klass), self._random_state)

    def fit(self):
        """Fits the internal Gaussian Process using the current stored data.

        Returns
        -------
        tuple
            Returns two lists, one for the standard messages and one for
            warning messages, as output during the fitting processes.
        """

        self._gps = []
        warning_messages = []
        messages = []

        for ii in range(self._n_targets):

            t0 = time()
            _gp = AutoscalingGaussianProcessRegressor(
                bounds=self._bounds, gp_kwargs=self._gp_kwargs
            )
            dt = time() - t0

            with warnings.catch_warnings(record=True) as caught_warnings:
                _gp.fit(
                    self._X,
                    self._y[:, ii].reshape(-1, 1),
                    self._alpha[:, ii].squeeze(),
                )
                self._gps.append(_gp)

            bad_fit = False
            for warn in caught_warnings:
                if FIT_WARNING in str(warn.message):
                    msg = f"{ii} (bad fit) {_gp.kernel} fit in {dt:.01} s"
                    warning_messages.append(msg)
                    bad_fit = True
            if not bad_fit:
                messages.append(f"{ii} {_gp.kernel} fit in {dt:.01} s")

        self._iteration += 1

        return messages, warning_messages

    def predict(self, X):
        """Runs a prediction using the most up-to-date version of the
        Gaussian Processes contained in the campaign.

        Parameters
        ----------
        X : numpy.array
        """

        N = X.shape[0]
        preds = []
        errors = []
        for ii in range(self._n_targets):
            gp = self._gps[ii]
            mu, sd = gp.predict(X)
            preds.append(mu)
            errors.append(sd)
        preds = np.array(preds).reshape(N, self._n_targets)
        errors = np.array(errors).reshape(N, self._n_targets)
        return preds, errors

    def _update(self, X, y, alpha):
        """Updates the data with new X, y and alpha values. The data is always
        assumed to be unscaled.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        alpha : np.ndarray
        """

        X = X.reshape(-1, self._n_features)
        y = y.reshape(-1, self._n_targets)
        alpha = alpha.reshape(-1, self._n_targets)
        self._X = np.concatenate([self._X, X], axis=0)
        self._y = np.concatenate([self._y, y], axis=0)
        self._alpha = np.concatenate([self._alpha, alpha], axis=0)

    def run(self, n=10, n_restarts=10, disable_tqdm=False):
        """Runs the campaign.

        Parameters
        ----------
        n : int, optional
            The number of experiments to run.
        n_restarts : int, optional
            The number of times the optimizer should be restarted in a new
            location while maximizing the acquisition function.
        disable_tqdm : bool, optional
            If True, force-disables the progress bar regardless of the
            debugging status.

        Returns
        -------
        dict
            A dictionary with the results of the campaign.
        """

        t0 = time()

        performance = []
        fit_info = []
        fit_warnings = []
        fit_errors = []

        logger.info(f"Beginning campaign (n={n})")
        logger.info(f"Policy is {self._policy.__class__.__name__}")

        for counter in tqdm(range(n), disable=disable_tqdm):

            logger.debug(f"Beginning iteration {counter:03}")

            # Start with setting ybest if needed
            if hasattr(self._policy, "set_ybest"):
                to_max = self._policy.objective(self._y)
                y_best = self._y[np.argmax(to_max, axis=0)].item()
                self._policy.set_ybest(y_best)
                logger.debug(f"y-best set to {y_best}")

            # No matter what, klass below will have a predict() method that
            # returns a prediction for mu and sd
            if self._target_function is None:
                if self._n_targets == 1:
                    klass = self._gps[0]
                else:
                    logger.critical("Invalid option for target_function")
            else:
                klass = _TargetMethod(
                    self._gps,
                    self._target_function,
                    self._target_function_uncertainty,
                )

            # Suggest a new point
            new_X = self._policy.suggest(klass, n_restarts)
            new_X = new_X.reshape(-1, self._n_features)

            # Get the performance given this new point
            p = self._performance_func(klass, self._truth, n_restarts)
            performance.append(p.item())

            # Get the new truth result for the suggested X value
            new_y = self._truth(new_X).reshape(-1, 1)
            logger.debug(f"y-value of proposed points {new_y}")

            # As for noise, use the average in the dataset plus/minus one
            # standard deviation
            avg_noise = np.array(
                [np.mean(self._alpha, axis=0)]
            ) + np.random.normal(scale=self._alpha.std(axis=0))

            # Update the datasets stored in _X, _y, and _alpha
            # print(new_X.shape, new_y.shape, avg_noise.shape)
            # print(self.X.shape, self.y.shape, self.alpha.shape)
            self._update(new_X, new_y, avg_noise)

            # Refit on the new data and keep track of any warnings
            messages, warning_messages = self.fit()
            fit_info.extend(messages)

            if counter == n - 1:
                fit_errors.extend(warning_messages)
            else:
                fit_warnings.extend(warning_messages)

        for msg in fit_warnings:
            logger.warning(msg)
        for msg in fit_errors:
            logger.error(msg)

        dt = time() - t0

        return {
            "performance": performance,
            "info": fit_info,
            "warnings": fit_warnings,
            "errors": fit_errors,
            "elapsed": dt,
            "pid": getpid(),
        }


class MultiCampaign:
    def __init__(self, campaigns):
        """Initializes the MultiCampaign.

        Parameters
        ----------
        campaigns : list of Campaign
            A list of the :class:`Campaign` classes. Each of these classes
            should have a different random state. If not, an error will be
            logged.
        """

        self._campaigns = campaigns
        random_states = [cc.random_state for cc in self._campaigns]
        if len(np.unique(random_states)) != len(random_states):
            logger.error("Campaigns do not contain all unique random_states")

    def run(self, n, n_restarts=10, n_jobs=cpu_count() // 2):
        """Executes the campaigns.

        Parameters
        ----------
        n_jobs : TYPE, optional
            Description
        """

        def _run_wrapper(xx, n=n, n_restarts=n_restarts):
            with disable_logger():
                res = xx.run(n, n_restarts, disable_tqdm=True)
                return res, deepcopy(xx)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_wrapper)(xx) for xx in self._campaigns
        )

        # The campaigns in the current class were not actually modified in
        # memory. They need to be reset to the status of what was returned by
        # joblib's Parallel + delayed
        self._campaigns = [xx[1] for xx in results]

        # The true results are in the first entry of the returned list:
        return [xx[0] for xx in results]
