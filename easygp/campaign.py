from copy import copy, deepcopy
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import getpid
from time import time
import warnings

import numpy as np
from tqdm import tqdm

from easygp import logger, disable_logger
from easygp.gp import EasyGP
from easygp.policy import _TargetPerformance


class GPSampler:
    """A method for deterministically sampling from a single instance of a
    Gaussian Process. Initializes the class with a Gaussian Process and random
    state variable, which should be set to not None to ensure deterministic
    calculations.

    Parameters
    ----------
    gp : easygp.gp.EasyGP
        The Gaussian Process regressor to use as the estimate for a single
        sample.
    random_state : int, optional
        The random state used for seeding the numpy random number
        generator.
    """

    @property
    def gp(self):
        return self._gp

    def __init__(self, gp, random_state=None):
        self._gp = gp
        self._random_state = random_state

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
            random_state=self._random_state,
        )


class Campaign:
    """Used for running optimization campaigns given some data. While choosing
    a policy is ultimately up to the user, it has been shown that running
    campaigns on minimal data can be useful in helping to choose an optimal
    policy given some objective. This class allows the user to rapidly test
    different policies (from :class:`easygp.policy`) and evaluate their
    effectiveness given some initial dataset.

    .. note::

        This campaign only allows for a single scalar target.

    Parameters
    ----------
    initial_X : numpy.ndarray
        Initial feature data. Should be of shape ``n`` x ``n_features``.
    initial_y : numpy.ndarray
        Initial target data. Should be of shape ``n`` x ``n_targets``.
    initial_alpha : numpy.ndarray
        Initial target noise (standard deviation). Should be of shape
        ``n`` x ``n_targets``, or a float.
    policy : easygp.policy.BasePolicy
        The policy for running the campaign. This defines the procedure by
        which new points are sampled.
    bounds : list of tuple
        The lower and upper bounds for each dimension. Should be of length
        of the number of features in the input data.
    random_state : int, optional
        The random_state for the underlying Gaussian Process instance that
        is treated as the ground truth.
    gp : easygp.gp.EasyGP, optional
        The EasyGP Gaussian Process object used for fitting the data.
    iteration : int, optional
        The current iteration of the campaign.
    """

    @property
    def gp(self):
        return self._gp

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
        policy,
        bounds=None,
        random_state=0,
        gp=EasyGP(n_restarts_optimizer=10),
        iteration=-1,
    ):
        self._X = initial_X.copy()
        self._y = initial_y.copy()
        self._n_features = self._X.shape[1]
        self._alpha = initial_alpha.copy()
        if bounds is None:
            mins = initial_X.min(axis=0)
            maxs = initial_X.max(axis=0)
            bounds = [(m0, mf) for m0, mf in zip(mins, maxs)]
        self._bounds = copy(bounds)
        self._policy = deepcopy(policy)
        self._random_state = random_state
        self._gp = deepcopy(gp)
        self._original_gp = deepcopy(gp)
        self._iteration = iteration
        self._fit()
        self._performance_func = _TargetPerformance()
        if self._policy._target is None:
            logger.warning(
                "Policy has no target- Saving performance function target to 0"
            )
            self._performance_func.set_target(0.0)
        else:
            self._performance_func.set_target(self._policy._target)

        # Set the truth function based on the GP sampler
        self._truth = GPSampler(deepcopy(self._gp), self._random_state)

    def _fit(self):
        """Fits the internal Gaussian Process using the current stored data.

        Returns
        -------
        dict
            A dict with two keys, one for the standard messages and one for
            warning messages, as output during the fitting processes.
        """

        self._gp = deepcopy(self._original_gp)
        warning_messages = []
        messages = []

        t0 = time()
        with warnings.catch_warnings(record=True) as caught_warnings:
            self._gp.fit(self._X, self._y, self._alpha)
        dt = time() - t0

        bad_fit = False
        FIT_WARNING = (
            "Decreasing the bound and calling fit again may find a better"
        )
        for warn in caught_warnings:
            if FIT_WARNING in str(warn.message):
                msg = f"(bad fit) {self._gp.kernel_} fit in {dt:.01} s"
                warning_messages.append(msg)
                bad_fit = True
        if not bad_fit:
            messages.append(f"{self._gp.kernel_} fit in {dt:.01} s")

        self._iteration += 1

        return {"messages": messages, "warnings": warning_messages}

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
        y = y.reshape(-1, 1)
        alpha = alpha.reshape(-1, 1)
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
                distance = self._policy.objective(self._y)
                y_best = self._y[np.argmax(distance, axis=0)].item()
                self._policy.set_ybest(y_best)
                logger.debug(f"y-best set to {y_best}")

            # Suggest a new point
            new_X = self._gp.suggest(self._policy, self._bounds, n_restarts)

            # Get the performance given this new point
            performance.append(
                self._performance_func(
                    self._gp, self._truth, self._bounds, n_restarts
                ).item()
            )

            # Get the new truth result for the suggested X value
            new_y = self._truth(new_X)
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
            fit_res = self._fit()
            fit_info.extend(fit_res["messages"])

            if counter == n - 1:
                fit_errors.extend(fit_res["warning_messages"])
            else:
                fit_warnings.extend(fit_res["warning_messages"])

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
