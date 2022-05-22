from contextlib import contextmanager

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from easygp import logger
from easygp.policy import MaxVariancePolicy


class EasyGP(GaussianProcessRegressor):
    """A lightweight wrapper for the `scikit-learn GaussianProcessRegressor
    <https://scikit-learn.org/stable/modules/generated/
    sklearn.gaussian_process.GaussianProcessRegressor.html
    #sklearn.gaussian_process.GaussianProcessRegressor>`_. Gaussian Processes
    can often be difficult to get working the first time a new user tries them.
    Part of this reason is due to ambiguities in choosing the kernels. This
    class abstracts away that difficulty by default, by making a simple choice
    for the form of the kernel function and by forcing data to be scaled by
    default. Specifically, the `default` assumptions of this class are:

    * Input features can be safely scaled to range :math:`[0, 1]` using
      :class:`sklearn.preprocessing.MinMaxScaler`.

    * Output targets can be safely scaled to a standard normal distribution
      using :class:`sklearn.preprocessing.StandardScaler`.

    * The form of the input features is `smooth`, and thus a radial basis
      function kernel (RBF) is a valid kernel choice with initial length scale
      of 1 (which is modified during fitting).

    We note however that any of these defaults can be overridden (though in
    that case it might make more sense to just use the standard
    ``sci-kit learn`` implementation).
    """

    @contextmanager
    def disable_scalers(self, *, disable_X, disable_y):
        """Context manager that disables the forward scaling of the input
        variable X."""

        self._scale_X = False if disable_X else True
        self._scale_y = False if disable_y else True
        try:
            yield
        finally:
            self._scale_X = True
            self._scale_y = True

    def __init__(self, **kwargs):
        if "kernel" not in kwargs.keys():
            kwargs["kernel"] = RBF(length_scale=1.0)
        super().__init__(**kwargs)
        if "alpha" in kwargs.keys():
            logger.warning(
                "alpha provided in __init__ will be overridden during fitting"
            )
        self._Xscaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._yscaler = StandardScaler()
        self._scale_X = True
        self._scale_y = True
        self._n_features = None
        self._n_targets = None

    def fit(self, X, y, alpha=1e-10):
        """A lightweight wrapper for `scikit-learn's Gaussian Process Regressor
        fit <https://scikit-learn.org/stable/modules/generated/
        sklearn.gaussian_process.GaussianProcessRegressor.html#
        sklearn.gaussian_process.GaussianProcessRegressor.fit>`_ method.
        The method scales the input and output values (and the noise term
        ``alpha``) as long as the user doesn't explicitly disable them).

        Parameters
        ----------
        X : np.ndarray
            Features to fit on.
        y : np.ndarray
            Targets to fit on.
        alpha : float or numpy.ndarray, optional
            Noise (variance), of the same shape as ``y``, or is a
            float (same noise for every target). Default is 1e-5 (to prevent
            numerical instability during the GP fitting process).
        """

        if len(X.shape) < 2:
            logger.error("X input shape should be 2d (samples x features)")
            return
        if len(y.shape) < 2:
            logger.error("y input shape should be 2d (samples x targets)")
            return
        if isinstance(alpha, np.ndarray):
            if y.shape[1] == 1 and len(alpha.shape) == 1:
                pass
            elif y.shape[1] == 1 and len(alpha.shape) == 2:
                alpha = alpha.squeeze()
            elif alpha.shape != y.shape:
                logger.error("alpha shape must be equal to y shape")
        if y.shape[1] > 1:
            logger.warning(
                "The sci-kit learn Gaussian Process (and EasyGP) does not "
                "truly handle multi-target objectives. Each target is treated "
                "as statistically independent from each other. This should be "
                "checked before fitting EasyGP on a multi-target y object."
            )

        self._n_features = X.shape[1]
        self._n_targets = y.shape[1]

        if self._scale_X:
            X = self._Xscaler.fit_transform(X)
        else:
            logger.warning("Disabling Xscaler for fitting is not recommended")

        if self._scale_y:
            y = self._yscaler.fit_transform(y)
            alpha = alpha / self._yscaler.scale_**2
        else:
            logger.warning("Disabling yscaler for fitting is not recommended")

        # self.alpha = (sd**2).squeeze()  # Override alpha
        self.alpha = alpha

        super().fit(X, y)

        if isinstance(self.kernel_, RBF):
            ls = self.kernel_.get_params()["length_scale"]
            if ls < 1e-3:
                logger.warning(
                    f"Small RBF kernel length scale {ls:.02f} could indicate a"
                    "poor fit"
                )

    def predict(self, X, return_std=False, return_cov=False):
        """A lightweight wrapper for `scikit-learn's Gaussian Process Regressor
        predict <https://scikit-learn.org/stable/modules/generated/
        sklearn.gaussian_process.GaussianProcessRegressor.html
        #sklearn.gaussian_process.GaussianProcessRegressor.predict>`_ method.
        This method performs the same scaling as
        :class:`.GaussianProcessRegressor.fit` unless the user explicitly
        disables the scalers.

        .. note::

            As per the scikit-learn API, ``return_std`` and ``return_cov``
            cannot both be specified simultaneously.

        Parameters
        ----------
        X : np.ndarray
            The data to predict on.
        return_std : bool, optional
            If True, returns the standard deviation of the model.
        return_cov : bool, optional
            If True, returns the covariance of the model.

        Returns
        -------
        tuple
            The mean prediction and either the standard deviation or covariance
            matrix.
        """

        # Preempt the error that will be thrown by sklearn
        if return_std and return_cov:
            logger.critical(
                "Both return_std and return_cov cannot be specified together"
            )

        X = X.reshape(-1, self._n_features)

        if self._scale_X:
            X = self._Xscaler.transform(X)

        if not return_std and not return_cov:
            pred = super().predict(X, return_std=False, return_cov=False)
            pred = pred.reshape(-1, self._n_targets)
            if self._scale_y:
                pred = self._yscaler.inverse_transform(pred)
            return pred.reshape(-1, self._n_targets)

        elif return_std:
            pred, std = super().predict(X, return_std=True, return_cov=False)
            pred = pred.reshape(-1, self._n_targets)
            if self._scale_y:
                pred = self._yscaler.inverse_transform(pred)
                std *= self._yscaler.scale_
            return pred.reshape(-1, self._n_targets), std.reshape(
                -1, self._n_targets
            )

        elif return_cov:
            pred, cov = super().predict(X, return_std=False, return_cov=True)
            pred = pred.reshape(-1, self._n_targets)
            if self._scale_y:
                pred = self._yscaler.inverse_transform(pred)
                cov *= self._yscaler.scale_**2
            return pred.reshape(-1, self._n_targets), cov

        else:
            logger.critical("Unknown error in predict")

    def sample_y(self, X, n_samples=1, random_state=0):
        """Takes a sample from the Gaussian Process. This code is custom
        written and does `not` wrap the ``scikit-learn`` library since it is
        easier to handle the scaling using the built-in
        :class:`.EasyGP.predict` method.

        Parameters
        ----------
        X : np.ndarray
            The data to predict on.
        n_samples : int, optional
            The number of random samples to draw from the Gaussian Processes.
        random_state : int, optional
            The random state which ensures reproducibility. Each unique number
            will produce a different sample.

        Returns
        -------
        np.ndarray
            The drawn samples. Will be of shape
            ``(len(X), num_targets, num_samples)`` if num_samples > 1, else
            just ``(len(X), num_targets)``.
        """

        X = X.reshape(-1, self._n_features)

        # Run the sampling myself -- This handles all of the scaling
        mu, cov = self.predict(X, return_cov=True)

        # Set the random state
        np.random.seed(random_state)

        # Sample for each target. Usually there will be only one target, but
        # this should be general
        if len(mu.shape) == 1 or mu.shape[1] == 1:
            mu = mu.reshape(-1, 1)
            cov = cov[..., None]

        return np.array(
            [
                np.random.multivariate_normal(
                    mu[:, ii], cov[..., ii], n_samples
                )
                for ii in range(mu.shape[1])
            ]
        ).squeeze()

    def sample_y_reproducibly(self, X, n_samples=1, random_state=0):
        """There is a subtlety when sampling from the Gaussian Process
        posterior that must be accounted for when using different sampling
        grids.

        .. warning::

            Even for the same random state, different input grids (X) will
            produce different samples from the posterior if using the
            scikit-learn `sample_y <https://scikit-learn.org/stable/modules/
            generated/sklearn.gaussian_process.GaussianProcessRegressor.html
            #sklearn.gaussian_process.GaussianProcessRegressor.
            sample_y>`_ method. This is likely due to how the random sampling
            works under the hood in sklearn. While it is not a bug, it is
            useful to have a completely reproducible method for producing the
            same curve even when grids differ in fineness.

        Here, the issue is addressed by explicitly setting the random state to
        the same value every time a new point x is sampled. This is slightly
        less efficient, but does consistent reproducible curves.

        Parameters
        ----------
        X : np.ndarray
            The data to predict on.
        n_samples : int, optional
            The number of random samples to take from the Gaussian Processes.
        random_state : int, optional
            The random state which ensures reproducibility. Each unique number
            will produce a different sample.

        Returns
        -------
        np.ndarray
            The drawn samples. Will be of shape
            ``(len(X), num_targets, num_samples)`` if num_samples > 1, else
            just ``(len(X), num_targets)``.
        """

        return np.array(
            [
                self.sample_y(xx.reshape(1, -1), n_samples, random_state)
                for xx in X
            ]
        )

    def suggest(self, policy=MaxVariancePolicy(), bounds=None, n_restarts=10):
        """This is a method used in Bayesian optimization and optimal
        experimental design for determining the optimal next point to sample
        such that some policy is followed and some acquisition function is
        maximized.

        Parameters
        ----------
        policy : easygp.policy.BasePolicy, optional
            The policy to follow.
        bounds : list of tuple, optional
            A list of tuple or lists that contains the minimum and maximum
            bound for those dimensions. The length of bounds should be equal to
            the number of input features of the function. E.g., for three
            features where each is between 0 and 1,
            ``bounds=[(0, 1), (0, 1), (0, 1)]``. If ``bounds`` is ``None``,
            then the default bounds are simply the minimum and maximum of what
            the EasyGP was fitted on.

            .. warning::

                The default behavior to find the ``bounds`` if they are not
                provided is to use the X-scaler to unscale the saved training
                data. If the EasyGP was originally fitted with scaling
                disabled, then scaling should also be disabled here.
        n_restarts : int, optional
            The number of times to randomly try the fitting procedure.

        Returns
        -------
        numpy.ndarray
            The suggested next point to sample.
        """

        if bounds is None:
            if self._scale_X:
                X = self._Xscaler.inverse_transform(self.X_train_)
            else:
                X = self.X_train_
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            bounds = [(m0, mf) for m0, mf in zip(mins, maxs)]

        return policy.suggest(self, bounds, n_restarts=n_restarts)
