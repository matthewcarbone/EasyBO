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
      :class:`sklearn.preprocessing.MinMaxScaler`. This allows for a consistent
      length scale between all features.

    * Output targets can be safely scaled to a standard normal distribution
      using :class:`sklearn.preprocessing.StandardScaler`.

    * The form of the input features is `smooth`, and thus a radial basis
      function kernel (RBF) is a valid kernel choice with initial length scale
      of 1 (which is modified during fitting).

    We note however that any of these defaults can be overridden (though in
    that case it might make more sense to just use the standard
    ``scikit-learn`` implementation).

    .. note::

        The :class:`.EasyGP` object only handles single-target objectives.
        This is because the ``scikit-learn`` implementation of the Gaussian
        Process Regressor does not handle correlated targets. In general,
        it is important to check that targets are uncorrelated when using this
        method, but in the case of :class:`.EasyGP`, it is simply prevented
        outright. Multi-target optimizations are highly nontrivial, though to
        predict multiple targets, one should simply train multiple
        :class:`.EasyGP` objects, one for each target, though correlation
        information between targets will not be utilized.
    """

    @contextmanager
    def disable_scalers(self, *, disable_X, disable_y):
        """Context manager that disables the forward scaling of the input
        variable ``X`` and the output variable ``y``.

        Parameters
        ----------
        disable_X : bool
            If ``True``, disables the forward scaling of ``X``.
        disable_y : bool
            If ``True``, disables the inverse scaling of ``y``.
        """

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
            float (same noise for every target). Default is 1e-10 (to prevent
            numerical instability during the GP fitting process).
        """

        if len(X.shape) < 2:
            logger.error("X input shape should be 2d (samples, features)")
            return

        if len(y.shape) == 1:
            pass
        elif len(y.shape) > 1 and y.shape[1] > 1:
            logger.error(
                "y input shape should be of either (samples, ) or (samples, 1)"
            )
            logger.error(
                "The scikit-learn Gaussian Process (and EasyGP) does not "
                "truly handle multi-target objectives. Each target is treated "
                "as statistically independent from each other. This should be "
                "checked before fitting EasyGP on a multi-target y object."
            )
            return

        if isinstance(alpha, np.ndarray):
            if len(alpha.shape) == 1:
                pass
            elif len(alpha.shape) == 2:
                alpha = alpha.squeeze()
            elif alpha.shape != y.shape:
                logger.error("alpha shape must be equal to y shape")
                return

        self._n_features = X.shape[1]

        if self._scale_X:
            X = self._Xscaler.fit_transform(X)
        else:
            logger.warning("Disabling Xscaler for fitting is not recommended")

        if self._scale_y:
            y = self._yscaler.fit_transform(y.reshape(-1, 1)).squeeze()
            alpha = alpha / self._yscaler.scale_**2
        else:
            logger.warning("Disabling yscaler for fitting is not recommended")

        # self.alpha = (sd**2).squeeze()  # Override alpha
        self.alpha = alpha

        logger.debug(f"Fitting on shapes X ({X.shape}) -> y ({y.shape})")
        super().fit(X, y)

    def predict(self, X, return_std=False, return_cov=False):
        """A lightweight wrapper for `scikit-learn's Gaussian Process Regressor
        predict <https://scikit-learn.org/stable/modules/generated/
        sklearn.gaussian_process.GaussianProcessRegressor.html
        #sklearn.gaussian_process.GaussianProcessRegressor.predict>`_ method.
        This method performs the same scaling as
        :class:`.GaussianProcessRegressor.fit` unless the user explicitly
        disables the scalers.

        .. note::

            As per the ``scikit-learn`` API, ``return_std`` and ``return_cov``
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
            logger.error(
                "Both return_std and return_cov cannot be specified together"
            )
            return

        X = X.reshape(-1, self._n_features)

        if self._scale_X:
            X = self._Xscaler.transform(X)

        if not return_std and not return_cov:
            pred = super().predict(X, return_std=False, return_cov=False)
            if self._scale_y:
                pred = self._yscaler.inverse_transform(pred.reshape(-1, 1))
            return pred

        elif return_std:
            pred, std = super().predict(X, return_std=True, return_cov=False)
            if self._scale_y:
                pred = self._yscaler.inverse_transform(pred)
                std *= self._yscaler.scale_
            return pred, std

        elif return_cov:
            pred, cov = super().predict(X, return_std=False, return_cov=True)
            if self._scale_y:
                pred = self._yscaler.inverse_transform(pred)
                cov *= self._yscaler.scale_**2
            return pred, cov

        else:
            logger.error("Unknown error in predict")

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
            The drawn samples, which will be of shape
            ``(X.shape[0], n_samples)``.
        """

        # Run the sampling myself -- This handles all of the scaling
        mu, cov = self.predict(X, return_cov=True)

        # Set the random state
        np.random.seed(random_state)

        # Get the sample
        return np.random.multivariate_normal(mu, cov, n_samples).T

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
                self.sample_y(
                    xx.reshape(1, self._n_features), n_samples, random_state
                )
                for xx in X
            ]
        ).reshape(X.shape[0], n_samples)

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
            the number of input features of the function.
        n_restarts : int, optional
            The number of times to randomly try the fitting procedure.

        Returns
        -------
        numpy.ndarray
            The suggested next point to sample.
        """

        if bounds is None:
            new_bounds = [[0.0, 1.0] for _ in range(self._n_features)]

        else:
            # Bounds is a list of tuples, where the list is of the same length
            # as the number of dimensions as X, and the list of each tuple
            # is 2 (min and max).
            new_bounds = np.array(bounds)  # n_dim x 2
            new_bounds = self._Xscaler.transform(new_bounds.T)  # 2 x n_dim
            new_bounds = new_bounds.T.tolist()

        with self.disable_scalers(disable_X=True, disable_y=False):
            suggested = policy.suggest(self, new_bounds, n_restarts=n_restarts)

        # Need to then unscale the suggested point
        return self._Xscaler.inverse_transform(
            suggested.reshape(-1, self._n_features)
        )
