import botorch  # noqa
from botorch.acquisition import UpperConfidenceBound
from botorch.utils.transforms import t_batch_mode_transform
import torch


def _acquisition_function_factory_regularization(cls):
    """Intended as a decorator for adding a ``custom_weight`` attribute to the
    provided class type. The user can set ``custom_weight`` as a function of
    the grid coordinate, which can allow one to weight the resultant
    acquisition function by e.g. the cost of moving a motor to a particular
    position.

    Returns
    -------
    type
        As this is essentially a metaclass, this returns the object
        ``AcqusitionFunction`` which needs to be instantiated, e.g.,
        ``AcquisitionFunction(...)``
    """

    class AcquisitionFunction(cls):
        def _regularized_result(self, x, acq_values):
            """Executes the weighting scheme before sending the result of ``x``
            to the ``super().forward()`` method.

            Parameters
            ----------
            x : torch.tensor
                A torch tensor of shape
                [num_restarts or raw_samples, q, n_dimensions].
            acq_values : torch.tensor
                The values of the acquisition function.

            Returns
            -------
            torch.tensor
            """

            if self._regularizing_function is None:
                return acq_values

            # Otherwise, we need to multiply the weight into the last
            # dimension element-wise
            N = x.shape[0]
            q = x.shape[1]
            d = x.shape[2]
            x_reshaped = x.reshape(-1, d)  # ~ [N x q, d]

            # _regularizing_function treats each dimension like x[0], x[1], ...
            # weights ~ [N x q]
            weights = self._regularizing_function(x_reshaped)
            weights = weights.reshape(N, q).mean(axis=1)

            return acq_values - weights * self._regularization_strength

        def __init__(
            self,
            *args,
            regularizing_function=None,
            regularization_strength=0.1,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            # A constant does not change the behavior of the acqusition
            # function
            if isinstance(regularizing_function, (int, float)):
                regularizing_function = None

            self._regularizing_function = regularizing_function
            self._regularization_strength = regularization_strength

        def forward(self, X):
            """Forward execution for the acquisition function. Note that
            ``X is a set of coordinates, where the dimensions are
            [num_restarts or raw_samples, q, dims]."""

            f = super().forward(X)
            return self._regularized_result(X, f)

    return AcquisitionFunction


class WeightedMaxVar(UpperConfidenceBound):
    def __init__(
        self, model, alpha=10.0, mu=0.2, sd=0.05, current_data=None, **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self._alpha = alpha
        self._mu = mu
        self._sd = sd
        if current_data is not None:
            self._current_data = torch.tensor(current_data, dtype=torch.double)
        else:
            self._current_data = None

    @staticmethod
    def regularizer(proposed, current, alpha, mu, sd):

        # dists is proposed[0] x current[0]
        # It represents the L2 distance between each proposed point
        # And each current point
        _proposed = proposed.view(proposed.shape[0], 1, proposed.shape[-1])
        _current = current.view(1, *current.shape)
        dists = _proposed - _current
        dists = torch.sqrt(torch.sum(dists**2, axis=2))
        # return dists

        gauss = 1.0 - torch.exp(-((dists - mu) ** 2) / 2.0 / sd**2) * alpha

        # return gauss.min(axis=1)[0].double()
        return gauss.mean(axis=1).double()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):

        f = super().forward(X)

        if self._current_data is None or self._alpha == 0.0:
            return f

        weight = WeightedMaxVar.regularizer(
            X, self._current_data, self._alpha, self._mu, self._sd
        )

        return f - weight

        # # Execute the weighting scheme
        # mean = self.model.posterior(
        #     X=X, posterior_transform=self.posterior_transform
        # ).mean.view(f.shape)

        # weighted_mean = mean * weight

        # if self.maximize:
        #     return f - mean + weighted_mean
        # else:
        #     return f + mean - weighted_mean


class ProximityWeightedUpperConfidenceBound(UpperConfidenceBound):
    def __init__(
        self,
        model,
        beta=0.1,
        current_data=None,
        sigmoid_cutoff=0.2,
        sigmoid_scale=100.0,
        **kwargs,
    ):
        super().__init__(model=model, beta=beta, **kwargs)
        if current_data is not None:
            self._current_data = torch.tensor(current_data, dtype=torch.double)
        else:
            self._current_data = None
        self._sigmoid_cutoff = sigmoid_cutoff
        self._sigmoid_scale = sigmoid_scale

    # @staticmethod
    # def closeness_weighting(proposed, current, cutoff=0.2, scale=100):

    #     # dists is proposed[0] x current[0]
    #     # It represents the L2 distance between each proposed point
    #     # And each current point
    #     _proposed = proposed.view(proposed.shape[0], 1, proposed.shape[-1])
    #     _current = current.view(1, *current.shape)
    #     dists = _proposed - _current
    #     dists = torch.sqrt(torch.sum(dists**2, axis=2))
    #     # return dists

    #     # Weighting scheme which goes to 0 as distance goes to 0
    #     # and goes to 1 as distnace gets larger past some cutoff
    #     sig = torch.nan_to_num(
    #         1.0 / (1.0 + torch.exp(-(dists - cutoff) * scale))
    #     )

    #     return sig.min(axis=1)[0].double()

    @staticmethod
    def closeness_weighting(proposed, current, mu=0.1, sd=0.2):

        # dists is proposed[0] x current[0]
        # It represents the L2 distance between each proposed point
        # And each current point
        _proposed = proposed.view(proposed.shape[0], 1, proposed.shape[-1])
        _current = current.view(1, *current.shape)
        dists = _proposed - _current
        dists = torch.sqrt(torch.sum(dists**2, axis=2))
        # return dists

        gauss = torch.exp(-((dists - mu) ** 2) / 2.0 / sd**2)

        # return gauss.min(axis=1)[0].double()
        return gauss.min(axis=1)[0].double()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):

        # If no data is passed, we just use an active learning strategy
        if self._current_data is None:
            return _MaxVariance._forward_with_args_as_explicit(
                self.model, self.posterior_transform, X
            )

        # Otherwise we use a much more sophisticated weighting scheme
        weights = ProximityWeightedUpperConfidenceBound.closeness_weighting(
            X, self._current_data, self._sigmoid_cutoff, self._sigmoid_scale
        )
        f = super().forward(X)
        return f * weights

