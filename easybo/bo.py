import botorch  # noqa
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils.transforms import (
    t_batch_mode_transform,
    concatenate_pending_points,
)
import torch

from easybo.utils import _to_float32_tensor, DEVICE
from easybo.logger import logger


def acquisition_function_factory(cls):
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
        def _weighted_result(self, x, acq_values):
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

            if self._custom_weight is None:
                return acq_values

            # Otherwise, we need to multiply the weight into the last
            # dimension element-wise
            N = x.shape[0]
            q = x.shape[1]
            d = x.shape[2]
            x_reshaped = x.reshape(-1, d)  # ~ [N x q, d]

            # _custom_weight treats each dimension like x[0], x[1], etc.
            # weights ~ [N x q]
            weights = self._custom_weight(x_reshaped)
            weights = weights.reshape(N, q).mean(axis=1)

            return acq_values * weights

        def __init__(self, *args, custom_weight=None, **kwargs):
            super().__init__(*args, **kwargs)

            # A constant does not change the behavior of the acqusition
            # function
            if isinstance(custom_weight, (int, float)):
                custom_weight = None

            self._custom_weight = custom_weight

        def forward(self, X):
            """Forward execution for the acquisition function. Note that
            ``X is a set of coordinates, where the dimensions are
            [num_restarts or raw_samples, q, dims]."""

            f = super().forward(X)
            return self._weighted_result(X, f)

    return AcquisitionFunction


class _MaxVariance(AnalyticAcquisitionFunction):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        view_shape = (
            mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        )
        return posterior.variance.view(view_shape)


class _qMaxVariance(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        sampler=None,
        objective=None,
        posterior_transform=None,
        X_pending=None,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = (obj - mean).abs()
        return ucb_samples.max(dim=-1)[0].mean(dim=0)


def ask(
    *,
    model,
    bounds=[[0, 1]],
    acquisition_function=UpperConfidenceBound,
    X_pending=None,
    fixed_features=None,
    acquisition_function_kwargs=dict(),
    optimize_acqf_kwargs={
        "q": 1,
        "num_restarts": 5,
        "raw_samples": 20,
    },
    weight=None,
    device=DEVICE,
):
    """Asks the model to sample the next point(s) based on the current state
    of the posterior and the given acquisition function.

    Parameters
    ----------
    model : SingleTaskGP
        The trained/conditioned model used to produce the next point.
    bounds : list, optional
        A list of tuple where the first entry of each tuple is the start of the
        bound and the second is the end.
    acquisition_function
        Either a ``botorch.acqusition`` function e.g. ``UpperConfidenceBound``,
        or a string matching the function name from that library.
    X_pending : array_like, optional
        These are samples that are "pending", meaning they will be run but have
        not been run yet. This is useful when doing joint optimization using
        MC-based acqusition functions. For example, if it is known that certain
        inputs will be run, but have not been run yet, ``X_pending`` should be
        set to those inputs. Optimization will then be performed assuming those
        points are pending. If the points to be run are given as an `n x d`
        matrix, this fixes certain rows. This is passed directly to the
        acquisition function.
    fixed_features : dict, optional
        This is a way to fix certain features during optimization. It is
        passed directly to the optimization scheme in BoTorch. If the points to
        be run are given as an `n x d` matrix, this fixes certain columns.
    acquisition_function_kwargs : dict, optional
        Other keyword arguments passed to the acquisition function.
    optimize_acqf_kwargs : dict, optional
        Other keyword arguments passed to ``optimize_acqf``.
    weight : callable, optional
        A custom weight applied to the acqusition function directly. This is
        callable function which takes the position as input. The larger the
        weight, the more that point is favored.
    device : str
        The device on which to place any arrays passed to ``ask``.

    Returns
    -------
    numpy.ndarray
        The next points to sample.

    Raises
    ------
    ValueError
        If an incorrect acqusition function name is provided.
    """

    logger.debug(f"ask provided with args: {locals()}")

    bounds = torch.tensor(bounds).float().reshape(-1, 2).T
    logger.debug(f"ask bounds set to {bounds}")

    # Instantiate assuming base of botorch.acquisition
    if isinstance(acquisition_function, str):

        try:
            acquisition_function = eval(
                f"botorch.acquisition.{acquisition_function}"
            )

        # Custom definitions
        except AttributeError:
            logger.debug(
                f"Acquisition function signature {acquisition_function} "
                "not found in botorch.acquisition"
            )
            if acquisition_function == "EI":
                acquisition_function = ExpectedImprovement
            elif acquisition_function == "UCB":
                acquisition_function = UpperConfidenceBound
            elif acquisition_function == "MaxVar":
                acquisition_function = _MaxVariance
            elif acquisition_function == "qMaxVar":
                acquisition_function = _qMaxVariance
            else:
                msg = (
                    "Unknown acquisition function alias "
                    f"{acquisition_function}"
                )
                logger.critical(msg)
                raise ValueError(msg)

    logger.debug(
        f"acquisition function in use: {acquisition_function.__name__}"
    )

    # Add custom_weight method
    acquisition_function = acquisition_function_factory(acquisition_function)

    if X_pending is not None:
        X_pending = _to_float32_tensor(X_pending, device=device)

    aq = acquisition_function(
        model,
        custom_weight=weight,
        X_pending=X_pending,
        **acquisition_function_kwargs,
    )

    candidate, acq_value = optimize_acqf(
        aq,
        bounds=bounds,
        fixed_features=fixed_features,
        **optimize_acqf_kwargs,
    )
    logger.debug(f"candidates: {candidate}")
    logger.debug(f"acquisition function value: {acq_value}")
    return candidate


def run_simulated_campaign(*, model, acquisition_functions, samples=10):
    """Summary

    Parameters
    ----------
    model : SingleTaskGP
        This is th emodel which has already been conditioned on some starting
        data and will be used in the campaign.
    acquisition_functions : list
        A list of acquisition function names
    samples : int, optional
        Description
    """

    ...
