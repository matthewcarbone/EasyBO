import botorch  # noqa
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils.transforms import (
    t_batch_mode_transform,
    concatenate_pending_points,
)

import torch

from easybo.utils import _to_float32_tensor, DEVICE
from easybo.logger import logger, _log_warnings
from easybo.gp import EasyGP


class XPendingError(Exception):
    ...


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


CUSTOM_AQ_MAPPING = {
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "MaxVar": _MaxVariance,
    "MaxVariance": _MaxVariance,
    "qMaxVar": _qMaxVariance,
    "qMaxVariance": _qMaxVariance,
}


@_log_warnings
def ask(
    *,
    model,
    bounds=[[0, 1]],
    acquisition_function="MaxVariance",
    X_pending=None,
    fixed_features=None,
    acquisition_function_kwargs=dict(),
    optimize_acqf_kwargs=dict(q=1, num_restarts=5, raw_samples=20),
    penalty_function=None,
    penalty_strength=0.1,
    terminate_on_fail=True,
    device=DEVICE,
):
    """Asks the model to sample the next point(s) based on the current state
    of the posterior and the given acquisition function.

    Parameters
    ----------
    model : EasyGP
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
    penalty_function : callable, optional
        A regularization applied to the acquisition funtion directly. This
        callable function takes the input coordinate as input. The larger
        the value of this function, the less that point is favored.
    penalty_strength : float, optional
        The strength of the penalty regularization.
    device : str
        The device on which to place any arrays passed to ``ask``.

    Returns
    -------
    numpy.ndarray
        The next point(s) to sample.

    Raises
    ------
    ValueError
        If an incorrect acqusition function name is provided.
    """

    logger.debug(f"ask queried with args: {locals()}")

    dims = len(bounds[0])
    bounds = torch.tensor(bounds).float().reshape(-1, 2).T
    logger.debug(f"ask bounds set to {bounds}")

    if isinstance(model, EasyGP):
        model = model.model

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

            acquisition_function = CUSTOM_AQ_MAPPING.get(acquisition_function)

            if acquisition_function is None:
                msg = (
                    "Unknown acquisition function alias "
                    f"{acquisition_function}"
                )
                logger.critical(msg)
                raise ValueError(msg)

    logger.debug(
        f"acquisition function in use: {acquisition_function.__name__}"
    )

    if X_pending is not None:
        X_pending = _to_float32_tensor(X_pending, device=device)

    aq = acquisition_function(
        model,
        X_pending=X_pending,
        **acquisition_function_kwargs,
    )

    if X_pending is not None and not isinstance(aq, MCAcquisitionFunction):
        klass = aq.__class__.__name__
        klass = klass.replace("_", "")
        logger.error(
            "You have passed X_pending to an acquisition function that does "
            "not inherit MCAcquisitionFunction. X_pending will be silently "
            "ignored! You passed acqusition function "
            f"{klass}, try e.g. q{klass}."
        )
        if terminate_on_fail:
            logger.critical("terminate_on_fail is True, throwing error")
            raise XPendingError

    if penalty_function is not None:
        aq = PenalizedAcquisitionFunction(
            aq, penalty_function, penalty_strength
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


# class SimulatedCampaign(MSONable):
#     """Runs a simulated Bayesian Optimization campaign."""

#     def __init__(
#         self,
#         *,
#         objective=lambda x: x,
#         bounds=[[0, 1]],
#         optimize_acqf_kwargs={
#             "q": 1,
#             "num_restarts": 5,
#             "raw_samples": 20,
#         },
#         weight=None,
#         device=DEVICE,
#         report=dict(),
#     ):
#         self._objective = objective
#         self._bounds = bounds
#         self._optimize_acqf_kwargs = optimize_acqf_kwargs
#         self._weight = weight
#         self._device = device
#         self._report = report

#     def run_single(
#         self,
#         *,
#         model,
#         acquisition_function,
#         acquisition_function_parameters,
#         iterations_per_dream=10,
#         total_dreams=10,
#     ):
#         keys, values = zip(*acquisition_function_parameters.items())
#         params = [dict(zip(keys, v)) for v in product(*values)]


# def run_simulated_campaign(*, model, acquisition_functions, samples=10):
#     """Summary

#     Parameters
#     ----------
#     model : SingleTaskGP
#         This is th emodel which has already been conditioned on some starting
#         data and will be used in the campaign.
#     acquisition_functions : list
#         A list of acquisition function names
#     samples : int, optional
#         Description
#     """

#     ...
