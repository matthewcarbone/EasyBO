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
        def __init__(self, *args, custom_weight=lambda x: 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self._custom_weight = custom_weight

        def forward(self, X):
            weight = self._custom_weight(X)
            if isinstance(weight, (float, int)):
                weight = torch.tensor(weight)
            weight = torch.atleast_1d(weight)
            forward_tensor = super().forward(X)

            # Hack to deal with the shapes
            returned = (weight.T * forward_tensor.T).T
            return returned.squeeze()

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
    acquisition_function_kwargs={"beta": 0.1},
    optimize_acqf_kwargs={
        "q": 1,
        "num_restarts": 5,
        "raw_samples": 20,
    },
    weight=lambda x: 1.0,
):
    """Asks the model to sample the next point(s) based on the current state
    of the posterior and the given acquisition function.

    Parameters
    ----------
    model : TYPE
        Description
    bounds : list, optional
        A list of tuple where the first entry of each tuple is the start of the
        bound and the second is the end.
    acquisition_function : TYPE
        Description
    acquisition_function_kwargs : dict, optional
        Description
    """

    bounds = torch.tensor(bounds).float().reshape(-1, 2).T

    # Instantiate assuming base of botorch.acquisition
    if isinstance(acquisition_function, str):

        try:
            acquisition_function = eval(
                f"botorch.acquisition.{acquisition_function}"
            )

        # Custom definitions
        except AttributeError:
            if acquisition_function == "EI":
                acquisition_function = ExpectedImprovement
            elif acquisition_function == "UCB":
                acquisition_function = UpperConfidenceBound
            elif acquisition_function == "MaxVar":
                acquisition_function = _MaxVariance
            elif acquisition_function == "qMaxVar":
                acquisition_function = _qMaxVariance
            else:
                raise ValueError(
                    "Unknown acquisition function alias "
                    f"{acquisition_function}"
                )

    # Add custom_weight method
    acquisition_function = acquisition_function_factory(acquisition_function)

    aq = acquisition_function(
        model, custom_weight=weight, **acquisition_function_kwargs
    )
    candidate, acq_value = optimize_acqf(
        aq, bounds=bounds, **optimize_acqf_kwargs
    )
    return candidate
