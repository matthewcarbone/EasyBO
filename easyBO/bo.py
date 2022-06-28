from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform, \
    concatenate_pending_points

import torch


class _MaxVariance(AnalyticAcquisitionFunction):

    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        view_shape = \
            mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
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
    }
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

    bounds = torch.tensor(bounds, dtype=torch.float32).reshape(-1, 2).T

    if isinstance(acquisition_function, str):
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
                f"Unknown acquisition function alias {acquisition_function}"
            )

    aq = acquisition_function(model, **acquisition_function_kwargs)
    candidate, acq_value = optimize_acqf(
        aq,
        bounds=bounds,
        **optimize_acqf_kwargs
    )
    return candidate
