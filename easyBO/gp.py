"""Wrappers for the gpytorch and botorch libraries. See
`here <https://botorch.org/docs/models>`_ for important details about the types
of models that we wrap.

Gaussian Processes can often be difficult to get working the first time a new
user tries them, e.g. ambiguities in choosing the kernels. The classes here
abstract away that difficulty (and others) by default.
"""

import torch
import gpytorch
from botorch.models import SingleTaskGP


from easyBO.utils import _to_float32_tensor, DEVICE


def get_single_task_gp_regressor(
    *,
    train_x,
    train_y,
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    mean_module=gpytorch.means.ConstantMean(),
    covar_module=gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel()
    ),
    device=DEVICE,
    **kwargs
):
    """Summary

    Parameters
    ----------
    train_x : array_like
        The inputs.
    train_y : array_like
        The targets.
    likelihood : gpytorch.likelihoods, optional
        Likelihood for initializing the GP. Recommended to keep the default:
        ``gpytorch.likelihoods.GaussianLikelihood()``.
    mean_module : gpytorch.means.Mean, optional
        The mean function of the GP. See `here <https://docs.gpytorch.ai/en/
        stable/means.html>`_ for more details.
    covar_module : gpytorch.kernels, optional
        Kernel used in the covariance function.
    device : str, optional
        The device to put the model on. Defaults to "cuda" if a GPU is
        available, else "cpu".
    **kwargs
        Extra keyword arguments to pass to ``SingleTaskGP``.
    """

    x = _to_float32_tensor(train_x, device=device)
    y = _to_float32_tensor(train_y, device=device)

    model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module,
        **kwargs,
    )

    return model.to(device)


def train_gp_hyperparameters(
    *,
    model,
    train_x=None,
    train_y=None,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.1},
    training_iter=100,
    print_frequency=5,
    device=DEVICE,
    verbose=True
):
    """Summary

    Parameters
    ----------
    model : TYPE
        Description
    train_x : None, optional
        Description
    train_y : None, optional
        Description
    optimizer : TYPE, optional
        Description
    optimizer_kwargs : dict, optional
        Description
    training_iter : int, optional
        Description
    print_frequency : int, optional
        Description
    verbose : bool, optional
        Description
    """

    train_x = _to_float32_tensor(train_x, device=device)
    train_y = _to_float32_tensor(train_y, device=device)

    model.train()

    _optimizer = optimizer(model.parameters(), **optimizer_kwargs)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood=model.likelihood, model=model
    )
    mll.to(train_x)

    losses = []
    for ii in range(training_iter + 1):

        # Standard training loop...
        _optimizer.zero_grad()
        output = model(train_x)

        # The train_y.flatten() will only work for single task models!
        loss = -mll(output, train_y.flatten())
        loss.backward()

        if verbose and ii % (training_iter // print_frequency) == 0:
            print(f"{ii}/{training_iter}")
            print(f"\t Loss        = {loss.item():.03f}")
            ls = model.covar_module.base_kernel.lengthscale.item()
            print(f"\t Lengthscale = {ls:.03f}")
            noise = model.likelihood.noise.item()
            print(f"\t Noise       = {noise:.03f}")

        _optimizer.step()
        losses.append(loss.item())

    return losses


def infer(*, model, grid, parsed=True, use_likelihood=True, device=DEVICE):
    """Summary

    Parameters
    ----------
    model : gpytorch.model
        Description
    grid : array_like
        Description
    parsed : bool, optional
        If True, returns a dictionary with the keys "mean",
        "mean-2sigma" and "mean+2sigma", representing the mean prediction of
        the posterior, as well as the mean +/- 2sigma, in addition to the
        ``gpytorch.distributions.MultivariateNormal`` object. If False, returns
        the full ``gpytorch.distributions.MultivariateNormal`` only.
    use_likelihood : bool, optional
        If True, applies the likelihood forward operation to the model forward
        operation. This is the recommended default behavior. Otherwise, just
        uses the model forward behavior without accounting for the likelihood.

    Returns
    -------
    dict or gpytorch.distributions.MultivariateNormal
    """

    grid = _to_float32_tensor(grid, device=device)

    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if use_likelihood:
            observed_pred = model.likelihood(model(grid))
        else:
            observed_pred = model(grid)

    if parsed:
        lower, upper = observed_pred.confidence_region()
        return {
            "mean": observed_pred.mean.detach().numpy(),
            "mean-2sigma": lower.detach().numpy(),
            "mean+2sigma": upper.detach().numpy(),
            "observed_pred": observed_pred
        }
    return observed_pred
