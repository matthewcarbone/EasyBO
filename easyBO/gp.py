"""Wrappers for the gpytorch and botorch libraries. See
`here <https://botorch.org/docs/models>`_ for important details about the types
of models that we wrap.

Gaussian Processes can often be difficult to get working the first time a new
user tries them, e.g. ambiguities in choosing the kernels. The classes here
abstract away that difficulty (and others) by default.
"""

from copy import deepcopy

import torch
import gpytorch
from botorch.models import SingleTaskGP

from easyBO.utils import _to_float32_tensor, _to_long_tensor, DEVICE
from easyBO.logger import logger


DEFAULT_MEAN_MODULE = gpytorch.means.ConstantMean()
DEFAULT_COVAR_MODULE = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel()
)


def _gp_type_from_str(gp_type):
    if gp_type not in ["regression", "classification"]:
        msg = (
            f"Unknown gp_type {gp_type}, must be either 'regression' or "
            "'classification'"
        )
        logger.critical(msg)
        raise ValueError(msg)
    return gp_type


def _get_likelihood_and_y(gp_type, likelihood, y):

    if gp_type == "regression":
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if likelihood is None:
            return gpytorch.likelihoods.GaussianLikelihood(), y
        else:
            return likelihood, y

    # Classification
    lh = gpytorch.likelihoods.DirichletClassificationLikelihood(
        y, learn_additional_noise=True
    )

    if likelihood is not None:
        logger.warning(
            f"Specified likelihood {likelihood} will be overwritten with "
            "the DirichletClassificationLikelihood, since classification "
            "was specified"
        )

    return lh, lh.transformed_targets.T


def get_gp(
    *,
    train_x,
    train_y,
    gp_type,
    likelihood=None,
    mean_module=DEFAULT_MEAN_MODULE,
    covar_module=DEFAULT_COVAR_MODULE,
    device=DEVICE,
    **kwargs,
):
    """Returns the appropriate botorch GP model depending on the type of
    gp (regression or classification).

    Parameters
    ----------
    train_x : array_like
        The inputs.
    train_y : array_like
        The targets. Note that for classification, these should be one-hot
        encoded, e.g. np.array([0, 1, 2, 1, 2, 0, 0]).
    gp_type : str
        The type of GP to return, either "regression" or "classification".
    likelihood : gpytorch.likelihoods, optional
        Likelihood for initializing the GP. Recommended to keep the default:
        ``gpytorch.likelihoods.GaussianLikelihood()`` for regression problems,
        or to just leave as None by default. The function will automatically
        choose the right likelihood depending on the type of calculation.
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

    Returns
    -------
    SingleTaskGP
    """

    gp_type = _gp_type_from_str(gp_type)

    # Convert the provided (likely numpy) arrays to the appropriate torch
    # tensors of the correct types
    x = _to_float32_tensor(train_x, device=device)
    if gp_type == "regression":
        y = _to_float32_tensor(train_y, device=device)
    else:
        y = _to_long_tensor(train_y, device=device)

    likelihood, y = _get_likelihood_and_y(gp_type, likelihood, y)

    model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module,
        **kwargs,
    )

    return deepcopy(model.to(device))


def _detect_gp_type_from_model(gp_type, model):
    if gp_type is None:
        if isinstance(
            model.likelihood,
            gpytorch.likelihoods.DirichletClassificationLikelihood,
        ):
            logger.debug(
                "DirichletClassificationLikelihood detected, assuming "
                "classification problem"
            )
            return "classification"
        logger.debug(
            f"{model.likelihood.__class__.__name__} detected, "
            "assuming regression problem"
        )
        return "regression"

    if gp_type not in ["regression", "classification"]:
        raise ValueError(f"Unknown gp_type {gp_type}")
    return gp_type


def train_gp_(
    *,
    model,
    train_x=None,
    train_y=None,
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.1},
    training_iter=100,
    print_frequency=5,
    device=DEVICE,
    gp_type=None,
):
    """Trains the provided botorch model. The methods used here are different
    than botorch's boilerplate ``fit_gpytorch_model``, and the function will
    automatically try to detect what type of problem (regression or
    classification) that is being performed if it's not specified explicitly.

    Parameters
    ----------
    model : SingleTaskGP
        The model to train.
    train_x : array_like
        The inputs. If None, defaults to the model.train_inputs.
    train_y : array_like
        The targets. Note that for classification, these should be one-hot
        encoded, e.g. np.array([0, 1, 2, 1, 2, 0, 0]). If None, defaults to
        model.train_targets.
    optimizer : torch.optim, optional
        The optimizer to use to train the GP.
    optimizer_kwargs : dict, optional
        Keyword arguments to pass to the optimizer.
    training_iter : int, optional
        The number of training iterations to perform.
    print_frequency : int, optional
        The frequency at which to log to the info logger during training. If
        0 does not print anything during training.
    device : str, optional
        Device on which to perform the training.
    gp_type : str, optional
        The type of GP to train. See :class:`get_gp`. If None, attempts to
        determine the type of GP from the model itself.
    """

    gp_type = _detect_gp_type_from_model(gp_type, model)

    # Handle setting the training data in case it wasn't provided.
    if train_x is None:
        train_x = model.train_inputs[0]
    else:
        train_x = _to_float32_tensor(train_x, device=device)
    if train_y is None:
        train_y = model.train_targets
    else:
        if gp_type == "regression":
            train_y = _to_float32_tensor(train_y, device=device)
        else:
            train_y = _to_long_tensor(train_y, device=device)

    # Get all of the training objects together
    model.train()
    _optimizer = optimizer(model.parameters(), **optimizer_kwargs)

    # Loss for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood=model.likelihood, model=model
    )
    mll.to(train_x)

    # Standard training loop...
    losses = []
    for ii in range(training_iter + 1):

        _optimizer.zero_grad()
        output = model(train_x)

        loss = -mll(output, train_y).sum()
        loss.backward()
        _loss = loss.item()
        ls = model.covar_module.base_kernel.lengthscale.mean().item()
        noise = model.likelihood.noise.mean().item()
        msg = (
            f"{ii}/{training_iter} loss={_loss:.03f} lengthscale="
            f"{ls:.03f} noise={noise:.03f}"
        )
        if print_frequency != 0:
            if ii % (training_iter // print_frequency) == 0:
                logger.info(msg)
        logger.debug(msg)

        _optimizer.step()
        losses.append(loss.item())

    return losses


def infer(*, model, grid, parsed=True, use_likelihood=True, device=DEVICE):
    """Summary

    Parameters
    ----------
    model : gpytorch.model
        The model on which to perform inference.
    grid : array_like
        The grid on which to perform inference.
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
            "observed_pred": observed_pred,
        }
    return observed_pred


def get_training_data(*, model):
    """Helper function for the user: easily extracts the training features and
    targets from the model.

    Parameters
    ----------
    model : botorch.models.Model

    Returns
    -------
    tuple
        Two numpy arrays, one for the training inputs, one for the training
        targets.
    """

    return (
        model.train_inputs[0].detach().numpy(),
        model.train_targets.detach().numpy(),
    )


def tell(
    *,
    model,
    new_x,
    new_y,
    gp_type=None,
    device=None,
):
    """Returns a new model with updated data. This implicitly conditions the
    model on the new data but without modifying the previous model's
    hyperparameters.

    .. warning::

         The input shapes of the new x and y values must be correct otherwise
         errors will be thrown.

    Parameters
    ----------
    model : botorch.models.Model
    new_x : array_like
        The new input data.
    new_y : array_like
        The new target data.
    device : None, optional
        Default or provided device.

    Returns
    -------
    botorch.models.Model
        A new model conditioned on the previous + new observations.
    """

    gp_type = _detect_gp_type_from_model(gp_type, model)
    new_x = _to_float32_tensor(new_x, device=device)
    if gp_type == "regression":
        new_y = _to_float32_tensor(new_y, device=device)
    else:
        new_y = _to_long_tensor(new_y, device=device)
    return model.condition_on_observations(new_x, new_y)
