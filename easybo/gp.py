"""Wrappers for the gpytorch and botorch libraries. See
`here <https://botorch.org/docs/models>`__ for important details about the
type of models that we wrap.

Gaussian Processes can often be difficult to get working the first time a new
user tries them, e.g. ambiguities in choosing the kernels. The classes here
abstract away that difficulty (and others) by default.
"""

from copy import deepcopy
from itertools import product

from botorch.exceptions.errors import ModelFittingError
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import gpytorch
from linear_operator.utils.errors import NotPSDError
import numpy as np
import torch

from easybo.utils import _to_float32_tensor, DEVICE, Timer
from easybo.logger import logger, _log_warnings


_TRAINING_WARN_MESSAGE = (
    "Last training attempt failed (or training was never performed): you are "
    "predicting on a model that might not be conditioned correctly on the "
    "data you provided to it."
)

_TRAINING_ERROR_MESSAGE = (
    "train_(...) failed to fit the model. Did you scale your inputs/targets"
    "properly? Is your kernel choice too strict?"
)


def _TRAINING_ERROR_MESSAGE_NLPD(nlpd):
    # https://stats.stackexchange.com/questions/547490/
    # gaussian-process-regression-normalization-of-data-worsens-fit-why
    return (
        f"fit_gpytorch_mll() did not fail, but the NLPD {nlpd:.02e} is high. "
        "Likely, this is due to the kernel producing extremely small length "
        "scales and interpreting each point as delta-like. This can happen "
        "in extremely data-limited conditions and/or when scaling your input "
        "data to the unit hypercube is TRUE. Try using "
        "normalize_inputs_to_unity=False in this calculation. Specifically, "
        "look for if parameters like the raw_outputscale and raw_lengthscale "
        "are very small (note they're on a log scale)."
    )


class NLPDModelFittingError(Exception):
    ...


class EasyGP:
    """Core base class for defining all the primary operations required for an
    "easy Gaussian Process"."""

    def __init__(self):
        self._training_state_successful = False

    @property
    def training_state_successful(self):
        """If True, then the last training instance of the model was
        successful. Note that the default (before training) is False.

        Returns
        -------
        bool
        """

        return self._training_state_successful

    def x_to_tensor(self, x):
        """Executes a forward transformation of some sort on the input data.
        This defaults to a simple conversion to a float32 tensor and placing
        that object on the correct device.

        Parameters
        ----------
        x : array_like

        Returns
        -------
        torch.tensor
        """

        return _to_float32_tensor(x, device=self.device)

    def y_to_tensor(self, y):
        """Executes a forward transformation of some sort on the output data.
        This defaults to a simple conversion to a float32 tensor and placing
        that object on the correct device.

        Parameters
        ----------
        y : array_like

        Returns
        -------
        torch.tensor
        """

        return _to_float32_tensor(y, device=self.device)

    @property
    def device(self):
        """The device on which to run the calculations and place the model.
        Follows the PyTorch standard, e.g. "cpu", "gpu:0", etc.

        Returns
        -------
        str
        """

        return self._device

    @device.setter
    def device(self, device):
        """Sets the device. This not only changes the device attribute, it will
        send the model to the new device.

        Parameters
        ----------
        device : str
        """

        self._model.to(device)
        self._device = device
        logger.debug(f"Model sent to {device}")

    @property
    def likelihood(self):
        """The likelihood function mapping the values f(x) to the observations
        y. See `here <https://docs.gpytorch.ai/en/latest/likelihoods.html>`__
        for more details.

        Returns
        -------
        TYPE
            Description
        """
        return self._model.likelihood

    @property
    def model(self):
        """Returns the GPyTorch model itself.

        Returns
        -------
        botorch.models
        """

        return self._model

    def _get_current_train_x(self, untransform=False):
        x = deepcopy(self._model.train_inputs[0])
        if untransform and hasattr(self._model, "input_transform"):
            x = self._model.input_transform.untransform(x)
        return x

    @property
    def train_x(self):
        """The training inputs. Should be of shape ``N_train x d_in``. This
        also unscales the data if the ``input_transform`` attribute is present
        in the model.

        Returns
        -------
        numpy.ndarray
        """

        # github.com/pytorch/botorch/issues/1435#issuecomment-1274974582
        t = self._model.training
        return self._get_current_train_x(untransform=not t).detach().numpy()

    def _get_current_train_y(self, untransform=False):
        y = deepcopy(self._model.train_targets.reshape(-1, 1))
        if untransform and hasattr(self._model, "outcome_transform"):
            y, _ = self._model.outcome_transform.untransform(y)
        return y

    @property
    def train_y(self):
        """The training targets. Should be of shape ``N_train x d_out``. Note
        that for classification, these should be one-hot encoded, e.g.
        ``np.array([0, 1, 2, 1, 2, 0, 0])``. This also unscales the data if
        the ``outcomes_transform`` attribute is present in the model.

        Returns
        -------
        numpy.ndarray
        """

        return self._get_current_train_y(untransform=True).detach().numpy()

    def _get_training_debug_information(self, model=None):
        if model is None:
            model = self._model

        info = {"hyperparameters": []}

        for p in model.named_parameters():
            p0 = p[0]
            p1 = str(p[1]).split("\n")[1]
            info["hyperparameters"].append(f"{p0}: {p1}")

        # Ensure we include the proper transforms
        if hasattr(model, "input_transform"):
            tmp = model.input_transform._buffers
            info["input_transform"] = tmp
        if hasattr(model, "outcome_transform"):
            tmp = model.outcome_transform._buffers
            info["outcome_transform"] = tmp

        return info

    def _log_training_debug_information(self, model=None):
        info = self._get_training_debug_information(model=model)
        for p in info["hyperparameters"]:
            logger.debug(p)

        # Ensure we include the proper transforms
        if "input_transform" in info.keys():
            logger.debug(f"INP TRANSFORM: {info['input_transform']}")
        if "outcome_transform" in info.keys():
            logger.debug(f"OUT TRANSFORM: {info['outcome_transform']}")

    def _get_posterior(self, grid):

        self._model.eval()
        self._model.likelihood.eval()

        grid = _to_float32_tensor(grid, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self._model.posterior(grid)

    def nlpd(self, train_x=None, train_y=None):
        """Gets the negative log predictive density of the model.

        Parameters
        ----------
        train_x : None, array_like
            If None, uses ``self.train_x``.
        train_y : None, array_like
            If None, uses ``self.train_y``.
        """

        if train_x is None:
            train_x = self.train_x
        train_x = _to_float32_tensor(train_x)

        if train_y is None:
            train_y = self.train_y
        train_y = _to_float32_tensor(train_y)

        posterior = self._get_posterior(train_x)
        try:
            _nlpd = -posterior.mvn.log_prob(train_y.squeeze()).item()
        except NotPSDError:
            _nlpd = 0
        return _nlpd

    @_log_warnings
    def train_(
        self,
        *,
        optimizer=None,
        optimizer_kwargs=None,
        log_error_on_fail=False,
        terminate_on_fail=False,
        **kwargs,
    ):
        """Trains model. This is a lightweight wrapper for ``botorch``'s
        ``fit_gpytorch_mll`` function. It simply initializes an
        exact marginal log likelihood and uses that to train the model.

        Parameters
        ----------
        optimizer : torch.optim, optional
            The optimizer to use to train the GP.
        optimizer_kwargs : dict, optional
            Keyword arguments to pass to the optimizer.
        **kwargs
            Extra keyword arguments to pass to ``fit_gpytorch_mll``.
        """

        self._training_state_successful = True

        logger.debug("------- PARAMETER INFO BEFORE TRAINING -------")
        self._log_training_debug_information()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=self._model.likelihood, model=self._model
        )

        try:
            with Timer() as timer:
                fit_gpytorch_mll(
                    mll,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    **kwargs,
                )

        except ModelFittingError:
            self._training_state_successful = False

            if log_error_on_fail:
                logger.exception(_TRAINING_ERROR_MESSAGE)
            else:
                logger.warning(_TRAINING_ERROR_MESSAGE)

            if terminate_on_fail:
                if not log_error_on_fail:
                    logger.exception(_TRAINING_ERROR_MESSAGE)

                logger.critical("terminate_on_fail is True, throwing error")
                raise ModelFittingError

        logger.debug("------- PARAMETER INFO AFTER TRAINING -------")
        self._log_training_debug_information()

        if self._training_state_successful:
            nlpd = self.nlpd()
            if nlpd > 100.0:
                self._training_state_successful = False
                logger.info(f"Model fit in {timer.dt:.01f} {timer.units}")
                training_info = self._get_training_debug_information()
                logger.error(_TRAINING_ERROR_MESSAGE_NLPD(nlpd))
                logger.info(f"model parameters: {training_info}")
                if terminate_on_fail:
                    logger.critical(
                        "terminate_on_fail is True, throwing error"
                    )
                    raise NLPDModelFittingError
            else:
                xshape = list(self._get_current_train_x().shape)
                yshape = list(self._get_current_train_y().shape)
                check = ""  # "\u2705"
                logger.success(
                    f"{check} Model fit on data, shapes: {xshape} -> {yshape} "
                    f"in {timer.dt:.01f} {timer.units}, NLPD: {nlpd:.02f}"
                )

    @_log_warnings
    def predict(self, *, grid):
        """Runs inference on the model in eval mode.

        Parameters
        ----------
        grid : array_like
            The grid on which to perform inference.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - ``"mean"``: the mean of the posterior on the provided ``grid``.
            - ``"mean+2sigma"``: the mean of the posterior on the provided
            ``grid``, plus 2 x one standard deviation.
            - ``"mean-2sigma"``: the mean of the posterior on the provided
            ``grid``, minus 2 x one standard deviation.
            - ``"posterior"``: the result of calling the model posterior
            on the grid, can be used for further debugging/inference.
        """

        posterior = self._get_posterior(grid)
        mean = posterior.mean.detach().numpy().squeeze()
        std = np.sqrt(posterior.variance.detach().numpy().squeeze())

        if not self._training_state_successful:
            logger.warning(_TRAINING_WARN_MESSAGE)

        return {
            "mean": mean,
            "std": std,
            "mean+2std": mean + 2.0 * std,
            "mean-2std": mean - 2.0 * std,
            "posterior": posterior,
        }

    @_log_warnings
    def sample(self, *, grid, samples=1, seed=None):
        """Samples from the provided model.

        Parameters
        ----------
        grid : array_like
            The grid from which to sample.
        samples : int, optional
            Number of samples to draw.
        seed : None, optional
            Seeds the random number generator via ``torch.manual_seed``.

        Returns
        -------
        numpy.array
            The array of sampled data, of shape ``samples x len(grid)``.
        """

        if seed is not None:
            torch.manual_seed(seed)
        posterior = self._get_posterior(grid)
        sampled = posterior.sample(torch.Size([samples]))
        return sampled.detach().numpy().reshape(samples, len(grid))

    def _condition(self, new_x, new_y):

        # Concatenate all of the untransformed data together
        x = self._get_current_train_x(untransform=True)
        logger.debug(f"old_x min max: {x.min(axis=0)} {x.max(axis=0)}")
        x = torch.cat([x, new_x], axis=0)
        y = self._get_current_train_y(untransform=True)
        logger.debug(f"old_y min max: {y.min(axis=0)} {y.max(axis=0)}")
        y = torch.cat([y, new_y], axis=0)

        # Get the model's state dict. This contains all of the state
        # information, including the parameters of the transforms
        state_dict = self._model.state_dict()

        # Get the keyword arguments that were initially used to construct
        # the model, but override with the new training data
        kwargs = deepcopy(self._initial_kwargs)
        kwargs["train_x"] = x
        kwargs["train_y"] = y

        # Initialize...
        logger.debug(f"Initialzing new model with keys: {kwargs.keys()}")
        new_model = self.__class__(**kwargs)

        # Old way -------------------------------------------------------------
        # # Condition the model with the right length scales but the WRONG
        # # information about the transform
        # new_model._model.load_state_dict(state_dict)

        # Retrain to get the right transform information
        # new_model.train_()
        # ---------------------------------------------------------------------

        # New way
        # github.com/pytorch/botorch/issues/1435#issuecomment-1268851771
        new_state_dict = {
            key: value
            for key, value in state_dict.items()
            if "outcome_transform" not in key and "input_transform" not in key
        }

        # `strict == False` needed since the state dict is now missing certain
        # keys
        new_model._model.load_state_dict(new_state_dict, strict=False)

        # Set any attributes that are not kwargs
        new_model._training_state_successful = self._training_state_successful

        return new_model

    @_log_warnings
    def tell(self, *, new_x, new_y, retrain=True):
        """Informs the GP about new data. This implicitly conditions the model
        on the new data but without modifying the previous model's
        hyperparameters.

        .. warning::

            The input shapes of the new x and y values must be correct
            otherwise errors will be thrown.

        This method will also utilize the previously defined transforms for
        the input and output data.

        Parameters
        ----------
        new_x : array_like
            The new input data.
        new_y : array_like
            The new target data.
        retrain : bool, optional
            If True, will automatically retrain via the ``train_`` method.

        Returns
        -------
        EasyGP
        """

        # The new data will be "untransformed"
        new_x = self.x_to_tensor(new_x)
        new_y = self.y_to_tensor(new_y)

        logger.debug(f"new_x min max {new_x.min(axis=0)} {new_x.max(axis=0)}")
        logger.debug(f"new_y min max {new_y.min(axis=0)} {new_y.max(axis=0)}")

        # try:
        #     self._model = self._model.condition_on_observations(new_x, new_y)

        # except RuntimeError as err:
        #     logger.debug(
        #         f"RuntimeError raised during conditioning: {err}. likely "
        #         "no predictions were made before trying to get a fantasy "
        #         "model: running predict to try and resolve"
        #     )
        #     self.predict(grid=self.train_x)
        #     self._model = self._model.condition_on_observations(new_x, new_y)

        # For why we do it this way, see:
        # github.com/pytorch/botorch/issues/1435#issuecomment-1265803038
        new_model = self._condition(new_x, new_y)

        if retrain:
            new_model.train_()

        return new_model

    @_log_warnings
    def dream(self, points_per_dimension=10, seed=123, **kwargs):
        """This is a simliar method to BoTorch's fantasize, but it's a bit
        simpler and is used for a specific purpose. This method returns a new
        instance of the :class:`EasyGP` (or its derived classes) which is
        constructed using a specific sample of the current model. The sample
        is dictated by the provided seed, and the new training data is
        determined by the minimum and maximum of the training data on each
        dimension, and ``points_per_dimension`` are used to construct a
        uniform grid. We call this method ``dream`` to differentiate it from
        ``fantasize``.

        Parameters
        ----------
        points_per_dimension : int, optional
            The number of points per dimension to use for training the dreamy
            model.
        seed : int, optional
            The seed used during sample.
        **kwargs
            Keyword arguments for the training procedure.

        Returns
        -------
        EasyGP
        """

        # train_x is of shape N x dims
        # train_x.t is of shape dims x N
        grids = [
            np.linspace(xx.min(), xx.max(), points_per_dimension)
            for xx in self.train_x.T
        ]
        coordinates = np.array([xx for xx in product(*grids)])

        logger.debug(f"dreamed coordinates shape: {coordinates.shape}")

        y = self.sample(grid=coordinates, samples=1, seed=seed).reshape(
            -1, self.train_y.shape[1]
        )

        kwargs = deepcopy(self._initial_kwargs)
        kwargs["train_x"] = coordinates
        kwargs["train_y"] = y

        # Initialize and train...
        new_model = self.__class__(**kwargs)
        new_model.train_(**kwargs)

        return new_model


class EasySingleTaskGPRegressor(EasyGP):
    def __init__(
        self,
        *,
        train_x,
        train_y,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        ),
        normalize_inputs_to_unity=True,
        standardize_outputs=True,
        device=DEVICE,
        **kwargs,
    ):
        self._initial_kwargs = deepcopy(
            {
                key: value
                for key, value in locals().items()
                if key not in ["self", "kwargs", "__class__"]
            }
        )
        kwargs = deepcopy(
            {key: value for key, value in kwargs.items() if key != "__class__"}
        )

        logger.debug(f"Initial kwargs: {kwargs.keys()}")
        super().__init__()
        self._initial_kwargs = {**self._initial_kwargs, **kwargs}
        self._device = device
        d = train_x.shape[1]  # Number of features
        input_transform = (
            Normalize(d, transform_on_eval=True)
            if normalize_inputs_to_unity
            else None
        )
        m = train_y.shape[1]  # Number of targets
        outcome_transform = Standardize(m) if standardize_outputs else None
        model = SingleTaskGP(
            train_X=self.x_to_tensor(train_x),
            train_Y=self.y_to_tensor(train_y),
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            **kwargs,
        )
        self._model = deepcopy(model.to(device))


# class MostLikelyHeteroskedasticGPRegressor(EasyGP):
#     def __init__(
#         self,
#         *,
#         train_x,
#         train_y,
#         likelihood=gpytorch.likelihoods.GaussianLikelihood(),
#         mean_module=gpytorch.means.ConstantMean(),
#         covar_module=gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel()
#         ),
#         device=DEVICE,
#         **kwargs,
#     ):
#         self._device = device

#         # Define the standard single task GP model
#         model = SingleTaskGP(
#             train_X=self.x_to_tensor(train_x),
#             train_Y=self.y_to_tensor(train_y),
#             likelihood=likelihood,
#             mean_module=mean_module,
#             covar_module=covar_module,
#             **kwargs,
#         )
#         model.likelihood.noise_covar.register_constraint(
#             "raw_noise", GreaterThan(1e-3)
#         )
#         self._model = deepcopy(model.to(device))

#         # Define a noise model - this is going to be initialized during
#         # training
#         self._noise_model = None

#     def train_(
#         self,
#         *,
#         optimizer=torch.optim.Adam,
#         optimizer_kwargs={"lr": 0.1},
#         training_iter=100,
#         print_frequency=5,
#     ):

#         # homoskedastic_loss = self._fit_model_(
#         #     model=self._model,
#         #     optimizer=optimizer,
#         #     optimizer_kwargs=optimizer_kwargs,
#         #     training_iter=training_iter,
#         #     print_frequency=print_frequency,
#         #     heteroskedastic_training=False
#         # )

#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(
#             likelihood=self._model.likelihood, model=self._model
#         )
#         botorch.fit.fit_gpytorch_model(mll)

#         # Now we have to fit the noise model; first we get the observed
#         # variance
#         self._model.eval()
#         c_train_x = self._get_current_train_x()
#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             post = self._model.posterior(c_train_x).mean.numpy()
#         observed_var = torch.tensor(
#             (post - self.train_y) ** 2, dtype=torch.float
#         )

#         # Now actually fit the noise model
#         self._noise_model = HeteroskedasticSingleTaskGP(
#             train_X=self._get_current_train_x(),
#             train_Y=self._get_current_train_y(),
#             train_Yvar=observed_var,
#         )
#         # heteroskedastic_loss = self._fit_model_(
#         #     model=self._noise_model,
#         #     optimizer=optimizer,
#         #     optimizer_kwargs=optimizer_kwargs,
#         #     training_iter=training_iter,
#         #     print_frequency=print_frequency,
#         #     heteroskedastic_training=True
#         # )

#         mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(
#             likelihood=self._noise_model.likelihood, model=self._noise_model
#         )
#         botorch.fit.fit_gpytorch_model(mll2, max_retries=10)

#         self._model.train()

# return heteroskedastic_loss


# class EasySingleTaskGPClassifier(EasyGP):
#     def y_to_tensor(self, y):
#         """Executes a forward transformation of some sort on the output data.
#         For the classifier, this is a conversion to a long tensor.

#         Parameters
#         ----------
#         y : array_like

#         Returns
#         -------
#         torch.tensor
#         """

#         return _to_long_tensor(y, device=self.device)

#     def __init__(
#         self,
#         *,
#         train_x,
#         train_y,
#         mean_module=gpytorch.means.ConstantMean(),
#         covar_module=gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel()
#         ),
#         device=DEVICE,
#         **kwargs,
#     ):
#         self._device = device
#         y = self.y_to_tensor(train_y)
#         lh = gpytorch.likelihoods.DirichletClassificationLikelihood(
#             y, learn_additional_noise=True
#         )
#         model = SingleTaskGP(
#             train_X=self.x_to_tensor(train_x),
#             train_Y=y,
#             likelihood=lh,
#             mean_module=mean_module,
#             covar_module=covar_module,
#             **kwargs,
#         )
#         self._model = deepcopy(model.to(device))
