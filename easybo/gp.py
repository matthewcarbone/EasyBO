"""Wrappers for the gpytorch and botorch libraries. See
`here <https://botorch.org/docs/models>`__ for important details about the
type of models that we wrap.

Gaussian Processes can often be difficult to get working the first time a new
user tries them, e.g. ambiguities in choosing the kernels. The classes here
abstract away that difficulty (and others) by default.
"""

from copy import deepcopy

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import gpytorch
import numpy as np
import torch

from easybo.utils import _to_float32_tensor, DEVICE
from easybo.logger import logger


class EasyGP:
    """Core base class for defining all the primary operations required for an
    "easy Gaussian Process"."""

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

    def _get_current_train_x(self):
        return self._model.train_inputs[0]

    @property
    def train_x(self):
        """The training inputs. Should be of shape ``N_train x d_in``. This
        also unscales the data if the ``input_transform`` attribute is present
        in the model.

        Returns
        -------
        numpy.ndarray
        """

        x = self._get_current_train_x()
        if hasattr(self._model, "input_transform"):
            x = self._model.input_transform.untransform(x)
        return x.detach().numpy()

    def _get_current_train_y(self):
        return self._model.train_targets.reshape(-1, 1)

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

        y = self._get_current_train_y()
        if hasattr(self._model, "outcome_transform"):
            y, _ = self._model.outcome_transform.untransform(y)
        return y.detach().numpy()

    def _log_training_debug_information(self):
        parameters = [param for param in self._model.named_parameters()]
        logger.debug(f"MODEL PARAMETERS: {parameters}")

        # Ensure we include the proper transforms
        if hasattr(self._model, "input_transform"):
            p = self._model.input_transform._buffers
            logger.debug(f"INPUT TRANSFORM: {p}")
        if hasattr(self._model, "outcome_transform"):
            p = self._model.outcome_transform._buffers
            logger.debug(f"OUTCOME TRANSFORM: {p}")

    def train_(self, *, optimizer=None, optimizer_kwargs=None, **kwargs):
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

        logger.debug("Parameter information before training:")
        self._log_training_debug_information()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            likelihood=self._model.likelihood, model=self._model
        )

        fit_gpytorch_mll(
            mll,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

        logger.debug("Parameter information after training:")
        self._log_training_debug_information()

        s1 = self._model.train_inputs[0].shape
        logger.debug(f"SAVED TRAIN INPUTS SHAPE: {s1}")

        s2 = self._model.train_targets.shape
        logger.debug(f"SAVED TRAIN OUTPUTS SHAPE: {s2}")

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
            - ``"observed_pred"``: the result of calling the model posterior
            on the grid, can be used for further debugging/inference.
        """

        self._model.eval()
        self._model.likelihood.eval()

        grid = _to_float32_tensor(grid, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._model.posterior(grid)

        mean = observed_pred.mean.detach().numpy().squeeze()
        std = np.sqrt(observed_pred.variance.detach().numpy().squeeze())

        return {
            "mean": mean,
            "mean+2sigma": mean + 2.0 * std,
            "mean-2sigma": mean - 2.0 * std,
            "observed_pred": observed_pred,
        }

    def sample(self, *, grid, samples=10, seed=None):
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

        self._model.eval()
        self._model.likelihood.eval()

        if seed is not None:
            torch.manual_seed(seed)
        grid = _to_float32_tensor(grid, device=self.device)
        result = self._model(grid).sample(sample_shape=torch.Size([samples]))
        return result.detach().numpy()

    def _condition(self, new_x, new_y):
        x = torch.cat([self._get_current_train_x(), new_x], axis=0)
        y = torch.cat([self._get_current_train_y(), new_y], axis=0)
        state_dict = self._model.state_dict()
        kwargs = deepcopy(self._initial_kwargs)
        kwargs["train_x"] = x
        kwargs["train_y"] = y
        model = self.__class__(**kwargs)
        model._model.load_state_dict(state_dict)
        return model

    def tell(self, *, new_x, new_y):
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

        Returns
        -------
        EasyGP
        """

        new_x = self.x_to_tensor(new_x)
        new_y = self.y_to_tensor(new_y)

        # Ensure we include the proper transforms
        if hasattr(self._model, "input_transform"):
            new_x = self._model.input_transform(new_x)
        if hasattr(self._model, "outcome_transform"):
            new_y, __ = self._model.outcome_transform(new_y)

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
        return self._condition(new_x, new_y)


class EasySingleTaskGPRegressor(EasyGP):
    def __init__(
        self,
        *,
        train_x,
        train_y,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
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
                if key not in ["self", "kwargs"]
            }
        )
        kwargs = deepcopy({key: value for key, value in kwargs.items()})
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
