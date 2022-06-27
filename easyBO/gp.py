"""Wrappers for the gpytorch and botorch libraries. See
`here <https://botorch.org/docs/models>`_ for important details about the types
of models that we wrap.

Gaussian Processes can often be difficult to get working the first time a new
user tries them, e.g. ambiguities in choosing the kernels. The classes here
abstract away that difficulty (and others) by default.
"""

from copy import deepcopy
from contextlib import contextmanager

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf


class EasyExactGP(gpytorch.models.ExactGP):
    """A lightweight wrapper for the ``gpytorch.models.ExactGP``. Basically
    ripped from the tutorial `here <https://docs.gpytorch.ai/en/stable/
    examples/01_Exact_GPs/Simple_GP_Regression.html>`_. It's the same class
    with a few extra defaults specified for convenience."""

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
    ):
        # Initialize with the scaled data
        super(EasyExactGP, self).__init__(train_x, train_y, likelihood)

        # Mean and covariance modules for the forward method later
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPEnv:

    @property
    def gp(self):
        return self._gp

    def __init__(
        self,
        train_x,
        train_y,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        ),
    ):

        # The training data is cached here for use in ``fit`` if no other
        # training data is provided
        if train_x is None:
            self._train_x = None
        elif isinstance(train_x, np.ndarray):
            self._train_x = torch.tensor(train_x.copy())
        else:
            self._train_x = train_x.clone().detach()  # Better be a tensor!

        if train_y is None:
            self._train_y = None
        elif isinstance(train_y, np.ndarray):
            self._train_y = torch.tensor(train_y.copy())
        else:
            self._train_y = train_y.clone().detach()

        self._likelihood = likelihood
        self._gp = EasyExactGP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
        )

    def fit(
        self,
        train_x=None,
        train_y=None,
        optimizer=torch.optim.Adam, optimizer_kwargs={"lr": 0.1},
        training_iter=100
    ):
        """Train the hyper-parameters of the Gaussian Process on the provided
        data.

        Parameters
        ----------
        optimizer : TYPE, optional
            Description
        optimizer_kwargs : dict, optional
            Description
        """

        if train_x is None:
            train_x = self._train_x
        if train_y is None:
            train_y = self._train_y

        self._gp.train()
        self._likelihood.train()

        _optimizer = optimizer(self._gp.parameters(), **optimizer_kwargs)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self._likelihood, self._gp
        )

        losses = []
        for ii in range(training_iter):
            # Zero gradients from previous iteration
            _optimizer.zero_grad()
            # Output from model
            output = self._gp(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()

            # if ii % (training_iter // 5) == 0:
            #     print(f"{ii:06}/{training_iter:06}")
            #     print(f"")
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                ii + 1, training_iter, loss.item(),
                self._gp.covar_module.base_kernel.lengthscale.item(),
                self._gp.likelihood.noise.item()
            ))
            _optimizer.step()
            losses.append(loss.item())

        return losses

    def predict(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        # Get into evaluation (predictive posterior) mode
        self._gp.eval()
        self._likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self._likelihood(self._gp(x))

        return observed_pred

    def ask(self):
        pass

    def tell(self):
        pass

    def report(self):
        pass



