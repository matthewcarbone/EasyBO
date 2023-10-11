===============================
Simple Gaussian Process example
===============================

We begin by emulating the simple example found `here <https://botorch.org/v/0.1.0/tutorials/fit_model_with_torch_optimizer>`__, where we attempt to model the simple function

.. math::

    f(x) = \sin(2 \pi x) + 0.15 \varepsilon

where :math:`\varepsilon` is i.i.d. Gaussian noise. In this tutorial, we'll need a few common libraries (as well as one from EasyBO), which we import at the start

.. code-block:: python

    import numpy as np
    import torch
    from easybo import gp

We then initialize some data following the above equation

.. code-block:: python

    np.random.seed(123)
    torch.manual_seed(123)

    # use regular spaced points on the interval [0, 1]
    train_x = torch.linspace(0, 1, 15)

    # training data needs to be explicitly multi-dimensional
    train_x = train_x.unsqueeze(1)

    # sample observed values and add some synthetic noise
    train_y = torch.sin(train_x * (2 * math.pi)) + 0.15 * torch.randn_like(train_x)

    # Testing grid
    grid = torch.linspace(0, 1, 101)

A quick note before continuing:

.. note::

    The entire EasyBO API is `keyword argument only`. If you look at some of the function signatures in EasyBO, you will see things like

    .. code-block:: python

        def get_gp(*, ...)

    where the ``*`` indicates that every argument afterwards provided to that function must be provided by keyword. We do this for two reasons. First, to avoid confusion. Second, it is instructive to `always` know which arguments you are providing.

The model
---------

Next, we make the first call to EasyBO and acquire the ``model`` object, which is our Gaussian Process (GP) model which we'll use for the rest of this tutorial.

.. code-block:: python

    model = gp.get_gp(train_x=train_x, train_y=train_y, gp_type="regression")

There are a few arguments to consider, and some other options that are left as default for the user (as most users will only need the defaults). First, and obviously, the training data is provided: ``train_x`` are the features of shape ``N x d``, where ``N`` is the number of training examples and ``d`` is the dimension of the input space, and ``train_y`` are the targets of shape ``N x 1``. Finally, the possibly strange argument of ``gp_type`` is also provided. This can be either ``"regression"`` (which we'll use in most tutorials here) or ``"classification"`` (see the classification tutorial for more details on this). The long story short is that this controls the type of likelihood used in the GP, and is a bit non-intuitive to initialize especially for classification problems.
