import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_float32_tensor(x, device=DEVICE):

    if x is None:
        return None

    if isinstance(x, np.ndarray):
        x = torch.tensor(x.copy()).float()

    x = x.clone().float()
    x = x.to(device)
    return x


def _to_long_tensor(x, device=DEVICE):

    if x is None:
        return None

    if isinstance(x, np.ndarray):
        x = torch.tensor(x.copy()).long()

    x = x.clone().long()
    x = x.to(device)
    return x


def get_dummy_1d_sinusoidal_data(seed=123):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # use regular spaced points on the interval [0, 1]
    train_x = torch.linspace(0, 1, 15)

    # training data needs to be explicitly multi-dimensional
    train_x = train_x.unsqueeze(1)

    # sample observed values and add some synthetic noise
    train_y = torch.sin(train_x * (2 * np.pi)) + 0.15 * torch.randn_like(
        train_x
    )

    # Testing grid
    grid = torch.linspace(0, 2.5, 110).reshape(-1, 1)

    return grid, train_x, train_y


def set_grids(
    ax,
    minorticks=True,
    grid=False,
    bottom=True,
    left=True,
    right=True,
    top=True,
):

    if minorticks:
        ax.minorticks_on()

    ax.tick_params(
        which="both",
        direction="in",
        bottom=bottom,
        left=left,
        top=top,
        right=right,
    )


def plot_1d_fit(
    *,
    ax,
    model,
    grid,
    scatter_kwargs={"color": "black", "s": 0.5, "label": "Obs", "zorder": 3},
    plot_kwargs={"color": "r", "linestyle": "-", "label": "$\\mu$"},
    fill_between_kwargs={
        "alpha": 0.2,
        "color": "red",
        "linewidth": 0,
        "label": "$\\mu \\pm 2\\sigma$",
    }
):
    """Plots results for a 1-dimensional input and output dataset.
    Specifically, plots a scatterplot of the training data (in black by
    default), the mean of the prediction (in red) and 2 x the spread (in
    red background).

    Parameters
    ----------
    ax : TYPE
        Description
    model : TYPE
        Description
    grid : TYPE
        Description
    scatter_kwargs : dict, optional
        Description
    plot_kwargs : dict, optional
        Description
    fill_between_kwargs : dict, optional
        Description
    """

    ax.scatter(
        model.train_x.squeeze(), model.train_y.squeeze(), **scatter_kwargs
    )
    preds = model.predict(grid=grid)

    ax.plot(grid.squeeze(), preds["mean"].squeeze(), **plot_kwargs)
    ax.fill_between(
        grid.squeeze(),
        preds["mean-2sigma"].squeeze(),
        preds["mean+2sigma"].squeeze(),
        **fill_between_kwargs
    )
