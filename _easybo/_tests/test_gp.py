import numpy as np

from easybo.gp import EasySingleTaskGPRegressor
from easybo.utils import get_dummy_1d_sinusoidal_data


def test_gp_scaling():
    """Makes sure that accessing model.train_x and model.train_y works properly
    with all of the transforms that botorch does under the hood."""

    grid, train_x, train_y = get_dummy_1d_sinusoidal_data()

    # Do some arbitrary scaling by constant factors
    train_y = train_y * 100.0
    train_x = train_x * 10.0
    grid = grid * 10.0

    model = EasySingleTaskGPRegressor(
        train_x=train_x,
        train_y=train_y,
        normalize_inputs_to_unity=True,
        standardize_outputs=True,
    )

    model.model.train()
    assert np.allclose(train_x, model.train_x)
    assert np.allclose(train_y, model.train_y)

    model.model.eval()
    assert np.allclose(train_x, model.train_x)
    assert np.allclose(train_y, model.train_y)

    model.train_()

    model.model.train()
    assert np.allclose(train_x, model.train_x)
    assert np.allclose(train_y, model.train_y)

    model.model.eval()
    assert np.allclose(train_x, model.train_x)
    assert np.allclose(train_y, model.train_y)

    new_x = np.array([2.25, 2.50]).reshape(-1, 1) * 10
    new_y = np.array([1, 2]).reshape(-1, 1) * 100
    new_model = model.tell(new_x=new_x, new_y=new_y)

    train_x = np.concatenate([train_x, new_x], axis=0)
    train_y = np.concatenate([train_y, new_y], axis=0)

    new_model.model.train()
    assert np.allclose(train_x, new_model.train_x)
    assert np.allclose(train_y, new_model.train_y)

    new_model.model.eval()
    assert np.allclose(train_x, new_model.train_x)
    assert np.allclose(train_y, new_model.train_y)

    new_model.train_()

    new_model.model.train()
    assert np.allclose(train_x, new_model.train_x)
    assert np.allclose(train_y, new_model.train_y)

    new_model.model.eval()
    assert np.allclose(train_x, new_model.train_x)
    assert np.allclose(train_y, new_model.train_y)
