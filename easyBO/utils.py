import numpy as np
import torch


def _to_tensor(x):

    if x is None:
        return None

    if isinstance(x, np.ndarray):
        return torch.tensor(x.copy(), dtype=torch.float32)

    return x.clone()
