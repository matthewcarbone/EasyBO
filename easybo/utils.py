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
