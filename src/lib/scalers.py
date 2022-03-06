import torch
from torch import nn


class BaseScaler(nn.Module):
    """Basic scaler class"""

    def __init__(self, size: int):
        super().__init__()

        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))
        self.device = torch.device("cpu")

    def fit(self, x):
        raise NotImplementedError()

    def transform(self, x):
        raise NotImplementedError()

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.device = device

    def __repr__(self):
        return f"mean: {self.mean}\tstd: {self.std}"


class MelScaler(BaseScaler):
    """Basic log scaler class"""

    def fit(self, x):
        clip_mask = (x < -100) | (x > 100)
        x = x[~clip_mask]
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)

    def transform(self, x):
        clip_mask = (x < -100) | (x > 100)
        transformed = (x - self.mean) / self.std
        transformed[clip_mask] = 0

        return transformed


class StandardScaler(BaseScaler):
    """Basic standard scaler class"""

    def fit(self, x):
        mask = x != 0
        self.mean = torch.mean(x[mask].view(-1))
        self.std = torch.std(x[mask].view(-1))

    def transform(self, x):
        return (x - self.mean) / self.std
