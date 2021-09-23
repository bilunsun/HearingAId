import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class VanillaCNN(nn.Module):
    """
    Vanilla Convolutional Neural Network architecture

    The model consists of 4 convolutional blocks (Conv2d -> ReLU -> MaxPool2d),
    followed by two linear layers.
    """

    def __init__(
        self,
        width: int,
        height: int,
        in_channels: int = 1,
        hidden_channels: List[int] = [16, 32, 64, 128],
        fc_hidden_dims: int = 128,
        n_classes: int = 6,
    ):
        super().__init__()

        self.width = width
        self.height = height
        self.in_channels = in_channels

        modules = [self._conv_block(in_channels, hidden_channels[0])]
        for i_c, o_c in zip(hidden_channels[:-1], hidden_channels[1:]):
            modules.append(self._conv_block(i_c, o_c))
        self.conv_blocks = nn.Sequential(*modules)

        self.flattened_dims = self._get_flattened_dims(self.width, self.height, self.in_channels)
        self.fc1 = nn.Linear(self.flattened_dims, fc_hidden_dims)
        self.fc2 = nn.Linear(fc_hidden_dims, n_classes)

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pooling_factor: int = 2,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_factor),
        )

    def _get_flattened_dims(self, width: int, height: int, in_channels: int) -> int:
        x = torch.randn(1, in_channels, width, height)
        x = self.conv_blocks(x)
        flattened_dims = x.view(1, -1).size(1)

        return flattened_dims

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.reshape(1, self.flattened_dims)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Testing only
    width = 128
    height = 64

    model = VanillaCNN(width, height)
    x = torch.randn(1, 1, width, height)
    out = model(x)

    print(out.shape)
