import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from typing import Tuple


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
        hidden_channels: Tuple[int] = (16, 32, 64, 128, 256),
        fc_hidden_dims: int = 256,
        n_classes: int = 10,
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
        self.dropout = nn.Dropout(p=0.3)
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
        x = x.reshape(-1, self.flattened_dims)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MobileNetV3Backbone(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.n_classes = n_classes

        self.backbone = models.mobilenet_v3_small(pretrained=True)

        # Freeze the backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Transfer learning with only the last few blocks/layers
        self.backbone.features[-2].requires_grad = True
        self.backbone.features[-1].requires_grad = True
        self.backbone.classifier.requires_grad = True

        self.backbone.classifier[-1] = nn.Linear(in_features=1024, out_features=self.n_classes)

    def forward(self, x):
        # Hacky reshaping and scaling
        x = F.interpolate(x, size=(244, 244))
        x = x.repeat(1, 3, 1, 1)
        x = (x - 30) / 300

        return self.backbone(x)


if __name__ == "__main__":
    # Testing only
    width = 64
    height = 44

    model = VanillaCNN(width, height)
    x = torch.randn(1, 1, width, height)
    out = model(x)

    print(out.shape)
