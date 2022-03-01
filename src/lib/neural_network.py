import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class EnvNet(nn.Module):
    CLASSIFIER_HIDDEN_DIM = 1024

    def __init__(self, width: int, height: int, n_classes: int, in_channels: int = 1):
        super().__init__()

        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 64), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 16), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 64), stride=(1, 64))

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(8, 8), stride=(1, 1))
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pool4 = nn.MaxPool2d(kernel_size=(5, 3), stride=(5, 3))

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), stride=(1, 1))
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 4), stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(num_features=64)
        self.pool6 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 2), stride=(1, 1))
        self.bn7 = nn.BatchNorm2d(num_features=128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 1))
        self.bn8 = nn.BatchNorm2d(num_features=128)
        self.pool8 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.conv9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 2), stride=(1, 1))
        self.bn9 = nn.BatchNorm2d(num_features=256)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 2), stride=(1, 1))
        self.bn10 = nn.BatchNorm2d(num_features=256)
        self.pool10 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        x = torch.randn(1, in_channels, height, width)
        x = self.backbone(x)
        print("Backbone output shape:", x.shape)
        x = x.reshape(x.size(0), -1)
        self.latent_dim = x.size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.CLASSIFIER_HIDDEN_DIM),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.CLASSIFIER_HIDDEN_DIM, self.CLASSIFIER_HIDDEN_DIM),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.CLASSIFIER_HIDDEN_DIM, n_classes),
        )

    def backbone(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = x.transpose(1, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool6(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool8(x)

        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool10(x)

        return x

    def forward(self, x):
        x = self.backbone(x)

        # x = x.max(dim=3).values
        x = x.view(x.size(0), self.latent_dim)

        x = self.classifier(x)

        return x


class Separable3x3Conv2d(nn.Module):
    """Inspired from https://arxiv.org/pdf/1704.04861.pdf"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, bias: bool = False) -> None:
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=stride, kernel_size=3, padding=1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)

        return x


class CustomCNN(nn.Module):
    """
    Convolutional Neural Network architecture with separable convolutions
    """

    def __init__(
        self,
        width: int,
        height: int,
        in_channels: int = 1,
        hidden_channels: List[int] = [32, 64, 128, 256, 512],
        classifier_hidden_dims: int = 1024,
        n_classes: int = 10,
        qat: bool = False
    ):
        super().__init__()

        self.width = width
        self.height = height

        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=3, stride=2)

        hidden_channels = [in_channels] + hidden_channels
        self.sep_conv_blocks = nn.ModuleList([
            Separable3x3Conv2d(in_channels=ic, out_channels=io, stride=2) for ic, io in zip(hidden_channels[:-1], hidden_channels[1:])
        ])

        x = torch.randn(1, in_channels, height, width)
        for conv in self.sep_conv_blocks:
            x = conv(x)
        x = x.reshape(x.size(0), -1)
        self.latent_dim = x.size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, classifier_hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_hidden_dims, n_classes)
        )

        self.quant = torch.quantization.QuantStub() if qat else nn.Identity()
        self.dequant = torch.quantization.DeQuantStub() if qat else nn.Identity()

    def forward(self, x):
        x = self.quant(x)

        for conv in self.sep_conv_blocks:
            x = conv(x)

        # Flatten
        x = x.reshape(x.size(0), self.latent_dim)
        x = self.classifier(x)

        x = self.dequant(x)

        return x


def test_custom_cnn():
    width = 44
    height = 64
    model = CustomCNN(width=width, height=height, n_classes=10)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_params)

    x = torch.randn(32, 1, height, width)
    out = model(x)
    print(out.shape)


def test_env_net():
    model = EnvNet(n_classes=10)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_params)

    x = torch.randn(32, 1, 1, 44_100)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    test_custom_cnn()
    # test_env_net()
