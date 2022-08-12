from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int, stride: int):
        super().__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9))
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        residual = x

        out = F.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = F.relu(out)
        return out + residual


# adapted from https://raw.githubusercontent.com/matthias-wright/cifar10-resnet/master/model.py
# under the MIT license
class ResNet9(nn.Module):
    """A 9-layer residual network, excluding BatchNorms and activation functions, as
    described in this blog post: https://myrtle.ai/learn/how-to-train-your-
    resnet-4-architecture/

    Args:
        num_classes: number of classes for the final classifier layer
        residual_factory: a callable that returns a residual block;
            defaults to the original ResNet9 residual block, but can be
            used to specify a custom one
    """

    def __init__(self, num_classes: int, residual_factory: Optional[Callable] = None):
        super().__init__()
        residual_factory = residual_factory or _ResidualBlock

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # residual_factory(in_channels=128,
            _ResidualBlock(in_channels=128,
                           out_channels=128,
                           kernel_size=3,
                           stride=1,
                           padding=1),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            residual_factory(in_channels=256,
                             out_channels=256,
                             kernel_size=3,
                             stride=1,
                             padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor):  # type: ignore
        out = self.body(x)
        out = out.reshape(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out