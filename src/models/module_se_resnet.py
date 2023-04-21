"""
module_se_resnet.py

This module contains the implementation of the SEResNet architecture.
"""
import torch.nn as nn


class SENet(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """
    Basic building block for squeeze-and-excitation networks with other layers.
    """
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, stride)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(output_channels, output_channels, 1)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.se = SENet(output_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        # Downsample if necessary
        if self.downsample is not None:
            residual = self.downsample(x)

        # Add residual connection and apply ReLU
        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    """
    Residual network with squeeze-and-excitation blocks

    Downsampling is performed by conv1 when stride != 1 or
    the input_channels size is not equal to the output size.
    """
    def __init__(self, output_channels, block_size):
        super(SEResNet, self).__init__()
        self.input_channels = 192
        self.block = SEBasicBlock
        self.layer = self._make_layer(self.block, output_channels, block_size)

    # Create a layer with 'blocks' number of SEBasicBlock instances
    def _make_layer(self, block, output_channels, blocks, stride = 1):
        downsample = self._downsample_layer(self.input_channels, output_channels * block.expansion, stride)

        layers = [block(self.input_channels, output_channels, stride, downsample)]
        self.input_channels = output_channels * block.expansion
        layers.extend(block(self.input_channels, output_channels) for _ in range(1, blocks))

        return nn.Sequential(*layers)

    @staticmethod
    def _downsample_layer(input_channels, output_channels, stride):
        if stride != 1 or input_channels != output_channels:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm1d(output_channels)
            )
        return None

    def forward(self, x):
        return self.layer(x)


