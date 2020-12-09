###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the AI85 implementation and custom PyTorch modules that take
the limits into account.
"""

import torch
import torch.nn as nn

import ai8x


class Fire(nn.Module):
    """
    AI8X - Fire Layer
    """
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes,
                 bias=True, **kwargs):
        super().__init__()
        self.squeeze_layer = ai8x.FusedConv2dReLU(in_channels=in_planes,
                                                  out_channels=squeeze_planes, kernel_size=1,
                                                  bias=bias, **kwargs)
        self.expand1x1_layer = ai8x.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand1x1_planes, kernel_size=1,
                                                    bias=bias, **kwargs)
        self.expand3x3_layer = ai8x.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand3x3_planes, kernel_size=3,
                                                    padding=1, bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.squeeze_layer(x)
        return torch.cat([
            self.expand1x1_layer(x),
            self.expand3x3_layer(x)
        ], 1)


class NoResidual(nn.Module):
    """
    Do nothing
    """
    def forward(self, *x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return x[0]


class Bottleneck(nn.Module):
    """
    AI8X - Bottleneck Layer
    """
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1, bias=False,
                 **kwargs):
        super().__init__()
        self.stride = stride
        hidden_channels = int(round(in_channels * expansion_factor))
        if hidden_channels == in_channels:
            self.conv1 = ai8x.Empty()
        else:
            self.conv1 = ai8x.FusedConv2dBNReLU(in_channels, hidden_channels, 1, padding=0,
                                                bias=bias, **kwargs)
        if stride == 1:
            self.conv2 = ai8x.FusedDepthwiseConv2dBNReLU(hidden_channels, hidden_channels, 3,
                                                         padding=1, stride=stride, bias=bias,
                                                         **kwargs)
        else:
            self.conv2 = ai8x.FusedAvgPoolDepthwiseConv2dBNReLU(hidden_channels, hidden_channels,
                                                                3, padding=1, pool_size=stride,
                                                                pool_stride=stride, bias=bias,
                                                                **kwargs)
        self.conv3 = ai8x.FusedConv2dBN(hidden_channels, out_channels, 1, bias=bias, **kwargs)

        if (stride == 1) and (in_channels == out_channels):
            self.resid = ai8x.Add()
        else:
            self.resid = NoResidual()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return self.resid(y, x)
