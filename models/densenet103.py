""" This file contains the defenition of the FC-DenseNet103 model.

The architucture and the definitions for all buildings blocks are introduced in
[Jegou et al., 2017](https://arxiv.org/abs/1611.09326).

"""

import torch
import torch.nn as nn


class FCDenseNet103(nn.Module):
    """ As defined in Table 2 of (Jegou et al., 2017). """

    def __init__(self, in_channels, n_classes):
        super(FCDenseNet103, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, 48, 3, padding=1)

        self.db0 = DenseBlock(48, 4, 16)
        self.td0 = TransitionDown(112)
        self.db1 = DenseBlock(112, 5, 16)
        self.td1 = TransitionDown(192)
        self.db2 = DenseBlock(192, 7, 16)
        self.td2 = TransitionDown(304)
        self.db3 = DenseBlock(304, 10, 16)
        self.td3 = TransitionDown(464)
        self.db4 = DenseBlock(464, 12, 16)
        self.td4 = TransitionDown(656)

        self.bottleneck = DenseBlock(656, 15, 16)

        self.tu0 = TransitionUp(240)
        self.db5 = DenseBlock(896, 12, 16)
        self.tu1 = TransitionUp(192)
        self.db6 = DenseBlock(656, 10, 16)
        self.tu2 = TransitionUp(160)
        self.db7 = DenseBlock(464, 7, 16)
        self.tu3 = TransitionUp(112)
        self.db8 = DenseBlock(304, 5, 16)
        self.tu4 = TransitionUp(80)
        self.db9 = DenseBlock(192, 4, 16)

        self.conv1 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        """ As defined in Figure 1 and Table 2 of (Jegou et al., 2017). """

        out_a = self.conv0(x)

        # beginning the downsampling path
        # trying to get clever with reuse of tensors to fit everything on GPU

        out_b = self.db0(out_a)
        cat0 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td0(cat0)

        out_b = self.db1(out_a)
        cat1 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td1(cat1)

        out_b = self.db2(out_a)
        cat2 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td2(cat2)

        out_b = self.db3(out_a)
        cat3 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td3(cat3)

        out_b = self.db4(out_a)
        cat4 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td4(cat4)

        # end of the downsampling path

        out_a = self.bottleneck(out_a)

        # beginning of the upsampling path

        out_a = self.tu0(out_a)
        out_a = torch.cat([out_a, cat4], dim=1)
        out_a = self.db5(out_a)

        out_a = self.tu1(out_a)
        out_a = torch.cat([out_a, cat3], dim=1)
        out_a = self.db6(out_a)

        out_a = self.tu2(out_a)
        out_a = torch.cat([out_a, cat2], dim=1)
        out_a = self.db7(out_a)

        out_a = self.tu3(out_a)
        out_a = torch.cat([out_a, cat1], dim=1)
        out_a = self.db8(out_a)

        out_a = self.tu4(out_a)
        out_a = torch.cat([out_a, cat0], dim=1)
        out_a = self.db9(out_a)

        out_a = self.conv1(out_a)

        return out_a


class DenseBlock(nn.Module):
    """ As defined in Figure 2 of (Jegou et al., 2017). """

    def __init__(self, in_channels, n_layers, growth_rate):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()

        in_channels = in_channels
        for i in range(n_layers):
            layer = Layer(in_channels, growth_rate)
            self.layers.append(layer)
            in_channels += growth_rate

    def forward(self, x):
        feature_maps = [x]

        for layer in self.layers:
            in_ = torch.cat(feature_maps, dim=1)
            out = layer(in_)
            feature_maps.append(out)

        # input is not concatenated with the output within the DenseBlock itself
        # (for the upsampling path, to combat feature map explosion)
        # instead, the concatanation is handled in the forward pass of the FC-DenseNet

        out = torch.cat(feature_maps[1:], dim=1)

        return out


class TransitionDown(nn.Module):
    """ As defined in Table 1 of (Jegou et al., 2017). """

    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout2d(0.2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = x.relu()
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)

        return x


class TransitionUp(nn.Module):
    """ As defined in Table 1 of (Jegou et al., 2017). """

    def __init__(self, in_channels):
        super(TransitionUp, self).__init__()

        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x = self.transposed_conv(x)

        return x


class Layer(nn.Module):
    """ As defined in Table 1 of (Jegou et al., 2017). """

    def __init__(self, in_channels, growth_rate):
        super(Layer, self).__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = x.relu()
        x = self.conv(x)
        x = self.dropout(x)

        return x
