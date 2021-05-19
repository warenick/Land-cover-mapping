""" This file contains the defenition of the ResNet-110 model.

The architecuture is an extension of the model from 
"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>.

"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet110(nn.Module):
    """ ResNet-110 image classification network. """

    def __init__(self, in_channels, n_classes):
        super(ResNet110, self).__init__()

        self.resnet = models.resnet101()
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.resnet.fc = nn.Linear(2048, n_classes)

        # to transform the ResNet-101 to ResNet-110, add 3 extra bottleneck blocks
        # each bottleneck block adds 3 layers, 101 + 3 * 3 = 110
        for i in range(3):
            bottleneck = models.resnet.Bottleneck(1024, 256)
            self.resnet.layer3.add_module(f"extra_{i}", bottleneck)

    def forward(self, x):
        return self.resnet(x)
