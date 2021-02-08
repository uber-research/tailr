""" Classes for the classifier of TAILR """

from absl import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Classifier(nn.Module):
    """ Simple CNN """

    def __init__(self,
                 in_channel,
                 out_channels,
                 kernel_sizes,
                 cnn_activation,
                 strides,
                 lin_activation,
                 n_classes,
                 name='conv_classifier'):
        """ Creates a CNN Classifier

        Args:
            in_channel: the dimensionality of the input space, int
            out_channels: the dimensionality of the output space (essentially, the channel sizes), List[int] 
            kernel_size: The size of the the kernels, List[List[int]]
            cnn_activation: The activation function used between the CNN layers
            strides: The stride size, int or Tuple[int] or List[int]
            lin_activation: The activation function used after the linear layer
            n_classes: The number of classes
            name: The name of the classifier
        """
        super(CNN_Classifier, self).__init__()

        self.layers = []

        for i, (out_channel, kernel_size, stride) in enumerate(zip(out_channels, kernel_sizes, strides)):
            self.layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride))
            self.layers.append(cnn_activation())
            self.layers.append(nn.MaxPool2d(kernel_size//2))
            in_channel = out_channel

        self.layers.append(nn.Flatten(start_dim=1))
        self.layers.append(nn.Linear(256, n_classes))
        # self.layers.append(lin_activation())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """ Applies the CNN classifier to a given input x

        Args: 
            x: the input tensor

        Returns:
            Class scores

        """

        h = self.layers(x)

        logging.info('Classifier output shape:')
        logging.info(h.shape)

        return h


class DGR_CNN(nn.Module):
    def __init__(self,
                 image_size,
                 image_channel_size, classes,
                 depth, channel_size, reducing_layers=3):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.classes = classes
        self.depth = depth
        self.channel_size = channel_size
        self.reducing_layers = reducing_layers

        # layers
        self.layers = nn.ModuleList([nn.Conv2d(
            self.image_channel_size, self.channel_size//(2**(depth-2)),
            3, 1, 1
        )])

        for i in range(self.depth-2):
            previous_conv = [
                l for l in self.layers if
                isinstance(l, nn.Conv2d)
            ][-1]
            self.layers.append(nn.Conv2d(
                previous_conv.out_channels,
                previous_conv.out_channels * 2,
                3, 1 if i >= reducing_layers else 2, 1
            ))
            self.layers.append(nn.BatchNorm2d(previous_conv.out_channels * 2))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten(start_dim=1))

        self.out = nn.Linear(
            16384,
            self.classes
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
