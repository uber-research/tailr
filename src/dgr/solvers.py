from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import dgr
from utils import dgr_utils


class CNN(dgr.Solver):
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

        self.layers.append(dgr_utils.LambdaModule(
            lambda x: x.view(x.size(0), -1)))
        self.layers.append(nn.Linear(
            (image_size//(2**reducing_layers))**2 * channel_size,
            self.classes
        ))

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)
