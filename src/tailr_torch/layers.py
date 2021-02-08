# Some building blocks of the various parts of the TAILR Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image

from utils import get_same_padding, generate_gaussian, crop_layer


class MLP(nn.Module):
    """A simple MLP"""

    def __init__(self, input_dim, hidden_dims, activation, activate_final=False):
        """
        A simple MLP

        Args:
          input_dim: int, dimension of the input layer
          hidden_dims: List[int], dimensions of the outputs of the hidden layers
          activation: activation used between the layers
        """

        super(MLP, self).__init__()
        # self.graph = nn.Sequential()
        self.graph = nn.ModuleList()

        prev_out_dim = input_dim

        for i, out_dim in enumerate(hidden_dims):
            layer = nn.Linear(
                in_features=prev_out_dim,
                out_features=out_dim,
            )
            self.graph.append(layer)
            if i != len(hidden_dims) - 1 or activate_final:
                self.graph.append(activation)
            prev_out_dim = out_dim

    def forward(self, x, printing=False):
        out = x
        for i, module in enumerate(self.graph):
            out = module(out)
            if printing:
                print(i, out)
        return out


class SharedConvModule(nn.Module):
    """Convolutional decoder."""

    def __init__(self,
                 in_channels,
                 layers_out_channels,
                 strides,
                 kernel_size,
                 activation):
        super(SharedConvModule, self).__init__()

        self._in_channels = in_channels
        self._layers_out_channels = layers_out_channels
        self._kernel_size = kernel_size
        self._activation = activation
        self.strides = strides
        assert len(strides) == len(layers_out_channels) - 1
        self.conv_shapes = None

        self.graph = nn.Sequential()

        prev_out_put_size = self._in_channels
        for i, (output_size_i,
                stride_i) in enumerate(zip(self._layers_out_channels, self.strides)):
            conv = nn.Conv2d(
                in_channels=prev_out_put_size,
                out_channels=output_size_i,
                kernel_size=self._kernel_size,
                stride=stride_i,
            )
            prev_out_put_size = output_size_i
            self.graph.add_module('enc_conv_%d' % i, conv)
            self.graph.add_module('activtion_%d' % i, self._activation)

        self.graph.add_module('flatten', nn.Flatten())

        # use last defined value of output_size_i as in_features size
        last_layer = nn.Linear(
            in_features=self._layers_out_channels[-1],
            out_features=self._layers_out_channels[-1],
        )

        self.graph.add_module('enc_mlp', last_layer)
        self.graph.add_module('mlp_activation', self._activation)

    def forward(self, x):
        assert len(x.shape) == 4

        self.conv_shapes = [x.shape]  # Needed by deconv module
        conv = x

        out = self.graph(conv)

        logging.info('Shared conv module layer shapes:')
        logging.info('\n'.join([str(el) for el in self.conv_shapes]))
        logging.info(out.shape)

        return out


class ConvDecoder(nn.Module):
    """Convolutional decoder.

    If `method` is 'deconv' apply transposed convolutions with stride 2,
    otherwise apply the `method` upsampling function and then smooth with a
    stride 1x1 convolution.

    Params:
    -------
    output_dims: list, where the first element is the output_dim of the initial
    MLP layer and the remaining elements are the output_dims of the
    upsampling layers.
    kernel_size: the size of the convolutional kernels. The same size will be
    used in all convolutions.
    activation: an activation function, applied to all layers but the last.
    dec_up_strides: list, the upsampling factors of each upsampling convolutional
    layer.
    enc_conv_shapes: list, the shapes of the input and of all the intermediate
    feature maps of the convolutional layers in the encoder.
    n_c: the number of output channels.
    """

    def __init__(self,
                 in_channels,
                 output_dims,
                 kernel_size,
                 activation,
                 dec_up_strides,
                 enc_conv_shapes,
                 n_c,
                 method='nn',
                 use_bn=False):

        super(ConvDecoder, self).__init__()

        assert len(output_dims) == len(dec_up_strides) + 1, (
            'The decoder\'s output_dims should contain one element more than the '
            'decoder\'s up stride list, but has %d elements instead of %d.\n'
            'Decoder output_dims: %s\nDecoder up strides: %s' %
            (len(output_dims), len(dec_up_strides) + 1, str(output_dims),
             str(dec_up_strides)))

        self._output_dims = output_dims
        self._kernel_size = kernel_size
        self._activation = activation
        self._use_bn = use_bn

        self._dec_up_strides = dec_up_strides
        self._enc_conv_shapes = enc_conv_shapes
        self._n_c = n_c

        # The first output_dim is an MLP.
        mlp_filter, conv_filters = self._output_dims[0], self._output_dims[1:]
        # The first shape is used after the MLP to go to 4D.

        mlp_hidden_dims = [mlp_filter, np.prod(enc_conv_shapes[0][1:])]
        self.dec_mlp = MLP(
            input_dim=in_channels,
            hidden_dims=mlp_hidden_dims,
            activation=self._activation)

        if method == 'deconv':
            self._conv_layer_fn = nn.ConvTranspose2d
            self._method = method
        else:
            self._conv_layer_fn = nn.Conv2d
            self._method = getattr(Image, method.upper())
        self._method_str = method.capitalize()

        # Upsampling layers
        self._conv_layers = []
        for i, (filter_i, stride_i) in enumerate(zip(conv_filters, self._dec_up_strides), 1):
            if i == 0:
                in_chans = mlp_hidden_dims[-1]
            self._conv_layers.append(self._conv_layer_fn(
                in_channels=in_chans,
                out_channels=filter_i,
                kernel_size=self._kernel_size,
                bias=not use_bn,
                stride=stride_i if method == 'deconv' else 1))
            in_chans = filter_i

        # Final layer, no upsampling.
        self.logits_layer = nn.Conv2d(
            in_channels=conv_filters[-1],
            out_channels=self._n_c,
            kernel_size=self._kernel_size,
            bias=not self._use_bn,
            stride=1)

    def forward(self, z, is_training=True, test_local_stats=True):
        batch_norm_args = {
            'track_running_stats': test_local_stats
        }

        method = self._method
        # Cycle over the encoder shapes backwards, to build a symmetrical decoder.
        enc_conv_shapes = self._enc_conv_shapes[::-1]
        strides = self._dec_up_strides
        # We store the heights and widths of the encoder feature maps that are
        # unique, i.e., the ones right after a layer with stride != 1. These will be
        # used as a target to potentially crop the upsampled feature maps.
        unique_hw = np.unique([(el[1], el[2])
                               for el in enc_conv_shapes], axis=0)
        unique_hw = unique_hw.tolist()[::-1]
        unique_hw.pop()  # Drop the initial shape

        layers = [z]

        upsample_mlp_flat = self.dec_mlp(z)
        if self._use_bn:
            upsample_mlp_flat = nn.BatchNorm1d(upsample_mlp_flat.shape[1],
                                               **batch_norm_args)(upsample_mlp_flat)
        layers.append(upsample_mlp_flat)
        upsample = upsample_mlp_flat.view(enc_conv_shapes[0])
        layers.append(upsample)

        for i, (conv_layer, stride_i) in enumerate(zip(self._conv_layers, strides), 1):
            if method != 'deconv' and stride_i > 1:
                upsample = Image.resize(
                    upsample, [stride_i * el for el in upsample.shape[1:3]],
                    method=method)
            padding = get_same_padding(
                self._kernel_size, upsample.shape, stride_i)
            upsample_padded = F.pad(upsample, padding, "constant", 0)
            upsample = conv_layer(upsample_padded)
            upsample = self._activation(upsample)
            if self._use_bn:
                upsample = nn.BatchNorm2d(
                    upsample.shape[1], **batch_norm_args)(upsample)
            if stride_i > 1:
                hw = unique_hw.pop()
                upsample = crop_layer(upsample, hw)
            layers.append(upsample)

        # Final layer, no upsampling.
        padding = get_same_padding(self._kernel_size, upsample.shape, 1)
        upsample_padded = F.pad(upsample, padding, "constant", 0)
        x_logits = self.logits_layer(upsample_padded)
        if self._use_bn:
            x_logits = nn.BatchNorm2d(
                x_logits.shape[1], **batch_norm_args)(x_logits)
        layers.append(x_logits)

        logging.info('%s upsampling module layer shapes', self._method_str)
        logging.info('\n'.join([str(v.shape) for v in layers]))

        return x_logits
