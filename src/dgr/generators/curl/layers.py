from absl import logging
import torch
import torch.nn as nn


class ResidualStack(nn.Module):
    """A stack of ResNet V2 blocks."""

    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 filter_size=3,
                 activation=nn.ReLU()):
        """Instantiate a ResidualStack."""
        super(ResidualStack, self).__init__()

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._filter_size = filter_size
        self._activation = activation

        self.residual_layers = []

        # TODO: check the in_channels values
        # Missing parameter entry for the in_channels
        for i in range(self._num_residual_layers):
            nxn_layer = nn.Conv2d(
                in_channels=self._num_hiddens,
                out_channels=self._num_residual_hiddens,
                kernel_size=(self._filter_size, self._filter_size),
                stride=(1, 1),
            )

            onexone_layer = nn.Conv2d(
                in_channels=self._num_residual_hiddens,
                out_channels=self._num_hiddens,
                kernel_size=(1, 1),
                stride=(1, 1),
            )

            self.residual_layers.append([nxn_layer, onexone_layer])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x_i = self._activation(x)

            x_i = self.residual_layers[i][0](x_i)
            x_i = self._activation(x_i)

            x_i = self.residual_layers[i][1](x_i)

            x += x_i

        return self._activation(x)


class SharedConvModule(nn.Module):
    """Convolutional decoder."""

    def __init__(self,
                 in_channels,
                 output_dims,
                 strides,
                 kernel_size,
                 activation):
        super(SharedConvModule, self).__init__()

        self._in_channels = in_channels
        self._output_dims = output_dims
        self._kernel_size = kernel_size
        self._activation = activation
        self.strides = strides
        assert len(strides) == len(output_dims) - 1
        self.conv_shapes = None

        self.graph = nn.Sequential()

        prev_out_put_size = self._in_channels
        for i, (output_size_i,
                stride_i) in enumerate(zip(self._output_dims, self.strides)):
            conv = nn.Conv2d(
                in_channels=prev_out_put_size,
                out_channels=output_size_i,
                kernel_size=self._kernel_size,
                stride=stride_i,
            )
            prev_out_put_size = output_size_i
            self.graph.add_module("enc_conv_%d" % i, conv)
            self.graph.add_module("activtion_%d" % i, self._activation)

        self.graph.add_module(nn.Flatten())

        # use last defined value of output_size_i as in_features size
        last_layer = nn.Linear(
            in_features=self._output_dims[-1],
            out_features=self._output_dims[-1],
        )

        self.graph.add_module("enc_mlp", last_layer)
        self.graph.add_module("mlp_activation", self._activation)

    def forward(self, x):
        assert len(x.shape) == 4

        self.conv_shapes = [x.shape]  # Needed by deconv module
        conv = x

        out = self.graph(conv)

        logging.info('Shared conv module layer shapes:')
        logging.info('\n'.join([str(el) for el in self.conv_shapes]))
        logging.info(out.shape)

        return out
