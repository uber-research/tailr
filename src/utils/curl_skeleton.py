from absl import logging
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import layers
import utils

#tfc = tf.compat.v1
tfc = tf


class SharedEncoder(snt.AbstractModule):
    """The shared encoder module, mapping input x to hiddens."""

    def __init__(self, encoder_type, n_enc, enc_strides, name='shared_encoder'):
        """The shared encoder function, mapping input x to hiddens.

        Args:
          encoder_type: str, type of encoder, either 'conv' or 'multi'
          n_enc: list, number of hidden units per layer in the encoder
          enc_strides: list, stride in each layer (only for 'conv' encoder_type)
          name: str, module name used for tf scope.
        """
        super(SharedEncoder, self).__init__(name=name)
        self._encoder_type = encoder_type

        if encoder_type == 'conv':
            self.shared_encoder = layers.SharedConvModule(
                filters=n_enc,
                strides=enc_strides,
                kernel_size=4,
                activation=tf.nn.relu)
                
        elif encoder_type == 'mixed':
            self.shared_encoder = layers.SharedMixedModule(
                filters=n_enc,
                strides=enc_strides,
                kernel_size=4,
                activation=tf.nn.relu)
        elif encoder_type == 'multi':
            self.shared_encoder = snt.nets.MLP(
                name='mlp_shared_encoder',
                output_sizes=n_enc,
                activation=tf.nn.relu,
                activate_final=True)
        else:
            raise ValueError('Unknown encoder_type {}'.format(encoder_type))

    def _build(self, x, is_training):
        if self._encoder_type == 'multi':
            self.conv_shapes = None
            x = snt.BatchFlatten()(x)
            return self.shared_encoder(x)
        else:
            output = self.shared_encoder(x)
            self.conv_shapes = self.shared_encoder.conv_shapes
            return output


def cluster_encoder_fn(hiddens, n_y_active, n_y, is_training=True):
    """The cluster encoder function, modelling q(y | x).

    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
      n_y_active: Tensor, the number of active components.
      n_y: int, number of maximum components allowed (used for tensor size)
      is_training: Boolean, whether to build the training graph or an evaluation
        graph.

    Returns:
      The distribution `q(y | x)`.
    """
    del is_training  # unused for now
    with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
        lin = snt.Linear(n_y, name='mlp_cluster_encoder_final')
        logits = lin(hiddens)

    # Only use the first n_y_active components, and set the remaining to zero.
    if n_y > 1:
        probs = tf.nn.softmax(logits[:, :n_y_active])
        logging.info('Cluster softmax active probs shape: %s',
                     str(probs.shape))
        paddings1 = tf.stack([tf.constant(0), tf.constant(0)], axis=0)
        paddings2 = tf.stack([tf.constant(0), n_y - n_y_active], axis=0)
        paddings = tf.stack([paddings1, paddings2], axis=1)
        probs = tf.pad(probs, paddings) + 0.0 * logits + 1e-12
    else:
        probs = tf.ones_like(logits)
    logging.info('Cluster softmax probs shape: %s', str(probs.shape))

    return tfp.distributions.OneHotCategorical(probs=probs)


def latent_encoder_fn(hiddens, y, n_y, n_z, is_training=True):
    """The latent encoder function, modelling q(z | x, y).

    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
      y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
      n_y: int, number of dims of y.
      n_z: int, number of dims of z.
      is_training: Boolean, whether to build the training graph or an evaluation
        graph.

    Returns:
      The Gaussian distribution `q(z | x, y)`.
    """
    del is_training  # unused for now

    with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
        # Logits for both mean and variance
        n_logits = 2 * n_z

        all_logits = []
        for k in range(n_y):
            lin = snt.Linear(n_logits, name='mlp_latent_encoder_' + str(k))
            all_logits.append(lin(hiddens))

    # Sum over cluster components.
    all_logits = tf.stack(all_logits)  # [n_y, B, n_logits]
    logits = tf.einsum('ij,jik->ik', y, all_logits)

    # Compute distribution from logits.
    return utils.generate_gaussian(
        logits=logits, sigma_nonlin='softplus', sigma_param='var')


def data_decoder_fn(z,
                    y,
                    output_type,
                    output_shape,
                    decoder_type,
                    n_dec,
                    dec_up_strides,
                    n_x,
                    n_y,
                    shared_encoder_conv_shapes=None,
                    is_training=True,
                    test_local_stats=True):
    """The data decoder function, modelling p(x | z).

    Args:
      z: Latent variables, `Tensor` of size `[B, n_z]`.
      y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
      output_type: str, output distribution ('bernoulli' or 'quantized_normal').
      output_shape: list, shape of output (not including batch dimension).
      decoder_type: str, 'single', 'multi', or 'deconv'.
      n_dec: list, number of hidden units per layer in the decoder
      dec_up_strides: list, stride in each layer (only for 'deconv' decoder_type).
      n_x: int, number of dims of x.
      n_y: int, number of dims of y.
      shared_encoder_conv_shapes: the shapes of the activations of the
        intermediate layers of the encoder,
      is_training: Boolean, whether to build the training graph or an evaluation
        graph.
      test_local_stats: Boolean, whether to use the test batch statistics at test
        time for batch norm (default) or the moving averages.

    Returns:
      The Bernoulli distribution `p(x | z)`.
    """

    if output_type == 'bernoulli':
        def output_dist(x): return tfp.distributions.Bernoulli(logits=x)
        n_out_factor = 1
        out_shape = list(output_shape)
    else:
        raise NotImplementedError
    if len(z.shape) != 2:
        raise NotImplementedError('The data decoder function expects `z` to be '
                                  '2D, but its shape was %s instead.' %
                                  str(z.shape))
    if len(y.shape) != 2:
        raise NotImplementedError('The data decoder function expects `y` to be '
                                  '2D, but its shape was %s instead.' %
                                  str(y.shape))

    # Upsample layer (deconvolutional, bilinear, ..).
    if decoder_type == 'deconv':

        # First, check that the encoder is convolutional too (needed for batchnorm)
        if shared_encoder_conv_shapes is None:
            raise ValueError('Shared encoder does not contain conv_shapes.')

        num_output_channels = output_shape[-1]
        conv_decoder = UpsampleModule(
            filters=n_dec,
            kernel_size=3,
            activation=tf.nn.relu,
            dec_up_strides=dec_up_strides,
            enc_conv_shapes=shared_encoder_conv_shapes,
            n_c=num_output_channels * n_out_factor,
            method=decoder_type)
        logits = conv_decoder(
            z, is_training=is_training, test_local_stats=test_local_stats)
        # n_out_factor in last dim
        logits = tf.reshape(logits, [-1] + out_shape)

    # Multiple MLP decoders, one for each component.
    elif decoder_type == 'multi':
        all_logits = []
        for k in range(n_y):
            mlp_decoding = snt.nets.MLP(
                name='mlp_latent_decoder_' + str(k),
                output_sizes=n_dec + [n_x * n_out_factor],
                activation=tf.nn.relu,
                activate_final=False)
            logits = mlp_decoding(z)
            all_logits.append(logits)

        all_logits = tf.stack(all_logits)
        logits = tf.einsum('ij,jik->ik', y, all_logits)
        logits = tf.reshape(logits, [-1] + out_shape)  # Back to 4D

    # Single (shared among components) MLP decoder.
    elif decoder_type == 'single':
        mlp_decoding = snt.nets.MLP(
            name='mlp_latent_decoder',
            output_sizes=n_dec + [n_x * n_out_factor],
            activation=tf.nn.relu,
            activate_final=False)
        logits = mlp_decoding(z)
        logits = tf.reshape(logits, [-1] + out_shape)  # Back to 4D
    else:
        raise ValueError('Unknown decoder_type {}'.format(decoder_type))

    return output_dist(logits)


def latent_decoder_fn(y, n_z, is_training=True):
    """The latent decoder function, modelling p(z | y).

    Args:
      y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
      n_z: int, number of dims of z.
      is_training: Boolean, whether to build the training graph or an evaluation
        graph.

    Returns:
      The Gaussian distribution `p(z | y)`.
    """
    del is_training  # Unused for now.
    if len(y.shape) != 2:
        raise NotImplementedError('The latent decoder function expects `y` to be '
                                  '2D, but its shape was %s instead.' %
                                  str(y.shape))

    lin_mu = snt.Linear(n_z, name='latent_prior_mu')
    lin_sigma = snt.Linear(n_z, name='latent_prior_sigma')

    mu = lin_mu(y)
    sigma = lin_sigma(y)

    logits = tf.concat([mu, sigma], axis=1)

    return utils.generate_gaussian(
        logits=logits, sigma_nonlin='softplus', sigma_param='var')


class UpsampleModule(snt.AbstractModule):
    """Convolutional decoder.

    If `method` is 'deconv' apply transposed convolutions with stride 2,
    otherwise apply the `method` upsampling function and then smooth with a
    stride 1x1 convolution.

    Params:
    -------
    filters: list, where the first element is the number of filters of the initial
      MLP layer and the remaining elements are the number of filters of the
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
                 filters,
                 kernel_size,
                 activation,
                 dec_up_strides,
                 enc_conv_shapes,
                 n_c,
                 method='nn',
                 name='upsample_module'):
        super(UpsampleModule, self).__init__(name=name)

        assert len(filters) == len(dec_up_strides) + 1, (
            'The decoder\'s filters should contain one element more than the '
            'decoder\'s up stride list, but has %d elements instead of %d.\n'
            'Decoder filters: %s\nDecoder up strides: %s' %
            (len(filters), len(dec_up_strides) + 1, str(filters),
             str(dec_up_strides)))

        self._filters = filters
        self._kernel_size = kernel_size
        self._activation = activation

        self._dec_up_strides = dec_up_strides
        self._enc_conv_shapes = enc_conv_shapes
        self._n_c = n_c
        if method == 'deconv':
            self._conv_layer = tf.layers.Conv2DTranspose
            self._method = method
        else:
            self._conv_layer = tf.layers.Conv2D
            self._method = getattr(tf.image.ResizeMethod, method.upper())
        self._method_str = method.capitalize()

    def _build(self, z, is_training=True, test_local_stats=True, use_bn=False):
        batch_norm_args = {
            'is_training': is_training,
            'test_local_stats': test_local_stats
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

        # The first filter is an MLP.
        mlp_filter, conv_filters = self._filters[0], self._filters[1:]
        # The first shape is used after the MLP to go to 4D.

        layers = [z]
        # The shape of the first enc is used after the MLP to go back to 4D.
        dec_mlp = snt.nets.MLP(
            name='dec_mlp_projection',
            output_sizes=[mlp_filter, np.prod(enc_conv_shapes[0][1:])],
            use_bias=not use_bn,
            activation=self._activation,
            activate_final=True)

        upsample_mlp_flat = dec_mlp(z)
        if use_bn:
            upsample_mlp_flat = snt.BatchNorm(scale=True)(upsample_mlp_flat,
                                                          **batch_norm_args)
        layers.append(upsample_mlp_flat)

        # Ignore the batch size
        enc_conv_shapes[0] = [upsample_mlp_flat.shape[0]] + \
            enc_conv_shapes[0][1:]

        upsample = tf.reshape(upsample_mlp_flat, [-1] + enc_conv_shapes[0][1:])
        layers.append(upsample)

        for i, (filter_i, stride_i) in enumerate(zip(conv_filters, strides), 1):
            if method != 'deconv' and stride_i > 1:
                upsample = tf.image.resize_images(
                    upsample, [stride_i *
                               el for el in upsample.shape.as_list()[1:3]],
                    method=method,
                    name='upsample_' + str(i))
            upsample = self._conv_layer(
                filters=filter_i,
                kernel_size=self._kernel_size,
                padding='same',
                use_bias=not use_bn,
                activation=self._activation,
                strides=stride_i if method == 'deconv' else 1,
                name='upsample_conv_' + str(i))(
                    upsample)
            if use_bn:
                upsample = snt.BatchNorm(scale=True)(
                    upsample, **batch_norm_args)
            if stride_i > 1:
                hw = unique_hw.pop()
                upsample = utils.maybe_center_crop(upsample, hw)
            layers.append(upsample)

        # Final layer, no upsampling.
        x_logits = tf.layers.Conv2D(
            filters=self._n_c,
            kernel_size=self._kernel_size,
            padding='same',
            use_bias=not use_bn,
            activation=None,
            strides=1,
            name='logits')(
                upsample)
        if use_bn:
            x_logits = snt.BatchNorm(scale=True)(x_logits, **batch_norm_args)
        layers.append(x_logits)

        logging.info('%s upsampling module layer shapes', self._method_str)
        logging.info('\n'.join([str(v.shape.as_list()) for v in layers]))

        return x_logits
