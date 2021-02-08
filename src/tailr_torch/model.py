# Collection of classes that will help build the TAILR model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import logging
from PIL import Image

from layers import MLP, SharedConvModule, ConvDecoder
from utils import get_same_padding, generate_gaussian, crop_layer, kl_gaussian


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class SharedEncoder(nn.Module):
    """The shared encoder module, mapping input x to hiddens."""

    def __init__(self, in_dims, n_enc, enc_strides, encoder_type):
        """The shared encoder function, mapping input x to hiddens.

        Args:
        encoder_type: str, type of encoder, either 'conv' or 'multi'
        n_enc: list, number of hidden units per layer in the encoder
        enc_strides: list, stride in each layer (only for 'conv' encoder_type)
        name: str, module name used for tf scope.
        """
        super(SharedEncoder, self).__init__()
        self._encoder_type = encoder_type

        if encoder_type == 'conv':
            self.encoder = SharedConvModule(
                in_channels=in_dims,
                layers_out_channels=n_enc,
                strides=enc_strides,
                kernel_size=3,
                activation=nn.ReLU())
        elif encoder_type == 'multi':
            self.encoder = MLP(
                input_dim=in_dims,
                hidden_dims=n_enc,
                activation=nn.ReLU(),
                activate_final=True)
        else:
            raise ValueError('Unknown encoder_type {}'.format(encoder_type))

    def forward(self, x):
        if self._encoder_type == 'multi':
            self.conv_shapes = None
            x = nn.Flatten(start_dim=1, end_dim=-1)(x)
            return self.encoder(x)
        else:
            output = self.encoder(x)
            self.conv_shapes = self.encoder.conv_shapes
            return output


class ClusterEncoder(nn.Module):
    """The cluster encoder, modelling q(y | x)."""

    def __init__(self, n_y, feature_size):
        """ Initialize the cluster encoder 

        Args:
            feature_size: Number of features
            n_y: int, number of maximum components allowed (used for tensor size)
        """
        super(ClusterEncoder, self).__init__()

        self.lin = nn.Linear(feature_size, n_y)
        self.n_y = n_y

    def forward(self, hiddens, n_y_active):
        assert hiddens.dim() == 2
        logits = self.lin(hiddens)

        # Only use the first n_y_active components, and set the remaining to zero.
        if self.n_y > 1:
            partial_cluster_score = F.softmax(logits[:, :n_y_active])

            paddings1 = torch.stack([torch.tensor(0), torch.tensor(0)], dim=0)
            paddings2 = torch.stack(
                [torch.tensor(0), torch.tensor(self.n_y - n_y_active)], dim=0)
            paddings = list(torch.cat([paddings2, paddings1], dim=0).numpy())

            cluster_score = F.pad(partial_cluster_score, paddings, value=1e-12)

        else:
            cluster_score = torch.ones_like(logits)

        return cluster_score


class LatentEncoder(nn.Module):
    """The latent encoder, modelling q(z | x, y).

    Args:
        n_y: number of dims of y (number of clusters), int
        n_z: number of dims of z (latent representation), int
        feature_size: number of features in the hidden representation, int
    """

    def __init__(self, n_y, n_z, feature_size):

        super(LatentEncoder, self).__init__()
        self.n_y = n_y
        self.lins = nn.ModuleList()

        # Logits for both mean and variance
        n_logits = 2 * n_z
        for k in range(n_y):
            self.lins.append(nn.Linear(feature_size, n_logits))

    def forward(self, hiddens, y):
        """See above

        Args:
            hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`
            y: Categorical cluster variable, `Tensor` of size `[B, n_y]`

        Returns:
            The Gaussian distribution `q(z | x, y)`
        """
        if y.dim() != 2:
            raise NotImplementedError(
                f'The latent decoder function expects `y` to be 2D, but its shape was {y.shape} instead.')
        all_logits = [lin(hiddens) for lin in self.lins]

        # Sum over cluster components.
        all_logits = torch.stack(all_logits)  # [n_y, B, n_logits]
        # TODO double check the einsum
        logits = torch.einsum('ij,jik->ik', y, all_logits)

        # Compute distribution from logits.
        return generate_gaussian(
            logits=logits, sigma_nonlin='softplus', sigma_param='var')


class LatentPrior(nn.Module):
    """Latent Prior, modelling p(z|y)

    Args:
        n_y: number of dims of y (number of clusters), int
        n_z: number of dims of z (latent representation), int
    """

    def __init__(self, n_y, n_z):
        super(LatentPrior, self).__init__()

        self.lin_mu = nn.Linear(n_y, n_z)
        self.lin_sigma = nn.Linear(n_y, n_z)

    def forward(self, y):
        """See above

        Args:
            y: the cluster score, Tensor [B, n_y]

        Returns:
            The Gaussian distribution `p(z | y)`

        """
        if y.dim() != 2:
            raise NotImplementedError(
                f'The latent decoder function expects `y` to be 2D, but its shape was {y.shape} instead.')

        mu = self.lin_mu(y)
        sigma = self.lin_sigma(y)

        logits = torch.cat([mu, sigma], dim=1)

        return generate_gaussian(
            logits=logits, sigma_nonlin='softplus', sigma_param='var')


class LatentDecoder(nn.Module):
    """The data decoder module, modelling p(x | z)."""

    def __init__(self,
                 z_shape,
                 output_shape,
                 decoder_type,
                 n_dec,
                 dec_up_strides,
                 n_x,
                 n_y,
                 shared_encoder_conv_shapes=None):
        """Module initialization

        Args:
            output_shape: list, shape of output (not including batch dimension).
            decoder_type: str, 'single', 'multi', or 'deconv'.
            n_dec: list, number of hidden units per layer in the decoder
            dec_up_strides: list, stride in each layer (only for 'deconv' decoder_type).
            n_x: int, number of dims of x.
            n_y: int, number of dims of y.
            shared_encoder_conv_shapes: the shapes of the activations of the
              intermediate layers of the encoder.

        Returns:
            Instance of the LatentDecoder 
        """

        super(LatentDecoder, self).__init__()
        self.decoder_type = decoder_type
        self.n_y = n_y

        n_out_factor = 1
        self.out_shape = list(output_shape)

        # Upsample layer (deconvolutional, bilinear, ..).
        if decoder_type == 'deconv':

            # First, check that the encoder is convolutional too (needed for batchnorm)
            if shared_encoder_conv_shapes is None:
                raise ValueError(
                    'Shared encoder does not contain conv_shapes.')

            num_output_channels = output_shape[-1]
            self.decoder = ConvDecoder(
                output_dims=n_dec,
                kernel_size=3,
                activation=nn.ReLU(),
                dec_up_strides=dec_up_strides,
                enc_conv_shapes=shared_encoder_conv_shapes,
                n_c=num_output_channels * n_out_factor,
                method=decoder_type)

        # Multiple MLP decoders, one for each component.
        # NOTE the 'multi' option is not in working condition and probably never will
        elif decoder_type == 'multi':
            self.decoder = []
            for k in range(n_y):
                mlp_decoding = MLP(input_dim=z_shape,
                                   hidden_dims=n_dec + [n_x * n_out_factor],
                                   activation=nn.ReLU(),
                                   activate_final=False)
                self.decoder.append(mlp_decoding)

        # Single (shared among components) MLP decoder.
        elif decoder_type == 'single':
            self.decoder = MLP(input_dim=z_shape,
                               hidden_dims=n_dec + [n_x * n_out_factor],
                               activation=nn.ReLU(),
                               activate_final=False,
                               )
        else:
            raise ValueError(f'Unknown decoder_type {decoder_type}')

    def forward(self, z, y, is_training=True, test_local_stats=True):
        """The Module's forward function

        Args:
            z: Latent variables, `Tensor` of size `[B, n_z]`.
            y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
            is_training: Boolean, whether to build the training graph or an evaluation
              graph.
            test_local_stats: Boolean, whether to use the test batch statistics at test
              time for batch norm (default) or the moving averages.   

        Returns:
            Bernouilli distribution 'p(x | z)'
        """
        if z.dim() != 2:
            raise NotImplementedError(
                f'The data decoder function expects `z` to be 2D, but its shape was {z.shape} instead.')
        if y.dim() != 2:
            raise NotImplementedError(
                f'The data decoder function expects `y` to be 2D, but its shape was {y.shape} instead.')

        if self.decoder_type == 'deconv':
            logits = self.decoder(
                z, is_training=is_training, test_local_stats=test_local_stats)
            # n_out_factor in last dim
            logits = logits.view([-1] + self.out_shape)

        elif self.decoder_type == 'multi':
            all_logits = []
            for k in range(n_y):
                logits = self.decoder[k](z)
                all_logits.append(logits)

            all_logits = torch.stack(all_logits)
            logits = torch.einsum('ij,jik->ik', y, all_logits)
            logits = logits.view([-1] + self.out_shape)  # Back to 4D

        elif self.decoder_type == 'single':
            logits = self.decoder(z)
            logits = logits.view([-1] + self.out_shape)  # Back to 4D

        return logits


class TAILR(nn.Module):
    """Main Module for TAILR, modelling q(x)"""

    # TODO Add the needed parameters as inputs to __init__
    def __init__(self,
                 n_x,
                 n_y,
                 n_y_active,
                 n_z,
                 n_enc,
                 n_dec,
                 decoder_in_dims,
                 shared_encoder_channels,
                 encoder_type,
                 decoder_type,
                 enc_strides,
                 dec_strides,
                 decoder_output_shapes,
                 feature_size,
                 shared_encoder_conv_shapes=None):
        """Model initalization"""

        super(TAILR, self).__init__()

        self.n_y = n_y
        self.n_z = n_z
        self.n_y_active = n_y_active

        self.shared_encoder = SharedEncoder(in_dims=decoder_in_dims,
                                            n_enc=n_enc,
                                            enc_strides=enc_strides,
                                            encoder_type=encoder_type)
        self.cluster_encoder = ClusterEncoder(n_y, feature_size)
        self.latent_encoder = LatentEncoder(n_y, n_z, feature_size)
        self.latent_prior = LatentPrior(n_y, n_z)
        self.latent_decoder = LatentDecoder(n_z,
                                            output_shape=decoder_output_shapes,
                                            decoder_type=decoder_type,
                                            n_dec=n_dec,
                                            dec_up_strides=dec_strides,
                                            n_x=n_x,
                                            n_y=n_y,
                                            shared_encoder_conv_shapes=shared_encoder_conv_shapes)

    def forward(self, x):
        hiddens = self.shared_encoder(x)
        cluster_score = self.cluster_encoder(hiddens, self.n_y_active)
        latent_sample, latent_mu, latent_sig = self.latent_encoder(
            hiddens, cluster_score)
        out = self.latent_decoder(latent_sample, cluster_score)

        return hiddens, cluster_score, latent_sample, out

    def generate_data_single_cluster(self, cluster_index):
        """Generates data logits from a single cluster

        Args:
            cluster_index: index of the cluster used for generation or Tensor with pre-filed cluster scores, int or Tensor

        Returns: 
            Logits for a Bernoulli distribution to sample the output from

        """
        if isinstance(cluster_index, int):
            if cluster_index >= self.n_y_active:
                raise ValueError(
                    'Cannot generate from a cluster that is unused yet')
            one_hot_cluster_score = torch.Tensor(
                [1e-12 if i != cluster_index else 1.0 for i in range(self.n_y)]).unsqueeze(0).to(device)
        elif isinstance(cluster_index, torch.Tensor):
            one_hot_cluster_score = cluster_index

        latent_prior_sample, latent_prior_mu, latent_prior_sig = self.latent_prior(
            one_hot_cluster_score)
        generated_logits = self.latent_decoder(
            latent_prior_sample, one_hot_cluster_score)

        return torch.distributions.bernoulli.Bernoulli(logits=generated_logits).sample(), torch.distributions.bernoulli.Bernoulli(logits=generated_logits)

    def get_cluster_score(self, x):
        """Computes and returns the cluster score for a given input
            note: you might want to use model.eval() before calling this function

        Args:
            x: input data, Tensor [B, *]

        Returns:
            cluster_score: the cluster score of data, Tensor [B]
        """

        hiddens = self.shared_encoder(x)
        return self.cluster_encoder(hiddens, self.n_y_active)

    def init_new_cluster(self, best_cluster_idx):
        """Initializes a new cluster by copy pasting the weights from the 'best' cluster
            into the new one

        Args:
            best_cluster_idx : cluster used for initializing the weights of the new cluster            
        """
        self.latent_encoder.lins[self.n_y_active].weight = self.latent_encoder.lins[best_cluster_idx].weight
        self.latent_encoder.lins[self.n_y_active].bias = self.latent_encoder.lins[best_cluster_idx].bias

    def _kl_and_z(self, hiddens, y):
        """Returns KL[q(z|y,x) || p(z|y)] and a sample for z from q(z|y,x).

        Returns the analytical KL divergence KL[q(z|y,x) || p(z|y)] if one is
        available (as registered with `kullback_leibler.RegisterKL`), or a sampled
        KL divergence otherwise (in this case the returned sample is the one used
        for the KL divergence).

        Args:
            hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
            y: Categorical cluster random variable, `Tensor` of size `[B, n_y]`.

        Returns:
            Pair `(kl, z)`, where `kl` is the KL divergence (a `Tensor` with shape
            `[B]`, where `B` is the batch size), and `z` is a sample from the encoding
            distribution.
        """
        assert hiddens.dim() == 2
        assert y.dim() == 2

        z, latent_mu, latent_sigma = self.latent_encoder(hiddens, y)
        _, latent_prior_mu, latent_prior_sigma = self.latent_prior(y)

        kl = kl_gaussian(latent_prior_mu, latent_prior_sigma,
                         latent_mu, latent_sigma)

        # Currently not used, will be moved to a testing area
        p = torch.distributions.normal.Normal(
            latent_prior_mu, latent_prior_sigma)
        q = torch.distributions.normal.Normal(
            latent_mu, latent_sigma)
        true_kl = torch.distributions.kl.kl_divergence(q, p)

        return true_kl, z

    def get_generative_snapshot(self, batch_size, gen_buffer_size, cluster_weights):
        """Get generated model data (in place of saving a model snapshot).

        Args:
            batch_size: batch size - pretty self-explanatory eh?, int
            gen_buffer_size: number of batches to generate, int 
            cluster_weights: prior logits over components, np.array

        Returns:
            A tuple of two numpy arrays
            The generated data
            The corresponding labels
        """

        # Sample based on the history of all components used.
        cluster_sample_probs = cluster_weights.astype(float)
        cluster_sample_probs = np.maximum(1e-12, cluster_sample_probs)
        cluster_sample_probs = cluster_sample_probs / \
            np.sum(cluster_sample_probs)

        # Now generate the data based on the specified cluster prior.
        gen_buffer_images = []
        gen_buffer_labels = []
        for _ in range(gen_buffer_size):
            gen_label = np.random.choice(
                np.arange(self.n_y),
                size=(batch_size,),
                replace=True,
                p=cluster_sample_probs)
            y_gen_posterior_vals = torch.zeros(
                (batch_size, self.n_y)).to(device)
            y_gen_posterior_vals[:, gen_label] = 1
            gen_sample, _ = self.generate_data_single_cluster(
                y_gen_posterior_vals)
            gen_sample = gen_sample.detach()
            gen_buffer_images.append(gen_sample)
            gen_buffer_labels.append(torch.tensor(gen_label))

        gen_buffer_images = torch.stack(gen_buffer_images)
        gen_buffer_labels = torch.stack(gen_buffer_labels)

        return gen_buffer_images, gen_buffer_labels


class Solver(nn.Module):
    """Simple CNN classifier"""

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
        return self.layers(x)
