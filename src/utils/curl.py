################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Implementation of Continual Unsupervised Representation Learning model."""

from absl import logging
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

import layers
import utils

#tfc = tf.compat.v1
tfc = tf

# pylint: disable=g-long-lambda
# pylint: disable=redefined-outer-name1


class Curl(object):
    """CURL model class."""

    def __init__(self,
                 prior,
                 latent_decoder,
                 data_decoder,
                 shared_encoder,
                 cluster_encoder,
                 latent_encoder,
                 n_y_active,
                 kly_over_batch=False,
                 is_training=True,
                 name='curl'):
        self.scope_name = name
        self._shared_encoder = shared_encoder
        self._prior = prior
        self._latent_decoder = latent_decoder
        self._data_decoder = data_decoder
        self._cluster_encoder = cluster_encoder
        self._latent_encoder = latent_encoder
        self._n_y_active = n_y_active
        self._kly_over_batch = kly_over_batch
        self._is_training = is_training
        self._cache = {}

    def sample(self, sample_shape=(), y=None, mean=False):
        """Draws a sample from the learnt distribution p(x).

        Args:
          sample_shape: `int` or 0D `Tensor` giving the number of samples to return.
            If  empty tuple (default value), 1 sample will be returned.
          y: Optional, the one hot label on which to condition the sample.
          mean: Boolean, if True the expected value of the output distribution is
            returned, otherwise samples from the output distribution.

        Returns:
          Sample tensor of shape `[B * N, ...]` where `B` is the batch size of
          the prior, `N` is the number of samples requested, and `...` represents
          the shape of the observations.

        Raises:
          ValueError: If both `sample_shape` and `n` are provided.
          ValueError: If `sample_shape` has rank > 0 or if `sample_shape`
          is an int that is < 1.
        """
        with tf.name_scope('{}_sample'.format(self.scope_name)):
            if y is None:
                y = tf.to_float(self.compute_prior().sample(sample_shape))

            if y.shape.ndims > 2:
                y = snt.MergeDims(
                    start=0, size=y.shape.ndims - 1, name='merge_y')(y)

            z = self._latent_decoder(y, is_training=self._is_training)
            if mean:
                samples = self.predict(z.sample(), y).mean()
            else:
                samples = self.predict(z.sample(), y).sample()
        return samples

    def reconstruct(self, x, use_mode=True, use_mean=False):
        """Reconstructs the given observations.

        Args:
          x: Observed `Tensor`.
          use_mode: Boolean, if true, take the argmax over q(y|x)
          use_mean: Boolean, if true, use pixel-mean for reconstructions.

        Returns:
          The reconstructed samples x ~ p(x | y~q(y|x), z~q(z|x, y)).
        """

        hiddens = self._shared_encoder(x, is_training=self._is_training)
        qy = self.infer_cluster(hiddens)
        y_sample = qy.mode() if use_mode else qy.sample()
        y_sample = tf.to_float(y_sample)
        qz = self.infer_latent(hiddens, y_sample)
        p = self.predict(qz.sample(), y_sample)

        if use_mean:
            return p.mean()
        else:
            return p.sample()

    def log_prob(self, x):
        """Redirects to log_prob_elbo with a warning."""
        logging.warn('log_prob is actually a lower bound')
        return self.log_prob_elbo(x)

    def log_prob_elbo(self, x):
        """Returns evidence lower bound."""
        log_p_x, kl_y, kl_z = self.log_prob_elbo_components(x)[:3]
        return log_p_x - kl_y - kl_z

    def log_prob_elbo_components(self, x, y=None, reduce_op=tf.reduce_sum):
        """Returns the components used in calculating the evidence lower bound.

        Args:
          x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
            a flattened input.
          y: Optional labels, `Tensor` of size `[B, I]` where `I` is the size of a
            flattened input.
          reduce_op: The op to use for reducing across non-batch dimensions.
            Typically either `tf.reduce_sum` or `tf.reduce_mean`.

        Returns:
          `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
          `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
          `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
        """
        cache_key = (x,)

        # Checks if the output graph for this inputs has already been computed.
        if cache_key in self._cache:
            return self._cache[cache_key]

        with tf.name_scope('{}_log_prob_elbo'.format(self.scope_name)):

            hiddens = self._shared_encoder(x, is_training=self._is_training)
            # 1) Compute KL[q(y|x) || p(y)] from x, and keep distribution q_y around
            kl_y, q_y = self._kl_and_qy(hiddens)  # [B], distribution

            # For the next two terms, we need to marginalise over all y.

            # First, construct every possible y indexing (as a one hot) and repeat it
            # for every element in the batch [n_y_active, B, n_y].
            # Note that the onehot have dimension of all y, while only the codes
            # corresponding to active components are instantiated
            bs, n_y = q_y.probs.shape
            all_y = tf.tile(
                tf.expand_dims(tf.one_hot(tf.range(self._n_y_active),
                                          n_y), axis=1),
                multiples=[1, tf.shape(x)[0], 1])

            # 2) Compute KL[q(z|x,y) || p(z|y)] (for all possible y), and keep z's
            # around [n_y, B] and [n_y, B, n_z]
            kl_z_all, z_all = tf.map_fn(
                fn=lambda y: self._kl_and_z(hiddens, y),
                elems=all_y,
                dtype=(tf.float32, tf.float32),
                name='elbo_components_z_map')
            kl_z_all = tf.transpose(kl_z_all, name='kl_z_all')

            # Now take the expectation over y (scale by q(y|x))
            y_logits = q_y.logits[:, :self._n_y_active]  # [B, n_y]
            y_probs = q_y.probs[:, :self._n_y_active]  # [B, n_y]
            y_probs = y_probs / tf.reduce_sum(y_probs, axis=1, keepdims=True)
            kl_z = tf.reduce_sum(y_probs * kl_z_all, axis=1)

            # 3) Evaluate logp and recon, i.e., log and mean of p(x|z,[y])
            # (conditioning on y only in the `multi` decoder_type case, when
            # train_supervised is True). Here we take the reconstruction from each
            # possible component y and take its log prob. [n_y, B, Ix, Iy, Iz]
            log_p_x_all = tf.map_fn(
                fn=lambda val: self.predict(val[0], val[1]).log_prob(x),
                elems=(z_all, all_y),
                dtype=tf.float32,
                name='elbo_components_logpx_map')

            # Sum log probs over all dimensions apart from the first two (n_y, B),
            # i.e., over I. Use einsum to construct higher order multiplication.
            log_p_x_all = snt.BatchFlatten(
                preserve_dims=2)(log_p_x_all)  # [n_y,B,I]
            # Note, this is E_{q(y|x)} [ log p(x | z, y)], i.e., we scale log_p_x_all
            # by q(y|x).
            log_p_x = tf.einsum('ij,jik->ik', y_probs, log_p_x_all)  # [B, I]

            # We may also use a supervised loss for some samples [B, n_y]
            if y is not None:
                self.y_label = tf.one_hot(y, n_y)
            else:
                self.y_label = tfc.placeholder(
                    shape=[bs, n_y], dtype=tf.float32, name='y_label')

            # This is computing log p(x | z, y=true_y)], which is basically equivalent
            # to indexing into the correct element of `log_p_x_all`.
            log_p_x_sup = tf.einsum('ij,jik->ik',
                                    self.y_label[:, :self._n_y_active],
                                    log_p_x_all)  # [B, I]
            kl_z_sup = tf.einsum('ij,ij->i',
                                 self.y_label[:, :self._n_y_active],
                                 kl_z_all)  # [B]
            # -log q(y=y_true | x)
            kl_y_sup = tf.nn.sparse_softmax_cross_entropy_with_logits(  # [B]
                labels=tf.argmax(self.y_label[:, :self._n_y_active], axis=1),
                logits=y_logits)

            # Reduce over all dimension except batch.
            dims_x = [k for k in range(1, log_p_x.shape.ndims)]
            log_p_x = reduce_op(log_p_x, dims_x, name='log_p_x')
            log_p_x_sup = reduce_op(log_p_x_sup, dims_x, name='log_p_x_sup')

            # Store values needed externally
            self.q_y = q_y
            self.log_p_x_all = tf.transpose(
                reduce_op(
                    log_p_x_all,
                    -1,  # [B, n_y]
                    name='log_p_x_all'))
            self.kl_z_all = kl_z_all
            self.y_probs = y_probs

        self._cache[cache_key] = (log_p_x, kl_y, kl_z, log_p_x_sup, kl_y_sup,
                                  kl_z_sup)
        return log_p_x, kl_y, kl_z, log_p_x_sup, kl_y_sup, kl_z_sup

    def _kl_and_qy(self, hiddens):
        """Returns analytical or sampled KL div and the distribution q(y | x).

        Args:
          hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.

        Returns:
          Pair `(kl, y)`, where `kl` is the KL divergence (a `Tensor` with shape
          `[B]`, where `B` is the batch size), and `y` is a sample from the
          categorical encoding distribution.
        """
        with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
            q = self.infer_cluster(hiddens)  # q(y|x)
        p = self.compute_prior()  # p(y)
        try:
            # Take the average proportions over whole batch then repeat it in each row
            # before computing the KL
            if self._kly_over_batch:
                probs = tf.reduce_mean(
                    q.probs, axis=0, keepdims=True) * tf.ones_like(q.probs)
                qmean = tfp.distributions.OneHotCategorical(probs=probs)
                kl = tfp.distributions.kl_divergence(qmean, p)
            else:
                kl = tfp.distributions.kl_divergence(q, p)
        except NotImplementedError:
            y = q.sample(name='y_sample')
            logging.warn('Using sampling KLD for y')
            log_p_y = p.log_prob(y, name='log_p_y')
            log_q_y = q.log_prob(y, name='log_q_y')

            # Reduce over all dimension except batch.
            sum_axis_p = [k for k in range(1, log_p_y.get_shape().ndims)]
            log_p_y = tf.reduce_sum(log_p_y, sum_axis_p)
            sum_axis_q = [k for k in range(1, log_q_y.get_shape().ndims)]
            log_q_y = tf.reduce_sum(log_q_y, sum_axis_q)

            kl = log_q_y - log_p_y

        # Reduce over all dimension except batch.
        sum_axis_kl = [k for k in range(1, kl.get_shape().ndims)]
        kl = tf.reduce_sum(kl, sum_axis_kl, name='kl')
        return kl, q

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
        with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
            q = self.infer_latent(hiddens, y)  # q(z|x,y)
        p = self.generate_latent(y)  # p(z|y)
        z = q.sample(name='z')
        try:
            kl = tfp.distributions.kl_divergence(q, p)
        except NotImplementedError:
            logging.warn('Using sampling KLD for z')
            log_p_z = p.log_prob(z, name='log_p_z_y')
            log_q_z = q.log_prob(z, name='log_q_z_xy')

            # Reduce over all dimension except batch.
            sum_axis_p = [k for k in range(1, log_p_z.get_shape().ndims)]
            log_p_z = tf.reduce_sum(log_p_z, sum_axis_p)
            sum_axis_q = [k for k in range(1, log_q_z.get_shape().ndims)]
            log_q_z = tf.reduce_sum(log_q_z, sum_axis_q)

            kl = log_q_z - log_p_z

        # Reduce over all dimension except batch.
        sum_axis_kl = [k for k in range(1, kl.get_shape().ndims)]
        kl = tf.reduce_sum(kl, sum_axis_kl, name='kl')
        return kl, z

    def infer_latent(self, hiddens, y=None, use_mean_y=False):
        """Performs inference over the latent variable z.

        Args:
          hiddens: The shared encoder activations, 4D `Tensor` of size `[B, ...]`.
          y: Categorical cluster variable, `Tensor` of size `[B, ...]`.
          use_mean_y: Boolean, whether to take the mean encoding over all y.

        Returns:
          The distribution `q(z|x, y)`, which on sample produces tensors of size
          `[N, B, ...]` where `B` is the batch size of `x` and `y`, and `N` is the
          number of samples and `...` represents the shape of the latent variables.
        """
        with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
            if y is None:
                y = tf.to_float(self.infer_cluster(hiddens).mode())

        if use_mean_y:
            # If use_mean_y, then y must be probabilities
            all_y = tf.tile(
                tf.expand_dims(tf.one_hot(
                    tf.range(y.shape[1]), y.shape[1]), axis=1),
                multiples=[1, y.shape[0], 1])

            # Compute z KL from x (for all possible y), and keep z's around
            z_all = tf.map_fn(
                fn=lambda y: self._latent_encoder(
                    hiddens, y, is_training=self._is_training).mean(),
                elems=all_y,
                dtype=tf.float32)
            return tf.einsum('ij,jik->ik', y, z_all)
        else:
            return self._latent_encoder(hiddens, y, is_training=self._is_training)

    def generate_latent(self, y):
        """Use the generative model to compute latent variable z, given a y.

        Args:
          y: Categorical cluster variable, `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `p(z|y)`, which on sample produces tensors of size
          `[N, B, ...]` where `B` is the batch size of `x`, and `N` is the number of
          samples asked and `...` represents the shape of the latent variables.
        """
        return self._latent_decoder(y, is_training=self._is_training)

    def get_shared_rep(self, x, is_training):
        """Gets the shared representation from a given input x.

        Args:
          x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
            a flattened input.
          is_training: bool, whether this constitutes training data or not.

        Returns:
          `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
          `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
          `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
        """
        return self._shared_encoder(x, is_training)

    def infer_cluster(self, hiddens):
        """Performs inference over the categorical variable y.

        Args:
          hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `q(y|x)`, which on sample produces tensors of size
          `[N, B, ...]` where `B` is the batch size of `x`, and `N` is the number of
          samples asked and `...` represents the shape of the latent variables.
        """
        with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
            return self._cluster_encoder(hiddens, is_training=self._is_training)

    def predict(self, z, y):
        """Computes prediction over the observed variables.

        Args:
          z: Latent variables, `Tensor` of size `[B, ...]`.
          y: Categorical cluster variable, `Tensor` of size `[B, ...]`.

        Returns:
          The distribution `p(x|z)`, which on sample produces tensors of size
          `[N, B, ...]` where `N` is the number of samples asked.
        """
        encoder_conv_shapes = getattr(
            self._shared_encoder, 'conv_shapes', None)
        return self._data_decoder(
            z,
            y,
            shared_encoder_conv_shapes=encoder_conv_shapes,
            is_training=self._is_training)

    def compute_prior(self):
        """Computes prior over the latent variables.

        Returns:
          The distribution `p(y)`, which on sample produces tensors of size
          `[N, ...]` where `N` is the number of samples asked and `...` represents
          the shape of the latent variables.
        """
        return self._prior()
