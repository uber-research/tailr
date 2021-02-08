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
"""Some common utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from absl import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from tensorboard import program
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def generate_gaussian(logits, sigma_nonlin, sigma_param):
    """Generate a Gaussian distribution given a selected parameterisation."""

    mu, sigma = tf.split(value=logits, num_or_size_splits=2, axis=1)

    if sigma_nonlin == 'exp':
        sigma = tf.exp(sigma)
    elif sigma_nonlin == 'softplus':
        sigma = tf.nn.softplus(sigma)
    else:
        raise ValueError('Unknown sigma_nonlin {}'.format(sigma_nonlin))

    if sigma_param == 'var':
        sigma = tf.sqrt(sigma)
    elif sigma_param != 'std':
        raise ValueError('Unknown sigma_param {}'.format(sigma_param))

    return tfp.distributions.Normal(loc=mu, scale=sigma)


def maybe_center_crop(layer, target_hw):
    """Center crop the layer to match a target shape."""
    l_height, l_width = layer.shape.as_list()[1:3]
    t_height, t_width = target_hw
    assert t_height <= l_height and t_width <= l_width

    if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
        logging.warn(
            'It is impossible to center-crop [%d, %d] into [%d, %d].'
            ' Crop will be uneven.', t_height, t_width, l_height, l_width)

    border = int((l_height - t_height) / 2)
    x_0, x_1 = border, l_height - border
    border = int((l_width - t_width) / 2)
    y_0, y_1 = border, l_width - border
    layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
    return layer_cropped


def launch_tb(output_dir: str) -> None:
    """ Launch TensorBoard 2.0 """

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', output_dir, '--bind_all'])
    url = tb.launch()
    print(f'Tensorboard is available from {url}')


def create_saving_folders(results_root: str, model_dir: str, tb_logdir: str, image_dir: str, csv_logdir: str, experiment_name: str) -> None:
    """Create the folder used to save the model, tb_logs and images in dedicated experiment folders
    """

    # Path definition
    Path(f'{results_root}/{experiment_name}/{model_dir}/CLF').mkdir(parents=True, exist_ok=True)
    Path(f'{results_root}/{experiment_name}/{model_dir}/GEN').mkdir(parents=True, exist_ok=True)

    Path(f'{results_root}/{experiment_name}/{tb_logdir}/CLF').mkdir(parents=True, exist_ok=True)
    Path(f'{results_root}/{experiment_name}/{tb_logdir}/GEN').mkdir(parents=True, exist_ok=True)

    Path(f'{results_root}/{experiment_name}/{image_dir}').mkdir(parents=True, exist_ok=True)

    Path(f'{results_root}/{experiment_name}/{csv_logdir}').mkdir(parents=True, exist_ok=True)


def get_cluster_probs(sess,
                      train_ops,
                      n_poor_batches,
                      poor_data_buffer,
                      x_train_raw,
                      batch_size):
    poor_cluster_probs = []

    for bs in range(n_poor_batches):
        poor_cluster_probs.append(
            sess.run(
                train_ops.cat_probs,
                feed_dict={
                    x_train_raw:
                        poor_data_buffer[bs * batch_size:(bs + 1) *
                                         batch_size]
                }))

    return poor_cluster_probs
