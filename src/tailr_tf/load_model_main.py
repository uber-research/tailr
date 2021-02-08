import os
import click
import yaml
import runpy
import numpy as np

import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow.compat.v1 as tf

import torch

import curl as model
import curl_skeleton as model_skeleton
import training_utils
import utils
import eval_utils
import classifier_utils
import viz_utils

from pathlib import Path

device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)


def load_tailr_and_gen_data(gen_model_load_file, clf_model_load_file, oracle_model_dir, experiment_name, n_y_active):
    """
    Loads a pre-trained TAILR instance and generates images and data from it.

    Args:
        gen_model_load_file: str, file path for the generator model
        clf_model_load_file: str, file path for the classifier model
        oracle_model_dir: str, file pat for the oracle model
        n_y_active: int, number of active components
    """
    results_root = '/results/tailr_tf'
    image_logdir = 'images'
    log_dir = 'log'

    Path(f'{results_root}/{experiment_name}/{image_logdir}').mkdir(parents=True, exist_ok=True)
    Path(f'{results_root}/{experiment_name}/{log_dir}/').mkdir(parents=True, exist_ok=True)

    batch_size = 128
    n_classes = 10
    n_y = 50
    n_z = 32
    n_x = 28 * 28
    output_type = 'bernoulli'
    output_shape = [28, 28, 1]
    encoder_kwargs = {
        'encoder_type': 'multi',
        'n_enc': [1200, 600, 300, 150],
        'enc_strides': [1],
    }
    decoder_kwargs = {
        'decoder_type': 'single',
        'n_dec': [500, 500],
        'dec_up_strides': None,
    }
    prior_size_placeholder = tf.placeholder(tf.float32, shape=[None, None])
    gen_train, gen_eval = training_utils.get_curl_modules(n_x=n_x, n_y=n_y, n_y_active=n_y_active,
                                                          n_z=n_z, output_type=output_type, output_shape=output_shape,
                                                          prior_size_placeholder=prior_size_placeholder,
                                                          encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)
    gen_save_image_count = 40
    gen_buffer_size = 1000
    cumulative_cluster_counts = np.array(
        [1. if i < n_y_active else 0. for i in range(n_y)]).astype(float)
    y_gen = tfp.distributions.OneHotCategorical(
        probs=np.ones((batch_size, n_y)) / n_y,
        dtype=tf.float32,
        name='extra_train_classes').sample()
    gen_samples = gen_train.sample(y=y_gen, mean=True)
    y_gen_image = tfp.distributions.OneHotCategorical(
        probs=np.ones((gen_save_image_count, n_y)) / n_y,
        dtype=tf.float32,
        name='gen_image_cluster').sample()
    gen_images = gen_train.sample(y=y_gen_image, mean=True)

    classifier, _ = classifier_utils.init_classifier_and_optim(
        'dgr_clf', n_classes)
    classifier.load_state_dict(torch.load(
        f'{clf_model_load_file}', map_location=device))
    oracle, _ = classifier_utils.init_classifier_and_optim(
        'dgr_clf', n_classes)
    oracle.load_state_dict(torch.load(
        f'{oracle_model_dir}', map_location=device))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, f'{gen_model_load_file}')

        gen_buffer_images, gen_buffer_labels, gen_buffer_class_labels = training_utils.get_generated_data(
            sess=sess,
            gen_op=gen_samples,
            y_input=y_gen,
            gen_buffer_size=gen_buffer_size,
            component_counts=cumulative_cluster_counts,
            labeling_clf=classifier,
            comparison_clf=oracle,
            log_file_name=f"{results_root}/{experiment_name}/{log_dir}/cluster_class_count_end")

        # Generated images for seen classes (need more than gen_save_image_count data point to be saved)
        save_path = f'{results_root}/{experiment_name}/{image_logdir}'

        viz_utils.save_images_per_cluster(
            sess,
            n_y,
            y_gen_image,
            gen_images,
            gen_save_image_count,
            n_y_active,
            save_path
        )


if __name__ == "__main__":
    experiment_model_dir = ['/results/tailr_tf/17122020/loss_1.7_25000']
    for exp_dir in experiment_model_dir:
        load_tailr_and_gen_data(gen_model_load_file=f'{exp_dir}/models/GEN/generator_end.ckpt', clf_model_load_file=f'{exp_dir}/models/CLF/classifier_end.ckpt',
                            oracle_model_dir=f'./src/tailr_tf/oracle_classifier/oracle_classifier.pth', experiment_name=exp_dir.split('/')[-1], n_y_active=30)
        print(f'Done with {exp_dir}')
