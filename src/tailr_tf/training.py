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
"""Script to train CURL."""

# from memory_profiler import profile
from profilehooks import profile
import collections
import functools
import numpy as np
import pandas
from pathlib import Path
import operator
import os
import sonnet as snt
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow.compat.v1 as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

import curl as model
import curl_skeleton as model_skeleton
import training_utils
import utils
import eval_utils
import classifier_utils
import viz_utils

from absl import logging
from classifier import CNN_Classifier, DGR_CNN
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# tfc = tf.compat.v1
device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)
TRAIN_LOG_INTERVAL = 1
tfc = tf

# pylint: disable=g-long-lambda

# paths for saving data
time_str = time.strftime('%d%m%Y', time.gmtime())
results_root = f'/results/tailr_tf/{time_str}'
tb_logdir = 'tb_logs'
model_dir = 'models'
image_logdir = 'images'
csv_logdir = 'csv_logs'

Path(f'{results_root}').mkdir(parents=True, exist_ok=True)
og_log = open(f'{results_root}/time_profile.log', 'w')
@profile(immediate=True, stdout=og_log)
def run_training(
        dataset,
        training_data_type,
        n_concurrent_classes,
        blend_classes,
        train_supervised,
        n_steps,
        random_seed,
        lr_init,
        lr_factor,
        lr_schedule,
        output_type,
        n_y,
        n_y_active,
        n_z,
        cluster_wait_steps,
        encoder_kwargs,
        decoder_kwargs,
        dynamic_expansion,
        ll_thresh,
        classify_with_samples,
        report_interval,
        knn_values,
        gen_replay_type,
        use_supervised_replay,
        batch_mix,
        experiment_name='toto',
        clf_mode='cluster_init',
        gen_save_image_count=40,
        max_gen_batches=5000,
        classifier_init_period=1,
        clf_thresh=0,
        class_order=[x for x in range(10)],
        class_conditioned=False,
        reverse_data=False,
        save_viz=False,
        need_oracle=False):
    """Run training script.

    Args:
      dataset: str, name of the dataset.
      training_data_type: str, type of training run ('iid' or 'sequential').
      n_concurrent_classes: int, # of classes seen at a time (ignored for 'iid').
      blend_classes: bool, whether to blend in samples from the next class.
      train_supervised: bool, whether to use supervision during training.
      n_steps: int, number of total training steps.
      random_seed: int, seed for tf and numpy RNG.
      lr_init: float, initial learning rate.
      lr_factor: float, learning rate decay factor.
      lr_schedule: float, epochs at which the decay should be applied.
      output_type: str, output distribution (currently only 'bernoulli').
      n_y: int, maximum possible dimensionality of discrete latent variable y.
      n_y_active: int, starting dimensionality of discrete latent variable y.
      n_z: int, dimensionality of continuous latent variable z.
      cluster_wait_steps: int, number of steps to wait between cluster creation
      encoder_kwargs: dict, parameters to specify encoder.
      decoder_kwargs: dict, parameters to specify decoder.
      dynamic_expansion: bool, whether to perform dynamic expansion.
      ll_thresh: float, log-likelihood threshold below which to keep poor samples.
      classify_with_samples: bool, whether to sample latents when classifying.
      report_interval: int, number of steps after which to evaluate and report.
      knn_values: list of ints, k values for different k-NN classifiers to run
      (values of 3, 5, and 10 were used in different parts of the paper).
      gen_replay_type: str, 'fixed', 'dynamic', or None.
      use_supervised_replay: str, whether to use supervised replay (aka 'SMGR').
      batch_mix: boolean, whether to use combined data during training.
      experiment_name: str, name of the experiment.
      clf_mode: str, the classifier initialization mode.
      gen_save_image_count: int, number of images that we want to save
      max_gen_batches: int, number of generated data batches
      classifier_init_period: int, period used for classifier initialization (only used if clf_mode is 'fixed_init' or 'cluster_init')
      clf_thresh: float, threshold for adding data to the poorly classified buffer (only used if clf_mode is 'loss_init')
      class_order: List[int], order used for the classes
      class_conditioned: bool, whether we condition the cluster on the classes
      reverse_data: bool, whether we reverse the order of the task
      save_viz: boolean, whether we want to save visuals of generated data
      need_oracle: boolean, whether we need to get information from the oracle
    """

    try:

        # clf_mode checks
        if clf_mode == 'fixed_init' or clf_mode == 'cluster_init':
            assert classifier_init_period > 0

        # Create saving folders
        utils.create_saving_folders(
            results_root, model_dir, tb_logdir, image_logdir, csv_logdir, experiment_name)

        # Save Argument values
        with open(f'{results_root}/{experiment_name}/argument_log.txt', 'w+') as arg_log_file:
            arg_log_file.write(str(locals()))

        # Set Classifier
        clf_type = 'dgr_clf'

        # Set tf random seed.
        tfc.set_random_seed(random_seed)
        torch.manual_seed(random_seed)
        np.set_printoptions(precision=2, suppress=True)

        # First set up the data source(s) and get dataset info.

        ll_thresh, batch_size, test_batch_size, dataset_kwargs, image_key, class_key = training_utils.get_training_params(
            dataset)

        dataset_ops, n_classes_from_dataset = training_utils.get_data_sources(dataset, dataset_kwargs, batch_size,
                                                                            test_batch_size, training_data_type,
                                                                            n_concurrent_classes, image_key, class_key,
                                                                            class_order=class_order)
        train_data = dataset_ops.train_data
        train_data_for_clf = dataset_ops.train_data_for_clf
        test_data = dataset_ops.test_data
        clf_test_data = dataset_ops.clf_test_data

        output_shape = dataset_ops.ds_info.features[image_key].shape
        n_x = np.prod(output_shape)
        num_train_examples = dataset_ops.ds_info.splits['train'].num_examples

        # Check that the number of classes is compatible with the training scenario
        assert n_classes_from_dataset % n_concurrent_classes == 0
        assert n_steps % (n_classes_from_dataset / n_concurrent_classes) == 0

        # Set specific params depending on the type of gen replay

        gen_every_n = 2  # Blend in a gen replay batch every 2 steps
        gen_refresh_period = 1e8  # Never refresh generated data periodically
        gen_refresh_on_expansion = True  # Refresh on dyn expansion instead

        # Define a global tf variable for the number of active components.
        n_y_active_np = n_y_active
        n_y_active = tfc.get_variable(
            initializer=tf.constant(n_y_active_np, dtype=tf.int32),
            trainable=False,
            name='n_y_active',
            dtype=tf.int32)

        logging.info('Starting CURL script on %s data.', dataset)

        # Set up placeholders for training.
        x_train_raw = tfc.placeholder(
            dtype=tf.float32, shape=(None,) + output_shape, name='x_train_raw')
        cluster_train = tfc.placeholder(
            dtype=tf.int32, shape=(None,), name='cluster_train')
        class_train = tfc.placeholder(
            dtype=tf.int32, shape=(None,), name='class_train')

        def binarize_fn(x):
            """Binarize a Bernoulli by rounding the probabilities.

            Args:
            x: tf tensor, input image.

            Returns:
            A tf tensor with the binarized image
            """
            return tf.cast(tf.greater(x, 0.5 * tf.ones_like(x)), tf.float32)

        if dataset == 'mnist':
            image_channel_size = 1
            x_train = binarize_fn(x_train_raw)
            x_test = binarize_fn(test_data[image_key])
            x_train_for_clf = binarize_fn(train_data_for_clf[image_key])
        elif 'cifar' in dataset or dataset == 'omniglot' or dataset == 'fashion_mnist':
            image_channel_size = 3
            x_train = x_train_raw
            x_test = test_data[image_key]
            x_train_for_clf = train_data_for_clf[image_key]
        else:
            raise ValueError('Unknown dataset {}'.format(dataset))

        cluster_test = test_data[class_key]

        # Set up CURL modules.

        prior_size_placeholder = tf.placeholder(tf.float32, shape=[None, None])
        model_train, model_eval = training_utils.get_curl_modules(n_x=n_x, n_y=n_y, n_y_active=n_y_active,
                                                                n_z=n_z, output_type=output_type, output_shape=output_shape,
                                                                prior_size_placeholder=prior_size_placeholder,
                                                                encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)

        # Set up training graph
        y_train = cluster_train if train_supervised else None
        y_test = cluster_test if train_supervised else None

        train_ops = training_utils.setup_training_and_eval_graphs(
            x_train,
            cluster_train,
            y_train,
            n_y,
            model_train,
            classify_with_samples,
            is_training=True,
            name='train')

        hiddens_for_clf = model_eval.get_shared_rep(x_train_for_clf,
                                                    is_training=False)
        cat_for_clf = model_eval.infer_cluster(hiddens_for_clf)

        # Set up test graph
        test_ops = training_utils.setup_training_and_eval_graphs(
            x_test,
            cluster_test,
            y_test,
            n_y,
            model_eval,
            classify_with_samples,
            is_training=False,
            name='test')

        # Set up Generator optimizer (with scheduler).
        global_step = tf.train.get_or_create_global_step()
        lr_schedule = [
            tf.cast(el * num_train_examples / batch_size, tf.int64)
            for el in lr_schedule
        ]
        num_schedule_steps = tf.reduce_sum(
            tf.cast(global_step >= lr_schedule, tf.float32))
        lr = float(lr_init) * float(lr_factor) ** num_schedule_steps
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_step = optimizer.minimize(train_ops.elbo)
            train_step_supervised = optimizer.minimize(train_ops.elbo_supervised)

        # How many generative batches will we use each period?
        gen_buffer_size = min(
            int(gen_refresh_period / gen_every_n), max_gen_batches)

        # Class each sample should be drawn from (default to uniform prior)
        y_gen = tfp.distributions.OneHotCategorical(
            probs=np.ones((batch_size, n_y)) / n_y,
            dtype=tf.float32,
            name='extra_train_classes').sample()

        gen_samples = model_train.sample(y=y_gen, mean=True)
        if dataset in ['mnist', 'omniglot']:
            gen_samples = binarize_fn(gen_samples)

        # Set up ops for image generation
        if save_viz and gen_save_image_count > 0:
            # Class each sample should be drawn from (default to uniform prior)
            y_gen_image = tfp.distributions.OneHotCategorical(
                probs=np.ones((gen_save_image_count, n_y)) / n_y,
                dtype=tf.float32,
                name='gen_image_cluster').sample()

            gen_images = model_train.sample(y=y_gen_image, mean=True)
            if dataset in ['mnist', 'omniglot']:
                gen_images = binarize_fn(gen_images)

        # Set up ops to dynamically modify parameters (for dynamic expansion)
        dynamic_ops = training_utils.setup_dynamic_ops(n_y)

        logging.info('Created computation graph.')

        n_steps_per_class = n_steps // n_classes_from_dataset  # pylint: disable=invalid-name

        cumulative_cluster_counts = np.array([0] * n_y).astype(float)
        recent_cluster_counts = np.array([0] * n_y).astype(float)

        # Set dynamic expansion parameters
        # Steps to wait after expansion before eligible again
        exp_wait_steps = cluster_wait_steps
        # Steps to wait at start of learning before eligible
        exp_burn_in = cluster_wait_steps

        # Size of the buffer of poorly explained data
        exp_buffer_size = max(100, batch_size)
        num_buffer_gen_train_steps = 20  # Num steps to train generator on buffer

        # Buffer of poorly explained data
        poor_data_count = collections.defaultdict(int)
        poor_data_dict = collections.defaultdict(list)
        has_expanded = False
        steps_since_expansion = 0
        gen_buffer_ind = 0
        eligible_for_expansion = False  # Flag to ensure we wait a bit after expansion

        num_buffer_clf_train_steps = 5  # Num steps to train classifier on buffer
        has_been_reinit = False

        if clf_mode == 'loss_init':
            # Set classifier reinit parameters
            clf_wait_steps = 500  # Steps to wait after initialization before eligible again
            clf_burn_in = 500  # Steps to wait at start of learning before eligible

            # Size of the buffer of poorly classified data
            clf_buffer_size = max(100, batch_size)
            clf_pretraining_n_batches = 10

            # Buffer of poorly classified data (if we are doing cluster-independent classifier init)
            clf_poor_data_buffer = []
            clf_poor_data_class = []
            steps_since_re_init = 0
            eligible_for_init = False

        # Set up basic ops to run and quantities to log.
        ops_to_run = {
            'train_ELBO': train_ops.elbo,
            'train_log_p_x': train_ops.log_p_x,
            'train_kl_y': train_ops.kl_y,
            'train_kl_z': train_ops.kl_z,
            'train_ll': train_ops.ll,
            'train_batch_purity': train_ops.purity,
            'train_probs': train_ops.cat_probs,
            'n_y_active': n_y_active,
            'train_confusion': train_ops.confusion
        }
        test_ops_to_run = {
            'test_ELBO': test_ops.elbo,
            'test_kl_y': test_ops.kl_y,
            'test_kl_z': test_ops.kl_z,
            'test_confusion': test_ops.confusion
        }
        to_log = ['train_batch_purity']
        to_log_eval = ['test_purity', 'test_ELBO', 'test_kl_y', 'test_kl_z']

        if train_supervised:
            # Track supervised losses, train on supervised loss.
            ops_to_run.update({
                'train_ELBO_supervised': train_ops.elbo_supervised,
                'train_log_p_x_supervised': train_ops.log_p_x_supervised,
                'train_kl_y_supervised': train_ops.kl_y_supervised,
                'train_kl_z_supervised': train_ops.kl_z_supervised,
                'train_ll_supervised': train_ops.ll_supervised
            })
            default_train_step = train_step_supervised
            to_log += [
                'train_ELBO_supervised', 'train_log_p_x_supervised',
                'train_kl_y_supervised', 'train_kl_z_supervised'
            ]
        else:
            # Track unsupervised losses, train on unsupervised loss.
            ops_to_run.update({
                'train_ELBO': train_ops.elbo,
                'train_kl_y': train_ops.kl_y,
                'train_kl_z': train_ops.kl_z,
                'train_ll': train_ops.ll
            })
            default_train_step = train_step
            to_log += ['train_ELBO', 'train_kl_y', 'train_kl_z']

        # Initialize Torch Classifier and Optimizer, and get the oracle
        classifier, torch_optim = classifier_utils.init_classifier_and_optim(
            clf_type, n_classes_from_dataset, image_channel_size=image_channel_size)
        if need_oracle:
            oracle, _ = classifier_utils.init_classifier_and_optim(
                clf_type, n_classes_from_dataset)
            oracle.load_state_dict(torch.load(
                './src/tailr_tf/oracle_classifier/oracle_classifier.pth'))
        else:
            oracle = None

        # Launch TB
        utils.launch_tb(tb_logdir)
        saver = tf.train.Saver(max_to_keep=None)
        loss_summary = tf.summary.scalar(
            name='CURL_train_loss', tensor=tf.reshape(train_ops.elbo, []))
        merged_summary = tf.summary.merge_all()
        tb_writer = SummaryWriter(
            f'{results_root}/{experiment_name}/{tb_logdir}/CLF')

        # training data buffers
        gen_train_data_array = None
        real_train_data_array = None

        # Cluster Creation and Clf re-init logging
        cluster_clf_creation_log = [[0, 0, 0]] 

        # Telling TF not to consume the whole GPU memory and leave some for Pytorch (bad kid!)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

        with tf.train.SingularMonitoredSession(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            print('Started TRAINING')
            file_writer = tf.summary.FileWriter(
                f'{results_root}/{experiment_name}/{tb_logdir}/CURL', graph=sess.graph)

            for step in range(1, n_steps+1):
                print(f'\nStep: {step}')
                feed_dict = {}

                # Use the default training loss, but vary it each step depending on the
                # training scenario (eg. for supervised gen replay, we alternate losses)
                ops_to_run['train_step'] = default_train_step

                ### 1) DECIDE WHICH DATA SOURCE TO USE ###
                # TODO Move whole section 1) to a function

                # If we want to combine data, first check if we have access to generated data
                periodic_refresh_started = gen_refresh_period and step >= gen_refresh_period
                refresh_on_expansion_started = gen_refresh_on_expansion and has_expanded

                # Check if we can use generated data and/or combined data
                can_use_gen_data = ((periodic_refresh_started or refresh_on_expansion_started) and
                                    gen_every_n > 0 and step % gen_every_n == 1)
                get_all_data = (batch_mix != 'alternate') and (
                    periodic_refresh_started or refresh_on_expansion_started)

                combine_data = get_all_data and (batch_mix == 'combined')
                semi_combine_data = get_all_data and (batch_mix == 'semi_combined')

                # get the Generated data if needed
                if get_all_data or can_use_gen_data:
                    if gen_train_data_array and gen_buffer_ind:
                        del gen_train_data_array

                    gen_train_data_array = training_utils.get_gen_train_data(
                        gen_buffer_images,
                        gen_buffer_clusters,
                        gen_buffer_classes,
                        gen_buffer_ind,
                        batch_size,
                    )

                    if use_supervised_replay:
                        # Convert cluster to one-hot before feeding in.
                        gen_cluster_onehot = np.eye(
                            n_y)[gen_train_data_array['cluster']]
                        feed_dict.update({model_train.y_label: gen_cluster_onehot})
                        ops_to_run['train_step'] = train_step_supervised

                    gen_buffer_ind = (gen_buffer_ind + 1) % gen_buffer_size

                # get the real data if needed
                if get_all_data or not can_use_gen_data:
                    used_real_data = True

                    real_train_data_array, current_data_period = training_utils.get_real_train_data(
                        sess,
                        training_data_type,
                        step,
                        n_steps_per_class,
                        n_y_active,
                        train_data,
                        batch_size,
                        image_key,
                        class_key,
                        n_concurrent_classes,
                        blend_classes,
                        train_supervised,
                        dynamic_expansion,
                    )

                # COMBINE REAL AND GEN DATA
                if combine_data:
                    final_images = np.concatenate(
                        (real_train_data_array[image_key], gen_train_data_array['image']), axis=0)
                    final_classes = np.concatenate(
                        (real_train_data_array[class_key], gen_train_data_array['class']), axis=0)

                    feed_dict.update({
                        x_train_raw: final_images,
                        class_train: final_classes,
                        # Filling cluster placeholder for the sake of filling, Ideally we would have the cluster labels here for the generated images only
                        cluster_train: final_classes,
                    })

                    # Get data prepared for Torch
                    torch_train_data = gen_train_data_array.copy()
                    torch_train_data[image_key] = np.concatenate(
                        (torch_train_data[image_key], real_train_data_array[image_key]), axis=0)
                    torch_train_data[class_key] = np.concatenate(
                        (torch_train_data['class'], real_train_data_array[class_key]), axis=0)

                # Non-combined data for the generator and combined data for the classifier
                elif semi_combine_data:
                    if can_use_gen_data:
                        final_images = gen_train_data_array['image']
                        final_classes = gen_train_data_array['class']
                    else:
                        final_images = real_train_data_array[image_key]
                        final_classes = real_train_data_array[class_key]

                    feed_dict.update({
                        x_train_raw: final_images,
                        class_train: final_classes,
                        # Filling cluster placeholder for the sake of filling, Ideally we would have the cluster labels here for the generated images only
                        cluster_train: final_classes,
                    })

                    # Get data prepared for Torch
                    torch_train_data = gen_train_data_array.copy()
                    torch_train_data[image_key] = np.concatenate(
                        (torch_train_data[image_key], real_train_data_array[image_key]), axis=0)
                    torch_train_data[class_key] = np.concatenate(
                        (torch_train_data['class'], real_train_data_array[class_key]), axis=0)

                # Or use only the generated data
                elif can_use_gen_data:
                    final_images = gen_train_data_array['image']
                    final_class = gen_train_data_array['class']

                    # Feed it as x_train because it's already reshaped and binarized.
                    feed_dict.update({
                        x_train_raw: final_images,
                        cluster_train: final_class,
                        class_train: final_class,
                    })

                    if use_supervised_replay:
                        # Convert label to one-hot before feeding in.
                        gen_cluster_onehot = np.eye(
                            n_y)[gen_train_data_array['cluster']]
                        feed_dict.update(
                            {model_train.y_label: gen_cluster_onehot})
                        ops_to_run['train_step'] = train_step_supervised

                    # Get data prepared for Torch
                    torch_train_data = gen_train_data_array.copy()
                    torch_train_data[class_key] = torch_train_data['class']

                # Else use only the real data
                else:
                    used_real_data = True
                    final_images = real_train_data_array[image_key]
                    final_classes = real_train_data_array[class_key]

                    feed_dict.update({
                        x_train_raw: final_images,
                        class_train: final_classes,
                        # We are filling this placeholder for the sake of filling
                        cluster_train: final_classes,
                    })

                    # Get data prepared for Torch
                    torch_train_data = real_train_data_array.copy()

                # train input data
                torch_train_input = torch.tensor(np.transpose(
                    torch_train_data[image_key], (0, 3, 1, 2))).to(device)
                torch_train_class = torch.tensor(
                    torch_train_data[class_key]).to(device)

                ### 2) PERFORM A GRADIENT STEP ###
                # Feed a random placeholder for the prior definition
                feed_dict.update({
                    prior_size_placeholder: np.random.rand(
                        final_classes.shape[0], n_y_active_np)
                })

                # Tensoflow Training
                results = sess.run(ops_to_run, feed_dict=feed_dict)

                # Torch training
                torch_out = classifier(torch_train_input)
                torch_loss = F.nll_loss(
                    F.log_softmax(torch_out), torch_train_class, reduction='none')
                mean_torch_loss = torch_loss.mean()
                torch_optim.zero_grad()
                mean_torch_loss.backward()
                torch_optim.step()

                # Print results
                print(f'Current Data Period: {current_data_period}')
                print(f'LOG {results["train_log_p_x"].mean()}')
                print(f'KL_Y {results["train_kl_y"]}')
                print(f'KL_Z {results["train_kl_z"]}')
                print(f'Classifier Loss: {mean_torch_loss.detach().cpu().numpy()}')

                # Tensorboard logs
                tb_writer.add_scalar(
                    'TAILR_clf_train_loss', mean_torch_loss.detach().cpu().numpy(), step)
                tb_writer.flush()
                loss_sum_eval = sess.run(merged_summary, feed_dict=feed_dict)
                file_writer.add_summary(loss_sum_eval, global_step=step)

                del results['train_step']

                ### 3) COMPUTE ADDITIONAL DIAGNOSTIC OPS ON VALIDATION/TEST SETS FOR CLASSIFIER AND GENERATOR ###
                if (step + 1) % report_interval == 0:

                    # Eval Classifier
                    valid_acc_dict, valid_loss_dict = eval_utils.eval_classifier(
                        sess, classifier, clf_test_data, image_key, class_key)
                    tb_writer.add_scalars(
                        'TAILR_clf_validation_acc_', valid_acc_dict, step)
                    tb_writer.add_scalars(
                        'TAILR_clf_validation_loss_', valid_loss_dict, step)

                    # Eval for the generator
                    feed_dict.update({
                        prior_size_placeholder: np.random.rand(
                            test_batch_size, n_y_active_np)
                    })

                    logging.info('Evaluating on test set!')
                    proc_ops = {
                        k: (np.sum if 'confusion' in k
                            else np.mean) for k in test_ops_to_run
                    }
                    results.update(training_utils.process_dataset(dataset_ops.test_iter,
                                                                test_ops_to_run,
                                                                sess,
                                                                feed_dict=feed_dict,
                                                                processing_ops=proc_ops))
                    results['test_purity'] = training_utils.compute_purity(
                        results['test_confusion'])
                    curr_to_log = to_log + to_log_eval
                else:
                    # copy to prevent in-place modifications
                    curr_to_log = list(to_log)

                ### 4) DYNAMIC EXPANSION AND CLASSIFIER INITIALIZATION ###
                if (clf_mode == 'fixed_init' and (step + 1) % classifier_init_period == 0) or \
                    (clf_mode == 'task_init' and (step + 1) % n_steps_per_class == 0):
                    with open(f'{results_root}/{experiment_name}/{model_dir}/CLF/classifier_checkpoint_{step}.ckpt', 'wb') as f:
                        torch.save(classifier.state_dict(), f)
                    del torch_optim
                    has_been_reinit = True
                    old_classifier = classifier
                    classifier, torch_optim = classifier_utils.init_classifier_and_optim(
                        clf_type, n_classes_from_dataset, image_channel_size=image_channel_size)
                    classifier.train()

                elif clf_mode == 'loss_init':
                    eligible_for_init = (steps_since_re_init >=
                                        clf_wait_steps and step >= clf_burn_in)
                    steps_since_re_init += 1

                    if eligible_for_init:
                        has_been_reinit, new_clf, new_optim = classifier_utils.loss_init_classifier(
                            clf_buffer_size,
                            torch_loss,
                            clf_thresh,
                            clf_poor_data_buffer,
                            clf_poor_data_class,
                            torch_train_data,
                            image_key,
                            class_key,
                            batch_size,
                            num_buffer_clf_train_steps,
                            clf_type,
                            n_classes_from_dataset
                        )


                    if eligible_for_init and has_been_reinit:
                        # Save old classifier
                        print('Model Saving')
                        with open(f'{results_root}/{experiment_name}/{model_dir}/CLF/classifier_checkpoint.ckpt', 'wb') as f:
                            torch.save(classifier.state_dict(), f)
                        # Set new classifier and optimizer
                        del torch_optim
                        old_classifier = classifier
                        classifier = new_clf
                        torch_optim = new_optim

                        # Reset the threshold flags so we have a burn in before the next
                        # component.
                        eligible_for_init = False
                        steps_since_re_init = 0

                if dynamic_expansion and used_real_data:
                    # If we're doing dynamic expansion and below max capacity then add
                    # poorly defined data points to a buffer.

                    # First check whether the model is eligible for expansion (the model
                    # becomes ineligible for a fixed time after each expansion, and when
                    # it has hit max capacity).
                    eligible_for_expansion = (
                        steps_since_expansion >= exp_wait_steps and step >= exp_burn_in and n_y_active_np < n_y)

                    steps_since_expansion += 1

                    if eligible_for_expansion:
                        # Add poorly explained data samples to a buffer.
                        poor_inds = results['train_ll'] < ll_thresh
                        for k, v in zip(feed_dict[class_train][poor_inds],
                                        feed_dict[x_train_raw][poor_inds]):
                            poor_data_dict[k].append(v)
                            poor_data_count[k] += 1

                        if class_conditioned:
                            curr_class, len_poor_data = max(
                                poor_data_count.items(), key=operator.itemgetter(1))
                        else:
                            len_poor_data = sum(
                                [x for _, x in poor_data_count.items()])

                        # If buffer is big enough, then add a new component and train just the
                        # new component with several steps of gradient descent.
                        # (We just feed in a onehot cluster vector to indicate which
                        # component).
                        if len_poor_data >= exp_buffer_size:
                            print(f'Cluster Creation')
                            # Save the cluster creation and clf init log
                            cluster_clf_creation_log.append([step, cluster_clf_creation_log[-1][1] + 1, cluster_clf_creation_log[-1][2]])
                            cluster_clf_creation_df = pandas.DataFrame(cluster_clf_creation_log)
                            cluster_clf_creation_df.columns = ['Step', 'CURL version', 'TAILR version']
                            cluster_clf_creation_df.to_csv(f'{results_root}/{experiment_name}/cluster_clf_creation.csv')

                            # Increment cumulative count and reset recent probs count.
                            cumulative_cluster_counts += recent_cluster_counts
                            recent_cluster_counts = np.zeros(n_y)

                            if has_expanded:
                                del gen_buffer_images
                                del gen_buffer_clusters
                                del gen_buffer_classes

                            gen_buffer_images, gen_buffer_clusters, gen_buffer_classes = training_utils.get_generated_data(
                                sess=sess,
                                gen_op=gen_samples,
                                y_input=y_gen,
                                gen_buffer_size=gen_buffer_size,
                                component_counts=cumulative_cluster_counts,
                                labeling_clf=classifier,
                                comparison_clf=oracle,
                                log_file_name=f"{results_root}/{experiment_name}/{csv_logdir}/cluster_{n_y_active_np}_{step}")

                            # Generated images for seen classes (need more than gen_save_image_count data point to be saved)
                            if save_viz:
                                save_path = f'{results_root}/{experiment_name}/{image_logdir}'
                                viz_utils.save_images_per_cluster(
                                    sess,
                                    n_y,
                                    y_gen_image,
                                    gen_images,
                                    gen_save_image_count,
                                    n_y_active_np,
                                    save_path
                                )
                                viz_utils.save_images_per_class(
                                    n_classes_from_dataset,
                                    gen_buffer_images,
                                    gen_buffer_classes,
                                    gen_save_image_count,
                                    n_y_active_np,
                                    save_path
                                )

                            # new cluster index
                            new_cluster = n_y_active_np

                            # Cull to a multiple of batch_size (keep the later data samples).
                            # will be removed in next commit (added for test purposes)
                            class_conditioned = False
                            curr_class = -1
                            if class_conditioned:
                                poor_data_buffer, poor_data_class, poor_data_cluster = \
                                    training_utils.get_poor_data_conditioned(poor_data_dict, batch_size,
                                                                            new_cluster, curr_class)
                            else:
                                poor_data_buffer, poor_data_class, poor_data_cluster = \
                                    training_utils.get_poor_data(poor_data_dict, len_poor_data, batch_size,
                                                                new_cluster)

                            n_poor_batches = len(poor_data_buffer) // batch_size
                            # Find most probable component (on poor batch).
                            poor_cprobs = utils.get_cluster_probs(
                                sess,
                                train_ops,
                                n_poor_batches,
                                poor_data_buffer,
                                x_train_raw,
                                batch_size,
                            )
                            best_cluster = np.argmax(
                                np.sum(np.vstack(poor_cprobs), axis=0))

                            # Initialize parameters of the new component from most prob
                            # existing.
                            training_utils.copy_component_params(best_cluster, new_cluster, sess,
                                                                **dynamic_ops)

                            # Increment mixture component count n_y_active.
                            n_y_active_np += 1
                            n_y_active.load(n_y_active_np, sess)

                            # Save old classifier and snapshot of Generator
                            print('Model Saving')
                            # Ugly but that's how the MonitoredSession works
                            saver.save(
                                sess._sess._sess._sess, f'{results_root}/{experiment_name}/{model_dir}/GEN/generator_cluster{n_y_active_np - 1}.ckpt')
                            with open(f'{results_root}/{experiment_name}/{model_dir}/CLF/classifier_cluster{n_y_active_np - 1}.ckpt', 'wb') as f:
                                torch.save(classifier.state_dict(), f)

                            # Perform a number of steps of gradient descent on the data buffer,
                            # training only the new component (supervised loss).
                            training_utils.train_new_cluster(
                                sess,
                                train_step_supervised,
                                x_train_raw,
                                model_train,
                                n_y,
                                num_buffer_gen_train_steps,
                                n_poor_batches,
                                poor_data_buffer,
                                poor_data_cluster,
                                batch_size,
                            )

                            if clf_mode == 'cluster_init' and n_y_active_np - 1 % classifier_init_period == 0:

                                has_been_reinit = True
                                old_classifier = classifier
                                classifier, torch_optim = classifier_utils.init_classifier_and_optim(
                                    clf_type, n_classes_from_dataset, image_channel_size=image_channel_size)
                                classifier.train()

                            # Empty the buffer.
                            if class_conditioned:
                                del poor_data_dict[curr_class]
                                del poor_data_count[curr_class]
                            else:
                                poor_data_dict = collections.defaultdict(list)
                                poor_data_count = collections.defaultdict(int)
                            eligible_for_expansion = False
                            has_expanded = True
                            steps_since_expansion = 0

                if has_been_reinit:
                    # Increment cumulative count and reset recent probs count.
                    cumulative_cluster_counts += recent_cluster_counts
                    recent_cluster_counts = np.zeros(n_y)

                    # Save the cluster creation and clf init log
                    cluster_clf_creation_log.append([step, cluster_clf_creation_log[-1][1], cluster_clf_creation_log[-1][2] + 1])
                    cluster_clf_creation_df = pandas.DataFrame(cluster_clf_creation_log)
                    cluster_clf_creation_df.columns = ['Step', 'CURL version', 'TAILR version']
                    cluster_clf_creation_df.to_csv(f'{results_root}/{experiment_name}/cluster_clf_creation.csv')

                    # Training the Classifier on the gen data
                    gen_buffer_images, gen_buffer_clusters, gen_buffer_classes = training_utils.get_generated_data(
                        sess=sess,
                        gen_op=gen_samples,
                        y_input=y_gen,
                        gen_buffer_size=gen_buffer_size,
                        component_counts=cumulative_cluster_counts,
                        labeling_clf=old_classifier,
                        comparison_clf=oracle,
                        log_file_name=f"{results_root}/{experiment_name}/{csv_logdir}/cluster_{n_y_active_np}_clf_init_{step}")


                    print('Pre-Training new Classifier on generated data')
                    classifier_utils.classifier_pre_train(
                        num_buffer_clf_train_steps,
                        clf_pretraining_n_batches,
                        gen_buffer_images,
                        gen_buffer_classes,
                        batch_size,
                        classifier,
                        torch_optim,
                    )
                    has_been_reinit = False

                # Accumulate cluster scores.
                if used_real_data:
                    train_cat_probs_vals = results['train_probs']
                    recent_cluster_counts += np.sum(
                        train_cat_probs_vals, axis=0).astype(float)

                ### 5) LOGGING AND EVALUATION ###

                # Periodically perform evaluation
                if (step + 1) % TRAIN_LOG_INTERVAL == 0:
                    def cleanup_for_print(x): return ', {}: %.{}f'.format(
                        x.capitalize().replace('_', ' '), 3)
                    log_str = 'Iteration %d'
                    log_str += ''.join([cleanup_for_print(el)
                                        for el in curr_to_log])
                    log_str += ' n_active: %d'
                    logging.info(
                        log_str,
                        *([(step + 1)] + [results[el] for el in curr_to_log] + [n_y_active_np]))

                # Periodically perform evaluation
                if (step + 1) % report_interval == 0:

                    # Report test purity and related measures
                    logging.info(
                        'Iteration %d, Test purity: %.3f, Test ELBO: %.3f, Test '
                        'KLy: %.3f, Test KLz: %.3f', (step +
                                                    1), results['test_purity'],
                        results['test_ELBO'], results['test_kl_y'], results['test_kl_z'])
                    # Flush data only once in a while to allow buffering of data for more
                    # efficient writes.
                    logging.info('Also training a classifier in latent space')

                file_writer.flush()

            # Last Log
            gen_buffer_images, gen_buffer_clusters, gen_buffer_classes = training_utils.get_generated_data(
                sess=sess,
                gen_op=gen_samples,
                y_input=y_gen,
                gen_buffer_size=gen_buffer_size,
                component_counts=cumulative_cluster_counts,
                labeling_clf=classifier,
                comparison_clf=oracle,
                log_file_name=f"{results_root}/{experiment_name}/cluster_class_count_last")

            if save_viz:
                save_path = f'{results_root}/{experiment_name}/{image_logdir}'
                viz_utils.save_images_per_cluster(
                    sess,
                    n_y,
                    y_gen_image,
                    gen_images,
                    gen_save_image_count,
                    n_y_active_np,
                    save_path
                )
                viz_utils.save_images_per_class(
                    n_classes_from_dataset,
                    gen_buffer_images,
                    gen_buffer_classes,
                    gen_save_image_count,
                    n_y_active_np,
                    save_path
                )

            # Save the cluster creation and clf init log
            cluster_clf_creation_df = pandas.DataFrame(cluster_clf_creation_log)
            cluster_clf_creation_df.columns = ['Step', 'CURL version', 'TAILR version']
            cluster_clf_creation_df.to_csv(f'{results_root}/{experiment_name}/cluster_clf_creation.csv')

            # Ugly but that's how the MonitoredSession works
            saver.save(sess._sess._sess._sess,
                    f'{results_root}/{experiment_name}/{model_dir}/GEN/generator_end.ckpt')
            with open(f'{results_root}/{experiment_name}/{model_dir}/CLF/classifier_end.ckpt', 'wb') as f:
                torch.save(classifier.state_dict(), f)

            print('End Saving')

    except Exception:
        with open(f'{results_root}/{experiment_name}/exception_log.txt', 'w+') as excp_log:
            traceback.print_exc()
            excp_log.write(traceback.format_exc())

    with open(f'/results/DONE', 'w+') as done_file:
        done_file.write(
            '-What is my purpose? -You terminate the run. -Oh my god...')

def move_time_log(experiment_name):
    """Moves the time profiling results to an experiment specific file
    """
    # Note: This is ugly but it works. Might change it later. 
    global og_log
    og_log.close()
    with open(f'{results_root}/time_profile.log', 'r') as log:
        with open(f'{results_root}/{experiment_name}/time_profile.log', 'a+') as dest_log:
            for line in log: 
                dest_log.write(line)