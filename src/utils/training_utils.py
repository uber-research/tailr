import collections
import functools
import os
import pandas

from absl import logging
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import sonnet as snt
import plotly.graph_objects as go
import plotly.express as px

import curl as model
import curl_skeleton
import utils

from classifier import CNN_Classifier, DGR_CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

MainOps = collections.namedtuple('MainOps', [
    'elbo', 'll', 'log_p_x', 'kl_y', 'kl_z', 'elbo_supervised', 'll_supervised',
    'log_p_x_supervised', 'kl_y_supervised', 'kl_z_supervised',
    'cat_probs', 'confusion', 'purity', 'latents'
])

DatasetTuple = collections.namedtuple('DatasetTuple', [
    'train_data', 'train_iter_for_clf', 'train_data_for_clf',
    'valid_iter', 'valid_data', 'test_iter', 'test_data', 'clf_test_data', 'ds_info'
])


device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)
TRAIN_LOG_INTERVAL = 1
tfc = tf


def compute_purity(confusion):
    return np.sum(np.max(confusion, axis=0)).astype(float) / np.sum(confusion)


def process_dataset(iterator,
                    ops_to_run,
                    sess,
                    feed_dict=None,
                    aggregation_ops=np.stack,
                    processing_ops=None):
    """Process a dataset by computing ops and accumulating batch by batch.

    Args:
      iterator: iterator through the dataset.
      ops_to_run: dict, tf ops to run as part of dataset processing.
      sess: tf.Session to use.
      feed_dict: dict, required placeholders.
      aggregation_ops: fn or dict of fns, aggregation op to apply for each op.
      processing_ops: fn or dict of fns, extra processing op to apply for each op.

    Returns:
      Results accumulated over dataset.
    """

    if not isinstance(ops_to_run, dict):
        raise TypeError('ops_to_run must be specified as a dict')

    if not isinstance(aggregation_ops, dict):
        aggregation_ops = {k: aggregation_ops for k in ops_to_run}
    if not isinstance(processing_ops, dict):
        processing_ops = {k: processing_ops for k in ops_to_run}

    out_results = collections.OrderedDict()
    sess.run(iterator.initializer)
    while True:
        # Iterate over the whole dataset and append the results to a per-key list.
        try:
            outs = sess.run(ops_to_run, feed_dict=feed_dict)
            for key, value in outs.items():
                out_results.setdefault(key, []).append(value)

        except tf.errors.OutOfRangeError:  # end of dataset iterator
            break

    # Aggregate and process results.
    for key, value in out_results.items():
        if aggregation_ops[key]:
            out_results[key] = aggregation_ops[key](value)
        if processing_ops[key]:
            out_results[key] = processing_ops[key](out_results[key], axis=0)

    return out_results


def get_data_sources(dataset, dataset_kwargs, batch_size, test_batch_size,
                     training_data_type, n_concurrent_classes, image_key,
                     label_key, class_order=[x for x in range(10)]):
    """Create and return data sources for training, validation, and testing.

    Args:
      dataset: str, name of dataset ('mnist', 'omniglot', etc).
      dataset_kwargs: dict, kwargs used in tf dataset constructors.
      batch_size: int, batch size used for training.
      test_batch_size: int, batch size used for evaluation.
      training_data_type: str, how training data is seen ('iid', or 'sequential').
      n_concurrent_classes: int, # classes seen at a time (ignored for 'iid').
      image_key: str, name if image key in dataset.
      label_key: str, name of label key in dataset.

    Returns:
      A namedtuple containing all of the dataset iterators and batches.

    """

    # Load training data sources
    ds_train, ds_info = tfds.load(
        name=dataset,
        split=tfds.Split.TRAIN,
        with_info=True,
        as_dataset_kwargs={'shuffle_files': False},
        **dataset_kwargs)

    # Validate assumption that data is in [0, 255]
    assert ds_info.features[image_key].dtype == tf.uint8

    n_classes = len(class_order)
    num_train_examples = ds_info.splits['train'].num_examples

    def preprocess_data(x):
        """Convert images from uint8 in [0, 255] to float in [0, 1]."""
        x[image_key] = tf.image.convert_image_dtype(x[image_key], tf.float32)
        return x

    # This is were we should be partitionning the data and get the "continual learning" part.
    if training_data_type == 'sequential':
        c = None  # The index of the class number, None for now and updated later
        if n_concurrent_classes == 1:
            def filter_fn(v): return tf.equal(v[label_key], c)
        else:
            # Define the lowest and highest class number at each data period.
            assert n_classes % n_concurrent_classes == 0, (
                'Number of total classes must be divisible by '
                'number of concurrent classes')
            cmin = []
            cmax = []
            for i in range(int(n_classes / n_concurrent_classes)):
                for _ in range(n_concurrent_classes):
                    cmin.append(i * n_concurrent_classes)
                    cmax.append((i + 1) * n_concurrent_classes)

            # Filter for getting only classes of interest
            def filter_fn(v): return tf.logical_and(
                tf.greater_equal(v[label_key], cmin[c]), tf.less(
                    v[label_key], cmax[c]))

        # Set up data sources/queues (one for each class).
        train_datasets = []
        train_iterators = []
        train_data = []

        full_ds = ds_train.repeat().shuffle(num_train_examples, seed=0)
        full_ds = ds_train.repeat()
        full_ds = full_ds.map(preprocess_data)
        for c in class_order:
            filtered_ds = full_ds.filter(filter_fn).batch(
                batch_size, drop_remainder=True)
            train_datasets.append(filtered_ds)
            train_iterators.append(train_datasets[-1].make_one_shot_iterator())
            train_data.append(train_iterators[-1].get_next())

    else:  # not sequential
        full_ds = ds_train.repeat().shuffle(num_train_examples, seed=0)
        full_ds = full_ds.map(preprocess_data)
        train_datasets = full_ds.batch(batch_size, drop_remainder=True)
        train_data = train_datasets.make_one_shot_iterator().get_next()

    # Set up data source to get full training set for classifier training
    full_ds = ds_train.repeat(1).shuffle(num_train_examples, seed=0)
    full_ds = full_ds.map(preprocess_data)
    train_datasets_for_classifier = full_ds.batch(
        test_batch_size, drop_remainder=True)
    train_iter_for_classifier = (
        train_datasets_for_classifier.make_initializable_iterator())
    train_data_for_classifier = train_iter_for_classifier.get_next()

    # Load validation dataset.
    try:
        valid_dataset = tfds.load(
            name=dataset, split=tfds.Split.VALIDATION, **dataset_kwargs)
        num_valid_examples = ds_info.splits[tfds.Split.VALIDATION].num_examples
        assert (num_valid_examples %
                test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                        num_valid_examples)
        valid_dataset = valid_dataset.repeat(1).batch(
            test_batch_size, drop_remainder=True)
        valid_dataset = valid_dataset.map(preprocess_data)
        valid_iter = valid_dataset.make_initializable_iterator()
        valid_data = valid_iter.get_next()
    except (KeyError, ValueError):
        logging.warning('No validation set!!')
        valid_iter = None
        valid_data = None

    # Load test dataset.
    test_dataset = tfds.load(
        name=dataset, split=tfds.Split.TEST, **dataset_kwargs)
    num_test_examples = ds_info.splits['test'].num_examples
    assert (num_test_examples %
            test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
                                    num_test_examples)
    test_dataset = test_dataset.repeat(1).batch(
        test_batch_size, drop_remainder=True)
    test_dataset = test_dataset.map(preprocess_data)
    test_iter = test_dataset.make_initializable_iterator()
    test_data = test_iter.get_next()

    # Second test set for the classifier
    if training_data_type == 'sequential':
        c = None  # The index of the class number, None for now and updated later
        if n_concurrent_classes == 1:
            def filter_fn(v): return tf.equal(v[label_key], c)
        else:
            # Define the lowest and highest class number at each data period.
            assert n_classes % n_concurrent_classes == 0, (
                'Number of total classes must be divisible by '
                'number of concurrent classes')
            cmin = []
            cmax = []
            for i in range(int(n_classes / n_concurrent_classes)):
                for _ in range(n_concurrent_classes):
                    cmin.append(i * n_concurrent_classes)
                    cmax.append((i + 1) * n_concurrent_classes)

            # Filter for getting only classes of interest
            def filter_fn(v): return tf.logical_and(
                tf.greater_equal(v[label_key], cmin[c]), tf.less(
                    v[label_key], cmax[c]))

        # Set up data sources/queues (one for each class).
        clf_test_datasets = []
        clf_test_iterators = []
        clf_test_data = []

        test_dataset = tfds.load(
            name=dataset, split=tfds.Split.TEST, **dataset_kwargs)
        full_ds = test_dataset.repeat().shuffle(num_test_examples, seed=0)
        full_ds = test_dataset.repeat()
        full_ds = full_ds.map(preprocess_data)
        for c in class_order:
            filtered_ds = full_ds.filter(filter_fn).batch(
                batch_size, drop_remainder=True)
            clf_test_datasets.append(filtered_ds)
            clf_test_iterators.append(
                clf_test_datasets[-1].make_one_shot_iterator())
            clf_test_data.append(clf_test_iterators[-1].get_next())
    logging.info('Loaded %s data', dataset)

    return DatasetTuple(train_data, train_iter_for_classifier,
                        train_data_for_classifier, valid_iter, valid_data,
                        test_iter, test_data, clf_test_data, ds_info), n_classes


def setup_training_and_eval_graphs(x, cluster_label, y, n_y, curl_model,
                                   classify_with_samples, is_training, name):
    """Set up the graph and return ops for training or evaluation.

    Args:
      x: tf placeholder for image.
      cluster_label: tf placeholder for ground truth cluster_label.
      y: tf placeholder for some self-supervised cluster_label/prediction.
      n_y: int, dimensionality of discrete latent variable y.
      curl_model: snt.AbstractModule representing the CURL model.
      classify_with_samples: bool, whether to *sample* latents for classification.
      is_training: bool, whether this graph is the training graph.
      name: str, graph name.

    Returns:
      A namedtuple with the required graph ops to perform training or evaluation.

    """
    # kl_y_supervised is -log q(y=y_true | x)
    (log_p_x, kl_y, kl_z, log_p_x_supervised, kl_y_supervised,
     kl_z_supervised) = curl_model.log_prob_elbo_components(x, y)

    ll = log_p_x - kl_y - kl_z
    elbo = -tf.reduce_mean(ll)

    # Supervised loss, either for SMGR, or adaptation to supervised benchmark.
    ll_supervised = log_p_x_supervised - kl_y_supervised - kl_z_supervised
    elbo_supervised = -tf.reduce_mean(ll_supervised)

    # Summaries
    kl_y = tf.reduce_mean(kl_y)
    kl_z = tf.reduce_mean(kl_z)
    log_p_x_supervised = tf.reduce_mean(log_p_x_supervised)
    kl_y_supervised = tf.reduce_mean(kl_y_supervised)
    kl_z_supervised = tf.reduce_mean(kl_z_supervised)

    # Evaluation.
    hiddens = curl_model.get_shared_rep(x, is_training=is_training)
    cat = curl_model.infer_cluster(hiddens)
    cat_probs = cat.probs

    # Not really a confusion matrix, more like a Class/Cluster relation matrix
    confusion = tf.confusion_matrix(cluster_label, tf.argmax(cat_probs, axis=1),
                                    num_classes=n_y, name=name + '_confusion')
    purity = (tf.reduce_sum(tf.reduce_max(confusion, axis=0))
              / tf.reduce_sum(confusion))

    if classify_with_samples:
        latents = curl_model.infer_latent(
            hiddens=hiddens, y=tf.to_float(cat.sample())).sample()
    else:
        latents = curl_model.infer_latent(
            hiddens=hiddens, y=tf.to_float(cat.mode())).mean()

    return MainOps(elbo, ll, log_p_x, kl_y, kl_z, elbo_supervised, ll_supervised,
                   log_p_x_supervised, kl_y_supervised, kl_z_supervised,
                   cat_probs, confusion, purity, latents)


def get_generated_data(sess, gen_op, y_input, gen_buffer_size,
                       component_counts, labeling_clf, comparison_clf=None,
                       log_file_name=None, gen_latent_op=None, knn_model=None):
    """Get generated model data (in place of saving a model snapshot).

    Args:
      sess: tf.Session.
      gen_op: tf op representing a batch of generated data.
      y_input: tf placeholder for which mixture components to generate from.
      gen_buffer_size: int, number of data points to generate.
      component_counts: np.array, prior probabilities over components.
      labeling_clf: torch.nn.Module, classifier used for class labeling
      comparison_clf: torch.nn.Module, classifier used for comparison
      log_file_name: str, file name for the log csv file without the extension

    Returns:
      A tuple of two numpy arrays
        The generated data
        The corresponding labels
    """
    batch_size, n_y = y_input.shape.as_list()

    # Sample based on the history of all components used.
    cluster_sample_probs = component_counts.astype(float)
    cluster_sample_probs = np.maximum(1e-12, cluster_sample_probs)
    cluster_sample_probs = cluster_sample_probs / np.sum(cluster_sample_probs)

    # Now generate the data based on the specified cluster prior.
    gen_buffer_latent = []
    gen_buffer_images = []
    gen_buffer_labels = []
    gen_buffer_class_labels = []
    gen_buffer_knn_labels = []
    gen_buffer_class_scores = []
    if comparison_clf is not None:
        gen_buffer_comparison_class_labels = []
        gen_buffer_comparison_class_scores = []

    for i in range(gen_buffer_size):
        # sample cluster labels
        gen_label = np.random.choice(
            np.arange(n_y),
            size=(batch_size,),
            replace=True,
            p=cluster_sample_probs)
        y_gen_posterior_vals = np.zeros((batch_size, n_y))
        y_gen_posterior_vals[np.arange(batch_size), gen_label] = 1
        gen_buffer_labels.append(gen_label)

        if knn_model is not None:
            # Get latent
            gen_latent = sess.run(gen_latent_op, feed_dict={
                                  y_input: y_gen_posterior_vals})
            gen_buffer_latent.append(gen_latent)

        # Get Image
        gen_image = sess.run(gen_op, feed_dict={y_input: y_gen_posterior_vals})
        gen_buffer_images.append(gen_image)

        # Get Predicted class
        if labeling_clf is not None:
            clf_output = labeling_clf(torch.tensor(
                np.transpose(gen_image, (0, 3, 1, 2))).to(device))
            gen_softmax_score, gen_class_label = torch.max(
                F.softmax(clf_output), dim=1)
            gen_softmax_score = gen_softmax_score.detach().detach().cpu().numpy()
            gen_class_label = gen_class_label.detach().detach().cpu().numpy()
            gen_buffer_class_labels.append(gen_class_label)
            gen_buffer_class_scores.append(gen_softmax_score)

        if comparison_clf is not None:
            clf_output = comparison_clf(torch.tensor(
                np.transpose(gen_image, (0, 3, 1, 2))).to(device))
            gen_comparison_softmax_score, gen_comparison_class_label = torch.max(
                F.softmax(clf_output), dim=1)
            gen_comparison_softmax_score = gen_comparison_softmax_score.detach().cpu().numpy()
            gen_comparison_class_label = gen_comparison_class_label.detach().cpu().numpy()
            gen_buffer_comparison_class_labels.append(
                gen_comparison_class_label)
            gen_buffer_comparison_class_scores.append(
                gen_comparison_softmax_score
            )

        if knn_model is not None:
            knn_pred = knn_model.predict(gen_latent)
            gen_buffer_knn_labels.append(knn_pred)

    gen_buffer_images = np.vstack(gen_buffer_images)
    gen_buffer_labels = np.concatenate(gen_buffer_labels)
    if labeling_clf is not None:
        gen_buffer_class_labels = np.concatenate(gen_buffer_class_labels)
        gen_buffer_class_scores = np.concatenate(gen_buffer_class_scores)
    if comparison_clf is not None:
        gen_buffer_comparison_class_labels = np.concatenate(
            gen_buffer_comparison_class_labels)
        gen_buffer_comparison_class_scores = np.concatenate(
            gen_buffer_comparison_class_scores
        )

    if knn_model is not None:
        gen_buffer_knn_labels = np.concatenate(gen_buffer_knn_labels)

    if log_file_name is not None:
        save_csv(log_file_name, gen_buffer_labels, gen_buffer_class_labels,
                 gen_buffer_class_scores)

        cluster_sample_probs_df = pandas.DataFrame(cluster_sample_probs)
        cluster_sample_probs_df.to_csv(f'{log_file_name}_cluster_score.csv')

        if comparison_clf is not None:
            save_csv(log_file_name, gen_buffer_labels, gen_buffer_comparison_class_labels,
                     gen_buffer_comparison_class_scores, is_comparison=True)

        if knn_model is not None:
            knn_class_count = collections.Counter(gen_buffer_knn_labels)
            knn_df = pandas.DataFrame.from_dict(
                knn_class_count, orient='index')
            knn_df.to_csv(f'{log_file_name}_knn.csv')

    return gen_buffer_images, gen_buffer_labels, gen_buffer_class_labels


def save_csv(log_file_name, gen_buffer_labels, gen_buffer_class_labels,
             gen_buffer_class_scores, is_comparison=False, top_score=.9, bot_score=.7):
    """Saves the given data into csv files"""

    cluster_class_count = collections.defaultdict(
        lambda: collections.defaultdict(int))
    cluster_class_avg_score = collections.defaultdict(
        lambda: collections.defaultdict(int))

    cluster_class_top_score_count = collections.defaultdict(
        lambda: collections.defaultdict(int))
    cluster_class_mid_score_count = collections.defaultdict(
        lambda: collections.defaultdict(int))
    cluster_class_bot_score_count = collections.defaultdict(
        lambda: collections.defaultdict(int))

    cluster_class_score_summary = collections.defaultdict(
        lambda: collections.defaultdict(str))

    clusters = set()
    classes = set()

    comp_suffix = '_comp' if is_comparison else ''

    for i, g_l in enumerate(gen_buffer_labels):
        clusters.add(g_l)
        classes.add(gen_buffer_class_labels[i])
        cluster_class_count[g_l][gen_buffer_class_labels[i]] += 1
        cluster_class_avg_score[g_l][gen_buffer_class_labels[i]
                                     ] += gen_buffer_class_scores[i]
        if gen_buffer_class_scores[i] > top_score:
            cluster_class_top_score_count[g_l][gen_buffer_class_labels[i]] += 1
        elif gen_buffer_class_scores[i] < bot_score:
            cluster_class_bot_score_count[g_l][gen_buffer_class_labels[i]] += 1
        else:
            cluster_class_mid_score_count[g_l][gen_buffer_class_labels[i]] += 1

    for clust in clusters:
        for cla in classes:
            if cluster_class_count[clust][cla] > 0:
                cluster_class_avg_score[clust][cla] = cluster_class_avg_score[clust][cla] / \
                    cluster_class_count[clust][cla]
                cluster_class_top_score_count[clust][cla] = cluster_class_top_score_count[clust][cla] / \
                    cluster_class_count[clust][cla]
                cluster_class_mid_score_count[clust][cla] = cluster_class_mid_score_count[clust][cla] / \
                    cluster_class_count[clust][cla]
                cluster_class_bot_score_count[clust][cla] = cluster_class_bot_score_count[clust][cla] / \
                    cluster_class_count[clust][cla]
                cluster_class_score_summary[clust][cla] = f'{cluster_class_top_score_count[clust][cla]:.2f}| \
                    {cluster_class_mid_score_count[clust][cla]:.2f}|{cluster_class_bot_score_count[clust][cla]:.2f}'

    clusters = sorted(list(clusters))
    count_df = pandas.DataFrame(cluster_class_count)
    count_df = count_df[clusters].sort_index()
    count_df = count_df.fillna(0).astype(int)
    count_df.to_csv(f'{log_file_name}{comp_suffix}.csv')

    class_count = count_df.sum(axis=1, skipna=True)
    class_count.to_csv(f'{log_file_name}_total_class{comp_suffix}.csv')

    score_df = pandas.DataFrame(cluster_class_avg_score)
    score_df = score_df[clusters].sort_index()
    score_df = score_df.fillna(0).round(4)
    score_df.to_csv(f'{log_file_name}_score{comp_suffix}.csv')

    score_summary_df = pandas.DataFrame(cluster_class_score_summary)
    score_summary_df = score_summary_df[clusters].sort_index()
    score_summary_df.to_csv(f'{log_file_name}_summary{comp_suffix}.csv')


def setup_dynamic_ops(n_y):
    """Set up ops to move / copy mixture component weights for dynamic expansion.

    Args:
      n_y: int, dimensionality of discrete latent variable y.

    Returns:
      A dict containing all of the ops required for dynamic updating.

    """
    # Set up graph ops to dynamically modify component params.
    graph = tf.get_default_graph()

    # 1) Ops to get and set latent encoder params (entire tensors)
    latent_enc_tensors = {}
    for k in range(n_y):
        latent_enc_tensors['latent_w_' + str(k)] = graph.get_tensor_by_name(
            'latent_encoder/mlp_latent_encoder_{}/w:0'.format(k))
        latent_enc_tensors['latent_b_' + str(k)] = graph.get_tensor_by_name(
            'latent_encoder/mlp_latent_encoder_{}/b:0'.format(k))

    latent_enc_assign_ops = {}
    latent_enc_phs = {}
    for key, tensor in latent_enc_tensors.items():
        latent_enc_phs[key] = tfc.placeholder(
            tensor.dtype, tensor.shape, name='latent_enc_phs')
        latent_enc_assign_ops[key] = tf.assign(tensor, latent_enc_phs[key])

    # 2) Ops to get and set cluster encoder params (columns of a tensor)
    # We will be copying column ind_from to column ind_to.
    cluster_w = graph.get_tensor_by_name(
        'cluster_encoder/mlp_cluster_encoder_final/w:0')
    cluster_b = graph.get_tensor_by_name(
        'cluster_encoder/mlp_cluster_encoder_final/b:0')

    ind_from = tfc.placeholder(dtype=tf.int32, name='ind_from')
    ind_to = tfc.placeholder(dtype=tf.int32, name='inf_to')

    # Determine indices of cluster encoder weights and biases to be updated
    w_indices = tf.transpose(
        tf.stack([
            tf.range(cluster_w.shape[0], dtype=tf.int32),
            ind_to * tf.ones(shape=(cluster_w.shape[0],), dtype=tf.int32)
        ]))
    b_indices = ind_to
    # Determine updates themselves
    cluster_w_updates = tf.squeeze(
        tf.slice(cluster_w, begin=(0, ind_from), size=(cluster_w.shape[0], 1)))
    cluster_b_updates = cluster_b[ind_from]
    # Create update ops
    cluster_w_update_op = tf.scatter_nd_update(cluster_w, w_indices,
                                               cluster_w_updates)
    cluster_b_update_op = tf.scatter_update(cluster_b, b_indices,
                                            cluster_b_updates)

    # 3) Ops to get and set latent prior params (columns of a tensor)
    # We will be copying column ind_from to column ind_to.
    latent_prior_mu_w = graph.get_tensor_by_name(
        'latent_decoder/latent_prior_mu/w:0')
    latent_prior_sigma_w = graph.get_tensor_by_name(
        'latent_decoder/latent_prior_sigma/w:0')

    mu_indices = tf.transpose(
        tf.stack([
            ind_to *
            tf.ones(shape=(latent_prior_mu_w.shape[1],), dtype=tf.int32),
            tf.range(latent_prior_mu_w.shape[1], dtype=tf.int32)
        ]))
    mu_updates = tf.squeeze(
        tf.slice(
            latent_prior_mu_w,
            begin=(ind_from, 0),
            size=(1, latent_prior_mu_w.shape[1])))
    mu_update_op = tf.scatter_nd_update(
        latent_prior_mu_w, mu_indices, mu_updates)
    sigma_indices = tf.transpose(
        tf.stack([
            ind_to *
            tf.ones(shape=(latent_prior_sigma_w.shape[1],), dtype=tf.int32),
            tf.range(latent_prior_sigma_w.shape[1], dtype=tf.int32)
        ]))
    sigma_updates = tf.squeeze(
        tf.slice(
            latent_prior_sigma_w,
            begin=(ind_from, 0),
            size=(1, latent_prior_sigma_w.shape[1])))
    sigma_update_op = tf.scatter_nd_update(latent_prior_sigma_w, sigma_indices,
                                           sigma_updates)

    dynamic_ops = {
        'ind_from_ph': ind_from,
        'ind_to_ph': ind_to,
        'latent_enc_tensors': latent_enc_tensors,
        'latent_enc_assign_ops': latent_enc_assign_ops,
        'latent_enc_phs': latent_enc_phs,
        'cluster_w_update_op': cluster_w_update_op,
        'cluster_b_update_op': cluster_b_update_op,
        'mu_update_op': mu_update_op,
        'sigma_update_op': sigma_update_op
    }

    return dynamic_ops


def copy_component_params(ind_from, ind_to, sess, ind_from_ph, ind_to_ph,
                          latent_enc_tensors, latent_enc_assign_ops,
                          latent_enc_phs,
                          cluster_w_update_op, cluster_b_update_op,
                          mu_update_op, sigma_update_op):
    """Copy parameters from component i to component j.

    Args:
      ind_from: int, component index to copy from.
      ind_to: int, component index to copy to.
      sess: tf.Session.
      ind_from_ph: tf placeholder for component to copy from.
      ind_to_ph: tf placeholder for component to copy to.
      latent_enc_tensors: dict, tensors in the latent posterior encoder.
      latent_enc_assign_ops: dict, assignment ops for latent posterior encoder.
      latent_enc_phs: dict, placeholders for assignment ops.
      cluster_w_update_op: op for updating weights of cluster encoder.
      cluster_b_update_op: op for updating biased of cluster encoder.
      mu_update_op: op for updating mu weights of latent prior.
      sigma_update_op: op for updating sigma weights of latent prior.

    """
    update_ops = []
    feed_dict = {}
    # Copy for latent encoder.
    new_w_val, new_b_val = sess.run([
        latent_enc_tensors['latent_w_' + str(ind_from)],
        latent_enc_tensors['latent_b_' + str(ind_from)]
    ])
    update_ops.extend([
        latent_enc_assign_ops['latent_w_' + str(ind_to)],
        latent_enc_assign_ops['latent_b_' + str(ind_to)]
    ])
    feed_dict.update({
        latent_enc_phs['latent_w_' + str(ind_to)]: new_w_val,
        latent_enc_phs['latent_b_' + str(ind_to)]: new_b_val
    })

    # Copy for cluster encoder softmax.
    update_ops.extend([cluster_w_update_op, cluster_b_update_op])
    feed_dict.update({ind_from_ph: ind_from, ind_to_ph: ind_to})

    # Copy for latent prior.
    update_ops.extend([mu_update_op, sigma_update_op])
    feed_dict.update({ind_from_ph: ind_from, ind_to_ph: ind_to})
    sess.run(update_ops, feed_dict)


def construct_prior_probs(size_place_holder, n_y, n_y_active):
    """Construct the uniform prior probabilities.

    Args:
      batch_size: int, the size of the batch.
      n_y: int, the number of categorical cluster components.
      n_y_active: tf.Variable, the number of components that are currently in use.

    Returns:
      Tensor representing the prior probability matrix, size of [batch_size, n_y].
    """
    probs = tf.ones_like(size_place_holder) / tf.cast(
        n_y_active, dtype=tf.float32)
    paddings1 = tf.stack([tf.constant(0), tf.constant(0)], axis=0)
    paddings2 = tf.stack([tf.constant(0), n_y - n_y_active], axis=0)
    paddings = tf.stack([paddings1, paddings2], axis=1)
    probs = tf.pad(probs, paddings, constant_values=1e-12)
    probs.set_shape((None, n_y))
    logging.info('Prior shape: %s', str(probs.shape))
    return probs


def get_curl_modules(n_x,
                     n_y,
                     n_y_active,
                     n_z,
                     output_type,
                     output_shape,
                     prior_size_placeholder,
                     encoder_kwargs,
                     decoder_kwargs):
    """Gets the training and testing model

    Args:
        n_x: flatten dimension of the input, int
        n_y: maximum number of clusters, int
        n_y_active: currently active number of clusters, int
        n_z: dimension of the latent space
        output_type: output distribution type, tf.distributions 
        output_shape: shape of the output image, List[int]
        encoder_kwargs: arguments for the encoder, dict
        decoder_kwargs: arguments for the decoder, dict


    Returns:
        model_train: 
        model_eval: 
    """

    shared_encoder = curl_skeleton.SharedEncoder(
        name='shared_encoder', **encoder_kwargs)
    latent_encoder = functools.partial(
        curl_skeleton.latent_encoder_fn, n_y=n_y, n_z=n_z)
    latent_encoder = snt.Module(latent_encoder, name='latent_encoder')
    latent_decoder = functools.partial(
        curl_skeleton.latent_decoder_fn, n_z=n_z)
    latent_decoder = snt.Module(latent_decoder, name='latent_decoder')
    cluster_encoder = functools.partial(
        curl_skeleton.cluster_encoder_fn, n_y_active=n_y_active, n_y=n_y)
    cluster_encoder = snt.Module(cluster_encoder, name='cluster_encoder')
    data_decoder = functools.partial(
        curl_skeleton.data_decoder_fn,
        output_type=output_type,
        output_shape=output_shape,
        n_x=n_x,
        n_y=n_y,
        **decoder_kwargs)
    data_decoder = snt.Module(data_decoder, name='data_decoder')

    # Uniform prior over y.
    prior_train_probs = construct_prior_probs(
        prior_size_placeholder, n_y, n_y_active)
    prior_train = snt.Module(
        lambda: tfp.distributions.OneHotCategorical(probs=prior_train_probs),
        name='prior_unconditional_train')
    prior_test_probs = construct_prior_probs(
        prior_size_placeholder, n_y, n_y_active)
    prior_test = snt.Module(
        lambda: tfp.distributions.OneHotCategorical(probs=prior_test_probs),
        name='prior_unconditional_test')

    model_train = model.Curl(
        prior_train,
        latent_decoder,
        data_decoder,
        shared_encoder,
        cluster_encoder,
        latent_encoder,
        n_y_active,
        is_training=True,
        name='curl_train')
    model_eval = model.Curl(
        prior_test,
        latent_decoder,
        data_decoder,
        shared_encoder,
        cluster_encoder,
        latent_encoder,
        n_y_active,
        is_training=False,
        name='curl_test')

    return model_train, model_eval


def get_training_params(dataset):
    """Gets the parameters for the training based on the given dataset

    Args:
        dataset: the name of the dataset that will be used, str

    Returns: 
        ll_thres: Loss Threshold used for adding data to the poor representation buffer
        batch_size: Batch size for training, int
        test_batch_size: Batch size for testing, int
        dataset_kwargs: Keyword arguments for a dataset, Dict
        image_key: dictionary key to get the images from dataset, str
        label_key: dictionary key to get the labels from dataset, str
    """

    if dataset == 'mnist':
        ll_thresh = -200
        batch_size = 128
        test_batch_size = 1000
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'label'
    elif dataset == 'fashion_mnist':
        ll_thresh = -250  # need ablation study
        batch_size = 128
        test_batch_size = 1000
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'label'
    elif dataset == 'cifar10':
        ll_thresh = -2000  # need ablation study
        batch_size = 32
        test_batch_size = 10000
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'label'
    elif dataset == 'omniglot':
        ll_thresh = -200  # same as mnist, see appendix of CURL paper
        batch_size = 15
        test_batch_size = 1318
        dataset_kwargs = {}
        image_key = 'image'
        label_key = 'alphabet'
    else:
        raise NotImplementedError

    return ll_thresh, batch_size, test_batch_size, dataset_kwargs, image_key, label_key


def get_real_train_data(sess,
                        training_data_type,
                        step,
                        n_steps_per_class,
                        n_y_active,
                        train_data,
                        batch_size,
                        image_key,
                        label_key,
                        n_concurrent_classes,
                        blend_classes,
                        train_supervised,
                        dynamic_expansion,
                        ):
    """Loads real train data for the a given step

    Args:
        sess: tf.Session, TensorFlow session
        training_data_type: str, structure of the training data (sequential or not) 
        step: int, the training step
        n_steps_per_class: int, number of steps for each class
        n_y_active: int, number of active cluster
        train_data: tf.Tensor, the training data
        batch_size: int, batch size
        image_key: str, key to access image
        label_key: str, key to access label
        n_concurrent_classes: int, number of classes in a single task (usually one or two)
        blend_classes: boolean, whether we want a continuous shift between tasks
        train_supervised: boolean, whether we fix the number of clusters based on the number of seen tasks (usually not used)
        dynamic_expansion: boolean, whether we want an automated creation of clusters (usually used)

    Return:
        Dictionary with images and labels
        Current data period (which class we are looking at)
    """

    if training_data_type == 'sequential':
        current_data_period = int(
            min(step / n_steps_per_class, len(train_data) - 1))

        # If training supervised, set n_y_active directly based on how many
        # classes have been seen
        if train_supervised:
            assert not dynamic_expansion
            n_y_active_np = n_concurrent_classes * (
                current_data_period // n_concurrent_classes + 1)
            n_y_active.load(n_y_active_np, sess)

        train_data_array = sess.run(
            train_data[current_data_period])

        # If we are blending classes, figure out where we are in the data
        # period and add some fraction of other samples.
        if blend_classes:
            # If in the first quarter, blend in examples from the previous class
            if (step % n_steps_per_class < n_steps_per_class / 4 and
                    current_data_period > 0):
                other_train_data_array = sess.run(
                    train_data[current_data_period - 1])

                num_other = int(
                    (n_steps_per_class / 2 - 2 *
                        (step % n_steps_per_class)) * batch_size / n_steps_per_class)
                other_inds = np.random.permutation(batch_size)[
                    :num_other]

                train_data_array[image_key][:num_other] = other_train_data_array[
                    image_key][other_inds]
                train_data_array[label_key][:num_other] = other_train_data_array[
                    label_key][other_inds]

            # If in the last quarter, blend in examples from the next class
            elif (step % n_steps_per_class > 3 * n_steps_per_class / 4 and
                    current_data_period < n_classes - 1):
                other_train_data_array = sess.run(
                    train_data[current_data_period + 1])

                num_other = int(
                    (2 * (step % n_steps_per_class) - 3 * n_steps_per_class / 2) *
                    batch_size / n_steps_per_class)
                other_inds = np.random.permutation(batch_size)[
                    :num_other]

                train_data_array[image_key][:num_other] = other_train_data_array[
                    image_key][other_inds]
                train_data_array[label_key][:num_other] = other_train_data_array[
                    label_key][other_inds]

            # Otherwise, just use the current class

    else:
        train_data_array = sess.run(train_data)

    return train_data_array, current_data_period


def get_gen_train_data(gen_buffer_images,
                       gen_buffer_labels,
                       gen_buffer_class_labels,
                       gen_buffer_ind,
                       batch_size,
                       ):
    """Loads generated train data for a given step

    Args:
        gen_buffer_images: np.array, buffer with generated images
        gen_buffer_labels: np.array, buffer with the cluster labels
        gen_buffer_class_labels: np.array, buffer with the class labels
        gen_buffer_ind: int, index used to select data from the buffers
        batch_size: int, the batch size 

    Return:
        Dictionary containing the images, cluster and class labels
        Index used to select data from the buffers
    """

    s = gen_buffer_ind * batch_size
    e = (gen_buffer_ind + 1) * batch_size

    gen_train_data_array = {
        'image': gen_buffer_images[s:e],
        'cluster': gen_buffer_labels[s:e],
        'class': gen_buffer_class_labels[s:e],
    }

    return gen_train_data_array


def train_new_cluster(sess,
                      train_step_supervised,
                      x_train_raw,
                      model_train,
                      n_y,
                      num_buffer_gen_train_steps,
                      n_poor_batches,
                      poor_data_buffer,
                      poor_data_cluster_labels,
                      batch_size):
    """Train a newly created cluster

    Args:
        sess: tf.Session, currently used session
        train_step_supervised: tf.optimizer.minimize, optimizing function
        x_train_raw: tf.placeholder, placeholder for the train data
        model_train: tf.placeholder, placeholder for the cluster labels
        n_y: int, number of maximum clusters
        num_buffer_gen_train_steps: int, number of training epochs over the poor data
        n_poor_batches: int, number of poor data buffers
        poor_data_buffer: List, poor data buffer
        poor_data_labels: List, poor data cluster labels
        batch_size: int, batch size
    """
    for _ in range(num_buffer_gen_train_steps):
        for bs in range(n_poor_batches):
            x_batch = poor_data_buffer[bs * batch_size:(bs + 1) *
                                       batch_size]
            label_batch = poor_data_cluster_labels[bs * batch_size:(bs + 1) *
                                                   batch_size]
            label_onehot_batch = np.eye(n_y)[label_batch]
            sess.run(
                train_step_supervised,
                feed_dict={
                    x_train_raw: x_batch,
                    model_train.y_label: label_onehot_batch
                })


def get_poor_data_conditioned(poor_data_dict, batch_size, new_cluster, curr_class):
    """Select and cull the poor data conditioned on a class to batch size for training the new cluster"""

    n_poor_batches = len(poor_data_dict[curr_class]) // batch_size
    poor_data_batches = poor_data_dict[curr_class][:n_poor_batches * batch_size]
    poor_class_labels = [curr_class for _ in range(
        n_poor_batches * batch_size)]
    poor_cluster_labels = [
        new_cluster for _ in range(n_poor_batches * batch_size)]

    return poor_data_batches, poor_class_labels, poor_cluster_labels


def get_poor_data(poor_data_dict, len_poor_data, batch_size, new_cluster):
    """Select and cull the poor data to batch size for training the new cluster"""

    n_poor_batches = len_poor_data // batch_size
    poor_data = []
    poor_class_labels = []
    for k, v in poor_data_dict.items():
        poor_data += v
        poor_class_labels += [k for _ in range(len(v))]

    poor_data_batches = poor_data[:n_poor_batches * batch_size]
    poor_class_labels = poor_class_labels[:n_poor_batches * batch_size]
    poor_cluster_labels = [
        new_cluster for _ in range(n_poor_batches * batch_size)]

    return poor_data_batches, poor_class_labels, poor_cluster_labels
