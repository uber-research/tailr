#!/usr/bin/env python3
import argparse
import os.path
import math
import numpy as np
import torch
from utils import dgr_utils
from training.data import get_dataset, DATASET_CONFIGS
from training.train_dgr import train
from dgr import Scholar
from solvers import CNN
from generators.wgan import WGAN

N_CLASSES = 3

parser = argparse.ArgumentParser(
    'PyTorch implementation of Deep Generative Replay'
)

parser.add_argument(
    '--experiment', type=str,
    choices=['permutated_mnist', 'svhn-mnist',
             'mnist-svhn', 'split_mnist', 'fashion_mnist'],
    default='split_mnist'
)
parser.add_argument('--mnist-permutation-number', type=int, default=5)
parser.add_argument('--mnist-permutation-seed', type=int, default=0)
parser.add_argument(
    '--replay-mode', type=str, default='generative-replay',
    choices=['exact-replay', 'generative-replay', 'none'],
)

parser.add_argument('--generator-lambda', type=float, default=10.)
parser.add_argument('--generator-z-size', type=int, default=100)
parser.add_argument('--generator-c-channel-size', type=int, default=64)
parser.add_argument('--generator-g-channel-size', type=int, default=64)
parser.add_argument('--solver-depth', type=int, default=5)
parser.add_argument('--solver-reducing-layers', type=int, default=3)
parser.add_argument('--solver-channel-size', type=int, default=1024)

parser.add_argument('--generator-c-updates-per-g-update', type=int, default=5)
parser.add_argument('--generator-iterations', type=int, default=1000)
parser.add_argument('--solver-iterations', type=int, default=1000)
parser.add_argument('--importance-of-new-task', type=float, default=.3)
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-05)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--test-size', type=int, default=1024)
parser.add_argument('--sample-size', type=int, default=36)

parser.add_argument('--sample-log', action='store_true')
parser.add_argument('--sample-log-interval', type=int, default=300)
parser.add_argument('--image-log-interval', type=int, default=100)
parser.add_argument('--eval-log-interval', type=int, default=50)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

parser.add_argument('--port', type=int, default=8097)
parser.add_argument('--server', type=str, default='http://localhost')
parser.add_argument('--vis-show', type=int, default=1)

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_true')
main_command.add_argument('--test', action='store_false', dest='train')


if __name__ == '__main__':
    args = parser.parse_args()

    if not math.log10(args.batch_size) / math.log10(2):
        raise ValueError("Batch size should be a power of 2")

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda
    experiment = args.experiment
    capacity = args.batch_size * max(
        args.generator_iterations,
        args.solver_iterations
    )

    if experiment == 'permutated_mnist':
        # generate permutations for the mnist classification tasks.
        np.random.seed(args.mnist_permutation_seed)
        permutations = [
            np.random.permutation(DATASET_CONFIGS['mnist']['size']**2) for
            _ in range(args.mnist_permutation_number)
        ]

        # prepare the datasets.
        train_datasets = [
            get_dataset('mnist', permutation=p, capacity=capacity)
            for p in permutations
        ]
        test_datasets = [
            get_dataset('mnist', train=False, permutation=p, capacity=capacity)
            for p in permutations
        ]

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['mnist']

    elif experiment == 'split_mnist':
        classes = [i for i in range(N_CLASSES)]
        # prepare the datasets.
        train_datasets = [
            get_dataset('mnist', label=c, capacity=capacity)
            for c in classes
        ]

        test_datasets = [
            get_dataset('mnist', train=False, label=c, capacity=capacity)
            for c in classes
        ]

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['mnist']

    elif experiment == 'fashion_mnist':
        classes = [i for i in range(10)]
        # prepare the datasets.
        train_datasets = [
            get_dataset('fashion_mnist', label=c, capacity=capacity)
            for c in classes
        ]

        test_datasets = [
            get_dataset('fashion_mnist', train=False,
                        label=c, capacity=capacity)
            for c in classes
        ]

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['fashion_mnist']

    elif experiment in ('svhn-mnist', 'mnist-svhn'):
        mnist_color_train = get_dataset(
            'mnist-color', train=True, capacity=capacity
        )
        mnist_color_test = get_dataset(
            'mnist-color', train=False, capacity=capacity
        )
        svhn_train = get_dataset('svhn', train=True, capacity=capacity)
        svhn_test = get_dataset('svhn', train=False, capacity=capacity)

        # prepare the datasets.
        train_datasets = (
            [mnist_color_train, svhn_train] if experiment == 'mnist-svhn' else
            [svhn_train, mnist_color_train]
        )
        test_datasets = (
            [mnist_color_test, svhn_test] if experiment == 'mnist-svhn' else
            [svhn_test, mnist_color_test]
        )

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['mnist-color']
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(experiment))

    # define the models.
    cnn = CNN(
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        classes=dataset_config['classes'],
        depth=args.solver_depth,
        channel_size=args.solver_channel_size,
        reducing_layers=args.solver_reducing_layers,
    )
    wgan = WGAN(
        z_size=args.generator_z_size,
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        c_channel_size=args.generator_c_channel_size,
        g_channel_size=args.generator_g_channel_size,
    )
    label = '{experiment}-{replay_mode}-r{importance_of_new_task}'.format(
        experiment=experiment,
        replay_mode=args.replay_mode,
        importance_of_new_task=(
            1 if args.replay_mode == 'none' else
            args.importance_of_new_task
        ),
    )
    scholar = Scholar(label, generator=wgan, solver=cnn)

    # initialize the model.
    dgr_utils.gaussian_intiailize(scholar, std=.02)

    # use cuda if needed
    if cuda:
        scholar.cuda()

    # determine whether we need to train the generator or not.
    train_generator = (
        args.replay_mode == 'generative-replay' or
        args.sample_log
    )

    # run the experiment.
    if args.train:
        train(
            scholar, train_datasets, test_datasets,
            replay_mode=args.replay_mode,
            generator_lambda=args.generator_lambda,
            generator_iterations=(
                args.generator_iterations if train_generator else 0
            ),
            generator_c_updates_per_g_update=(
                args.generator_c_updates_per_g_update
            ),
            solver_iterations=args.solver_iterations,
            importance_of_new_task=args.importance_of_new_task,
            batch_size=args.batch_size,
            test_size=args.test_size,
            sample_size=args.sample_size,
            lr=args.lr, weight_decay=args.weight_decay,
            beta1=args.beta1, beta2=args.beta2,
            loss_log_interval=args.loss_log_interval,
            eval_log_interval=args.eval_log_interval,
            image_log_interval=args.image_log_interval,
            sample_log_interval=args.sample_log_interval,
            sample_log=args.sample_log,
            sample_dir=args.sample_dir,
            checkpoint_dir=args.checkpoint_dir,
            collate_fn=dgr_utils.label_squeezing_collate_fn,
            cuda=cuda,
            server=args.server,
            port=args.port,
            vis_show=(args.vis_show == 1)
        )
    else:
        path = os.path.join(args.sample_dir, '{}-sample'.format(scholar.name))
        dgr_utils.load_checkpoint(scholar, args.checkpoint_dir)
        dgr_utils.test_model(scholar.generator, args.sample_size, path)
