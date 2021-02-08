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
"""Training file to run most of the experiments in the paper.

The default parameters corresponding to the first set of experiments in Section
4.2.

For the expansion ablation, run with different ll_thresh values as in the paper.
Note that n_y_active represents the number of *active* components at the
start, and should be set to 1, while n_y represents the maximum number of
components allowed, and should be set sufficiently high (eg. n_y = 100).

For the MGR ablation, setting use_sup_replay = True switches to using SMGR,
and the gen_replay_type flag can switch between fixed and dynamic replay. The
generative snapshot period is set automatically in the train_curl.py file based
on these settings (ie. the data_period variable), so the 0.1T runs can be
reproduced by dividing this value by 10.
"""
import os
import click
import yaml
import runpy

import training

from absl import logging
from typing import List
from click_list import ClickList

# with open('/src/tailr_tf/config_training.yaml') as temp:
#     config_training = yaml.safe_load(temp)

encoder_kwargs_dict = {'mixed': {
                                    'encoder_type': 'mixed',
                                    'n_enc': [64, 64, 64, 300, 150],
                                    'enc_strides': [1, 1, 1],
                        },
                        'conv': {
                                    'encoder_type': 'conv',
                                    'n_enc': [16, 32, 150],
                                    'enc_strides': [1, 1],
                        },
                        'multi':{
                                    'encoder_type': 'multi',
                                    'n_enc': [1200, 600, 300, 150],
                                    'enc_strides': [1, 1],
                        }
}

decoder_kwargs_dict = { 'deconv': {
                                    'decoder_type': 'deconv',
                                    'n_dec': [16, 32, 150],
                                    'dec_up_strides': [1, 1],
                        },
                        'single': {
                                    'decoder_type': 'single',
                                    'n_dec': [500, 500],
                                    'dec_up_strides': None,
                        },
                        'multi':{
                                    'decoder_type': 'multi',
                                    'n_dec': [500, 500],
                                    'dec_up_strides': None,
                        }
}


@click.command()
@click.option(
    '--log_dir',
    default='./',
    type=click.STRING,
    help='log saving folder',
)
@click.option(
    '--dataset',
    default='mnist',
    type=click.Choice(
        [
            'mnist', 'omniglot', 'fashion_mnist', 'cifar10'
        ]
    ),
    help='Training dataset',
)
@click.option(
    '--clfmode',
    default='og_init',
    type=click.Choice(
        [
            'task_init', 'no_init',
            'cluster_init', 'loss_init', 'fixed_init'
        ]
    ),
    help='Classifier re-initialization mode',
)
@click.option(
    '--classifier_init_period',
    default=1,
    type=click.INT,
    help='The period used for classifier re-initialization under \'fixed_init\' and \'cluster_init\'',
)
@click.option(
    '--clf_thresh',
    default=0.0,
    type=click.FLOAT,
    help='The threshold for adding data to the classifier poorly classified buffer. Used with  \'loss_init\'',
)
@click.option(
    '--n_steps',
    default=25000,
    type=click.INT,
    help='Number of train steps for TAILR',
)
@click.option(
    '--max_gen_batches',
    default=5000,
    type=click.INT,
    help='Number of generated batches for TAILR',
)
@click.option(
    '--class_conditioned',
    is_flag=True,
    help='Use of class conditioned clusters',
)
@click.option(
    '--cluster_wait_steps',
    default=100,
    type=click.INT,
    help='Number of minimum steps between two cluster creations'
)
@click.option(
    '--experiment_name',
    default='',
    help='Name for the experiment',
)
@click.option(
    '--save_viz',
    is_flag=True,
    help='whether to save images or not',
)
@click.option(
    '--batch_mix',
    default='combined',
    type=click.Choice(
        ['combined', 'semi_combined', 'alternate']
    ),
    help='how to mix the data in the training batches'
)
@click.option(
    '--class_order',
    cls=ClickList,
    default=[x for x in range(10)],
    help='Orders of the classes to be seen'
)
@click.option(
    '--encoder_type',
    default='multi',
    type=click.Choice(
        ['multi', 'conv', 'mixed']
    ),
    help='Encoder architecture',
)
@click.option(
    '--decoder_type',
    default='single',
    type=click.Choice(
        ['multi', 'deconv', 'single']
    ),
    help='Decoder architecture',
)
@click.option(
    "--ll_thresh",
    default=-200,
    type=click.INT, 
    help='Loss threshold for poorly expained data for the generator',
)
def main(log_dir: str, 
         dataset: str, 
         clfmode: str,
         classifier_init_period: int, 
         clf_thresh: float,
         n_steps: int, 
         max_gen_batches: int,
         class_conditioned: bool,
         cluster_wait_steps: int,
         experiment_name: str, 
         save_viz: bool,
         batch_mix: str, 
         class_order: List[int],
         encoder_type: str,
         decoder_type: str,
         ll_thresh: int, 
         ) -> None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file('CURL_TRAIN_LOG', log_dir)
    experiment_name = experiment_name if experiment_name else f'{clfmode}_csv_{dataset}_{n_steps}'
    training.run_training(
        dataset=dataset,
        output_type='bernoulli',
        n_y=50,
        n_y_active=1,
        n_z=32,
        cluster_wait_steps=cluster_wait_steps,
        training_data_type='sequential',
        n_concurrent_classes=1,
        lr_init=1e-3,
        lr_factor=1.,
        lr_schedule=[1],
        blend_classes=False,
        train_supervised=False,
        n_steps=n_steps,
        report_interval=10,
        knn_values=[10],
        random_seed=1,
        encoder_kwargs=encoder_kwargs_dict[encoder_type],
        decoder_kwargs=decoder_kwargs_dict[decoder_type],
        dynamic_expansion=True,
        ll_thresh=ll_thresh,
        classify_with_samples=False,
        gen_replay_type='dynamic',
        use_supervised_replay=False,
        batch_mix=batch_mix,
        experiment_name=experiment_name,
        clf_mode=clfmode,
        gen_save_image_count=40,
        max_gen_batches=max_gen_batches,
        classifier_init_period=classifier_init_period,
        clf_thresh=clf_thresh,
        class_order=class_order,
        class_conditioned=class_conditioned,
        save_viz=save_viz,
        need_oracle=True,
    )

    training.move_time_log(experiment_name)

if __name__ == '__main__':
    main()
