import os.path
import copy
import numpy as np
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils import dgr_utils
from utils import dgr_visual

tb_logdir = '/results/dgr/tb_logs'
image_dir = '/results/dgr/images'


def train(scholar, train_datasets, test_datasets, replay_mode,
          generator_lambda=10.,
          generator_c_updates_per_g_update=5,
          generator_iterations=2000,
          solver_iterations=1000,
          importance_of_new_task=.5,
          batch_size=32,
          test_size=1024,
          sample_size=36,
          lr=1e-03, weight_decay=1e-05,
          beta1=.5, beta2=.9,
          loss_log_interval=30,
          eval_log_interval=50,
          image_log_interval=100,
          sample_log_interval=300,
          sample_log=False,
          sample_dir='/results/samples',
          checkpoint_dir='/results/checkpoints',
          collate_fn=None,
          cuda=False,
          port=8097,
          server='http://localhost',
          vis_show=True):

    # Create saving folders
    tb_logdir_path = Path(tb_logdir)
    tb_logdir_path.mkdir(parents=True, exist_ok=True)
    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(f'{tb_logdir}/CLF')

    # define solver criterion and generators for the scholar model.
    solver_criterion = nn.CrossEntropyLoss()
    solver_optimizer = optim.Adam(
        scholar.solver.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    generator_g_optimizer = optim.Adam(
        scholar.generator.generator.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    generator_c_optimizer = optim.Adam(
        scholar.generator.critic.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )

    # set the criterion, optimizers, and training configurations for the
    # scholar model.
    scholar.solver.set_criterion(solver_criterion)
    scholar.solver.set_optimizer(solver_optimizer)
    scholar.generator.set_lambda(generator_lambda)
    scholar.generator.set_generator_optimizer(generator_g_optimizer)
    scholar.generator.set_critic_optimizer(generator_c_optimizer)
    scholar.generator.set_critic_updates_per_generator_update(
        generator_c_updates_per_g_update
    )
    scholar.train()

    # define the previous scholar who will generate samples of previous tasks.
    previous_scholar = None
    previous_datasets = None

    for task, train_dataset in enumerate(train_datasets, 1):
        # define callbacks for visualizing the training process.
        generator_training_callbacks = [_generator_training_callback(
            loss_log_interval=loss_log_interval,
            image_log_interval=image_log_interval,
            sample_log_interval=sample_log_interval,
            sample_log=sample_log,
            sample_dir=sample_dir,
            sample_size=sample_size,
            current_task=task,
            total_tasks=len(train_datasets),
            total_iterations=generator_iterations,
            batch_size=batch_size,
            replay_mode=replay_mode,
            env=scholar.name,
            vis_show=vis_show,
        )]
        solver_training_callbacks = [_solver_training_callback(
            loss_log_interval=loss_log_interval,
            eval_log_interval=eval_log_interval,
            current_task=task,
            total_tasks=len(train_datasets),
            total_iterations=solver_iterations,
            batch_size=batch_size,
            test_size=test_size,
            test_datasets=test_datasets,
            replay_mode=replay_mode,
            cuda=cuda,
            collate_fn=collate_fn,
            env=scholar.name,
            vis_show=vis_show,
            tb_writer=tb_writer,
        )]

        # train the scholar with generative replay.
        scholar.train_with_replay(
            train_dataset,
            scholar=previous_scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            generator_iterations=generator_iterations,
            generator_training_callbacks=generator_training_callbacks,
            solver_iterations=solver_iterations,
            solver_training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

        previous_scholar = (
            copy.deepcopy(scholar) if replay_mode == 'generative-replay' else
            None
        )
        previous_datasets = (
            train_datasets[:task] if replay_mode == 'exact-replay' else
            None
        )

        buffer = scholar.sample(40)[0].cpu()
        dgr_visual.save_buffer_images(
            buffer, image_dir, f'Generator for the task #{task}')

    # save the model after the experiment.
    dgr_utils.save_checkpoint(scholar, checkpoint_dir)

    with open(f'/results/DONE', 'w+') as done_file:
        done_file.write(
            '-What is my purpose? -You terminate the run. -Oh my god...')


def _generator_training_callback(
        loss_log_interval,
        image_log_interval,
        sample_log_interval,
        sample_log,
        sample_dir,
        current_task,
        total_tasks,
        total_iterations,
        batch_size,
        sample_size,
        replay_mode,
        env,
        server='http://localhost',
        port=8097,
        vis_show=True):

    if vis_show:
        vis_ = dgr_visual.visualization(env=env, server=server, port=port)

    def cb(generator, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            '<Training Generator> '
            'task: {task}/{tasks} | '
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'loss => '
            'g: {g_loss:.4} / '
            'w: {w_dist:.4}'
        ).format(
            task=current_task,
            tasks=total_tasks,
            trained=batch_size * batch_index,
            total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            g_loss=result['g_loss'],
            w_dist=-result['c_loss'],
        ))

        if vis_show:
            # log the losses of the generator.
            if iteration % loss_log_interval == 0:
                vis_.visualize_scalar(
                    result['g_loss'], 'generator g loss', iteration, env=env
                )
                vis_.visualize_scalar(
                    -result['c_loss'], 'generator w distance', iteration, env=env
                )

            # log the generated images of the generator.
            if iteration % image_log_interval == 0:
                vis_.visualize_images(
                    generator.sample(sample_size).data,
                    'generated samples ({replay_mode})'
                    .format(replay_mode=replay_mode), env=env,
                )

        # log the sample images of the generator
        if iteration % sample_log_interval == 0 and sample_log:
            dgr_utils.test_model(generator, sample_size, os.path.join(
                sample_dir,
                env + '-sample-logs',
                str(iteration)
            ), verbose=False)

    return cb


def _solver_training_callback(
        loss_log_interval,
        eval_log_interval,
        current_task,
        total_tasks,
        total_iterations,
        batch_size,
        test_size,
        test_datasets,
        cuda,
        replay_mode,
        collate_fn,
        env,
        server='http://localhost',
        port=8097,
        vis_show=True,
        tb_writer=None):

    if vis_show:
        vis_ = dgr_visual.visualization(env=env, server=server, port=port)

    def cb(solver, progress, batch_index, result):
        iteration = (current_task-1)*total_iterations + batch_index
        progress.set_description((
            '<Training Solver>    '
            'task: {task}/{tasks} | '
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'loss: {loss:.4} | '
            'prec: {prec:.4}'
        ).format(
            task=current_task,
            tasks=total_tasks,
            trained=batch_size * batch_index,
            total=batch_size * total_iterations,
            percentage=(100.*batch_index/total_iterations),
            loss=result['loss'],
            prec=result['precision'],
        ))

        # log the loss of the solver.
        if iteration % loss_log_interval == 0 and vis_show:
            vis_.visualize_scalar(
                result['loss'], 'solver loss', iteration, env=env
            )

        if tb_writer:
            tb_writer.add_scalar(
                'clf_train_loss', result['loss'], iteration)

        # evaluate the solver on multiple tasks.
        if iteration % eval_log_interval == 0:
            names = ['task {}'.format(i+1) for i in range(len(test_datasets))]
            accuracies = [
                dgr_utils.validate(
                    solver, test_datasets[i], test_size=test_size,
                    cuda=cuda, verbose=False, collate_fn=collate_fn,
                ) if i+1 <= current_task else 0 for i in
                range(len(test_datasets))
            ]
            title = f'Accuracy ({replay_mode})'
            if vis_show:
                vis_.visualize_scalars(
                    accuracies, names, title,
                    iteration, env=env
                )
            if tb_writer:
                acc_dict = {f'Task_{i}': acc for i,
                            acc in enumerate(accuracies)}
                tb_writer.add_scalars(
                    'clf_validation_accuracy', acc_dict, iteration)

    return cb
