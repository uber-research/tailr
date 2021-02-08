# Training TAILR model

import torch
import numpy as np
import tqdm


from data import get_dataset, DATASET_CONFIGS
from model import TAILR

# Break utils into utils and viz_utils
from utils import log_prob_elbo, log_prob_elbo_sup, launch_tb, binarize_fn

from vis_utils import save_loss_info, save_single_cluster_generation_img, save_buffer_images, \
    save_all_clusters_multi_generations, save_fig, save_cluster_evolution

from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

EPOCHS = 1
LR = 1e-3

BATCH_SIZE = 100
SAMPLES_PER_CLASS = 10000
BATCH_PER_CLASS = SAMPLES_PER_CLASS // BATCH_SIZE
LABELS = [i for i in range(1)]

GENERATION_SNAPSHOT_N_BATCHES = 5000

# NOTE About 1000 batches per class from CURL
UNEXP_LL_THRES = -200
CLUSTER_CREATE = True
CLUSTER_CREATION_THRES = 100
assert CLUSTER_CREATION_THRES >= 100
N_NEW_CLUSTER_TRAIN_EPOCHS = 20
BURN_IN_STEPS = 100
WAIT_STEPS = BURN_IN_STEPS

OUTPUT_PATH = '/results'


def reinforce_cluster(data_batches, cluster_label, model, n_y, n_y_active, optimizer, new_cluster_epochs, tb_writer=None, debug=False):
    """Reinforce existing cluster with a given dataset

    Args:
        data_batches: List of batches of data used to reinforce the given cluster, List[Tensor [B, *]]
        cluster_label: cluster idx to be reinforced, int 
    """
    # Form the labels tensor
    label_tensor = torch.tensor(
        [0 if i != cluster_label else 1 for i in range(n_y)]).unsqueeze(0).type(torch.float).to(device)
    label_tensors = label_tensor.expand(data_batches[0].shape[0], -1)

    step = 0

    # optimize the model
    for e in range(new_cluster_epochs):
        for batch in data_batches:
            optimizer.zero_grad()

            hiddens, cluster_score, latent_sample, out = model(batch)

            kl_y, kl_z, log_p_x = log_prob_elbo(batch, model)
            kl_y_sup, kl_z_sup, log_p_x_sup = log_prob_elbo_sup(
                batch, hiddens, cluster_score, label_tensors, n_y, n_y_active, model)

            basic_loss = (log_p_x - kl_y - kl_z)
            sup_loss = (log_p_x_sup - kl_y_sup - kl_z_sup)
            batch_loss = sup_loss.mean()

            loss_for_grad = - batch_loss
            loss_for_grad.backward()
            optimizer.step()

            if tb_writer:
                tb_writer.add_scalar(
                    f'Sup_Loss/train cluster {cluster_label} creation', batch_loss.detach().cpu().numpy(), step)
                tb_writer.add_scalar(
                    f'UnSup_Loss/train cluster {cluster_label} creation', basic_loss.mean().detach().cpu().numpy(), step)

            step += 1

        if debug:
            for i in range(1, 4 + 1):
                print(
                    f'Shared Encoder {4 - i + 1}th Weight grad={model.shared_encoder.encoder.graph[-(i * 2)].weight.grad}')
            print(
                f'Cluster Encoder Weight grad={model.cluster_encoder.lin.weight.grad}')
            print(
                f'Latent Encoder Weight grad={model.latent_encoder.lins[0].weight.grad}')
            print(
                f'Latent Decoder Weight grad={model.latent_decoder.decoder.graph[0].weight.grad}')


def train(dataset_name, debug=False):
    """Training the TAILR model with a given dataset

    Args:
        dataset_name: Name of the dataset used for training, str
    """

    try:
        train_ds = get_dataset(dataset_name)
        test_ds = get_dataset(dataset_name, train=False)
    except:
        raise ValueError('The given dataset is not available yet')
    else:
        train_ds_temp = get_dataset(dataset_name)

        idx = train_ds_temp.targets == LABELS[0]
        n_repeats = SAMPLES_PER_CLASS // train_ds_temp.targets[idx].shape[0]
        train_ds.targets = train_ds_temp.targets[idx].repeat(n_repeats)
        train_ds.targets = torch.cat(
            [train_ds.targets, train_ds_temp.targets[idx][:(SAMPLES_PER_CLASS - len(train_ds.targets))]])
        train_ds.data = train_ds_temp.data[idx].repeat(n_repeats, 1, 1)
        train_ds.data = torch.cat(
            [train_ds.data, train_ds_temp.data[idx][:(SAMPLES_PER_CLASS - len(train_ds.data))]])

        for LABEL in LABELS[1:]:
            idx = train_ds_temp.targets == LABEL
            n_repeats = SAMPLES_PER_CLASS // train_ds_temp.targets[idx].shape[0]
            targets_to_add = train_ds_temp.targets[idx].repeat(n_repeats)
            targets_to_add = torch.cat(
                [targets_to_add, train_ds_temp.targets[idx][:(SAMPLES_PER_CLASS - len(targets_to_add))]])
            data_to_add = train_ds_temp.data[idx].repeat(n_repeats, 1, 1)
            data_to_add = torch.cat(
                [data_to_add, train_ds_temp.data[idx][:(SAMPLES_PER_CLASS - len(data_to_add))]])
            train_ds.targets = torch.cat(
                [train_ds.targets, targets_to_add])
            train_ds.data = torch.cat([train_ds.data, data_to_add])

        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True)

    output_shape = [DATASET_CONFIGS[dataset_name]['channels'], DATASET_CONFIGS[dataset_name]['size'],
                    DATASET_CONFIGS[dataset_name]['size']]  # Torch image size is channel, height, width
    n_x = np.prod(output_shape)
    n_y = 50
    n_z = 32
    start_n_y_active = 1
    n_enc = [1200, 600, 300, 150]
    n_dec = [500, 500]

    model = TAILR(n_x=n_x,
                  n_y=n_y,
                  n_y_active=start_n_y_active,
                  n_z=n_z,
                  n_enc=n_enc,
                  n_dec=n_dec,
                  decoder_in_dims=n_x,
                  shared_encoder_channels=-1,
                  encoder_type='multi',
                  decoder_type='single',
                  enc_strides=None,
                  dec_strides=None,
                  decoder_output_shapes=output_shape,
                  feature_size=n_enc[-1],
                  )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    # Loss History
    losses = []
    log_ps = []
    kl_ys = []
    kl_zs = []

    # Data Generation infos and buffers
    generative_replay = False
    gen_every_n = 1
    generative_replay_counter = 0
    cluster_generation_weights = np.zeros((n_y))
    recently_used_clusters = np.zeros((n_y))

    # Unexplained inputs buffers and history
    unexplained_data_buffer = []
    unexplained_label_buffer = []
    unexplained_data_history = []
    unexplained_label_history = []

    # Step counters for Cluster Creation
    step = 0
    steps_since_expansion = 0
    eligible_for_expansion = False
    cluster_creation_count = 0

    # Set up TB and tqdm
    folder_name = f'{SAMPLES_PER_CLASS}_{"_".join(map(str,LABELS))}'
    tb_path = f'{OUTPUT_PATH}/tb/{folder_name}/'
    launch_tb(tb_path)
    tb_writer = SummaryWriter(tb_path)
    tb_writer.add_graph(model, train_ds[0][0].detach().to(device))
    pbar = tqdm.tqdm(total=len(train_loader))

    # Investigation on cluster scores for class 1
    class_one_cluster_scores = None

    # "Validation" batches
    validation_batch = []

    for e in range(EPOCHS):
        for batch_idx, (data, label) in enumerate(train_loader):

            if dataset_name == 'mnist':
                data = binarize_fn(data)

            if batch_idx % BATCH_PER_CLASS == 0:
                validation_batch.append(data)

            # If we created a cluster, mix data from other clusters to avoid forgetting
            if generative_replay and (gen_every_n > 0 and step % gen_every_n == 0):

                generative_replay_data = generated_data[generative_replay_counter]
                generative_replay_data = generative_replay_data.to(device)
                generative_replay_counter += 1

                hiddens, cluster_score, latent_sample, out = model(
                    generative_replay_data)
                kl_y, kl_z, log_p_x, cluster_score = log_prob_elbo(generative_replay_data,
                                                                   model=model)
                loss = log_p_x - kl_y - kl_z
                batch_loss = loss.mean()
                loss_for_grad = - batch_loss

                optimizer.zero_grad()
                loss_for_grad.backward()
                optimizer.step()

            data = data.to(device)
            optimizer.zero_grad()

            kl_y, kl_z, log_p_x, cluster_score = log_prob_elbo(
                data, model=model)
            loss = log_p_x - kl_y - kl_z
            batch_loss = loss.mean()

            loss_for_grad = - batch_loss
            loss_for_grad.backward()
            optimizer.step()

            # Record cluster score for class 1
            class_one_idx = label == 1
            if cluster_score[class_one_idx].nelement() > 0:
                if class_one_cluster_scores is None:
                    class_one_cluster_scores = cluster_score[class_one_idx].detach(
                    ).cpu().mean(dim=0).unsqueeze(0)

                else:
                    class_one_cluster_scores = torch.cat(
                        [class_one_cluster_scores, cluster_score[class_one_idx].detach().cpu()])

            # Record cluster scores for data generation
            recently_used_clusters += torch.sum(cluster_score,
                                                dim=0).cpu().detach().numpy()

            # Unexplained Buffers and Cluster Creation
            if CLUSTER_CREATE:

                steps_since_expansion += 1

                # Check eligibility for adding data to unexplained buffer and creating a new cluster
                if (steps_since_expansion >= WAIT_STEPS and step >= BURN_IN_STEPS and
                        model.n_y_active < n_y):
                    eligible_for_expansion = True

                if eligible_for_expansion:

                    unexplained_idx = loss < UNEXP_LL_THRES
                    unexplained_data_buffer += data[unexplained_idx]
                    unexplained_label_buffer += label[unexplained_idx]

                    # If size threshold is reached, create and initialize cluster
                    # TODO maybe move part of this into a separate function later for the sake of readability
                    if len(unexplained_data_buffer) > CLUSTER_CREATION_THRES:
                        print(
                            f'CLUSTER CREATION at step #{step} for class {label.cpu().detach().numpy()}')

                        cluster_creation_count += 1

                        # Save mutliple generated images for comparison
                        save_all_clusters_multi_generations(
                            model, path=OUTPUT_PATH, name=f'multi_creation{cluster_creation_count}')

                        # Take a generative snapshot for later cluster reinforcement
                        cluster_generation_weights += recently_used_clusters
                        recently_used_clusters = np.zeros(n_y)

                        generated_data, generated_cluster_label = model.get_generative_snapshot(
                            batch_size=BATCH_SIZE,
                            gen_buffer_size=GENERATION_SNAPSHOT_N_BATCHES,
                            cluster_weights=cluster_generation_weights,
                        )

                        # Build history of unexplained buffer
                        unexplained_data_history.append(
                            unexplained_data_buffer)
                        unexplained_label_history.append(
                            unexplained_label_buffer)

                        # Resize the buffers to accomodate batch size
                        n_unexplained_batches = len(
                            unexplained_data_buffer) // BATCH_SIZE
                        unexplained_data_buffer = unexplained_data_buffer[-(
                            n_unexplained_batches * BATCH_SIZE):]
                        unexplained_label_buffer = unexplained_label_buffer[-(
                            n_unexplained_batches * BATCH_SIZE):]

                        # Build the batches
                        batches = []
                        for bs in range(n_unexplained_batches):
                            batches.append(torch.stack(
                                unexplained_data_buffer[bs * BATCH_SIZE:(bs + 1) * BATCH_SIZE]).to(device))

                        # Get the cluster "closest" to the data
                        unexplained_cluster_scores = []
                        for batch in batches:
                            unexplained_cluster_scores.append(
                                model.get_cluster_score(batch).cpu().detach())
                        best_cluster = np.argmax(
                            np.sum(np.vstack(unexplained_cluster_scores), axis=0))

                        # Initialize new cluster with the the closest cluster weights and biases
                        model.init_new_cluster(best_cluster)
                        model.n_y_active += 1

                        save_buffer_images(unexplained_data_buffer, path=OUTPUT_PATH,
                                           name=f'unexplained_buffer_creation{cluster_creation_count}')
                        # Train the newly created cluster
                        reinforce_cluster(batches, model.n_y_active - 1, model, model.n_y,
                                          model.n_y_active, optimizer, N_NEW_CLUSTER_TRAIN_EPOCHS, tb_writer=tb_writer)

                        save_single_cluster_generation_img(
                            model, path=OUTPUT_PATH, name=f'post_creation{cluster_creation_count}', cluster_idx=model.n_y_active - 1)

                        # Reset cluster creation flags and buffers
                        unexplained_data_buffer = []
                        unexplained_label_buffer = []
                        steps_since_expansion = 0
                        eligible_for_expansion = False
                        generative_replay = True

            pbar.update(1)
            pbar.set_description(desc=f'Epoch {e}')

            losses.append(batch_loss.detach().cpu().numpy())
            tb_writer.add_scalar(
                'Loss/train', batch_loss.detach().cpu().numpy(), step)

            log_ps.append(log_p_x.mean().cpu().detach().numpy())
            kl_ys.append(kl_y.mean().cpu().detach().numpy())
            kl_zs.append(kl_z.mean().cpu().detach().numpy())

            print(f'Iteration {batch_idx+1}, Train elbo: {loss_for_grad.cpu().detach().numpy():.3f}, kl y: {kl_y.mean().cpu().detach().numpy():.3f}, kl z: {kl_z.mean().cpu().detach().numpy():.3f}')

            if debug:
                print(f'LOG {log_p_x.mean().cpu().detach().numpy()}')
                print(f'KL_Y {kl_y.mean().cpu().detach().numpy()}')
                print(f'KL_Z {kl_z.mean().cpu().detach().numpy()}')
                if False:
                    for i in range(len(n_enc), 0, -1):
                        print(
                            f'Shared Encoder {len(n_enc) - i + 1}th Weight absolute sum grad={model.shared_encoder.encoder.graph[-(i * 2)].weight.grad.abs().sum()}')
                    print(
                        f'Cluster Encoder Weight absolute sum grad={model.cluster_encoder.lin.weight.grad.abs().sum()}')
                    print(
                        f'Latent Encoder Weight absolute sum grad={model.latent_encoder.lins[0].weight.grad.abs().sum()}')
                    for i in range(len(n_dec) + 1, 0, -1):
                        print(
                            f'Latent Decoder {len(n_dec) - i + 1}th Weight average grad={model.latent_decoder.decoder.graph[-(i * 2) + 1].weight.grad.abs().sum() / torch.numel(model.latent_decoder.decoder.graph[-(i * 2) + 1].weight.grad)}')

        print('\n')
        pbar.reset()

    tb_writer.flush()
    tb_writer.close()

    model.eval()

    # Save the model
    torch.save(model.state_dict(), f'{OUTPUT_PATH}/models')

    # Save cluster score evolaution for class 1
    if 1 in LABELS:
        save_cluster_evolution(
            np.array(class_one_cluster_scores), path=OUTPUT_PATH, name='cluster_evol_class1', title='Cluster Score for class 1', x_label='train batches', y_label='Cluster Score', n_y_active=model.n_y_active, smoothing_window=801, smoothing_degree=3)

    # Save loss visualization and cluster visualization
    save_loss_info(log_ps, kl_ys, kl_zs, losses, path=OUTPUT_PATH)
    save_all_clusters_multi_generations(model, path=OUTPUT_PATH, name='end')

    for i, label in enumerate(LABELS):
        test_batch = validation_batch[i].to(device)

        hiddens, cluster_score, latent_sample, out = model(
            test_batch)

        kl_y, kl_z, log_p_x, cluster_score = log_prob_elbo(test_batch,
                                                           model=model)

        print(f'VALIDATION for class {label} \n \
                kl_y: {kl_y.mean()}, \n \
                kl_z: {kl_z.mean()}, \n \
                log_p: {log_p_x.mean()}, \n \
                ELBO: {(log_p_x - kl_y - kl_z).mean()} \n \
                cluster Score: {torch.mean(cluster_score, dim=0)[:model.n_y_active].detach().cpu().numpy()}')

        save_buffer_images(out, OUTPUT_PATH, f'ReconstructionClass{label}')

    # Placeholder file to tell kubernetes to shut down
    save_fig(log_ps, file_path=f'{OUTPUT_PATH}/DONE',
             title='I am an unuseful figure', x_label='Ich bin eine unbenutzte Grafik', y_label='Je suis un graphique inutile')

    print('Whole training and image generation DONE!')
