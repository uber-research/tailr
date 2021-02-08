# Utilitary functions for the TAILR model
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from tensorboard import program
from typing import List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_same_padding(kernel_size: List[int], input_size: List[int], stride: int) -> Tuple[int]:
    """Generates the padding to simulate the 'same' padding from TF"""

    if input_size[2] % stride == 0:
        pad1 = max(kernel_size[3] - stride, 0)
    else:
        pad1 = max(kernel_size[3] - (input_size[2] % stride), 0)

    if input_size[3] % stride == 0:
        pad2 = max(kernel_size[2] - stride, 0)
    else:
        pad2 = max(kernel_size[2] - (input_size[3] % stride), 0)

    padding = [0, 0, 0, 0]
    if pad1 % 2 == 0:
        pad_val1 = pad1 // 2
        padding[0] = pad_val1
        padding[1] = pad_val1
    else:
        pad_val_start = pad1 // 2
        pad_val_end = pad1 - pad_val_start
        padding[0] = pad_val_start
        padding[1] = pad_val_end

    if pad2 % 2 == 0:
        pad_val2 = pad2 // 2
        padding[2] = pad_val2
        padding[3] = pad_val2
    else:
        pad_val_start = pad2 // 2
        pad_val_end = pad2 - pad_val_start
        padding[2] = pad_val_start
        padding[3] = pad_val_end

    return padding


def generate_gaussian(logits, sigma_nonlin, sigma_param):
    """Generate a Gaussian distribution given a selected parameterisation."""

    if isinstance(logits.shape[1], int):
        mu, sigma = torch.split(logits, logits.shape[1] // 2, dim=1)
    elif isinstance(logits.shape[1], torch.Tensor):
        mu, sigma = torch.split(logits, logits.shape[1].item() // 2, dim=1)
    else:
        TypeError('You\'ve used a weird type of size apparently')

    if sigma_nonlin == 'exp':
        sigma = torch.exp(sigma)
    elif sigma_nonlin == 'softplus':
        sigma = F.softplus(sigma)
    else:
        raise ValueError(f'Unknown sigma_nonlin {sigma_nonlin}')

    if sigma_param == 'var':
        sigma = torch.sqrt(sigma)
    elif sigma_param != 'std':
        raise ValueError(f'Unknown sigma_param {sigma_param}')

    return generate_gaussian_sample(mu, sigma), mu, sigma


def generate_gaussian_sample(mu, sigma):
    """Generates a sample from Gaussian parameters, sigma should already be passed through nonlinearity and parametrization (see generate_gaussian)"""

    eps = torch.randn(mu.size()).to(device)  # autograd False by default

    return mu + sigma * eps


def tile(a, dim, n_tile):
    """Acts as the tile function from tensorflow"""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)


def log_prob_elbo(x, model, cluster_prior=None):
    """Returns the components used in calculating the ELBO.

    Args:
      x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
        a flattened input.
      model: the tailr model.
      cluster_prior: Prior used for the KL for cluster scores, `Tensor` [B, n_y]
    Returns:
      `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
      `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
      `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
    """

    # Get info about cluster score size from the model
    n_y = model.n_y
    n_y_active = model.n_y_active

    # Get the hidden representation and the cluster score
    hiddens = model.shared_encoder(x)
    cluster_score = model.cluster_encoder(hiddens, n_y_active)

    # 1) Compute KL[q(y|x) || p(y)]
    if cluster_prior is None:
        cluster_prior = construct_prior_probs(x.shape[0], n_y, n_y_active)
    kl_y = kl_onehotcategorical(
        cluster_score, cluster_prior)  # [B], cluster_score

    # For the next two terms, we need to marginalise over all possible y.

    # First, construct every possible y indexing (as a one hot) and repeat it
    # for every element in the batch [n_y_active, B, n_y].
    # Note that the onehot have dimension of all y, while only the codes
    # corresponding to active components are instantiated
    bs, n_y = cluster_score.shape

    one_hot_tensor = F.one_hot(torch.arange(
        start=0, end=n_y_active), n_y)  # Get n_y_active rows
    one_y = one_hot_tensor.unsqueeze(1).type(torch.float).to(device)
    # Repeat for each element in a batch
    all_y = tile(one_y, dim=1, n_tile=x.shape[0])

    # 2) Compute KL[q(z|x,y) || p(z|y)] (for all possible y), and keep z's
    # around [n_y, B] and [n_y, B, n_z]
    kl_z_all = []
    z_all = []
    for y in all_y:
        kl_z, z = model._kl_and_z(hiddens, y)
        kl_z_all.append(kl_z)
        z_all.append(z)
    kl_z_all = torch.stack(kl_z_all, dim=0)
    z_all = torch.stack(z_all, dim=0)

    kl_z_all = kl_z_all.permute(
        *[i for i in reversed(range(kl_z_all.dim()))])

    # Now take the expectation over y (scale by q(y|x))
    y_probs = cluster_score[:, :n_y_active]  # [B, n_y_active]
    y_probs = y_probs / y_probs.sum(dim=1, keepdim=True)
    kl_z = torch.sum(y_probs * kl_z_all, dim=1)

    # 3) Evaluate logp and recon, i.e., log and mean of p(x|z,[y])
    # (conditioning on y only in the `multi` decoder_type case, when
    # train_supervised is True). Here we take the reconstruction from each
    # possible component y and take its log prob. [n_y, B, Ix, Iy, Iz] where Ix, Iy and Iz and for width, height and channel_dim respectively.
    log_p_x_all = []
    for i, val in enumerate(zip(z_all, all_y)):
        logits = model.latent_decoder(val[0], val[1])
        # replicate the log prob of bernoulli distributions
        log_p = - \
            F.binary_cross_entropy_with_logits(logits, x, reduction='none')
        log_p_x_all.append(log_p)
    log_p_x_all = torch.stack(log_p_x_all, dim=0)

    # Sum log probs over all dimensions apart from the first two (n_y, B),
    # i.e., over I. Use einsum to construct higher order multiplication.
    log_p_x_all = nn.Flatten(start_dim=2)(log_p_x_all)  # [n_y,B,I]
    # Note, this is E_{q(y|x)} [ log p(x | z, y)], i.e., we scale log_p_x_all
    # by q(y|x).
    log_p_x = torch.einsum('ij,jik->ik', y_probs, log_p_x_all)  # [B, I]
    log_p_x = torch.sum(log_p_x, dim=1)

    return kl_y, kl_z, log_p_x, cluster_score


def log_prob_elbo_sup(x, hiddens, cluster_score, y_label, n_y, n_y_active, model, reduce_op=torch.sum, cluster_prior=None):
    """Returns the components used in calculating the ELBO.

    Args:
      x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
        a flattened input.
      hiddens: features of the input computed with the SharedEncoder
      cluster_score: labels, `Tensor` of size `[B, I]` where `I` is the size of a
        flattened input.
      y_label: the onehot cluster score targets, Tensot [B, n_y]
      n_y: the maximum number of clusters.
      n_y_active: the number of active components.
      model: the tailr model.
      reduce_op: The op to use for reducing across non-batch dimensions.
        Typically either `torch.sum` or `torch.mean`.

    Returns:
      `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
      `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
      `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
    """

    # First, construct every possible y indexing (as a one hot) and repeat it
    # for every element in the batch [n_y_active, B, n_y].
    # Note that the onehot have dimension of all y, while only the codes
    # corresponding to active components are instantiated
    kl_z_sup, z = model._kl_and_z(hiddens, y_label)

    # Now take the expectation over y (scale by q(y|x))
    y_probs = cluster_score[:, :n_y_active]  # [B, n_y_active]
    y_probs = y_probs / y_probs.sum(dim=1, keepdim=True)
    y_logits = torch.log(y_probs)

    # 3) Evaluate logp and recon, i.e., log and mean of p(x|z,[y])
    # (conditioning on y only in the `multi` decoder_type case, when
    # train_supervised is True). Here we take the reconstruction from each
    # possible component y and take its log prob. [n_y, B, Ix, Iy, Iz] where Ix, Iy and Iz and for width, height and channel_dim respectively.

    logits = model.latent_decoder(z, y_label)
    # replicate the log prob of bernoulli distributions
    log_p_x_sup = - \
        F.binary_cross_entropy_with_logits(logits, x, reduction='none')

    # Sum log probs over all dimensions apart from the first two (n_y, B),
    # i.e., over I. Use einsum to construct higher order multiplication.
    log_p_x_sup = nn.Flatten(start_dim=2)(log_p_x_sup).squeeze()  # [n_y,B,I]
    # Note, this is E_{q(y|x)} [ log p(x | z, y)], i.e., we scale log_p_x_all
    # by q(y|x).

    log_p_x_sup = torch.sum(log_p_x_sup, dim=1)

    kl_y_sup = F.cross_entropy(
        input=y_logits,
        target=torch.argmax(y_label[:, :n_y_active], dim=1),
        reduction='none',
    )  # [B]

    return kl_y_sup, kl_z_sup, log_p_x_sup


def one_hot_loss(logits, target):
    """Computes the log prob of a target for a given OneHotCategorical dist

    Args:
        logits: logits of the OneHotCategorical dist, Tensor
        target: target probabilities, Tensor

    Returns:
        Log P(target | logits) 
    """
    p = target.max(-1)[1]
    p = p.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(p, logits)
    value = value[..., :1]

    return log_pmf.gather(-1, value).squeeze(-1)


def kl_onehotcategorical(cluster_score, prior, kly_over_batch=False):
    """Returns analytical or sampled KL div and the distribution q(y | x).

    Args:
    cluster_score: The cluster_scores, 2D `Tensor` of size `[B, n_y]`.
    prior: The cluster_score prior, 2D `Tensor` of size `[B, n_y]`.

    Returns:
    KL divergence 
    """
    q = cluster_score  # q(y|x)
    p = prior  # p(y)
    # Take the average proportions over whole batch then repeat it in each row
    # before computing the KL
    if kly_over_batch:
        probs = q * torch.ones_like(q).to(device)
        qmean = probs
        kl = qmean * (torch.log(qmean) - torch.log(p))

    else:
        kl = q * (torch.log(q) - torch.log(p))

        # Currently not used, will be moved to a testing area
        true_q = torch.distributions.one_hot_categorical.OneHotCategorical(
            probs=q)
        true_p = torch.distributions.one_hot_categorical.OneHotCategorical(
            probs=p)
        true_kl = torch.distributions.kl.kl_divergence(true_p, true_q)

    kl = torch.sum(kl, dim=1)

    return kl


def kl_gaussian(p_mu, p_sigma, q_mu, q_sigma):
    """KL divergence between two Gaussians P and Q. 
        note: here we are computing KL(P||Q)

    Args:
        p_mu: mean of the target distribution, Tensor [B, n_z]
        p_sigma: standard dev of the target distribution, Tensor [B, n_z]
        q_mu: mean of the optimized distribution, Tensor [B, n_z]
        q_sigma: standard dev of the optimized distribution, Tensor [B, n_z]

    Returns:
        KL(P||Q), Tensor [B]

    """
    q_sigma_mat = torch.diag_embed(q_sigma.pow(2))
    p_sigma_mat = torch.diag_embed(p_sigma.pow(2))

    q_sig_det = q_sigma_mat.det()
    p_sig_det = p_sigma_mat.det()

    log_det = (q_sig_det / p_sig_det).log()
    n = q_mu.shape[1]
    trace = torch.einsum(
        'bii->b', torch.matmul(q_sigma_mat.inverse(), p_sigma_mat))  # trace
    last_elem = torch.bmm(torch.bmm((q_mu - p_mu).unsqueeze(1),
                                    q_sigma_mat.inverse()), (q_mu - p_mu).unsqueeze(2)).squeeze()

    kl = 1/2 * (log_det - n + trace + last_elem)

    return kl


def construct_prior_probs(batch_size, n_y, n_y_active):
    """Construct the uniform prior probabilities.

    Args:
        batch_size: int, the size of the batch.
        n_y: int, the number of categorical cluster components.
        n_y_active: tf.Variable, the number of components that are currently in use.

    Returns:
        Tensor representing the prior probability matrix, size of [batch_size, n_y].
    """
    probs = (torch.ones((batch_size, n_y_active)) / n_y_active).to(device)

    paddings1 = torch.stack([torch.tensor(0), torch.tensor(0)], dim=0)
    paddings2 = torch.stack(
        [torch.tensor(0), torch.tensor(n_y - n_y_active)], dim=0)
    paddings = list(torch.cat([paddings2, paddings1], dim=0).numpy())

    probs = F.pad(probs, paddings, value=1e-12)
    probs = probs.view(batch_size, n_y)
    logging.info(f'Prior shape: {probs.shape}')

    probs = probs.to(device)

    return probs


def crop_layer(layer, target_hw):
    """Center crop the layer to match a target shape."""
    l_height, l_width = layer.shape[1:3]
    t_height, t_width = target_hw
    assert t_height <= l_height and t_width <= l_width

    if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
        logging.warning(
            f'It is impossible to center-crop [{t_height},{t_width}] into [{l_height}, {l_width}]. Crop will be uneven.')

    border = int((l_height - t_height) / 2)
    x_0, x_1 = border, l_height - border
    border = int((l_width - t_width) / 2)
    y_0, y_1 = border, l_width - border
    layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
    return layer_cropped


def launch_tb(output_dir):
    # Tensorboard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', output_dir, '--host', '0.0.0.0'])
    url = tb.launch()
    print(f'Tensorboard is available from {url}')


def binarize_fn(x):
    """Binarize a Bernoulli by rounding the probabilities.

    Args:
        x: tf tensor, input image.

    Returns:
        A tf tensor with the binarized image
    """
    return torch.gt(x, 0.5 * torch.ones_like(x)).type(torch.float32)
