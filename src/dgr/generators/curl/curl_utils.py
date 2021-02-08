"""Some common utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import torch
import torch.nn.functional as F


# MAKE SURE THAT ALL FUNCTION STILL LET THE GRADIENT TRAVEL DURING BACKPROP

def generate_gaussian(logits, sigma_nonlin, sigma_param):
    """Generate a Gaussian distribution given a selected parameterisation."""

    mu, sigma = torch.split(logits, logits.shape[1] // 2, dim=1)

    if sigma_nonlin == 'exp':
        sigma = torch.exp(sigma)
    elif sigma_nonlin == 'softplus':
        sigma = F.softplus(sigma)
    else:
        raise ValueError('Unknown sigma_nonlin {}'.format(sigma_nonlin))

    if sigma_param == 'var':
        sigma = torch.sqrt(sigma)
    elif sigma_param != 'std':
        raise ValueError('Unknown sigma_param {}'.format(sigma_param))

    return torch.distributions.normal.Normal(loc=mu, scale=sigma)


def construct_prior_probs(batch_size, n_y, n_y_active):
    """Construct the uniform prior probabilities.

    Args:
        batch_size: int, the size of the batch.
        n_y: int, the number of categorical cluster components.
        # TODO: Need to check how we will define n_y_active later
        n_y_active: tf.Variable, the number of components that are currently in use.

    Returns:
        Tensor representing the prior probability matrix, size of [batch_size, n_y].
    """
    probs = torch.ones((batch_size, n_y_active)) / n_y_active

    paddings1 = torch.stack([torch.tensor(0), torch.tensor(0)], dim=0)
    paddings2 = torch.stack(
        [torch.tensor(0), torch.tensor(n_y - n_y_active)], dim=0)
    paddings = list(torch.cat([paddings2, paddings1], dim=0).numpy())

    probs = F.pad(probs, paddings, value=1e-12)
    probs = probs.view(batch_size, n_y)
    logging.info('Prior shape: %s', str(probs.shape))

    return probs


def maybe_center_crop(layer, target_hw):
    """Center crop the layer to match a target shape."""
    l_height, l_width = layer.shape[1:3]
    t_height, t_width = target_hw
    assert t_height <= l_height and t_width <= l_width

    if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
        logging.warning(
            'It is impossible to center-crop [%d, %d] into [%d, %d].'
            ' Crop will be uneven.', t_height, t_width, l_height, l_width)

    border = int((l_height - t_height) / 2)
    x_0, x_1 = border, l_height - border
    border = int((l_width - t_width) / 2)
    y_0, y_1 = border, l_width - border
    layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
    return layer_cropped


def get_padding(kernel_size: List[int], input_size: List[int], stride: int) -> Tuple[int]:

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
