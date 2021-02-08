import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

from classifier import DGR_CNN, CNN_Classifier

device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)


def loss_init_classifier(clf_buffer_size,
                         torch_loss,
                         clf_thresh,
                         clf_poor_data_buffer,
                         clf_poor_data_labels,
                         train_data,
                         image_key,
                         label_key,
                         batch_size,
                         num_buffer_clf_train_steps,
                         clf_type,
                         n_classes):
    """Initializes Classifier based on classification loss

    Args:
        clf_buffer_size: int, Buffer size needed to re init a new classifier 
        torch_loss: torch.Tensor, losses for the last seen batch
        clf_thresh: int, loss threshold for holding poorly classified data 
        clf_poor_data_buffer: List, Poorly classified data
        clf_poor_data_labels: List, Poorly classified labels
        train_data: dict, train data
        image_key: str, key to get the images from the train data
        label_key: str, key to get the labels from the train data
        batch_size: int, the batch size
        num_buffer_clf_train_steps: int, number of iteration for training the classifier on the unexplained data
        clf_type: str, type of the classifier
        n_classes: int, number of classes

    Returns:
        Whether we re_initialized a new classifier or not
        The new classifier or None
        The new optimizer or None
    """

    has_been_reinit = False
    classifier = None
    torch_optim = None

    clf_poor_inds = torch_loss > clf_thresh
    clf_poor_inds = clf_poor_inds.cpu()
    clf_poor_data_buffer.extend(
        train_data[image_key][clf_poor_inds])
    clf_poor_data_labels.extend(
        train_data[label_key][clf_poor_inds])

    n_poor_classification = len(clf_poor_data_buffer)

    if n_poor_classification >= clf_buffer_size:

        print(f'Classifier Init')

        # Cull to a multiple of batch_size (keep the later data samples).
        n_poor_batches = int(
            n_poor_classification / batch_size)
        clf_poor_data_buffer = clf_poor_data_buffer[-(
            n_poor_batches * batch_size):]
        clf_poor_data_labels = clf_poor_data_labels[-(
            n_poor_batches * batch_size):]

        classifier, torch_optim = init_classifier_and_optim(
            clf_type, n_classes)
        classifier.train()

        # Empty the buffers.
        clf_poor_data_buffer = []
        clf_poor_data_labels = []

        # Reset the threshold flag so we have a burn in before the next
        # component.
        has_been_reinit = True

    return has_been_reinit, classifier, torch_optim


def init_classifier_and_optim(clf_type, n_classes, image_channel_size=1):
    """Initializes a classifier and its optimizer"""

    if clf_type == 'dgr_clf':
        classifier = DGR_CNN(
            image_size=28, image_channel_size=image_channel_size, classes=n_classes, depth=5, channel_size=1024, reducing_layers=3).to(device)
    elif clf_type == 'generic_clf':
        classifier = CNN_Classifier(in_channel=image_channel_size, out_channels=[8, 16], kernel_sizes=[
            4, 4], cnn_activation=nn.ReLU, strides=[1, 1], lin_activation=nn.Softmax, n_classes=n_classes).to(device)
    torch_optim = torch.optim.Adam(
        classifier.parameters(), lr=0.01)

    return classifier, torch_optim


def classifier_pre_train(n_clf_pre_train_steps,
                         n_batches,
                         data_buffer,
                         data_labels,
                         batch_size,
                         classifier,
                         torch_optim,
                         ):
    for _ in range(n_clf_pre_train_steps):
        for bs in range(n_batches):
            x_batch = np.array(data_buffer[bs * batch_size:(bs + 1) *
                                           batch_size])
            label_batch = data_labels[bs * batch_size:(bs + 1) *
                                      batch_size]
            torch_train_input = torch.tensor(
                np.transpose(x_batch, (0, 3, 1, 2))).to(device)
            torch_train_label = torch.tensor(
                label_batch).to(device)
            torch_out = classifier(torch_train_input)
            torch_loss = F.nll_loss(
                F.log_softmax(torch_out), torch_train_label, reduction='none')
            mean_torch_loss = torch_loss.mean()
            torch_optim.zero_grad()
            mean_torch_loss.backward()
            torch_optim.step()
