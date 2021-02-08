import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(device)


def eval_classifier(sess, classifier, clf_test_data, image_key, label_key):
    """Evaluate the Classifier with test data"""
    classifier.eval()

    valid_acc_dict = {}
    valid_loss_dict = {}

    for i, test_data in enumerate(clf_test_data):
        test_data_array = sess.run(test_data)

        test_input_array = torch.tensor(np.transpose(
            test_data_array[image_key], (0, 3, 1, 2))).to(device)
        test_label_array = torch.tensor(
            test_data_array[label_key]).to(device)

        test_pred = classifier(test_input_array)
        test_loss = F.nll_loss(
            F.log_softmax(test_pred), test_label_array)
        test_loss = test_loss.detach().cpu().numpy()

        valid_acc_dict[f'task_{test_data_array[label_key][0]}'] = (F.log_softmax(test_pred).argmax(
            dim=1) == test_label_array).sum().item() / test_label_array.shape[0]
        valid_loss_dict[f'task_{test_data_array[label_key][0]}'] = test_loss

    classifier.train()

    return valid_acc_dict, valid_loss_dict
