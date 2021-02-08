import torch
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import ConcatDataset
from data import get_dataset

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

BATCH_SIZE = 128
EPOCH = 5


def train():
    train_ds = get_dataset(name='mnist')
    test_ds = get_dataset(name='mnist', train=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)
    pbar = tqdm.tqdm(total=len(train_loader))

    my_model = TAILR_CLF(image_size=28, image_channel_size=1, classes=10,
                         depth=5, channel_size=1024, reducing_layers=3).to(device)
    my_model.load_state_dict(torch.load(
        '/tailr_project/src/tailr_tf/oracle_classifier/oracle_classifier.pth'))
    # my_model = MOD_TAILR_CLF(image_size=28, image_channel_size=1, classes=10, depth=3, channel_size=32).to(device)
    # my_model = CNN().to(device)
    optim = torch.optim.Adam(
        my_model.parameters(), lr=0.0001, weight_decay=1e-05, betas=(.5, .9))
    # optim = torch.optim.Adam(
    #    my_model.parameters(), lr=0.003)
    # optim = torch.optim.SGD(
    #     my_model.parameters(), lr=0.0001)
    my_model.train()

    for e in range(EPOCH):
        all_train_acc = list()
        all_train_loss = list()
        for _, batch in enumerate(train_loader):

            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()

            pred = my_model(x)
            loss = F.cross_entropy(pred, y, reduction='mean')
            loss.backward()
            optim.step()

            acc = get_acc(torch.argmax(pred, dim=1), y)
            all_train_acc.append(acc)
            all_train_loss.append(loss.detach().cpu())

            pbar.update(1)
            pbar.set_description(
                desc=f'Epoch {e+1}/{EPOCH} | Train Loss: {np.mean(all_train_loss):.5f}')

        all_test_acc = list()
        for _, batch in enumerate(test_loader):

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = my_model(x)
            acc = get_acc(torch.argmax(pred, dim=1), y)
            all_test_acc.append(acc)

        pbar.set_description(
            desc=f'Epoch {e+1}/{EPOCH} | Train Acc: {np.mean(all_train_acc):.5f} | Test Acc: {np.mean(all_test_acc):.5f}')
        all_test_acc = []
        all_train_acc = []
        print('\n')
        pbar.reset()

    torch.save(my_model.cpu().state_dict(
    ), '/tailr_project/src/tailr_tf/oracle_classifier/oracle_classifier.pth')


def get_acc(pred, ground_truth, i=0):
    return (pred == ground_truth).sum().item() / len(pred)


class TAILR_CLF(nn.Module):

    def __init__(self,
                 image_size,
                 image_channel_size, classes,
                 depth, channel_size, reducing_layers=3):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.classes = classes
        self.depth = depth
        self.channel_size = channel_size
        self.reducing_layers = reducing_layers

        # layers
        self.layers = nn.ModuleList([nn.Conv2d(
            self.image_channel_size, self.channel_size//(2**(depth-2)),
            3, 1, 1
        )])

        for i in range(self.depth-2):
            previous_conv = [
                l for l in self.layers if
                isinstance(l, nn.Conv2d)
            ][-1]
            self.layers.append(nn.Conv2d(
                previous_conv.out_channels,
                previous_conv.out_channels * 2,
                3, 1 if i >= reducing_layers else 2, 1
            ))
            self.layers.append(nn.BatchNorm2d(previous_conv.out_channels * 2))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Flatten(start_dim=1))

        self.out = nn.Linear(
            16384,
            self.classes
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class MOD_TAILR_CLF(nn.Module):

    def __init__(self,
                 image_size,
                 image_channel_size, classes,
                 depth, channel_size):
        # configurations
        super().__init__()
        self.image_channel_size = image_channel_size
        self.classes = classes
        self.depth = depth
        self.channel_size = channel_size

        # layers
        self.layers = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1
        )])

        self.layers.append(nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1
        ))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2, 2))
        self.layers.append(nn.Dropout2d(p=.5))
        # self.layers.append(nn.BatchNorm2d(previous_conv.out_channels * 2))

        self.layers.append(nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1
        ))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout2d(p=.5))
        self.layers.append(nn.MaxPool2d(2, 2))
        # self.layers.append(nn.BatchNorm2d(previous_conv.out_channels * 2))

        self.layers.append(nn.Flatten(start_dim=1))
        self.layers.append(nn.Linear(
            3200,
            128
        ))

        self.out = nn.Linear(
            128,
            self.classes
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):

        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)

        return out


if __name__ == "__main__":
    train()
