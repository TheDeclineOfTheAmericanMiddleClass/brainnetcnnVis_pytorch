import torch.nn
import torch.nn.functional as F
import torch.utils.data.dataset

from analysis.Load_model_data import *


# Definite E2E
class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


# BrainNetCNN Network
class BrainNetCNN(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 32, example, bias=True)  # TODO: change initial dim for multilayer
        self.e2econv2 = E2EBlock(32, 64, example, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)  # init
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 1)  # change final layer to number of predictions model should make

    def forward(self, x):
        out = F.dropout(F.leaky_relu(self.e2econv1(x), negative_slope=0.33), p=.5)
        out = F.dropout(F.leaky_relu(self.e2econv2(out), negative_slope=0.33), p=.5)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.relu(self.dense1(out)), p=0.5)
        out = F.dropout(F.relu(self.dense2(out)), p=0.5)
        out = F.relu(self.dense3(out))

        return out

class HCPDataset(torch.utils.data.Dataset):

    def __init__(self, mode="train", transform=False, class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): train = 90% Train, validation=10% Train, train+validation=100% train else test.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.transform = transform

        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        if self.mode == "train":
            x = X[train_ind]
            y = Y[train_ind]  # Y_train

        elif self.mode == "test":
            x = X[test_ind]
            y = Y[test_ind]  # Y_test

        elif mode == "valid":
            x = X[val_ind]
            y = Y[val_ind]

        else:
            x = X
            y = Y

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float64))
        # self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float64))

        print(self.mode, self.X.shape, (self.Y.shape))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample
