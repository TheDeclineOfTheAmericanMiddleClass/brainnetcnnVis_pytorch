import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data.dataset

from analysis.load_model_data import *


class HCPDataset(torch.utils.data.Dataset):

    def __init__(self, mode="train", transform=False, class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): train = 75% Train, validation=15% Train, train+validation=100% train else test.
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

        self.X = torch.FloatTensor(np.expand_dims(x, 1))  # removed .astype(np.float64)
        self.Y = torch.FloatTensor(y)

        print(self.mode, self.X.shape, (self.Y.shape))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


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


# BrainNetCNN edge to edge layer
class Edge2Edge(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Edge, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

    # implemented by two conv2d with line filter
    def forward(self, x):
        size = x.size()
        row = self.row_conv(x)
        col = self.col_conv(x)
        row_ex = row.expand(size[0], self.filters, self.dim, self.dim)
        col_ex = col.expand(size[0], self.filters, self.dim, self.dim)
        return row_ex + col_ex


# BrainNetCNN edge to node layer
class Edge2Node(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Edge2Node, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.row_conv = nn.Conv2d(channel, filters, (1, dim))
        self.col_conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        row = self.row_conv(x)
        col = self.col_conv(x)
        return row + col.permute(0, 1, 3, 2)


# BrainNetCNN node to graph layer
class Node2Graph(nn.Module):
    def __init__(self, channel, dim, filters):
        super(Node2Graph, self).__init__()
        self.channel = channel
        self.dim = dim
        self.filters = filters
        self.conv = nn.Conv2d(channel, filters, (dim, 1))

    def forward(self, x):
        return self.conv(x)


# Parvathy's self-written script
class ParvathySex_BNCNN_original(nn.Module):
    def __init__(self, e2e, e2n, n2g, f_size, dropout):
        super(ParvathySex_BNCNN_original, self).__init__()
        self.n2g_filter = n2g
        self.e2e = Edge2Edge(1, f_size, e2e)
        self.e2n = Edge2Node(e2e, f_size, e2n)
        self.dropout = nn.Dropout(p=dropout)
        self.n2g = Node2Graph(e2n, f_size, n2g)

        if one_hot:
            self.fc = nn.Linear(n2g, num_classes)
        else:
            self.fc = nn.Linear(n2g, 1)
        self.BatchNorm = nn.BatchNorm1d(n2g)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.e2e(x)
        x = self.dropout(x)
        x = self.e2n(x)
        x = self.dropout(x)
        x = self.n2g(x)
        x = self.dropout(x)
        x = x.view(-1, self.n2g_filter)
        x = self.fc(self.BatchNorm(x))

        return x


# Adu's self-written script, using parameters of Parvathy's netowkr
class ParvathySex_BNCNN_v2byAdu(torch.nn.Module):
    def __init__(self, example):  # removed num_classes=10
        super(ParvathySex_BNCNN_v2byAdu, self).__init__()
        print('\nInitializing BNCNN: Parvathy_Sex Architecture')
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 16, example, bias=True)  # TODO: change initial dim for multilayer
        self.E2N = torch.nn.Conv2d(16, 128, (1, self.d))
        self.N2G = torch.nn.Conv2d(128, 26, (self.d, 1))
        if one_hot:
            self.dense1 = torch.nn.Linear(26, num_classes)
        else:
            self.dense1 = torch.nn.Linear(26, 1)

        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(self.e2econv1(x), p=0.6)
        out = F.dropout(self.E2N(out), p=0.6)
        out = F.dropout(self.N2G(out), p=0.6)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.dense1(out))  # adding sigmoid for binary sex

        return out

    # def predict(self, x):
    #     """This function takes an input and predicts the class, (0 or 1)"""
    #     # Apply softmax to output.
    #     pred = F.softmax(self.forward(x))
    #     ans = []
    #     # Pick the class with maximum weight
    #     for t in pred:
    #         if t[0] > t[1]:
    #             ans.append(0)
    #         else:
    #             ans.append(1)
    #     return torch.tensor(ans)


# Yeo-version BNCNN for Multiclass Sex Classification (output = num_classes)
class YeoSex_BNCNN(torch.nn.Module):
    def __init__(self, example):  # removed num_classes=10
        super(YeoSex_BNCNN, self).__init__()
        print('\nInitializing BNCNN: Yeo_Sex Architecture...')
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 38, example, bias=True)  # TODO: change initial dim for multilayer
        self.E2N = torch.nn.Conv2d(38, 58, (1, self.d))
        self.N2G = torch.nn.Conv2d(58, 7, (self.d, 1))
        if one_hot:
            self.dense1 = torch.nn.Linear(7, num_classes)
        else:
            self.dense1 = torch.nn.Linear(7, 1)


        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        print('BNCNN instance initialized.\n')

    def forward(self, x):
        out = F.dropout(self.e2econv1(x), p=0.463)
        out = F.dropout(self.E2N(out), p=0.463)
        out = F.dropout(self.N2G(out), p=0.463)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        out = torch.sigmoid(out)  # adding sigmoid for binary sex

        return out

    # def predict(self, x):
    #     """This function takes an input and predicts the class, (0 or 1)"""
    #     # Apply softmax to output.
    #     pred = F.softmax(self.forward(x))
    #     ans = []
    #     # Pick the class with maximum weight
    #     for t in pred:
    #         if t[0] > t[1]:
    #             ans.append(0)
    #         else:
    #             ans.append(1)
    #     return torch.tensor(ans)


# BrainNetCNN Network
class BNCNN(torch.nn.Module):
    def __init__(self, example):  # removed num_classes=10
        super(BNCNN, self).__init__()
        print('\nInitializing BNCNN: Usama Architecture')
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 32, example, bias=True)  # TODO: change initial dim for multilayer
        self.e2econv2 = E2EBlock(32, 64, example, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)  # init
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, num_outcome)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


class FNN(nn.Module):

    def __init__(self, input_size, n_l1, n_l2, output_size, drop_out):
        super(FNN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(input_size, n_l1),
            nn.Sigmoid(),
            nn.BatchNorm1d(n_l1),

        )

        self.fc2 = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(n_l1, n_l2),

            nn.Sigmoid(),
            nn.BatchNorm1d(n_l2),

        )

        self.fc3 = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(n_l2, 1),

        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
