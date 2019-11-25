#!/usr/bin/env python
# coding: utf-8

# In[27]:


import torch
import numpy as np

input = np.array(
    [[[
        [1, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1],
        [0.5, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
    ]
    ],
        [[
            [1, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1],
            [0.5, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
        ]
        ]
    ]
)

kernel = np.transpose(np.array(
    [[[
        [1, 1, 1, 1, 1, 1.0],
        [1, 1, 1, 1, 1, 1]
    ]]]), [0, 1, 2, 3])

# In[42]:


# One image of one feature map 6x6
# The kernel has 1 feature map out, 1 feature map in, 2 vectors of size 6
input.shape, kernel.shape

# In[43]:


input_torch = torch.FloatTensor(input)
kernel_torch = torch.FloatTensor(kernel)

# In[44]:


import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable

cnn = torch.nn.Conv2d(1, 1, (1, 6), bias=False)
cnn.weight.data.copy_(kernel_torch[:, :, 0, :])
a = cnn.forward(Variable(input_torch))
print(a.size())

# In[56]:


cnn2 = torch.nn.Conv2d(1, 1, (6, 1), bias=False)
cnn2.weight.data.copy_(kernel_torch[:, :, 0, :])

b = cnn2.forward(Variable(input_torch))
print(b.size())

# In[14]:


a

# In[15]:


torch.cat([a] * 6, 3)

# In[16]:


b

# In[17]:


torch.cat([b] * 6, 2)

# In[ ]:


torch.cat([a] * 6, 3) + torch.cat([b] * 6, 2)


# In[7]:


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=True):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


# In[13]:


block = E2EBlock(1, 1, input_torch, False)
block(Variable(input_torch))

# BrainNetCNN Network for fitting Gold-MSI on LSD dataset

# In[89]:


"""" KERAS MODEL 
model.add(E2E_conv(2,32,(2,n_feat),kernel_regularizer=reg,input_shape=(n_feat,n_feat,1),input_dtype='float32',data_format="channels_last"))
print("First layer output shape :"+str(model.output_shape))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(E2E_conv(2,32,(2,n_feat),kernel_regularizer=reg,data_format="channels_last"))
print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(64,(1,n_feat),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(256,(n_feat,1),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))


model.add(Dense(128,kernel_regularizer=reg,kernel_initializer=kernel_init))
#print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(30,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(2,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.33))
"""


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = E2EBlock(1, 32, example)
        self.e2econv2 = E2EBlock(32, 64, example)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out


# In[134]:


net = BrainNetCNN(input_torch)
net(Variable(input_torch))

# In[152]:


input_torch.size()

# refs : carlos , voir [ici](https://github.com/brain-bzh/MCNN/blob/master/proposed/pines_aux.py) et [ici](https://github.com/brain-bzh/MCNN/blob/master/proposed/cifar.py)
# 

# Loader for GoldMSI-LSD77 dataset

# In[161]:


behavdir = "/Users/nicolasfarrugia/Documents/recherche/git/Gold-MSI-LSD77/behav"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os

import torch.utils.data.dataset


class GoldMSI_LSD_Dataset(torch.utils.data.Dataset):

    def __init__(self, directory=behavdir, mode="train", transform=False, class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): train = 90% Train, validation=10% Train, train+validation=100% train else test.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = directory
        self.mode = mode
        self.transform = transform

        x = np.load(os.path.join(directory, "X_y_lsd77_static_tangent.npz"))['X']
        y_all = np.load(os.path.join(directory, "X_y_lsd77_static_tangent.npz"))['y']
        y_2 = y_all[:, [3, 4]]
        y = normalize(y_2, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

        if self.mode == "train":

            x = X_train
            y = y_train

        elif self.mode == "validation":
            x = X_test
            y = y_test
        elif mode == "train+validation":
            x = x
            y = y
        else:
            x = x
            y = y

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        # self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float32))

        print(self.mode, self.X.shape, (self.Y.shape))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


# In[162]:


trainset = GoldMSI_LSD_Dataset(mode="train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=14, shuffle=True, num_workers=1)

testset = GoldMSI_LSD_Dataset(mode="validation")
testloader = torch.utils.data.DataLoader(testset, batch_size=14, shuffle=False, num_workers=1)

# Training

# In[223]:


net = BrainNetCNN(trainset.X)

momentum = 0.9
lr = 0.01
# wd = 0.0005 ## Decay for L2 regularization
wd = 0

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)


# In[224]:


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if batch_idx % 10 == 9:  # print every 10 mini-batches
            print('Training loss: %.6f' % (running_loss / 10))
            running_loss = 0.0
        # _, predicted = torch.max(outputs.data, 1)

        # total += targets.size(0)

        # correct += predicted.eq(targets.data).cpu().sum()


def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0

    preds = []

    for batch_idx, (inputs, targets) in enumerate(testloader):

        # if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            preds.append(outputs.numpy())

        # print statistics
        running_loss += loss.item()
        if batch_idx % 5 == 4:  # print every 5 mini-batches
            print('Test loss: %.6f' % (running_loss / 5))
            running_loss = 0.0

        # _, predicted = torch.max(outputs.data, 1)
        # total += targets.size(0)
        # correct += predicted.eq(targets.data).cpu().sum()

    return np.vstack(preds)
    # Save checkpoint.
    # acc = 100.*correct/total


# Run Epochs of training and testing 

# In[225]:


from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr

nbepochs = 100

y_true = testset.Y.numpy()

for epoch in range(nbepochs):
    train(epoch)
    preds = test()
    print("Epoch %d" % epoch)
    mae_1 = 100 * mae(preds[:, 0], y_true[:, 0])
    pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
    print("Test Set : MAE for Engagement : %0.2f %%" % (mae_1))
    print("Test Set : pearson R for Engagement : %0.2f, p = %0.2f" % (pears_1[0], pears_1[1]))

    mae_2 = 100 * mae(preds[:, 1], y_true[:, 1])
    pears_2 = pearsonr(preds[:, 1], y_true[:, 1])
    print("Test Set : MAE for Training : %0.2f %%" % (mae_2))
    print("Test Set : pearson R for Training : %0.2f, p = %0.2f" % (pears_2[0], pears_2[1]))

# Calculate Mean Absolute Error on Test Set 

# In[226]:


from sklearn.metrics import mean_absolute_error as mae
from scipy.stats import pearsonr

y_true = testset.Y.numpy()

mae_1 = 100 * mae(preds[:, 0], y_true[:, 0])
pears_1 = pearsonr(preds[:, 0], y_true[:, 0])
print("Test Set : MAE for Engagement : %0.2f %%" % (mae_1))
print("Test Set : pearson R for Engagement : %0.2f, p = %0.2f" % (pears_1[0], pears_1[1]))

mae_2 = 100 * mae(preds[:, 1], y_true[:, 1])
pears_2 = pearsonr(preds[:, 1], y_true[:, 1])
print("Test Set : MAE for Training : %0.2f %%" % (mae_2))
print("Test Set : pearson R for Training : %0.2f, p = %0.2f" % (pears_2[0], pears_2[1]))

# In[ ]:
