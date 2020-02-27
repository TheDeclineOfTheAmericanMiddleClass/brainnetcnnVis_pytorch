import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.dataset

from analysis.Define_model import YeoSex_BrainNetCNN, BrainNetCNN, HCPDataset
from preprocessing.Main_preproc import use_cuda, multi_outcome, lr, momentum, wd, predicted_outcome, architecture, \
    multiclass

# Defining train, test, validation sets
trainset = HCPDataset(mode="train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True,
                                          num_workers=2)  # TODO: look for changes in num_workers size

testset = HCPDataset(mode="test")
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

valset = HCPDataset(mode="valid")
valloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

# Creating the model
if predicted_outcome == 'sex' and architecture == 'yeo':
    net = YeoSex_BrainNetCNN(trainset.X)
else:
    net = BrainNetCNN(trainset.X)

# Putting the model on the GPU
if use_cuda:
    net = net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=[0]) # for multiple GPUS
    cudnn.benchmark = True

# check if model parameters are on GPU or no
next(net.parameters()).is_cuda

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)

if multiclass:  # TODO: implement more general way to do multiclass classification
    y_unique = trainset.Y.unique(sorted=True)
    y_unique_count = torch.stack([trainset.Y[:, i].sum() for i, y_u in enumerate(y_unique)]) / trainset.Y.__len__()
    y_unique_count = y_unique_count.float().cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=y_unique_count)

else:
    criterion = torch.nn.MSELoss()  # shows loss for each outcome


def train(epoch):  # training in mini batches
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            if not multi_outcome:
                inputs, targets = inputs.cuda(), targets.cuda().unsqueeze(1)  # unsqueezing for vstack
            else:
                inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        # inputs, targets = Variable(inputs), Variable(targets)  # variable deprecated in Pytorch 0.4.0

        outputs = net(inputs)
        loss = criterion(outputs, targets.squeeze().long())
        loss.backward()
        optimizer.step()

        # running_loss += loss.data.item()  # only predicting 1 feature
        running_loss += loss.data.mean(0)  # only predicting 1 feature

        if batch_idx % 10 == 9:  # print every 10 mini-batches
            print('Training loss: %.6f' % (running_loss / 10))
            running_loss = 0.0
        _, predicted = torch.max(outputs.data, 1, keepdim=True)  # TODO: ensure change to keepdim=True works

        # total += targets.size(0)

        # correct += predicted.eq(targets.data).cpu().sum()

    return running_loss / batch_idx


def test():
    global loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    running_loss = 0.0

    preds = []
    ytrue = []

    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            if not multi_outcome:
                inputs, targets = inputs.cuda(), targets.cuda().unsqueeze(1)  # unsqueezing for vstack
            else:
                inputs, targets = inputs.cuda(), targets.cuda()

            # with torch.no_grad():
            # inputs, targets = Variable(inputs), Variable(targets)

            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            # test_loss += loss.data.item()  # only predicting 1 feature
            test_loss += loss.data.mean(0)  # only predicting 1 feature

            preds.append(outputs.data.cpu().numpy())
            ytrue.append(targets.data.cpu().numpy())

            # # converting nan to largest float 32 number
            # preds.append(np.nan_to_num(outputs.data.cpu().numpy(), nan=np.finfo('float32').max))
            # ytrue.append(np.nan_to_num(targets.data.cpu().numpy(), nan=np.finfo('float32').max))

        # print statistics
        # running_loss += loss.data.item()
        running_loss += loss.data.mean(0)  # only predicting 1 feature

        # if batch_idx % 5 == 4:    # print every 5 mini-batches
        if batch_idx == len(valloader):  # just print for final batch
            print('Test loss: %.6f' % (running_loss / 5))
            running_loss = 0.0

        # _, predicted = torch.max(outputs.data, 1)
        # total += targets.size(0)
        # correct += predicted.eq(targets.data).cpu().sum()
    if not multi_outcome:
        return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
    else:
        return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

    # Save checkpoint.
    # acc = 100.*correct/total


### Weights initialization for the dense layers using He Uniform initialization
### He et al., http://arxiv.org/abs/1502.01852


# TODO: inspect why only the first dense layer dims are used for weight intiialization
def init_weights_he(m):
    # https://keras.io/initializers/#he_uniform
    print(m)
    if type(m) == torch.nn.Linear:
        fan_in = net.dense1.in_features
        print(f'In features for dense 1: {fan_in}')
        he_lim = np.sqrt(6 / fan_in)  # Note: fixed error in he limit calculation (Feb 10, 2020)
        print(f'he limit {he_lim}')
        m.weight.data.uniform_(-he_lim, he_lim)
        print(f'\nWeight initializations: {m.weight}')


def init_weights_XU(m):
    """Init weights per xavier uniform method"""
    print(m)
    if type(m) == torch.nn.Linear:
        fan_in = net.dense1.in_features
        fan_out = net.dense1.out_features
        print(f'In features for dense 1: {fan_in}')
        he_lim = np.sqrt(6 / fan_in + fan_out)  # Note: fixed error in he limit calculation (Feb 10, 2020)
        print(f'he limit {he_lim}')
        m.weight.data.uniform_(-he_lim, he_lim)
        print(f'\nWeight initializations: {m.weight}')
