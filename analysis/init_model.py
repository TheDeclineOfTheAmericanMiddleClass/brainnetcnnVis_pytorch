import torch.backends.cudnn as cudnn
import torch.utils.data.dataset

from analysis.define_models import *
from preprocessing.read_data import *

# Defining train, test, validation sets
trainset = HCPDataset(mode="train")
testset = HCPDataset(mode="test")
valset = HCPDataset(mode="valid")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)
valloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

# Creating the model
if predicted_outcome == 'sex' and architecture == 'yeo':
    net = YeoSex_BNCNN(trainset.X)
elif predicted_outcome == 'sex' and architecture == 'parvathy_v2':
    net = ParvathySex_BNCNN_v2byAdu(trainset.X)
elif predicted_outcome == 'sex' and architecture == 'parvathy_orig':
    net = ParvathySex_BNCNN_original(e2e=16, e2n=128, n2g=26, f_size=trainset.X.shape[3], dropout=.6)
else:
    net = BNCNN(trainset.X)

# Putting the model on the GPU
if use_cuda:
    net = net.cuda()
    cudnn.benchmark = True

# check if model parameters are on GPU or no
assert next(net.parameters()).is_cuda, 'Parameters are not on the GPU !'

# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

if multiclass and num_classes <= 2:
    y_unique = trainset.Y.unique(sorted=True).numpy()
    y_propor = (torch.stack([trainset.Y.T[i].sum() for i in range(len(y_unique))]) / trainset.Y.__len__()).numpy()
    y_propor = dict((int(key), y_propor[int(key)]) for key in y_unique)

    if one_hot:
        criterion = nn.BCELoss().cuda(device)  # balanced Binary Cross Entropy as loss function
    else:
        # criterion = torch.nn.BCELoss().cuda(device)
        criterion = nn.CrossEntropyLoss().cuda(device)

else:
    criterion = torch.nn.MSELoss().cuda(device)  # shows loss for each outcome


def train():  # training in mini batches
    net.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            if not multi_outcome and multiclass:
                # print('unsqueezing target for vstack...')
                inputs, targets = inputs.cuda(), targets.cuda().unsqueeze(1)  # unsqueezing for vstack
            else:
                # print('target left alone...')
                inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)

        # TODO: remove all traces of rounding before loss is calcualted
        # if multiclass and num_classes == 2:
        #     outputs = torch.round(outputs)

        targets = targets.view(outputs.size())

        try:
            loss = criterion(input=outputs, target=targets)
        except RuntimeError:
            if multiclass:
                loss = criterion(input=torch.argmax(outputs.data, 1),
                                 target=torch.argmax(targets.data, 1),
                                 weight=[y_propor[x] for x in targets])

        loss.backward()
        optimizer.step()

        running_loss += loss.data.mean(0)  # only predicting 1 feature

        if batch_idx % 10 == 9:  # print every 10 mini-batches
            print('Training loss: %.6f' % (running_loss / 10))
            running_loss = 0.0
        _, predicted = torch.max(outputs.data, 1, keepdim=True)

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
            if not multi_outcome and not multiclass:
                inputs, targets = inputs.cuda(), targets.cuda().unsqueeze(1)  # unsqueezing for vstack
            else:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            # TODO: remove all traces of rounding before loss is calcualted
            # if multiclass and num_classes == 2:  # for binary classification
            #     outputs = torch.round(outputs)

            targets = targets.view(outputs.size())

            try:
                loss = criterion(input=outputs, target=targets)
            except RuntimeError:
                if multiclass:
                    loss = criterion(input=torch.argmax(outputs.data, 1),
                                     target=torch.argmax(targets.data, 1),
                                     weight=[y_propor[x] for x in targets])

            test_loss += loss.data.mean(0)  # only predicting 1 feature

            preds.append(outputs.data.cpu().numpy())
            ytrue.append(targets.data.cpu().numpy())

        running_loss += loss.data.mean(0)  # only predicting 1 feature

        # print statistics
        if batch_idx == len(valloader):  # print for final batch
            print('Test loss: %.6f' % (running_loss / 5))
            running_loss = 0.0

    if not multi_outcome or multiclass:
        # print('y_true left well enough alone...')
        return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
    else:
        # print('squeezing y_true...')
        return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx


# Following function are only applied to linear layers
def init_weights_he(m):
    """ Weights initialization for the dense layers using He Uniform initialization
     He et al., http://arxiv.org/abs/1502.01852
    https://keras.io/initializers/#he_uniform
"""
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
        print(f'In/out features for dense 1: {fan_in}/{fan_out}')
        he_lim = np.sqrt(6 / fan_in + fan_out)  # Note: fixed error in he limit calculation (Feb 10, 2020)
        print(f'he limit {he_lim}')
        m.weight.data.uniform_(-he_lim, he_lim)
        print(f'\nWeight initializations: {m.weight}')
