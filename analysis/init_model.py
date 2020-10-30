import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data.dataset
from torch.autograd import Variable

from utils.util_funcs import Bunch


def main(args):
    bunch = Bunch(args)

    # limiting CPU usage
    torch.set_num_threads(1)

    # Defining train, test, validation sets
    trainset = bunch.HCPDataset(mode="train")
    testset = bunch.HCPDataset(mode="test")
    valset = bunch.HCPDataset(mode="valid")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, pin_memory=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=False, num_workers=1)
    valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False, pin_memory=False, num_workers=1)

    # Creating the model
    if bunch.predicted_outcome == ['Gender'] and bunch.architecture == 'yeo_sex':
        net = bunch.YeoSex_BNCNN(trainset.X)
    elif bunch.predicted_outcome == ['Gender'] and bunch.architecture == 'parvathy_v2':
        net = bunch.ParvathySex_BNCNN_v2byAdu(trainset.X)
    # elif predicted_outcome == 'Gender' and architecture == 'parvathy_orig':
    #     net = ParvathySex_BNCNN_original(e2e=16, e2n=128, n2g=26, f_size=trainset.X.shape[3], dropout=.6)
    elif bunch.architecture == 'kawahara':
        net = bunch.Kawahara_BNCNN(trainset.X)
    elif bunch.architecture == 'usama':
        net = bunch.Usama_BNCNN(trainset.X)
    elif bunch.architecture == 'FC90Net':
        net = bunch.FC90Net_YeoSex(trainset.X)
    elif bunch.architecture == 'yeo_58':
        net = bunch.Yeo58behaviors_BNCNN(trainset.X)
    else:
        print(
            f'"{bunch.architecture}" architecture not available for outcome(s) {", ".join(bunch.predicted_outcome)}. Using default \'usama\' architecture...\n')
        net = bunch.Usama_BNCNN(trainset.X)

    # Putting the model on the GPU
    if bunch.use_cuda:
        net = net.to(bunch.device)
        cudnn.benchmark = True

        # ensure model parameters are on GPU
        assert next(net.parameters()).is_cuda, 'Parameters are not on the GPU !'

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

    # net.apply(init_weights_he)

    if bunch.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=bunch.lr, momentum=bunch.momentum, nesterov=True,
                                    weight_decay=bunch.wd)
    elif bunch.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=bunch.lr, weight_decay=bunch.wd)
    else:
        raise KeyError(f'{bunch.optimizer} is not a valid optimizer. Please try again.')

    # # defining loss functions
    if bunch.multiclass:
        if bunch.num_classes == 2:
            criterion = nn.BCELoss(weight=torch.Tensor(bunch.y_weights).cuda())  # balanced Binary Cross Entropy
        elif bunch.num_classes > 2:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor(bunch.y_weights).cuda())
    else:
        criterion = torch.nn.MSELoss()

    def train():  # training in mini batches
        net.train()
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            if bunch.use_cuda:
                if not bunch.multi_outcome and not bunch.multiclass:
                    # print('unsqueezing target for vstack...')
                    inputs, targets = inputs.to(bunch.device), targets.to(bunch.device).unsqueeze(
                        1)  # unsqueezing for vstack
                else:
                    # print('target left alone...')
                    inputs, targets = inputs.to(bunch.device), targets.to(bunch.device)

            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = net(inputs)
            targets = targets.view(outputs.size())

            if bunch.multiclass and bunch.num_classes > 2:  # targets is encoded as one-hot by CrossEntropyLoss
                loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
            else:
                loss = criterion(input=outputs, target=targets)

            loss.backward()

            # print('\ngradient after backward: ')
            # for name, param in net.named_parameters(): # CONFIRMED! gradient issue
            #     print(name, param.grad.abs().sum())

            # TODO: see if max_norm size appropriate
            # prevents a vanishing / exploding gradient problem
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=bunch.max_norm)

            for p in net.parameters():
                p.data.add_(-bunch.lr, p.grad.data)

            optimizer.step()

            running_loss += loss.data.mean(0)  # only predicting 1 feature

            # TODO: see if accumulated gradient speeds up training
            # # 16 accumulated gradient steps
            # scaled_loss = 0
            # for accumulated_step_i in range(16):
            #     outputs = net(inputs)
            #     targets = targets.view(outputs.size())
            #     loss = criterion(input=outputs, target=targets)
            #     loss.backward()
            #     scaled_loss += loss.data.mean(0)
            #
            # # update weights after 8 steps. effective batch = 8*16
            # optimizer.step()
            #
            # # loss is now scaled up by the number of accumulated batches
            # actual_loss = scaled_loss / 16
            # running_loss += actual_loss

            preds.append(outputs.data.cpu().numpy())
            ytrue.append(targets.data.cpu().numpy())

            if batch_idx % 10 == 9:  # print every 10 mini-batches
                print('Training loss: %.6f' % (running_loss / 10))
                running_loss = 0.0
            _, predicted = torch.max(outputs.data, 1, keepdim=True)

        # return running_loss / batch_idx

        if not bunch.multi_outcome and not bunch.multiclass:
            # print('y_true left well enough alone...')
            return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
        else:
            # print('squeezing y_true...')
            return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

    def test():
        global loss
        net.eval()
        test_loss = 0
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if bunch.use_cuda:
                if not bunch.multi_outcome and not bunch.multiclass:
                    # print('unsqueezing target for vstack...')
                    inputs, targets = inputs.to(bunch.device), targets.to(bunch.device).unsqueeze(
                        1)  # unsqueezing for vstack
                else:
                    # print('target left alone...')
                    inputs, targets = inputs.to(bunch.device), targets.to(bunch.device)

            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)

                outputs = net(inputs)
                targets = targets.view(outputs.size())

                if bunch.multiclass and bunch.num_classes > 2:  # targets is encoded as one-hot by CrossEntropyLoss
                    loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                else:
                    loss = criterion(input=outputs, target=targets)

                test_loss += loss.data.mean(0)  # only predicting 1 feature

                preds.append(outputs.data.cpu().numpy())
                ytrue.append(targets.data.cpu().numpy())

            running_loss += loss.data.mean(0)  # only predicting 1 feature

            # print statistics
            if batch_idx == len(testloader) - 1:  # print for final batch
                print('\nTest loss: %.6f' % (running_loss / len(testloader)))
                running_loss = 0.0

        if not bunch.multi_outcome and not bunch.multiclass:
            # print('y_true left well enough alone...')
            return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
        else:
            # print('squeezing y_true...')
            return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

    def val():
        net.eval()
        test_loss = 0
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(valloader):
            if bunch.use_cuda:
                if not bunch.multi_outcome and not bunch.multiclass:
                    # print('unsqueezing target for vstack...')
                    inputs, targets = inputs.to(bunch.device), targets.to(bunch.device).unsqueeze(
                        1)  # unsqueezing for vstack
                else:
                    # print('target left alone...')
                    inputs, targets = inputs.to(bunch.device), targets.to(bunch.device)

            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)

                outputs = net(inputs)
                targets = targets.view(outputs.size())

                if bunch.multiclass and bunch.num_classes > 2:  # targets is encoded as one-hot by CrossEntropyLoss
                    loss = criterion(input=outputs, target=torch.argmax(targets.data, 1))
                else:
                    loss = criterion(input=outputs, target=targets)

                test_loss += loss.data.mean(0)  # only predicting 1 feature

                preds.append(outputs.data.cpu().numpy())
                ytrue.append(targets.data.cpu().numpy())

            running_loss += loss.data.mean(0)  # only predicting 1 feature

            # print statistics
            if batch_idx == len(valloader) - 1:  # print for final batch # TODO: fix batching
                print('Val loss: %.6f' % (running_loss / len(valloader)))
                running_loss = 0.0

        if not bunch.multi_outcome and not bunch.multiclass:
            # print('y_true left well enough alone...')
            return np.vstack(preds), np.vstack(ytrue), running_loss / batch_idx
        else:
            # print('squeezing y_true...')
            return np.vstack(preds), np.vstack(ytrue).squeeze(), running_loss / batch_idx

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

    return dict(train=train, test=test, val=val, net=net,
                valloader=valloader, testloader=testloader, criterion=criterion)

if __name__ == '__main__':
    main()
