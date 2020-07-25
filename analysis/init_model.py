import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data.dataset

from utils.util_funcs import Bunch


def main(args):
    bunch = Bunch(args)

    # limiting CPU usage
    torch.set_num_threads(1)

    # Defining train, test, validation sets
    trainset = bunch.HCPDataset(mode="train")
    testset = bunch.HCPDataset(mode="test")
    valset = bunch.HCPDataset(mode="valid")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, pin_memory=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=False, num_workers=0)
    valloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, pin_memory=False, num_workers=0)

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
        net = net.cuda()
        cudnn.benchmark = True

    # ensure model parameters are on GPU
    assert next(net.parameters()).is_cuda, 'Parameters are not on the GPU !'

    if bunch.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=bunch.lr, momentum=bunch.momentum, nesterov=True,
                                    weight_decay=bunch.wd)
    elif bunch.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=bunch.lr, weight_decay=bunch.wd)
    else:
        raise KeyError(f'{bunch.optimizer} is not a valid optimizer. Please try again.')

    if bunch.multiclass:
        if bunch.num_classes == 2:
            criterion = nn.BCELoss(weight=torch.Tensor(bunch.y_weights)).cuda(
                bunch.device)  # balanced Binary Cross Entropy as loss function
        elif bunch.num_classes > 2:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor(bunch.y_weights)).cuda(bunch.device)

    else:
        criterion = torch.nn.MSELoss().cuda(bunch.device)  # shows loss for each outcome

    def train(net):  # training in mini batches

        net.train()
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            if bunch.use_cuda:
                if not bunch.multi_outcome and not bunch.multiclass:
                    # print('unsqueezing target for vstack...')
                    inputs, targets = inputs.cuda(), targets.cuda().unsqueeze(1)  # unsqueezing for vstack
                else:
                    # print('target left alone...')
                    inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            targets = targets.view(outputs.size())

            try:
                loss = criterion(input=outputs, target=targets)
            except RuntimeError:
                if bunch.multiclass:
                    loss = criterion(input=outputs,
                                     target=torch.argmax(targets.data, 1))

            loss.backward()

            # This line is used to prevent the vanishing / exploding gradient problem
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           max_norm=bunch.max_norm)  # TODO: see if max_norm size appropriate

            for p in net.parameters():
                p.data.add_(-bunch.lr, p.grad.data)

            optimizer.step()

            running_loss += loss.data.mean(0)  # only predicting 1 feature

            # TODO: see iff accumulated gradient speeds up training
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

    def test(net):
        global loss
        net.eval()
        test_loss = 0
        running_loss = 0.0

        preds = []
        ytrue = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if bunch.use_cuda:
                if not bunch.multi_outcome and not bunch.multiclass:
                    inputs, targets = inputs.cuda(), targets.cuda().unsqueeze(1)  # unsqueezing for vstack
                else:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = net(inputs)
                targets = targets.view(outputs.size())

                try:
                    loss = criterion(input=outputs, target=targets)
                except RuntimeError:
                    if bunch.multiclass:
                        loss = criterion(input=outputs,
                                         target=torch.argmax(targets.data, 1))

                test_loss += loss.data.mean(0)  # only predicting 1 feature

                preds.append(outputs.data.cpu().numpy())
                ytrue.append(targets.data.cpu().numpy())

            running_loss += loss.data.mean(0)  # only predicting 1 feature

            # print statistics
            if batch_idx == len(valloader):  # print for final batch
                print('Test loss: %.6f' % (running_loss / 5))
                running_loss = 0.0

        if not bunch.multi_outcome and not bunch.multiclass:
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

    return dict(train=train, test=test, net=net)


if __name__ == '__main__':
    main()
