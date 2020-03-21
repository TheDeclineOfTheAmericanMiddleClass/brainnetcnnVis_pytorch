import numpy as np

losses_train = []
losses_test = []
accs_test = []
maes_test = []
pears_test = []
pvals_test = []
rundate = []

def main():
    # Adjust to load desired model
    loadpath = input('Filepath of model/stats (without _model.pt or _stats.npz) : ')

    # model = torch.load(f'{loadpath}_model.pt')
    # model.eval()

    modelnpz = np.load(f'{loadpath}_stats.npz', allow_pickle=True)

    losses_train.extend(modelnpz.f.train_losses)
    losses_test.extend(modelnpz.f.test_losses)
    rundate.extend(modelnpz.f.rundate)
    maes_test.extend(modelnpz.f.mae_eng)
    pears_test.extend(modelnpz.f.pears_eng)
    accs_test.extend(modelnpz.f.acc_eng)


# def main():
#     # Adjust to load desired model
#     loadpath = input('Filepath of model/stats (without _model.pt or _stats.npz : ')
#
#     # model = torch.load(f'loadpath}_model.pt')
#     # model.eval()
#
#     modelnpz = np.load(f'{loadpath}_stats.npz')
#
#     allloss_train = modelnpz.f.train_losses
#     allloss_test = modelnpz.f.test_losses
#     rundate = modelnpz.f.rundate
#
#     try:
#         allmae_test = modelnpz.f.mae_eng
#         allpears_test = modelnpz.f.pears_eng
#     except ValueError:
#         pass
#
#     try:
#         allacc_test = modelnpz.f.acc_eng
#     except ValueError:
#         pass

if __name__ == '__main__':
    main()
