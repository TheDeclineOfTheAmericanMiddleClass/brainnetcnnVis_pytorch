import numpy as np

savepath = 'models/BNCNN_usama_allFFI_Adu_rsfc_pCorr50_300_tangent_X0Y0__es502-18-21-27'

# model = torch.load(f'{savepath}_model.pt')
# model.eval()

modelnpz = np.load(f'{savepath}_stats.npz')  # TODO: adjust to load desired model

losses_train = modelnpz.f.train_losses
losses_test = modelnpz.f.test_losses
maes_test = modelnpz.f.mae_eng
pears_test = modelnpz.f.pears_eng
acc_test = modelnpz.f.acc_test
rundate = modelnpz.f.rundate
