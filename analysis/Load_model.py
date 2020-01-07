import torch
import numpy as np

# from analysis.Init_model import trainset
# from analysis.Define_model import BrainNetCNN

# model = torch.load('models/BNCNN_usama_tangent264_300e_12-16-21-11_model.pt')
# model.eval()

modelnpz = np.load('models/BNCNN_usama_tangent264_300e_12-16-21-11_stats.npz')

allloss_train = modelnpz.f.train_losses
allloss_test = modelnpz.f.test_losses
allmae_test1 = modelnpz.f.mae_eng
allpears_test1 = modelnpz.f.pears_eng
rundate = modelnpz.f.rundate
