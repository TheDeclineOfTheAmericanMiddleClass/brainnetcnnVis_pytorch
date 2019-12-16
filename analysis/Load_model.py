import torch
import numpy as np
from analysis.Init_model import trainset
from analysis.Define_model import BrainNetCNN

model = torch.load('models/BNCNN12-10-01-25_model.pt')
model.eval()

modelnpz = np.load('models/BNCNN_kawahara_300p_300e_12-16-11-56_stats.npz')

allloss_train = modelnpz.f.train_losses
allloss_test = modelnpz.f.test_losses
allmae_test1 = modelnpz.f.mae_eng
allpears_test1 = modelnpz.f.pears_eng