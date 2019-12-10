import datetime
from analysis.Run_Model import net, allloss_train, allloss_test, allmae_test1, allpears_test1 #, allmae_test2, allpears_test2
import numpy as np
import torch

mystring = datetime.datetime.now().strftime("%m-%d-%H-%M")

filename_pt = "BNCNN" + mystring + "_model.pt"
filename_stats = "BNCNN" + mystring + "_stats.npz"

torch.save(net, "models/"+filename_pt)

np.savez_compressed("models/"+filename_stats, test_losses=allloss_test, train_losses=allloss_train, mae_eng=allmae_test1,
                    pears_eng=allpears_test1)
                    # mae_training=allmae_test2, pears_train=allpears_test2)
