# from analysis.Run_Model import *
import datetime

import numpy as np
import torch

from preprocessing.Model_DOF import *

rundate = datetime.datetime.now().strftime("%m-%d-%H-%M")

filename_pt = f"BNCNN_usama_{predicted_outcome}_{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}" \
              f"_{data_to_use}_{deconfound_flavor}{scl}__es{ep_int}" + rundate + "_model.pt"
filename_stats = f"BNCNN_usama_{predicted_outcome}_{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}" \
                 f"_{data_to_use}_{deconfound_flavor}{scl}__es{ep_int}" + rundate + "_stats.npz"

torch.save(net, "models/"+filename_pt)

# np.savez_compressed("models/"+filename_stats, test_losses=allloss_test, train_losses=allloss_train, mae_eng=allmae_test1,
#                     pears_eng=allpears_test1, pears_pval=allpval_test1, final_pears=pears_1, final_mae=mae_1,
#                     rundate=rundate)
#                     # mae_training=allmae_test2, pears_train=allpears_test2)

np.savez_compressed("models/" + filename_stats, test_losses=losses_test, train_losses=losses_train,
                    mae_eng=maes_test,
                    pears_eng=pears_test, pears_pval=pvals_test, final_pears=final_pears, final_mae=final_mae,
                    rundate=rundate)
