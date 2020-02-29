# from analysis.Run_Model import *
import datetime

import numpy as np

from preprocessing.Model_DOF import *

rundate = datetime.datetime.now().strftime("%m-%d-%H-%M")

filename_pt = f"BNCNN_{architecture}_{predicted_outcome}_{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}" \
              f"_{data_to_use}_{deconfound_flavor}{scl}__es{ep_int}" + rundate + "_model.pt"
filename_stats = f"BNCNN_{architecture}_{predicted_outcome}_{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}" \
                 f"_{data_to_use}_{deconfound_flavor}{scl}__es{ep_int}" + rundate + "_stats.npz"

torch.save(net, "models/"+filename_pt)

np.savez_compressed("models/" + filename_stats, test_losses=losses_test, train_losses=losses_train,
                    mae_eng=maes_test,
                    pears_eng=pears_test,
                    pears_pval=pvals_test,
                    final_pears=final_pears,
                    final_mae=final_mae,
                    acc_eng=acc_test,
                    rundate=rundate)
