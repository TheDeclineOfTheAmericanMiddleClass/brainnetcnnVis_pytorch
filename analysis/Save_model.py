# from analysis.Run_Model import net, allloss_train, allloss_test, allmae_test1, allpears_test1, pears_1, mae_1, allpval_test1
import datetime
import numpy as np
import torch

rundate = datetime.datetime.now().strftime("%m-%d-%H-%M")

filename_pt = "BNCNN_usama_tangent300_earlystop_" + rundate + "_model.pt"
filename_stats = "BNCNN_usama_tangent300_earlystop_" + rundate + "_stats.npz"

torch.save(net, "models/"+filename_pt)

# np.savez_compressed("models/"+filename_stats, test_losses=allloss_test, train_losses=allloss_train, mae_eng=allmae_test1,
#                     pears_eng=allpears_test1, pears_pval=allpval_test1, final_pears=pears_1, final_mae=mae_1,
#                     rundate=rundate)
#                     # mae_training=allmae_test2, pears_train=allpears_test2)

np.savez_compressed("models/" + filename_stats, test_losses=stoploss_test, train_losses=stoploss_train,
                    mae_eng=stopmae_test1,
                    pears_eng=stoppears_test1, pears_pval=stoppval_test1, final_pears=stoppears_1, final_mae=stopmae_1,
                    rundate=rundate)
