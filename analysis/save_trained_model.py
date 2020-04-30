import datetime

from analysis.train_model import *


def main():
    rundate = datetime.datetime.now().strftime("%m-%d-%H-%M")

    filename_pt = f"BNCNN_{architecture}_{'_'.join(predicted_outcome)}_{list(directories.keys())[list(directories.values()).index(dataDir)]}" \
                  f"_{transformations}_{deconfound_flavor}{scl}__es{ep_int}" + rundate + "_model.pt"

    torch.save(net, "models/" + filename_pt)

    try:  # old method of saving performance
        filename_stats = f"BNCNN_{architecture}_{'_'.join(predicted_outcome)}_{'_'.join(chosen_Xdatavars)}" \
                         f"_{transformations}_{deconfound_flavor}{scl}__es{ep_int}" + rundate + "_stats.npz"
        np.savez_compressed("models/" + filename_stats,
                            test_losses=losses_test,
                            train_losses=losses_train,
                            mae_eng=maes_test,
                            pears_eng=pears_test,
                            pears_pval=pvals_test,
                            acc_eng=accs_test,
                            rundate=rundate)

    except NameError:
        filename_performance = f"BNCNN_{architecture}_{'_'.join(predicted_outcome)}_{'_'.join(chosen_Xdatavars)}" \
                               f"_{transformations}_{deconfound_flavor}{scl}__es{ep_int}_" + rundate + '_performance'

        performance = performance.assign_coords(rundate=rundate)
        performance = performance.expand_dims('rundate')
        performance.rename(filename_performance)

        performance.to_netcdf(f'models/{filename_performance}.nc')

if __name__ == '__main__':
    main()
