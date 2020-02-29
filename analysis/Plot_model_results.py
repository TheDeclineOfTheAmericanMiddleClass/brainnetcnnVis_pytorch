import numpy as np
from matplotlib import pyplot as plt

from preprocessing.Model_DOF import *

def main():

    # TODO: edit plot_model_results to accommodate multiple outcomes
    def plot_model_results(allloss_train=[], allloss_test=[], allmae_test=[], allpears_test=[],
                           deconfound_flavor='', scl='', predicted_outcome='', outcome_names=[],
                           arc='', input_data='', transformations='', ep_int=666, es=True):
        # defining location of all plots and titles
        nPlots = 4
        plt.rcParams.update({'font.size': 9})
        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        fig.subplots_adjust(hspace=.2, wspace=.3, top=.9, bottom=.24)
        axs = axs.ravel()

        # string formatting, if there was early stopping
        if es:
            TFstop = 'Yes'
        else:
            TFstop = 'No'


        # Setting plot title and caption
        fig.suptitle(f'BrainNetCNN: {predicted_outcome} Prediction Performance', fontsize=11, fontweight='bold')
        caption = f'{predicted_outcome} prediction for Brain Net CNN. Architecture by {arc}.     Trained over {len(allloss_test)} epochs.' \
                  f'\nInput matrices: {input_data},     Transformations: {transformations}\n     Deconfound method:{deconfound_flavor}{scl}' \
                  f'    Early stopping: {TFstop}, checks every {ep_int} epochs.\nMAE and MSE are plotted on a log-10 scale.' \
                  f'The displayed final and minimum values are on a linear scale.'
        plt.figtext(0.5, 0.01, caption, wrap=True, ha='center', va='bottom', fontsize=9, fontstyle='italic')

        # Plotting for BrainNetCNN
        ylabs = ['Training MSE', 'Test MSE', 'Prediction (MAE)', 'Prediction (Pearson r)']

        plotties = [np.array(x) for i, x in enumerate([allloss_train, allloss_test, allmae_test, allpears_test])]
        plotties = [plotties[i].astype(float) for i, p in enumerate(plotties)]

        # setting colors for plots
        color_idx = np.linspace(0, 1, nPlots)

        # TODO: check dimensionality of sex accuracy, add clause here if necessary
        if not multi_outcome:
            for i in range(4):
                if i < 3:  # for test and training MSE
                    axs[i].plot(np.log10(plotties[i]), color=plt.cm.cool(color_idx[i]))
                    axs[i].plot(np.repeat(np.log10(plotties[i])[-1], len(plotties[i])),
                                color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
                    axs[i].plot(np.repeat(np.log10(plotties[i]).min(), len(plotties[i])),
                                color='g', linestyle='--', label=f'min. = {plotties[i].min():.3}')
                    axs[i].legend()

                    if i == 2:
                        axs[i].set_ylabel(f'{ylabs[i]}')

                    elif i < 2:
                        # matching y-axis limits
                        llim = np.log10(plotties[0]).min() - (
                                np.log10(plotties[0].mean()) - np.log10(
                            plotties[0].min())) / 5  # lower limit # TODO: see why yticks are changing
                        ulim = np.log10(plotties[1]).max() + np.abs(
                            np.log10(plotties[1]).mean() - np.log10(plotties[1]).max()) / 5  # upper limit
                        axs[i].set_ylim(llim, ulim)  # training set min and test set max
                        axs[i].set_ylabel(f'{ylabs[i]}\n(log 10 scale)')
                        print(llim, ulim)

                else:
                    axs[i].set_xlabel('Epoch')
                    axs[i].plot(plotties[i], color=plt.cm.cool(color_idx[i]))  # test mean absolute error
                    axs[i].plot(np.repeat(plotties[i][-1], len(plotties[i])),
                                color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
                    axs[i].set_ylabel(ylabs[i])
                    axs[i].legend()

                    # if i == 2:
                    #     axs[i].plot(np.repeat(plotties[i].min(), len(plotties[i])),
                    #                 color='g', linestyle='--', label=f'final = {plotties[i].min():.3}')
                    #     axs[i].legend()

        elif multi_outcome:
            num_out = int(plotties[-2].shape[-1])
            assert num_out == len(outcome_names), 'Number of outcome names must match dimensions of data !'
            multicolor_idx = np.linspace(0, 1, num_out)
            for i in range(4):
                if i < 2:  # (0) Training  and (1) Test MSE

                    axs[i].plot(np.log10(plotties[i]), color=plt.cm.cool(color_idx[i]))  # plotting results
                    # plotting final TODO: could also do hlines here
                    axs[i].plot(np.repeat(np.log10(plotties[i])[-1], len(plotties[i])),
                                color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
                    axs[i].plot(np.repeat(np.log10(plotties[i]).min(), len(plotties[i])),  # plotting lowest error
                                color='g', linestyle='--', label=f'min. = {plotties[i].min():.3}')
                    axs[i].legend()
                    axs[i].set_ylabel(f'{ylabs[i]}\n(log 10 scale)')

                    # matching y-axis limits
                    llim = np.log10(plotties[0]).min() - (
                            np.log10(plotties[0].mean()) - np.log10(plotties[0].min())) / 5  # lower limit
                    ulim = np.log10(plotties[1]).max() + np.abs(
                        np.log10(plotties[1]).mean() - np.log10(plotties[1]).max()) / 5  # upper limit
                    axs[i].set_ylim(llim, ulim)  # training set min and test set max
                    axs[i].set_ylabel(f'{ylabs[i]}\n(log 10 scale)')

                elif i == 2:  # (2) Test MAE
                    axs[i].set_xlabel('Epoch')
                    for j in range(num_out):
                        axs[i].plot(np.log10(plotties[i][:, j]), color=plt.cm.cool(multicolor_idx[j]))
                        # axs[i].plot(np.repeat(np.log10(plotties[i][:,j])[-1], len(plotties[i])),  # plotting final
                        #             color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
                    axs[i].plot(np.repeat(np.log10(plotties[i]).min(), len(plotties[i])),  # plotting lowest error
                                color='g', linestyle='--', label=f'min. = {plotties[i].min():.3}')
                    axs[i].set_ylabel(f'{ylabs[i]}\n(log 10 scale)')
                    handles, labels = axs[i].get_legend_handles_labels()
                    axs[i].legend()  # , ncol=num_out)

                else:  # (3) Pearson R
                    axs[i].set_xlabel('Epoch')
                    for j in range(num_out):
                        axs[i].plot(plotties[i][:, j], color=plt.cm.cool(multicolor_idx[j]),
                                    label=f'{outcome_names[j]}')  # test mean absolute error
                    # axs[i].plot(np.repeat(plotties[i][-1], len(plotties[i])),
                    #             color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
                    axs[i].set_ylabel(ylabs[i])
                    handles, labels = axs[i].get_legend_handles_labels()
                    axs[i].legend(handles, labels, loc='center right', bbox_to_anchor=(1.2, .5))  # , ncol=num_out)


    # TODO: enable plot model results to do accuracy as well
    plot_model_results(allloss_train=losses_train,
                       allloss_test=losses_test,
                       allmae_test=maes_test,
                       allpears_test=pears_test,
                       arc=architecture,
                       predicted_outcome=predicted_outcome,
                       outcome_names=outcome_names,
                       deconfound_flavor=deconfound_flavor,
                       scl=scl,
                       input_data=f'{list(dataDirs.keys())[list(dataDirs.values()).index(dataDir)]}',
                       transformations=data_to_use,
                       ep_int=ep_int,
                       es=True)

if __name__ == "__main__":
    main()

# # PRINTING RESULTS TO CONSOLE
# colhs = ['BrainNetCNN,', 'ElasticNet', 'SVM']
# rowhs = ['pearson r', 'mean absolute error']
# tabel = np.array([[f'{allpears_test1[-1]:.2}, p-value: {pears_1[1]:.2}', f'{elastic_r:.2}, p-value: {elastic_p:.2}',
#                    f'{svm_r:.2}, p-value: {svm_p:.2}'],
#                   [f'{allmae_test1[-1]:.2}', f'{elastic_mae:.2}', f'{svm_mae:.2}']])
#
# for i, x in enumerate(colhs):
#     for j, y in enumerate(rowhs):
#         print(f'{x} {y}: {tabel[j, i]}')
#     print('')
#