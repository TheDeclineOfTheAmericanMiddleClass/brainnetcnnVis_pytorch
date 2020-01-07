from matplotlib import pyplot as plt
# from analysis.Load_model import allloss_train, allloss_test, allmae_test1, \
#     allpears_test1
# from analysis.ElasticSVM_models import *
import numpy as np


def plot_model_results(allloss_train, allloss_test, allmae_test1, allpears_test1,
                       arc='Usama', input_data='power 264', transformations='positive-definite, then tangent', es=True):
    # defining location of all plots and titles
    nPlots = 4
    plt.rcParams.update({'font.size': 9})
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    fig.subplots_adjust(hspace=.4, wspace=.3, top=.9, bottom=.18)
    axs = axs.ravel()

    # string formatting, if there was early stopping
    if es:
        TFstop = ''
    else:
        TFstop = 'not '

    # Setting plot title and caption
    fig.suptitle(f'BrainNetCNN: AGE Prediction Performance', fontsize=11, fontweight='bold')
    caption = f'Age prediction for Brain Net CNN. The network used architecture from {arc} and was trained over' \
              f' {len(allloss_test)} epochs. \nThe {input_data} matrices were transformed into the {transformations} ' \
              f'space, then used as input data.\nMAE and MSE are plotted on a log-10 scale. The dispalyed final and' \
              f' minimum values are on a linear scale. Early stopping was {TFstop}used'
    plt.figtext(0.5, 0.01, caption, wrap=True, ha='center', va='bottom', fontsize=8, fontstyle='italic')

    # setting colors for plots
    color_idx = np.linspace(0, 1, nPlots)

    # Plotting for BrainNetCNN
    ylabs = ['Training MSE', 'Test MSE', 'Prediction (MAE)', 'Prediction (Pearson r)']
    plotties = np.array([allloss_train, allloss_test, allmae_test1, allpears_test1])

    for i in range(4):
        axs[i].set_xlabel('Epoch')

        if i < 3:  # for test and training MSE

            axs[i].plot(np.log10(plotties[i]), color=plt.cm.cool(color_idx[i]))
            axs[i].plot(np.repeat(np.log10(plotties[i])[-1], len(plotties[i])),
                        color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
            axs[i].plot(np.repeat(np.log10(plotties[i]).min(), len(plotties[i])),
                        color='g', linestyle='--', label=f'min. = {plotties[i].min():.3}')
            axs[i].legend()
            axs[i].set_ylabel(f'{ylabs[i]}\n(log 10 scale)')

            if i < 2:
                # matching y-axis limits
                llim = np.log10(plotties[0]).min() - (
                        np.log10(plotties[0].mean()) - np.log10(plotties[0].min())) / 5  # lower limit
                ulim = np.log10(plotties[1]).max() + np.abs(
                    np.log10(plotties[1]).mean() - np.log10(plotties[1]).max()) / 5  # upper limit
                axs[i].set_ylim(llim, ulim)  # training set min and test set max
                axs[i].set_ylabel(f'{ylabs[i]}\n(log 10 scale)')

        else:
            axs[i].plot(plotties[i], color=plt.cm.cool(color_idx[i]))  # test mean absolute error
            axs[i].plot(np.repeat(plotties[i][-1], len(plotties[i])),
                        color='k', linestyle='solid', label=f'final = {plotties[i][-1]:.3}')
            axs[i].set_ylabel(ylabs[i])
            axs[i].legend()

            # if i == 2:
            #     axs[i].plot(np.repeat(plotties[i].min(), len(plotties[i])),
            #                 color='g', linestyle='--', label=f'final = {plotties[i].min():.3}')
            #     axs[i].legend()


plot_model_results(stoploss_train, stoploss_test, stopmae_test1, stoppears_test1,
                   arc='Usama', input_data='ICA 300', transformations='positive-definite, then tangent', es=True)

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

# PRINTING ONLY THE SHALLOW RESULTS
# s_colhs = ['ElasticNet', 'SVM']
# s_rowhs = ['pearson r', 'mean absolute error']
# s_tabel = np.array([[f'{elastic_r:.2}, p-value: {elastic_p:.2}',
#                    f'{svm_r:.2}, p-value: {svm_p:.2}'],
#                    [f'{elastic_mae:.2}', f'{svm_mae:.2}']])
#
# for i, x in enumerate(s_colhs):
#     for j, y in enumerate(s_rowhs):
#         print(f'{x} {y}: {s_tabel[j, i]}')
#     print('')
