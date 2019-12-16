from matplotlib import pyplot as plt
# from analysis.Run_Model import allloss_train, allloss_test, allmae_test1, \
#     allpears_test1, pears_1, mae_1 #, allmae_test2, allpears_test2
from analysis.Load_model import allloss_train, allloss_test, allmae_test1, \
    allpears_test1
from analysis.ElasticSVM_models import *
import numpy as np

nPlots = 4
plt.rcParams.update({'font.size':9})
fig, axs = plt.subplots(2, 2, figsize=(8, 4))
fig.subplots_adjust(hspace=.2, wspace=.3)
axs = axs.ravel()

fig.suptitle('BrainNetCNN  **Age** Prediction Performance\n')
color_idx = np.linspace(0, 1, nPlots)

# Plotting for BrainNetCNN
ylabs = ['Training MSE','Test MSE','Prediction (MAE)','Prediction (Pearson r)']
plotties = np.array([allloss_train, allloss_test, allmae_test1, allpears_test1])

for i in range(4):
    axs[i].set_xlabel('Epoch')

    if i < 2:  # for test and training MSE
        axs[i].plot(np.log10(plotties[i]), color=plt.cm.cool(color_idx[i]))
        axs[i].plot(np.repeat(np.log10(plotties[i]).min(), len(plotties[i])),
                    color='g', linestyle='--', label=f'minimum ={plotties[i].min():.3}')
        axs[i].set_ylabel(f'{ylabs[i]}\n(log10-scale)')
        axs[i].legend()

        # matching y-axis limits
        llim = np.log10(plotties[0]).min() - (np.log10(plotties[0].mean()) - np.log10(plotties[0].min()))/5  # lower limit
        ulim = np.log10(plotties[1]).max() + np.abs(np.log10(plotties[1]).mean() - np.log10(plotties[1]).max())/5  # upper limit
        axs[i].set_ylim(llim, ulim)  # training set min and test set max

    else:
        axs[i].plot(plotties[i], color=plt.cm.cool(color_idx[i]))  # test mean absolute error
        axs[i].set_ylabel(ylabs[i])

colhs = ['BrainNetCNN,', 'ElasticNet', 'SVM']
rowhs = ['pearson r', 'mean absolute error']
tabel = np.array([[f'{allpears_test1[-1]:.2}, p-value: {pears_1[1]:.2}', f'{elastic_r:.2}, p-value: {elastic_p:.2}',
          f'{svm_r:.2}, p-value: {svm_p:.2}'],
         [f'{allmae_test1[-1]:.2}', f'{elastic_mae:.2}', f'{svm_mae:.2}']])

for i, x in enumerate(colhs):
    for j, y in enumerate(rowhs):
        print(f'{x} {y}: {tabel[j,i]}')
    print('')