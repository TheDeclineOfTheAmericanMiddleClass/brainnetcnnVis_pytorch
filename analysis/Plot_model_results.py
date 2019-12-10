from matplotlib import pyplot as plt
from analysis.Run_Model import allloss_train, allloss_test, allmae_test1, \
    allpears_test1, pears_1, mae_1 #, allmae_test2, allpears_test2
from analysis.ElasticSVM_models import *
import numpy as np

nPlots = 4
plt.rcParams.update({'font.size':9})
fig, axs = plt.subplots(2, 2, figsize=(8, 4))
fig.subplots_adjust(hspace=.2, wspace=.2)
axs = axs.ravel()

fig.suptitle('BrainNetCNN Prediction Performance\n **Age**')
color_idx = np.linspace(0, 1, nPlots)

ylabs = ['Training MSE','Test MSE','Prediction (MAE)','Prediction (Pearson r)']
plotties = [allloss_train, allloss_test, allmae_test1, allpears_test1]

for i in range(4):
    axs[i].plot(plotties[i], color=plt.cm.cool(color_idx[i]))  # test mean absolute error
    axs[i].set_ylabel(ylabs[i])
    axs[i].set_xlabel('Epoch')

colhs = ['BrainNetCNN,', 'ElasticNet', 'SVM']
rowhs = ['pearson r', 'mean absolute error']
tabel = np.array([[f'{pears_1[0]:.2}, p-value: {pears_1[1]:.2}', f'{elastic_r:.2}, p-value: {elastic_p:.2}',
          f'{svm_r:.2}, p-value: {svm_p:.2}'],
         [f'{mae_1:.2}', f'{elastic_mae:.2}', f'{svm_mae:.2}']])

for i, x in enumerate(colhs):
    for j, y in enumerate(rowhs):
        print(f'{x} {y}: {tabel[j,i]}')
    print('')