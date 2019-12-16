import torch
import numpy as np
from preprocessing.Main_preproc import restricted, subnums

# Parvathy's partitions
final_test_list = np.loadtxt('Subject_Splits/final_test_list.txt')
final_train_list = np.loadtxt('Subject_Splits/final_train_list.txt')
final_val_list = np.loadtxt('Subject_Splits/final_val_list.txt')

# Info for all subjects (age, family number, subjectID)
AiY = restricted["Age_in_Yrs"]
Family_ID = restricted['Family_ID']
Subject = restricted['Subject']

## Defining train, test, validaton sets
# 70-15-15 train test validation split
train_ind = np.where(np.isin(subnums, final_train_list))[0]
test_ind = np.where(np.isin(subnums, final_test_list))[0]
val_ind = np.where(np.isin(subnums, final_val_list))[0]

ages = np.array(AiY)
# data = torch.tensor(data)