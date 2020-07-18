# Directories of posssible datasets to use
data_directories = {'HCP_rsfc_pCorr01_300': 'data/3T_HCP1200_MSMAll_d300_ts2_RIDGE',
                    # HCP partial correlation @ rho = .01
                    'HCP_rsfc_pCorr01_264': 'data/POWER_264_FIXEXTENDED_RIDGEP',  # power 264 resting PCORR matrices
                    'HCP_rsfc_Corr_300': 'data/HCP_created_ICA300_mats/corr',  # HCP-created ICA300 rsfc
                    'HCP_alltasks_268': 'data/cfHCP900_FSL_GM',  # all HCP tasks, 268x268, z-scored
                    'Lea_EB_rsfc_264': 'data/edge_betweenness',  # edge-betweenness created by Lea
                    'Adu_rsfc_pCorr50_300': 'data/self_created_HCP_mats/ICA300_rho50_pcorr',
                    # Adu's ICA 300 resting PCORR with rho of 0.5
                    'Adu_rsfc_Corr_300': 'data/self_created_HCP_mats/ICA300_corr',
                    # should be equivalent to 'HCP_rsfc_Corr_300'
                    'Johann_mega_graph': 'data/Send_to_Tim/HCP_IMAGEN_ID_mega_file.txt'
                    }  # TODO: add data_directories and DoF for HCP vs IMAGEN dataset_to_cluster

# Tasks in alltasks_268 aka cfHCP900_FSL_GM dataset_to_cluster
HCP268_tasks = {'rest1': 'rfMRI_REST1',
                'working_memory': 'tfMRI_WM',
                'gambling': 'tfMRI_GAMBLING',
                'motor': 'tfMRI_MOTOR',
                'rest2': 'rfMRI_REST2',
                'language': 'tfMRI_LANGUAGE',
                'social': 'tfMRI_SOCIAL',
                'relational': 'tfMRI_RELATIONAL',
                'faces': 'tfMRI_EMOTION',
                'NA': ''}

# Outcomes to predict
predict_choices = ['NEOFAC_O', 'NEOFAC_C', 'NEOFAC_E', 'NEOFAC_A', 'NEOFAC_N', 'Gender',
                   'Age_in_Yrs', 'PMAT24_A_CR'] + [f'softcluster_{i}' for i in range(1, 14)]

multiclass_outcomes = ['Gender', 'hardcluster']

# Subject splits, by twin status, in .txt format
subnum_paths = dict(test='Subject_Splits/final_test_list.txt',
                    train='Subject_Splits/final_train_list.txt',
                    val='Subject_Splits/final_val_list.txt')
