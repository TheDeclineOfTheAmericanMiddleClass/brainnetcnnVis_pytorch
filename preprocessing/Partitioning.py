import numpy as np
from preprocessing.Main_preproc import *
from analysis.Load_model_data import *

# def partition(restricted):
#     '''Partitioning data so one families twins remain in test/validation/training sets'''
#
#     ZygositySR = restricted["ZygositySR"]
#     ZygosityGT = restricted['ZygosityGT']
#     HasGT = restricted['HasGT']
#
#     # GTyes = np.where(np.isin(HasGT, True))[0]  # subjects with genetic tests
#     # GTno = np.where(np.isin(HasGT, False))[0]  # subject wo/ genetic tests
#     # assert len(GTyes) <= 1142
#     # assert len(GTno) <= 64
#     #
#     # srMZ = np.where(np.isin(ZygositySR, "MZ"))[0]  # self-reported monozygotic
#     # srNotMZ = np.where(np.isin(ZygositySR, "NotMZ"))[0]  # self-reported non-monozygotic
#     # srNotTwin = np.where(np.isin(ZygositySR, "NotTwin"))[0]  # self-reported not twin
#     # srBlank = np.where(np.isin(ZygositySR, " "))[0]  # no self-report on twin status
#     # assert len(srMZ) + len(srNotMZ) + len(srNotTwin) + len(srBlank) == 1003
#     #
#     # gcMZ = np.where(np.isin(ZygosityGT, "MZ"))[0]  # genetically confirmed monozygotic
#     # gcDZ = np.where(np.isin(ZygosityGT, "DZ"))[0]  # genetically confirmed dizygotic
#     # gcBlank = np.where(np.isin(ZygosityGT, " "))[0]  # not genetically confirmed
#     # assert len(gcMZ) + len(gcDZ) + len(gcBlank) == 1003
#     #
#     # gcMZTwins = set(gcMZ) & set(GTyes)  # genetically confirmed MZ twins
#     # assert len(gcMZTwins) <= 298  # 298 MZ in 1200 dataset
#     #
#     # gcDZTwins = set(gcDZ) & set(GTyes)  # genetically confirmed DZ twins
#     # assert len(gcDZTwins) <= 188  # 188 DZ twins in 1200 dataset
#     #
#     # # p5 and p6 refer to points 5 and 6 of pg 89 in HCP release reference manual
#     # # https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf
#     #
#     # p5 = set(srMZ) & set(gcBlank)  # point 5 of pg 89
#     # assert len(p5) <= 66  # 66 subjects with ZygositySR=MZ, but ZygosityGT=Blank in 1200 dataset
#     #
#     # p6 = set(srNotMZ) & set(gcBlank)
#     # assert len(p6) <= 65  # 65 subjects with ZygositySR=NotMZ, but ZygosityGT=Blank in 1200 dataset
#     #
#     # # subjects whose putative twin is not part of the 1206 released study subjects.
#     # nsMZ = p5 & set(GTno)  # subjects whose putative MZ twin IS  part of the 1206, but HasGT=FALSE for one of the pair,
#     # nsDZ = p6 & set(GTno)  # subjects whose putative DZ twin IS  part of the 1206, but HasGT=FALSE for one or both of the pair
#     # noGTTwins = nsMZ|nsDZ
#     # assert len(noGTTwins) <= 56, 'More non-singular twins than expected. Should be fewer than 56.'
#     #
#     # # Creating full list of non-singular twins (adding genetically confirmed twins)
#     # nsTwins = noGTTwins|gcMZTwins|gcDZTwins
#     #
#     # # TODO: figure out if problem in finding twins from nsTwinFams or familymems or Family_ID
#     # # family IDs of families with self-reported, non-singular twins
#     # nsTwinfams = list(np.unique(list(Family_ID.iloc[list(nsTwins)])))
#     # assert len(nsTwinfams) <= len(nsTwins)/2
#     #
#     # # Sanity check, confirming self-reported but not-genetically confirmed twins are of the same family
#     # sib1 = int(np.where(Family_ID == nsTwinfams[0])[0][0])
#     # sib2 = int(np.where(Family_ID == nsTwinfams[0])[0][1])
#     # assert Family_ID.iloc[sib1] == Family_ID.iloc[sib2]
#     #
#     # # Grouping twins together
#     # twingroups = []  # groups of subjects who are twins
#     #
#     # for i, x in enumerate(nsTwinfams):  # For each family with twins...
#     #     familymems = list(np.where(Family_ID == x)[0])  # list indices of all members
#     #     # print(familymems)
#     #     for j, y in enumerate([noGTTwins, gcMZTwins, gcDZTwins]):
#     #         twins = set(familymems) & y   # find if/which family members are twins...
#     #         if len(twins) > 2:  # (if more than two twins, tell us, but still add them all)
#     #             print(f'family found with {len(twins)} twins')
#     #         elif len(twins) < 2:
#     #             print('only one twin in family!')
#     #         if twins:
#     #             twingroups.append(twins)  # and add them to our list
#     #
#     # # TODO: Random shuffling of standalone participants and twins into 70-10-20 train-validation-test split
#
#     return test, train, validation # subject IDs