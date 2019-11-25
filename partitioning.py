import numpy as np

# info for all subjects (age, genetic testing, self-reported twin status, genetically tested twin status, family number)
AiY = Rdata["Age_in_Yrs"]
HasGT = Rdata["HasGT"]
ZygositySR = Rdata["ZygositySR"]
ZygosityGT = Rdata['ZygosityGT']
Family_ID = Rdata['Family_ID']
Subject = Rdata['Subject']

GTdone = np.where(np.isin(HasGT, True))[0]  # subjects with genetic tests
srMZ = np.where(np.isin(ZygositySR, "MZ"))[0]  # self-reported monozygotic
srNotMZ = np.where(np.isin(ZygositySR, "NotMZ"))[0]  # self-reported non-monozygotic
boolsrTwins = np.isin(ZygositySR, "NotMZ") + np.isin(ZygositySR, "MZ")
srTwins = np.where(np.isin(ZygositySR, "NotMZ") + np.isin(ZygositySR, "MZ"))[0]  # self-reported twins
srNotTwin = np.where(np.isin(ZygositySR, "NotTwin"))[0]
ZgGTblank = np.where(np.isin(ZygosityGT, ' '))[0]  # unverified twin zygosity

assert np.sum(np.isin(srNotTwin, srTwins)) == 0  # sanity check, ensuring no overlap between twins and non twins

# Identifying singular twins (those without their twin(s) in the study)
singTwins = np.where(np.isin(ZgGTblank, GTdone))[0]
singTwins = np.where(np.isin(singTwins, srTwins))[0]

# Grouping twins together, taking self-report as fact
twingroups = []
families = list(np.unique(Family_ID))
familieswithTwins = np.unique(list(Family_ID.reindex(srTwins)))  # ID of families with self-reported twins

# # Testing this works
# sib1 = int(np.where(Family_ID==familieswithTwins[0])[0][0])
# sib2 = int(np.where(Family_ID==familieswithTwins[0])[0][1])
#
# assert Family_ID.loc[sib1] == Family_ID.loc[sib2]

for i, x in enumerate(familieswithTwins):  # For each family with twins...
    familymems = list(np.where(Family_ID == x)[0])  # list indices of all members
    if len(familymems) > 1:  # if the family is bigger than 1 person...
        fam = np.zeros(len(ZygosityGT), dtype=bool)
        for j, y in enumerate(familymems):
            fam[y] = True
        print(fam.sum())

        twins = list(np.where(boolsrTwins * fam)[0])  # find if/which family members are twins...
        if len(twins) > 2:  # (if more than two twins, tell us, but still add them all)
            print(f'family found with {len(twins)} twins')
    twingroups.append(twins)  # and add them to our list

# Random shuffling of standalone participants and twins into 70-10-20 train-validation-test split
