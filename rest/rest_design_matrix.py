"""


"""

#%% Imports & SetVars
from glob import glob
import os
import pandas as pd
from natsort import natsorted
import numpy as np

#%% Combined Subjects & Exposure Data
exposure = 'meana13'
covfile = '/mnt/elysium/IRC805/data/gludata.csv'
covariates = ['sex', 'ageyr']
destination = '/mnt/elysium/IRC805/rest'

file = pd.read_csv(covfile)
file.set_index('Subject', inplace=True, drop=True)

rest_files = natsorted(glob('/mnt/elysium/IRC805/rest/proc/*.nii.gz'))
subjects = []
for i in rest_files:
    s1 = i.split('/')
    s2 = s1[-1].split('_')
    subject = s2[0]
    subjects.append(subject)
subject_df = pd.DataFrame()
subject_df['Subject'] = subjects
subject_df['rest'] = 1
subject_df['Subject'] = subject_df['Subject'].astype('int')
subject_df.set_index('Subject', inplace=True)
subject_df.sort_values(by='Subject')


combined_df = file.merge(subject_df, on='Subject')
combined_df = combined_df.sort_values(by='Subject')
#%% Create Group .mat File (Group)
# Group .mat file
dummies = pd.get_dummies(combined_df[exposure])
dummies[covariates] = combined_df[covariates]

dmean = lambda x: x - x.mean(skipna=True)
dummies['ageyr'] = dummies['ageyr'].transform(dmean)
dummies['sex'] = dummies['sex'] - 1

#%%

dummies = combined_df[[exposure] + covariates]
dmean = lambda x: x - x.mean(skipna=True)
dummies['ageyr'] = dummies['ageyr'].transform(dmean)
dummies['sex'] = dummies['sex'] - 1
dummies['intercept'] = 1
dummies = dummies.dropna()
#%% Create .mat File
# .mat file

os.makedirs('/mnt/elysium/IRC805/rest/'+exposure, exist_ok=True)
dummies.to_csv('/mnt/elysium/IRC805/rest/'+exposure+'/design.txt', index=False, header=None, sep=' ')

#%% Create file list
d = {}
for item in dummies.index:
        file = '/mnt/elysium/IRC805/rest/proc/'+str(item)+'_rest.nii.gz'
        d[item] = file
df = pd.DataFrame.from_dict(d, orient='index')
df[0].to_csv('/mnt/elysium/IRC805/rest/'+exposure+'/filelist.txt', index=False, header=None)


