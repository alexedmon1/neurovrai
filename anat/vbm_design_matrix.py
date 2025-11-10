"""


"""

#%% Imports & SetVars
from glob import glob
import os
import pandas as pd
import numpy as np
from natsort import natsorted

#%% Combined Subjects & Exposure Data
exposure = 'mriglu'
visit = 1
covfile = '/mnt/elysium/IRC805/data/gludata.csv'
covariates = ['sex', 'ageyr']
destination = '/mnt/elysium/IRC805/morph/vbm/'

file = pd.read_csv(covfile)
file.set_index('Subject', inplace=True, drop=True)
#file = file[file['cares_session'] == visit]

t1w_files = natsorted(glob('/mnt/elysium/IRC805/morph/t1w/*/*.nii.gz'))
subjects = []
for i in t1w_files:
    s1 = i.split('/')
    s2 = s1[-1].split('_')
    subject = s2[0]
    subjects.append(subject)
subject_df = pd.DataFrame()
subject_df['Subject'] = subjects
subject_df['t1w'] = 1
subject_df['Subject'] = subject_df['Subject'].astype('int')
subject_df.set_index('Subject', inplace=True)
subject_df.sort_values(by='Subject')

combined_df = file.merge(subject_df, on='Subject')
combined_df = combined_df.sort_values(by='Subject')
#%% Regression .mat file
df1 = combined_df[[exposure] + covariates]

#df1['Gender'] = df1['Gender'].apply(lambda x: 1 if x == "Male" else 0)

dmean = lambda x: x - x.mean(skipna=True)
df1['ageyr'] = df1['ageyr'].transform(dmean)
df1['mriglu'] = df1['mriglu'] - 1
df1['sex'] = df1['sex'] - 1
df1['intercept'] = 1
df1 = df1.dropna()
df1 = df1.sort_values(by='Subject')

#%% Create .mat File
# .mat file

os.makedirs('/mnt/elysium/IRC805/morph/vbm/'+exposure, exist_ok=True)
#dummies.to_csv(destination+exposure+'/3group_design.txt', index=False, header=None, sep=' ')

df1.to_csv('/mnt/elysium/IRC805/morph/vbm/'+exposure+'/design.txt', index=False, header=None, sep=' ')

#%% Create file list
d = {}
for item in df1.index:
        file = '/mnt/elysium/IRC805/morph/t1w/'+str(item)+'/'+str(item)+'_t1w.nii.gz'
        d[item] = file

df = pd.DataFrame.from_dict(d, orient='index')
df[0].to_csv('/mnt/elysium/IRC805/morph/vbm/'+exposure+'/filelist.txt', index=False, header=None)
