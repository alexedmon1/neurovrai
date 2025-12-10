#!/usr/bin/env python3
"""
Fix ReHo and fALFF design matrices to match actual 4D data (15 subjects)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib

# Paths
ANALYSIS_ROOT = Path('/mnt/bytopia/IRC805/analysis')
DESIGN_ROOT = Path('/mnt/bytopia/IRC805/data/designs')

# Get actual number of subjects in 4D files
reho_4d = ANALYSIS_ROOT / 'func' / 'reho' / 'all_reho_normalized_z.nii.gz'
falff_4d = ANALYSIS_ROOT / 'func' / 'falff' / 'all_falff_normalized_z.nii.gz'

reho_img = nib.load(reho_4d)
falff_img = nib.load(falff_4d)

n_reho = reho_img.shape[3]
n_falff = falff_img.shape[3]

print(f"ReHo 4D: {n_reho} subjects")
print(f"fALFF 4D: {n_falff} subjects")

# Load the 16-subject list (which should match the 4D data order minus 1)
subject_list_file = DESIGN_ROOT / 'func_reho' / 'subject_list.txt'
with open(subject_list_file) as f:
    subjects_16 = [line.strip() for line in f if line.strip() and not line.startswith('#')]

print(f"Subject list has {len(subjects_16)} subjects")
print(f"Need to remove 1 subject to match {n_reho}")

# Load full participants data
participants_file = DESIGN_ROOT / 'func_reho' / 'participants_matched.tsv'
df_full = pd.read_csv(participants_file, sep='\t')

# Filter to only the subjects in the 16-subject list
df_16 = df_full[df_full['participant_id'].isin(subjects_16)].copy()
print(f"\nFiltered participants: {len(df_16)} subjects")

# The 4D file has 15, so we need to figure out which of the 16 is missing
# Let's assume the first 15 in the list order are the ones in the 4D
subjects_15 = subjects_16[:15]
print(f"\nUsing first 15 subjects from list:")
for s in subjects_15:
    print(f"  {s}")

# Filter to 15 subjects
df_15 = df_full[df_full['participant_id'].isin(subjects_15)].copy()
# Sort by the order in subjects_15
df_15['sort_order'] = df_15['participant_id'].apply(lambda x: subjects_15.index(x))
df_15 = df_15.sort_values('sort_order').drop('sort_order', axis=1)

print(f"\nCreating design matrix for {len(df_15)} subjects")

# Create design matrix (sex, age, mriglu_1, mriglu_2)
sex = (df_15['sex'] - df_15['sex'].mean()).values
age = (df_15['age'] - df_15['age'].mean()).values
mriglu_1 = (df_15['mriglu'] == 1).astype(float).values
mriglu_2 = (df_15['mriglu'] == 2).astype(float).values

design = np.column_stack([sex, age, mriglu_1, mriglu_2])

# Contrasts (same as before)
contrasts = np.array([
    [0, 0, 1, -1],  # controlled > uncontrolled
    [0, 0, -1, 1],  # uncontrolled > controlled
    [1, 0, 0, 0],   # sex positive
    [-1, 0, 0, 0],  # sex negative
    [0, 1, 0, 0],   # age positive
    [0, -1, 0, 0],  # age negative
])

# Save for ReHo
reho_design_dir = ANALYSIS_ROOT / 'func' / 'reho' / 'randomise_5000perm'
reho_design_dir.mkdir(parents=True, exist_ok=True)

design_mat_file = reho_design_dir / 'design.mat'
with open(design_mat_file, 'w') as f:
    f.write('/NumWaves 4\n')
    f.write(f'/NumPoints {len(df_15)}\n')
    f.write('/Matrix\n')
    for row in design:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

design_con_file = reho_design_dir / 'design.con'
with open(design_con_file, 'w') as f:
    f.write('/NumWaves 4\n')
    f.write(f'/NumContrasts {len(contrasts)}\n')
    f.write('/Matrix\n')
    for row in contrasts:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

print(f"\n✓ Created ReHo design files:")
print(f"  {design_mat_file}")
print(f"  {design_con_file}")

# Save for fALFF (same subjects)
falff_design_dir = ANALYSIS_ROOT / 'func' / 'falff' / 'randomise_5000perm'
falff_design_dir.mkdir(parents=True, exist_ok=True)

design_mat_file = falff_design_dir / 'design.mat'
with open(design_mat_file, 'w') as f:
    f.write('/NumWaves 4\n')
    f.write(f'/NumPoints {len(df_15)}\n')
    f.write('/Matrix\n')
    for row in design:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

design_con_file = falff_design_dir / 'design.con'
with open(design_con_file, 'w') as f:
    f.write('/NumWaves 4\n')
    f.write(f'/NumContrasts {len(contrasts)}\n')
    f.write('/Matrix\n')
    for row in contrasts:
        f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

print(f"\n✓ Created fALFF design files:")
print(f"  {design_mat_file}")
print(f"  {design_con_file}")

print("\nDone! Design matrices now match 4D data.")
