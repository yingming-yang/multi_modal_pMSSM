"Import Data"
import os
import uproot
import numpy as np
import pandas as pd

def split_constant_nonconstant_columns(df: pd.DataFrame, return_names = False):
    """
    Splits a DataFrame into:
    - constant_df: columns where all values are identical
    - nonconstant_df: columns where at least one value differs
    """

    # A column is constant if it has exactly one unique value
    constant_columns = df.nunique(dropna=False) == 1

    constant_df = df.loc[:, constant_columns]
    nonconstant_df = df.loc[:, ~constant_columns]

    if return_names:
        return constant_df.columns, nonconstant_df.columns
    else:
        return constant_df, nonconstant_df

# Get the absolute path of the directory where the notebook is located
data_folder = '/scratch/users/yinya/0_test_exercise'
file_name = os.path.join(data_folder, 'ntuple.18192176.0.root')

with uproot.open(file_name) as file:
    key = file.keys()[0]
    tree = uproot.open(f"{file_name}:" + key)
    print("File opened successfully!")
    print(f"Available classes: {file.classnames()}")
    print(f"Imported Class: {key}")
    print(f"Tree information: {tree}")

# Select Branches
theory_param = [branch for branch in tree.keys() if branch.startswith('IN')]

theory_tree = tree.arrays(theory_param, library='pd')
_, theory_param_nonconst = split_constant_nonconstant_columns(theory_tree, return_names=True)
del theory_tree

#If the data is saved as float64, there will be an error double!=float later
parameters = tree.arrays(theory_param_nonconst, library='pd').astype(np.float32)
mo_omega = tree['MO_Omega'].array(library="np").astype(np.float32)
mo_omega = mo_omega[:, None] #This is to prevent tensor dimension conflict when calculating loss

#We get rid of the samples where mo_omega is -1
mask = (mo_omega != -1).flatten()
parameters = parameters[mask]
mo_omega = mo_omega[mask]

# To make training easy for the normalizing flow, we standardize the parameters:

parameters_mean = np.mean(parameters, axis=0).to_numpy()
parameters_std = np.std(parameters, axis=0).to_numpy()

parameters_standardized = (parameters - parameters_mean) / parameters_std