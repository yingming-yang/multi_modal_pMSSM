"""
Loads and preprocesses pMSSM ntuple ROOT files for deep learning training.

Reads ROOT files named ntuple.<jobid>.<index>.root, extracts non-constant theory
parameters (IN* branches) and the relic density observable (MO_Omega), filters
unphysical samples (MO_Omega == -1), and standardizes the parameterst.
"""
import os
import re
import uproot
import numpy as np
import pandas as pd

def _split_constant_nonconstant_columns(df: pd.DataFrame, return_names = False):
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

def _ntuple_index(filename):
    match = re.search(r'ntuple\.\d+\.(\d+)\.root', filename)
    return int(match.group(1)) if match else -1

def _collect_files(data_folders):
    file_names = []
    for folder in data_folders:
        file_names.extend(sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.root')],
            key=lambda p: _ntuple_index(os.path.basename(p))
        ))
    return file_names

def load_data(data_folder='/ptmp/yinya/gen_flat_28346825/ntuples'):
    """
    Load and preprocess pMSSM ntuple ROOT files.

    Args:
        data_folder: Path or list of paths to directories containing ntuple ROOT files.

    Returns:
        parameters:           (N, D) float32 array of non-constant theory parameters, filtered.
        mo_omega:             (N, 1) float32 array of relic density values, filtered.
        parameters_mean:      (D,) mean of parameters after filtering, used for standardization.
        parameters_std:       (D,) std of parameters after filtering, used for standardization.
        parameters_standardized: (N, D) standardized parameters.
        theory_param:         List of all IN* branch names found in the files.
        theory_param_nonconst: Subset of theory_param with non-constant values across the dataset.
    """
    print('---Load pMSSM dataset---')

    data_folders = [data_folder] if isinstance(data_folder, str) else data_folder
    file_names = _collect_files(data_folders)

    # Determine non-constant theory parameters from the first file
    with uproot.open(file_names[0]) as file:
        key = file.keys()[0]
        tree = file[key]
        theory_param = [branch for branch in tree.keys() if branch.startswith('IN')]
        theory_tree = tree.arrays(theory_param, library='pd')
        _, theory_param_nonconst = _split_constant_nonconstant_columns(theory_tree, return_names=True)

    # Load and combine all files
    all_parameters = []
    all_mo_omega = []
    for i, file_name in enumerate(file_names):
        with uproot.open(file_name) as file:
            key = file.keys()[0]
            tree = file[key]
            chunk = tree.arrays(theory_param_nonconst, library='np')
            all_parameters.append({k: v.astype(np.float32) for k, v in chunk.items()})
            all_mo_omega.append(tree['MO_Omega'].array(library="np").astype(np.float32))
            print(f"Loading files: {i+1}/{len(file_names)}", end='\r', flush=True)

    parameters = np.concatenate([np.stack(list(chunk.values()), axis=1) for chunk in all_parameters])
    mo_omega = np.concatenate(all_mo_omega)[:, None] #This is to prevent tensor dimension conflict when calculating loss
    print(f"Loaded {len(file_names)} files from {data_folders}\nTotal samples: {len(parameters)}")

    #We get rid of the samples where mo_omega is -1
    mask = (mo_omega != -1).flatten()
    parameters = parameters[mask]
    mo_omega = mo_omega[mask]
    print(f"Number of samples after filter on relic density: {len(parameters)}")

    # To make training easy for the normalizing flow, we standardize the parameters:
    parameters_mean = np.mean(parameters, axis=0)
    parameters_std = np.std(parameters, axis=0)

    parameters_standardized = (parameters - parameters_mean) / parameters_std

    return parameters, mo_omega, parameters_mean, parameters_std, parameters_standardized, theory_param,  theory_param_nonconst

parameters, mo_omega, parameters_mean, parameters_std, parameters_standardized, theory_param,  theory_param_nonconst = load_data()
