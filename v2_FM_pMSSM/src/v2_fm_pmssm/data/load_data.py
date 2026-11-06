import os
import numpy as np
import re
import uproot
from pathlib import Path

ID_DTYPE = np.dtype([("job", "i8"), ("file", "i4"), ("entry", "i4")])

def _ntuple_index(filename):
    match = re.fullmatch(r'ntuple\.(\d+)\.(\d+)\.root', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')

def _collect_files(data_folders, max_files=None):
    """
    Collects all .root files from the data folders and their direct subfolders
    Sorts them based on the ntuple index extracted from their filenames
    - data_folders: a list of folder paths or a single folder path
    - max_files: limits the maximum number of files; Default: no limit
    """
    if isinstance(data_folders, (str, os.PathLike)):
        data_folders = [data_folders]

    file_names = []
    for folder in data_folders:
        folder = Path(folder)
        file_names.extend(
            path
            for pattern in ("ntuple.*.*.root", "*/ntuple.*.*.root")
            for path in folder.glob(pattern)
            if path.is_file()
        )

    file_names.sort(
        key=lambda path: _ntuple_index(os.path.basename(path))
    )

    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be a positive integer")
        file_names = file_names[:max_files]

    return file_names

def load_data(
        data_folder='/u/yinya/archive/gen_flat_v2/ntuples',
        param_names=None,
        max_files=None,
        exact_paths=None
    ):
    """
    Load and preprocess pMSSM ntuple ROOT files.

    Args:
        data_folder:    Path or list of paths to directories containing ntuple ROOT files;
                        ROOT files in the direct subdirectories can also be found.
        param_names:    List of parameter names needed.
        max_files:      Max number of files used in the training. (Default: no limit)
        exact_paths:    List of String. Paths of files that should be loaded; Used for exact
                        reproduction; Overrides data_folder and max_files. (Default: None)

    Returns:
        parameters_dict:      A dictionary of param_names and parameters_table
        parameters_table:     (N, D) float32 array of parameters.
        file_names:           List of Paths; Can be saved for later reproduction.
        point_ids:            (N,) structured array (job, file, entry), row-aligned with parameters_table.
    """
    print('---Load pMSSM dataset v2---')

    if exact_paths is None:
        data_folders = [data_folder] if isinstance(data_folder, str) else data_folder
        file_names = _collect_files(data_folders, max_files)
    else:
        file_names = exact_paths
    
    param_names = [param_names] if isinstance(param_names, str) else param_names
    if param_names is None:
        param_names = ['IN_M_1', 'IN_M_2', 'IN_M_3', 'IN_At', 'IN_Ab', 'IN_Atau', 'IN_mu', 
                       'IN_tanb', 'IN_mA',  'IN_mqL3', 'IN_mtR', 'IN_mbR', 'MO_Omega']

    # Load and combine all files
    parameters_table = []
    point_ids = []
    for i, file_name in enumerate(file_names):
        job, file_idx = _ntuple_index(os.path.basename(file_name))
        if job == float('inf'):
            raise ValueError(f"Cannot read job/file index from file name: {file_name}")
        with uproot.open(file_name) as file:
            key = file.keys()[0]
            tree = file[key]
            chunk = tree.arrays(param_names, library='np') # (N, D)
            parameters_table.append({k: v.astype(np.float32) for k, v in chunk.items()})

            ids = np.empty(tree.num_entries, dtype=ID_DTYPE)
            ids["job"], ids["file"], ids["entry"] = job, file_idx, np.arange(tree.num_entries)
            point_ids.append(ids)
            print(f"Loading files: {i+1}/{len(file_names)}", end='\r', flush=True)
    print()
    parameters_table = np.concatenate([np.stack([chunk[n] for n in param_names], axis=1) for chunk in parameters_table])
    point_ids = np.concatenate(point_ids) # (N,) ID_DTYPE, same row order as parameters_table
    parameters_dict = {
        name: parameters_table[:, i]
        for i, name in enumerate(param_names)
    }
    print(f"Loaded {len(file_names)} files from {data_folders if exact_paths is None else 'given paths'}")
    print(f"Total samples: {len(parameters_table)}")

    return parameters_dict, parameters_table, point_ids, file_names