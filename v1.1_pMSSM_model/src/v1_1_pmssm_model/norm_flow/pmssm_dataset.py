import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

# Mask sampling for the arbitrary-conditioning dataset below; see masking.py
# in this package for the corresponding masked coupling transform/flow.
from .masking import sample_conditioning_mask


class pMSSM_Dataset(Dataset):
    def __init__(self, parameters, observables, device):
        # device: str or torch.device
        if isinstance(parameters, pd.core.frame.DataFrame):
            self.parameters = torch.as_tensor(parameters.to_numpy(), dtype=torch.float32, device=torch.device(device))
            self.columns = parameters.columns
        else:
            self.parameters = torch.as_tensor(parameters, dtype=torch.float32, device=torch.device(device))
        self.observables = torch.as_tensor(observables, dtype=torch.float32, device=torch.device(device))

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        return self.observables[idx], self.parameters[idx]
        
    def setDevice(self, device):
        self.parameters = self.parameters.to(device)
        self.observables = self.observables.to(device)


class MaskedPMSSM_Dataset(Dataset):
    """Joint (parameters, MO_Omega) dataset for the arbitrary-conditioning flow.

    Each __getitem__ call returns the full standardized joint vector
    `v = concat(parameters, mo_omega)` together with a freshly-sampled
    per-example "unknown" mask (see masking.sample_conditioning_mask), so the
    conditioning direction varies every epoch rather than being fixed to
    params-given-omega. Both must already be standardized (mean 0, std 1)
    before construction, since they share tail_bound/spline range in
    masking.MaskedRQCouplingStep.
    """

    def __init__(self, parameters_standardized, mo_omega_standardized, device,
                 min_unknown=1, canonical_masks=None, canonical_prob=0.0):
        if isinstance(parameters_standardized, pd.core.frame.DataFrame):
            parameters_standardized = parameters_standardized.to_numpy()
        self.joint = np.concatenate([parameters_standardized, mo_omega_standardized], axis=-1)
        self.total_dim = self.joint.shape[-1]
        self.device = torch.device(device)
        self.min_unknown = min_unknown
        self.canonical_masks = canonical_masks
        self.canonical_prob = canonical_prob

    def __len__(self):
        return len(self.joint)

    def __getitem__(self, idx):
        dtype = torch.float32 if self.device.type == "mps" else None
        v = torch.tensor(self.joint[idx], dtype=dtype, device=self.device)
        mask = sample_conditioning_mask(
            batch_size=1,
            total_dim=self.total_dim,
            device=self.device,
            min_unknown=self.min_unknown,
            canonical_masks=self.canonical_masks,
            canonical_prob=self.canonical_prob,
        ).squeeze(0).to(v.dtype)
        return v, mask

    def setDevice(self, device):
        self.device = torch.device(device)

