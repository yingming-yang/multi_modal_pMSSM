import torch
from torch.utils.data import Dataset

class pMSSM_Dataset(Dataset):
    def __init__(self, parameters, observables, device):
        self.parameters = parameters
        self.observables = observables
        self.device = torch.device(device) # also works if device has type torch.device

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        params = self.parameters.iloc[idx]
        data = self.observables[idx]

        if self.device.type == "mps":
            return torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(params,  dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(data, device=self.device), torch.tensor(params, device=self.device)
        
    def setDevice(self, device):
        self.device = torch.device(device)

