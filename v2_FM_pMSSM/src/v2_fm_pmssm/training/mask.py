import torch

def make_masks(device):
    # 0: inputs known -> predict Omega; 1: Omega known -> predict inputs; 2: unconditional
    return torch.tensor([
        [True] * 12 + [False],
        [False] * 12 + [True],
        [False] * 13,
    ], device=device)

