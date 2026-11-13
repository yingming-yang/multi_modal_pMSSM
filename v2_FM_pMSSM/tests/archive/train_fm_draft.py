# Script version of tests/training.ipynb.
# Changes vs. the notebook are marked with "# CHANGED:".
import os  # CHANGED: needed for output file names
from v2_fm_pmssm.data import load_data, pMSSM_Dataset, select, all_var, split_indices
from v2_fm_pmssm.training import Encoder_Transformer, make_masks

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from x_transformers import ContinuousTransformerWrapper, Encoder

import numpy as np
import matplotlib
matplotlib.use("Agg")  # CHANGED: no display on the compute node
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import sklearn.datasets as skld
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# CHANGED: outputs go next to this script, named after the Slurm job id
out_prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"fm_{os.environ.get('SLURM_JOB_ID', 'local')}")

# parameter, parameter_table, point_id, _ = load_data('/u/yinya/archive/gen_flat_v2_test/ntuples')
parameter, parameter_table, point_id, _ = load_data('/u/yinya/archive/gen_flat_v2/ntuples', max_files=20)  # CHANGED: all files (notebook: max_files=20)

var_registry = select(all_var)
n_features = len(var_registry)
dataset = pMSSM_Dataset(parameter_table, point_id, var_registry, device)

dataset = dataset.filter_invalid(branches=['MO_Omega'])
print(len(dataset))  # CHANGED: print instead of bare expression

train_idx, val_idx, test_idx = split_indices(len(dataset), (0.8, 0.1, 0.1), seed=42)
train, val, test = (dataset.subset(i) for i in (train_idx, val_idx, test_idx))
mean, std = train.transform()
val.transform(mean, std); test.transform(mean, std)

# hyperparameters
batch_size = 1000
learning_rate = 3e-4   # CHANGED: notebook 1e-3, lowered for a 16-layer model without warmup
max_iters = 1000      # CHANGED: notebook 120, ~17 epochs over the full training set
eval_interval = 100    # CHANGED: notebook 10, gives 100 points on the loss curve
n_embd = 256
n_heads = 16
n_layers = 16
dropout = 0
# ------------

# TODO: temperary, last small batch discarded. Fix it.
index = 0
def sample_data(N, dataset:pMSSM_Dataset):
    global index
    if index + N <= len(dataset):
        index = index + N
        return dataset[index - N:index]
    index = N
    return dataset[0:N]

masks = make_masks(device)

val_indices = torch.randint(len(masks), (batch_size,), device=device)
val_target = sample_data(batch_size, val)
val_times = torch.rand(batch_size, 1, device=device)
val_noise = torch.randn(batch_size, 13, device=device)
val_batch = (val_indices, val_target, val_times, val_noise)

path = AffineProbPath(scheduler=CondOTScheduler())

def train_val_loop(model, validation=False):

    if validation:
        indices, target, times, noise = val_batch
    else:
        indices = torch.randint(len(masks), (batch_size,), device=device)
        target = sample_data(batch_size, train)
        times = torch.rand(batch_size, 1, device=device)
        noise = torch.randn(batch_size, 13, device=device)

    known_masks = masks[indices]
    path_sample = path.sample(x_0=noise, x_1=target, t=times.squeeze(-1))
    positions = torch.where(known_masks, target, path_sample.x_t)
    xt = torch.column_stack([positions, times])
    model_field = model(xt, known=known_masks)

    unknown = ~known_masks
    loss = ((model_field - path_sample.dx_t)**2 * unknown).sum() / unknown.sum()

    return loss

def train_transformer_FM(model):
    opt = torch.optim.Adam(model.parameters(), learning_rate)

    loss_history = []
    for i in range(max_iters):

        loss = train_val_loop(model)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % eval_interval == 0 or i == max_iters - 1:
            with torch.no_grad():
                model.eval()

                loss_val = train_val_loop(model, validation=True)
                loss_history.append([loss.item(), loss_val.item()])

                if i % eval_interval == 0:
                    print(f"step {i}: train loss {loss:.4f}, val loss {loss_val:.4f}", flush=True)  # CHANGED: flush for the Slurm log
                
                model.train()
    return model, loss_history


vf = Encoder_Transformer(n_embd, n_layers, n_heads, n_features).to(device)
vf = vf.to(device)

vf.train()
vf, loss_history = train_transformer_FM(vf)

loss_history = torch.tensor(loss_history).cpu().detach().numpy()
plt.plot(np.arange(loss_history.shape[0])*eval_interval, loss_history[:,0])
plt.plot(np.arange(loss_history.shape[0])*eval_interval, loss_history[:,1])
plt.savefig(f"{out_prefix}_loss.png")  # CHANGED: save instead of plt.show()

# CHANGED: save weights and loss history
torch.save(vf.state_dict(), f"{out_prefix}_weights.pt")
np.save(f"{out_prefix}_loss_history.npy", loss_history)
print(f"saved {out_prefix}_weights.pt, {out_prefix}_loss_history.npy, {out_prefix}_loss.png")
