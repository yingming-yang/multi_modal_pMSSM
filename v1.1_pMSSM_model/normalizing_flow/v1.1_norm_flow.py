"""
Training script for a conditional normalizing flow on pMSSM data.

Trains a rational-quadratic coupling flow (RQ-NSF) to learn the conditional
distribution p(parameters | MO_Omega). Hyperparameters are logged as JSON and
model weights are saved as checkpoints. Loss curves are saved as figures and
numpy arrays for later analysis.
"""
import os
import json
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import subprocess

from torch.utils.data import DataLoader, random_split

from glasflow.nflows import distributions, flows

from v1_1_pmssm_model.load_data import load_data
from v1_1_pmssm_model.norm_flow import *

print("Project: Multi-Modal pMSSM")
print("Architecture: Normalizing Flows")
# returns the serial number of the gpu with most free memory
def get_free_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    free_mem = [int(x) for x in result.stdout.strip().split("\n")]
    best_gpu = free_mem.index(max(free_mem))
    print(f"Selected GPU {best_gpu} ({free_mem[best_gpu]} MiB free)")
    return best_gpu

# If running on Apple Silicon or CUDA is available, use the GPU.

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    current_gpu_number = get_free_gpu()
    device = torch.device(f"cuda:{current_gpu_number}")
else:
    device = torch.device("cpu")

# Hyperparameters and documentation

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to yaml config file. If not given, defaults in this file are used.')
parser.add_argument('--test_run', action='store_true', help='If set, results are saved under id_training=test and overwritten each run')
parser.add_argument('--comment', type=str, default=None, help='Free-text note about this run, saved alongside the hyperparameters')
args = parser.parse_args()

# Default hyperparameters — overridden by --config if provided
num_flow_steps      = 3
hidden_dim          = 128
num_transform_blocks = 2
batch_norm          = False
base_transform_type = "rq-coupling"
num_bins            = 8  # only used when base_transform_type == "rq-coupling"
learning_rate       = 1e-5
weight_decay        = 1e-5
num_epochs          = 2
data_fraction       = 1  # small fraction used for testing
train_fraction      = 0.8
batch_size          = 64
doc_folder          = '/u/yinya/workspace/multi_modal_pMSSM/v1.1_pMSSM_model/normalizing_flow'
data_folder         = '/ptmp/yinya/gen_flat_28346825/ntuples'

if args.config is not None:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    num_flow_steps      = cfg.get('num_flow_steps',      num_flow_steps)
    hidden_dim          = cfg.get('hidden_dim',          hidden_dim)
    num_transform_blocks = cfg.get('num_transform_blocks', num_transform_blocks)
    batch_norm          = cfg.get('batch_norm',          batch_norm)
    base_transform_type = cfg.get('base_transform_type', base_transform_type)
    num_bins            = cfg.get('num_bins',            num_bins)
    learning_rate       = cfg.get('learning_rate',       learning_rate)
    weight_decay        = cfg.get('weight_decay',        weight_decay)
    num_epochs          = cfg.get('num_epochs',          num_epochs)
    data_fraction       = cfg.get('data_fraction',       data_fraction)
    train_fraction      = cfg.get('train_fraction',      train_fraction)
    batch_size          = cfg.get('batch_size',          batch_size)
    doc_folder          = cfg.get('doc_folder',          doc_folder)
    data_folder         = cfg.get('data_folder',         data_folder)

param_folder = os.path.join(doc_folder, 'hyperparams')
os.makedirs(param_folder, exist_ok=True)

if args.test_run:
    id_training = 'test'
else:
    existing_ids = [
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(param_folder)
        if f.startswith('training_') and f.endswith('.json') and f.split('_')[1].split('.')[0].isdigit()
    ]
    id_training = max(existing_ids) + 1 if existing_ids else 1

checkpoints_folder = os.path.join(doc_folder, 'checkpoints')
figures_folder = os.path.join(doc_folder, 'figures', f'training_{id_training}')
os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

hyperparams = {
    "model": "one-direction ResNet-based normalizing flow",
    "id_training": id_training,
    "num_flow_steps": num_flow_steps,
    "hidden_dim": hidden_dim,
    "num_transform_blocks": num_transform_blocks,
    "batch_norm": batch_norm,
    "base_transform_type": base_transform_type,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "num_epochs": num_epochs,
    "data_fraction": data_fraction,
    "train_fraction": train_fraction,
    "batch_size": batch_size,
    "data_folder": data_folder,
}
if base_transform_type == "rq-coupling":
    hyperparams["num_bins"] = num_bins
if args.comment:
    hyperparams["comment"] = args.comment
with open(os.path.join(param_folder, f'training_{id_training}.json'), 'w') as f:
    json.dump(hyperparams, f, indent=4)

# ### Data loading and preprocessing for training
parameters, mo_omega, parameters_mean, parameters_std, parameters_standardized, theory_param, theory_param_nonconst = load_data(data_folder=data_folder, data_fraction=data_fraction)

mo_omega_dataset = pMSSM_Dataset(parameters_standardized, mo_omega, device)

# ### Split data into train and test dataset
# Determine the number of training and test samples from a `train_fraction` and use the function `random_split()` to separate the data into a training and test dataset. Afterwards define the training and test dataloaders.

num_samples = len(mo_omega_dataset)
num_train = int(round(train_fraction * num_samples))
num_test = num_samples - num_train
generator1 = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(mo_omega_dataset, [num_train, num_test], generator=generator1)

print(f"Training sample size: {num_train}")

# The DataLoader is used in training
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ### Conditional Normalizing Flow
# We use the `glasflow` library. It is a wrapper for [`nflows`](https://github.com/bayesiains/nflows), which implements different flow types (e.g. autoregressive flows and coupling flows) and transforms (e.g. rational-quadratic splines).
# 

base_transform_kwargs = {
    "base_transform_type": base_transform_type,
    "hidden_dim": hidden_dim,
    "num_transform_blocks": num_transform_blocks,
    "batch_norm": batch_norm,
}
if base_transform_type == "rq-coupling":
    base_transform_kwargs["num_bins"] = num_bins

transform = create_transform(num_flow_steps = num_flow_steps,
                             param_dim = np.shape(parameters_standardized)[-1],
                             context_dim = np.shape(mo_omega)[-1], #np.shape(data) instead of data.shape to avoid error by one-dim array
                             base_transform_kwargs = base_transform_kwargs)

base_distribution = distributions.StandardNormal(shape=[parameters_standardized.shape[-1]])

model = flows.Flow(transform, base_distribution)
model = model.to(device) #Is equivalent to just "model.to(device)", but gets rid of the verbose log

# ### Training

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_history = []
test_history = []
best_test_loss = np.inf
min_delta = 5e-3
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_history.append(loss)
    loss = test_loop(test_dataloader, model)
    test_history.append(loss)
    if t > 0 and (t + 1) % 200 == 0: torch.save(model.state_dict(), os.path.join(checkpoints_folder, f'training_{id_training}_epoch_{t+1}.pt'))
    if t % 10 == 0:
        torch.save({'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_loss': best_test_loss,
                    'train_history': train_history,
                    'test_history': test_history,
                    }, os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt.tmp'))
        os.replace(os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt.tmp'),
                   os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt'))
    if loss < best_test_loss - min_delta:
        torch.save(model.state_dict(), os.path.join(checkpoints_folder, f'training_{id_training}_best.pt'))
        best_test_loss = loss
print("Training done!")

torch.save({'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_loss': best_test_loss,
            'train_history': train_history,
            'test_history': test_history,
            }, os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt.tmp'))
os.replace(os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt.tmp'),
           os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt'))
print(f"Final weights saved under {os.path.join(checkpoints_folder, f'training_{id_training}_latest.pt')}")

epochs = np.arange(1, len(train_history) + 1)
plt.plot(epochs, train_history, '-o', label = 'train loss')
plt.plot(epochs, test_history, '-o', label = 'test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(figures_folder, f'loss_{id_training}.png'))
