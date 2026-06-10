import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import subprocess

from torch.utils.data import DataLoader, random_split

from glasflow.nflows import distributions, flows

from python.transforms import *
from python.pmssm_dataset import *
from python.loops import *
from python.load_data import *

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


id_training = 7
num_flow_steps = 3
learning_rate = 1e-5
weight_decay = 1e-5
num_epochs = 2

doc_folder = '/scratch/users/yinya/1.0_toy_model/'

with open(os.path.join(doc_folder, 'figures/document.txt'), 'a') as file:
    file.write(f'training_{id_training} = {{num_flow_steps={num_flow_steps}, lr={learning_rate}, weight_decay={weight_decay}, epochs={num_epochs}}}' + '\n')

# ### Data loading and preprocessing for training

mo_omega_dataset = pMSSM_Dataset(parameters_standardized, mo_omega, device)

# ### Split data into train and test dataset
# Determine the number of training and test samples from a `train_fraction` and use the function `random_split()` to separate the data into a training and test dataset. Afterwards define the training and test dataloaders.

num_samples = len(mo_omega_dataset)
train_fraction = 0.8
num_train = int(round(train_fraction * num_samples))
num_test = num_samples - num_train
generator1 = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(mo_omega_dataset, [num_train, num_test], generator=generator1)

# The DataLoader is used in training
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ### Conditional Normalizing Flow
# We use the `glasflow` library. It is a wrapper for [`nflows`](https://github.com/bayesiains/nflows), which implements different flow types (e.g. autoregressive flows and coupling flows) and transforms (e.g. rational-quadratic splines).
# 

transform = create_transform(num_flow_steps = num_flow_steps,
                             param_dim = np.shape(parameters)[-1],
                             context_dim = np.shape(mo_omega)[-1], #np.shape(data) instead of data.shape to avoid error by one-dim array
                             base_transform_kwargs = {"base_transform_type": "rq-coupling"})

base_distribution = distributions.StandardNormal(shape=[parameters.shape[-1]])

model = flows.Flow(transform, base_distribution)
model = model.to(device) #Is equivalent to just "model.to(device)", but gets rid of the verbose log

# ### Training

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_history = []
test_history = []
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_history.append(loss)
    loss = test_loop(test_dataloader, model)
    test_history.append(loss)
    if t % 1000 == 0: torch.save(model.state_dict(), f"{doc_folder}checkpoints/training_{id_training}_epoch_{t+1}.pt")
print("Done!")

torch.save(model.state_dict(), f"{doc_folder}checkpoints/training_{id_training}.pt")

epochs = np.arange(1, len(train_history) + 1)
plt.plot(epochs, train_history, '-o', label = 'train loss')
plt.plot(epochs, test_history, '-o', label = 'test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{doc_folder}figures/loss_{id_training}.png')

np.save(f'{doc_folder}figures/train_history_{id_training}.npy', train_history)
np.save(f'{doc_folder}figures/test_history_{id_training}.npy', test_history)

