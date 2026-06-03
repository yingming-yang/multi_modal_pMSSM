import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import corner
import uproot
import pandas as pd
import subprocess

from torch.utils.data import Dataset, DataLoader, random_split

from glasflow.nflows import transforms, distributions, flows
import glasflow.nflows as nflows
import glasflow.nflows.nn.nets as nflows_nets

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


# THe Code is adapted from examples in the [neural spline flow repository](https://github.com/bayesiains/nsf).
# 
# First, we define a function that creates the basic MAF or RQS-coupling transform:


def create_base_transform(
    i: int,
    param_dim: int,
    context_dim: int = None,
    hidden_dim: int = 512,
    num_transform_blocks: int = 2,
    batch_norm: bool = False,
    base_transform_type: str = "rq-coupling",
):

    activation_fn = torch.nn.ELU()

    if base_transform_type == "maf":
        return transforms.MaskedAffineAutoregressiveTransform(
            param_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_blocks=num_transform_blocks,
            activation=activation_fn,
            use_batch_norm=batch_norm,
        )
    elif base_transform_type == "rq-coupling":
        if param_dim == 1:
            mask = torch.tensor([1], dtype=torch.uint8)
        else:
            mask = nflows.utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)
            )
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=(
                lambda in_features, out_features: nflows_nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=num_transform_blocks,
                    activation=activation_fn,
                    use_batch_norm=batch_norm,
                )
            ),
            num_bins=8,
            tails="linear",
            tail_bound=1.0,
            apply_unconditional_transform=False,
        )

    else:
        raise ValueError


# Next, we need to define a transform the randomly permutes the dimensions to ensure that all dimensions interact sufficiently with each other:

def create_linear_transform(param_dim: int):
    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


# Finally, we can define the `create_transform` function which composes a list of `num_flow_steps` transforms where each transform consists of a linear transform (to permute dimensions) and a base transform (MAF or RQ-coupling).
# We start and finish the transforms with a linear transform that ensures that the flow outputs samples that have the correct dimensionality (`param_dim`).

def create_transform(
    num_flow_steps: int, 
    param_dim: int, 
    context_dim: int, 
    base_transform_kwargs: dict
):
    return transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(param_dim),
                    create_base_transform(
                        i, param_dim, context_dim=context_dim, **base_transform_kwargs
                    ),
                ]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform(param_dim)]
    )


# The entire flow model consists of a standard normal base distribution, followed by the flow transform.

# Training and test loops

def train_loop(dataloader, model, optimizer):

    model.train()

    size = len(dataloader.dataset)
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute negative log probability loss
        loss = - model.log_prob(y, X)

        train_loss += loss.detach().sum()
        loss = loss.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d} samples]")

    average_loss = train_loss.item() / size
    print('Average loss: {:.4f}'.format(average_loss))
    return average_loss


def test_loop(dataloader, model):

    model.eval()

    size = len(dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            loss = - model.log_prob(y, X)
            test_loss += loss.sum()

    test_loss = test_loss.item() / size
    print(f"Test loss: {test_loss:>8f} \n")
    return test_loss



class pMSSM_Dataset(Dataset):
    def __init__(self, parameters, observables):
        self.parameters = parameters
        self.observables = observables

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        params = self.parameters.iloc[idx]
        signal = self.observables[idx]

        # Add unit normal noise to the signal
        # noise = np.random.normal(size = signal.shape).astype(np.float32)
        # data = signal + noise
        data = signal
        if device == "mps":
            return torch.tensor(data, dtype=torch.float32, device=device), torch.tensor(params,  dtype=torch.float32, device=device)
        else:
            return torch.tensor(data, device=device), torch.tensor(params, device=device)

id_training = 6
num_flow_steps = 10
learning_rate = 1e-5
weight_decay = 1e-5
epochs = 3000

with open('../figures/document.txt', 'a') as file:
    file.write(f'training_{id_training} = {{num_flow_steps={num_flow_steps}, lr={learning_rate}, weight_decay={weight_decay}, epochs={epochs}}}' + '\n')

# ### Data loading and preprocessing for training
# Import Data

# If running on Apple Silicon or CUDA is available, use the GPU.

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
    current_device_number = get_free_gpu()
    torch.cuda.set_device(current_device_number)
else:
    device = "cpu"

# Get the absolute path of the directory where the notebook is located
cwd = os.getcwd()
print("Current working directory", cwd)
data_folder = os.path.dirname(os.path.dirname(cwd))
data_folder = os.path.join(data_folder, '0_test_exercise')
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

parameters_mean = np.mean(parameters, axis=0)
parameters_std = np.std(parameters, axis=0)

parameters_standardized = (parameters - parameters_mean) / parameters_std

mo_omega_dataset = pMSSM_Dataset(parameters_standardized, mo_omega)

# Get first sample
signal, params = mo_omega_dataset[0]
signal = signal.cpu()  # Careful to put data back onto CPU.
print(signal)
print(params)


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


device = 'cpu'  # For some reason, "mps" does not work here.

transform = create_transform(num_flow_steps = num_flow_steps,
                             param_dim = np.shape(parameters)[-1],
                             context_dim = np.shape(mo_omega)[-1], #np.shape(data) instead of data.shape to avoid error by one-dim array
                             base_transform_kwargs = {"base_transform_type": "rq-coupling"})

base_distribution = distributions.StandardNormal(shape=[parameters.shape[-1]])

model = flows.Flow(transform, base_distribution)
model.to(device)

# ### Training


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_history = []
test_history = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_history.append(loss)
    loss = test_loop(test_dataloader, model)
    test_history.append(loss)
    if t % 1000 == 0: torch.save(model.state_dict(), f"../checkpoints/training_{id_training}_epoch_{t+1}.pt")
print("Done!")

torch.save(model.state_dict(), f"../checkpoints/training_{id_training}.pt")

epochs = np.arange(1, len(train_history) + 1)
plt.plot(epochs, train_history, '-o', label = 'train loss')
plt.plot(epochs, test_history, '-o', label = 'test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'../figures/loss_{id_training}.png')

np.save(f'../figures/train_history_{id_training}.npy', train_history)
np.save(f'../figures/test_history_{id_training}.npy', test_history)

# ### Visualize posteriors

# text_x, _ = test_dataset[0]

# num_posteriors = 3
# num_eval_samples = 10_000

# model.eval()

# for n in range(num_posteriors):

#     with torch.no_grad():

#         test_x, test_y = test_dataset[n]

#         # Sample the posterior
#         test_x = test_x.expand(num_eval_samples, *test_x.shape)
#         pred_samples = model.sample(1, test_x).squeeze(1).cpu().numpy()

#         # Undo the standardization
#         truth = parameters_std * test_y.cpu().numpy() + parameters_mean

#         # Plot
#         corner.corner(pred_samples, truths=truth, labels=['$m_1$', '$m_2$'])
#         plt.show()

