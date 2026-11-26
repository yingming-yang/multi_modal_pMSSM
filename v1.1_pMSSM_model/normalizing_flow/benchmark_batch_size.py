"""
Benchmark training throughput as a function of batch size.

Builds the same flow and the same data pipeline as v1.1_norm_flow.py, then for
each candidate batch size times a handful of real training steps (forward,
backward, gradient clipping, optimizer step) and a handful of evaluation steps.
From those it extrapolates the per-epoch cost and the total cost of a full run.

Nothing is trained to convergence and no checkpoints are written -- this only
answers "how long would an epoch take at batch size B".

Why measure instead of scaling analytically: at these batch sizes the model is
CUDA-launch-overhead bound rather than arithmetic bound, so time-per-step is
almost flat in B and cannot be extrapolated from a single measurement. The
per-sample cost of pMSSM_Dataset.__getitem__ also grows with B, which is why the
real DataLoader is used here rather than pre-batched tensors.

Run (needs a GPU node):
    sbatch slurm/benchmark_batch_size.sbatch
or directly:
    pixi run python -u normalizing_flow/benchmark_batch_size.py -c slurm/train_norm_flow.yaml

Useful overrides:
    --batch_sizes 256 512 1024 2048 4096   candidate batch sizes
    --timed_steps 50                       training steps timed per batch size
    --data_fraction 0.05                   subsample AFTER loading; all 250 ROOT files
                                           are still read, so this shortens the timing
                                           loop but not the ~3 min startup
    --budget_hours 24                       wall-clock budget for the last column
"""
import os
import time
import argparse
import yaml
import numpy as np
import torch
import subprocess

from torch.utils.data import DataLoader, random_split

from glasflow.nflows import distributions, flows

from v1_1_pmssm_model.load_data import load_data
from v1_1_pmssm_model.norm_flow import *


def get_free_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    free_mem = [int(x) for x in result.stdout.strip().split("\n")]
    best_gpu = free_mem.index(max(free_mem))
    print(f"Selected GPU {best_gpu} ({free_mem[best_gpu]} MiB free)")
    return best_gpu


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None,
                    help='Path to the same yaml config used for training')
parser.add_argument('--batch_sizes', type=int, nargs='+',
                    default=[256, 512, 1024, 2048, 4096],
                    help='Batch sizes to compare')
parser.add_argument('--timed_steps', type=int, default=50,
                    help='Number of training steps to time per batch size')
parser.add_argument('--warmup_steps', type=int, default=5,
                    help='Steps run before timing starts (CUDA init, autotuning, caches)')
parser.add_argument('--eval_steps', type=int, default=20,
                    help='Number of evaluation steps to time per batch size')
parser.add_argument('--data_fraction', type=float, default=None,
                    help='Override data_fraction from the config. load_data reads every ROOT '
                         'file regardless, so this shortens the timing loop, not the startup.')
parser.add_argument('--num_epochs', type=int, default=None,
                    help='Override num_epochs from the config, for the total-time column')
parser.add_argument('--budget_hours', type=float, default=24.0,
                    help='Wall-clock budget used for the "epochs in budget" column')
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{get_free_gpu()}")
else:
    device = torch.device("cpu")
    print("WARNING: no GPU found. Timings from a CPU run say nothing about the "
          "GPU cost and must not be used to plan a job.")
print(f"Device: {device}")

# ---- hyperparameters (mirrors v1.1_norm_flow.py so the model matches) --------
num_flow_steps       = 3
hidden_dim           = 128
num_transform_blocks = 2
batch_norm           = False
base_transform_type  = "rq-coupling"
num_bins             = 8
tail_bound           = 3.0
learning_rate        = 1e-5
weight_decay         = 1e-5
max_grad_norm        = 5.0
num_epochs           = 1000
data_fraction        = 1
train_fraction       = 0.8
data_folder          = '/ptmp/yinya/gen_flat_28346825/ntuples'

if args.config is not None:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    num_flow_steps       = cfg.get('num_flow_steps',       num_flow_steps)
    hidden_dim           = cfg.get('hidden_dim',           hidden_dim)
    num_transform_blocks = cfg.get('num_transform_blocks', num_transform_blocks)
    batch_norm           = cfg.get('batch_norm',           batch_norm)
    base_transform_type  = cfg.get('base_transform_type',  base_transform_type)
    num_bins             = cfg.get('num_bins',             num_bins)
    tail_bound           = cfg.get('tail_bound',           tail_bound)
    learning_rate        = cfg.get('learning_rate',        learning_rate)
    weight_decay         = cfg.get('weight_decay',         weight_decay)
    max_grad_norm        = cfg.get('max_grad_norm',        max_grad_norm)
    num_epochs           = cfg.get('num_epochs',           num_epochs)
    data_fraction        = cfg.get('data_fraction',        data_fraction)
    train_fraction       = cfg.get('train_fraction',       train_fraction)
    data_folder          = cfg.get('data_folder',          data_folder)

if args.data_fraction is not None:
    data_fraction = args.data_fraction
if args.num_epochs is not None:
    num_epochs = args.num_epochs

# ---- data (loaded once and shared across all batch sizes) -------------------
load_start = time.perf_counter()
parameters, mo_omega, _, _, parameters_standardized, _, _ = load_data(
    data_folder=data_folder, data_fraction=data_fraction)
load_time = time.perf_counter() - load_start
print(f"Data loading and preprocessing took {load_time:.1f} s "
      f"(paid once per job, before any epoch)")

dataset = pMSSM_Dataset(parameters_standardized, mo_omega, device)

num_samples = len(dataset)
num_train = int(round(train_fraction * num_samples))
num_test = num_samples - num_train
# Same seed as the training script, so the split matches the real run.
train_dataset, test_dataset = random_split(
    dataset, [num_train, num_test], generator=torch.Generator().manual_seed(42))
print(f"Train samples: {num_train}   test samples: {num_test}")


def build_model():
    """Fresh model and optimizer, so no state carries between batch sizes."""
    base_transform_kwargs = {
        "base_transform_type": base_transform_type,
        "hidden_dim": hidden_dim,
        "num_transform_blocks": num_transform_blocks,
        "batch_norm": batch_norm,
    }
    if base_transform_type == "rq-coupling":
        base_transform_kwargs["num_bins"] = num_bins
        base_transform_kwargs["tail_bound"] = tail_bound

    transform = create_transform(
        num_flow_steps=num_flow_steps,
        param_dim=np.shape(parameters_standardized)[-1],
        context_dim=np.shape(mo_omega)[-1],
        base_transform_kwargs=base_transform_kwargs)
    model = flows.Flow(
        transform,
        distributions.StandardNormal(shape=[parameters_standardized.shape[-1]])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    return model, optimizer


def sync():
    """CUDA kernels are asynchronous; without this the timer would measure only
    how long it took to *queue* the work, not to run it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_train_steps(model, optimizer, loader, warmup, timed):
    """Median seconds per training step. Median, not mean, so that one stray
    slow step (page fault, another job on the node) does not skew the estimate."""
    model.train()
    per_step = []
    done = 0
    for X, y in loader:
        if done >= warmup + timed:
            break
        sync()
        t0 = time.perf_counter()
        loss = (- model.log_prob(y, X)).mean()
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        sync()
        dt = time.perf_counter() - t0
        if done >= warmup:          # discard warmup steps
            per_step.append(dt)
        done += 1
    return float(np.median(per_step)), len(per_step)


def time_eval_steps(model, loader, steps):
    """Median seconds per evaluation step (no backward, no optimizer)."""
    model.eval()
    per_step = []
    with torch.no_grad():
        done = 0
        for X, y in loader:
            if done >= steps:
                break
            sync()
            t0 = time.perf_counter()
            (- model.log_prob(y, X)).sum()
            sync()
            per_step.append(time.perf_counter() - t0)
            done += 1
    return float(np.median(per_step))


rows = []
for bs in args.batch_sizes:
    if bs > num_train:
        print(f"Skipping batch size {bs}: larger than the training set ({num_train})")
        continue

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model, optimizer = build_model()
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    try:
        t_train, n_timed = time_train_steps(
            model, optimizer, train_loader, args.warmup_steps, args.timed_steps)
        t_eval = time_eval_steps(model, test_loader, args.eval_steps)
    except torch.cuda.OutOfMemoryError:
        print(f"Batch size {bs}: out of memory, skipped")
        del model, optimizer
        continue

    peak_gb = (torch.cuda.max_memory_allocated(device) / 1024**3
               if device.type == "cuda" else float('nan'))

    train_steps = int(np.ceil(num_train / bs))
    test_steps = int(np.ceil(num_test / bs))
    epoch_s = train_steps * t_train + test_steps * t_eval
    total_h = epoch_s * num_epochs / 3600
    epochs_in_budget = args.budget_hours * 3600 / epoch_s
    # Optimizer steps and samples seen within the wall-clock budget: the two
    # quantities that actually compete when choosing a batch size.
    steps_in_budget = epochs_in_budget * train_steps
    samples_per_s = num_train / (train_steps * t_train)

    rows.append(dict(bs=bs, t_train=t_train, t_eval=t_eval, n_timed=n_timed,
                     train_steps=train_steps, epoch_s=epoch_s, total_h=total_h,
                     epochs_in_budget=epochs_in_budget,
                     steps_in_budget=steps_in_budget,
                     samples_per_s=samples_per_s, peak_gb=peak_gb))
    print(f"  batch {bs:>5d}: {t_train*1000:7.1f} ms/train step "
          f"({n_timed} steps timed), {t_eval*1000:6.1f} ms/eval step")

    del model, optimizer, train_loader, test_loader

# ---- report ----------------------------------------------------------------
print()
print(f"Model: {num_flow_steps} flow steps, hidden {hidden_dim}, "
      f"{num_transform_blocks} blocks, {num_bins} bins, tail_bound {tail_bound}")
print(f"Extrapolated over num_epochs = {num_epochs}, "
      f"budget = {args.budget_hours:g} h, excluding the "
      f"{load_time:.0f} s one-off data load")
print()
hdr = (f"{'batch':>6} {'ms/step':>9} {'steps/ep':>9} {'epoch [s]':>10} "
       f"{'total [h]':>10} {'epochs/budget':>14} {'steps/budget':>13} "
       f"{'samples/s':>11} {'peak [GB]':>10}")
print(hdr)
print('-' * len(hdr))
for r in rows:
    print(f"{r['bs']:>6d} {r['t_train']*1000:>9.1f} {r['train_steps']:>9d} "
          f"{r['epoch_s']:>10.1f} {r['total_h']:>10.2f} "
          f"{r['epochs_in_budget']:>14.0f} {r['steps_in_budget']:>13.0f} "
          f"{r['samples_per_s']:>11.0f} {r['peak_gb']:>10.2f}")

if rows:
    print()
    fastest = min(rows, key=lambda r: r['epoch_s'])
    most_steps = max(rows, key=lambda r: r['steps_in_budget'])
    print(f"Cheapest epoch:                  batch {fastest['bs']} "
          f"({fastest['epoch_s']:.1f} s/epoch)")
    print(f"Most optimizer steps in {args.budget_hours:g} h: batch "
          f"{most_steps['bs']} ({most_steps['steps_in_budget']:.0f} steps)")
    print()
    print("These two are usually different batch sizes, and that is the whole")
    print("trade-off: a large batch processes more samples per hour, a small one")
    print("takes more optimizer steps per hour. Which wins is empirical -- to")
    print("settle it, train two batch sizes for the SAME wall-clock time with the")
    print("learning rate retuned for each, then compare test loss.")
