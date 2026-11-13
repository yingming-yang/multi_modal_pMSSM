"""Flow-matching training of a transformer velocity field on pMSSM data.

Converted from tests/training.ipynb. Differences from the notebook:
  * training batches come from the normalized training split, shuffled each epoch
  * validation uses the whole validation split with fixed noise/times/masks
  * AdamW with linear warmup + cosine decay, gradient clipping

Outputs (prefix = <out_dir>/<run_name>/<run_name>, out_dir defaults to tests/checkpoints):
  <prefix>_weights.pt       dict with model_state, hparams, mean, std, file_names, ...
  <prefix>_loss_history.npz arrays: step, train_loss, val_loss, lr
  <prefix>_loss.png         train/val loss curve
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from v2_fm_pmssm.data import load_data, pMSSM_Dataset, select, all_var, split_indices
from v2_fm_pmssm.training import Encoder_Transformer, make_masks, EpochSampler, fm_loss, evaluate, lr_at

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML file with hyperparameters; command-line arguments override it")
    p.add_argument("--data_folder", default="/u/yinya/archive/gen_flat_v2/ntuples")
    p.add_argument("--branches", nargs="+", default=None)  # passed to select(); None = all variables
    p.add_argument("--steps", nargs="+", default=None)     # passed to select(), used only when branches is None
    p.add_argument("--data_type", nargs="+", default=None) # passed to select() (modality), used only when branches is None
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"))  # tests/checkpoints
    p.add_argument("--run_name", default=f"fm_{os.environ.get('SLURM_JOB_ID', 'local')}")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--warmup_frac", type=float, default=0.02)   # fraction of max_iters
    p.add_argument("--warmup_iters", type=int, default=None)    # optional absolute override of warmup_frac
    p.add_argument("--eval_interval", type=int, default=1_000)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    config_args, _ = p.parse_known_args()
    if config_args.config is not None:  # YAML values replace the defaults above; the command line still wins
        with open(config_args.config) as f:
            config = yaml.safe_load(f) or {}
        unknown = set(config) - set(vars(config_args))
        if unknown:
            raise ValueError(f"unknown keys in {config_args.config}: {sorted(unknown)}")
        p.set_defaults(**config)
    args = p.parse_args()
    if args.warmup_iters is None:
        args.warmup_iters = int(args.warmup_frac * args.max_iters)
    if args.warmup_iters >= args.max_iters:
        raise ValueError(f"warmup_iters ({args.warmup_iters}) must be smaller than max_iters ({args.max_iters})")
    hparams = vars(args)
    print(json.dumps(hparams, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(args.seed)
    g = torch.Generator(device=device).manual_seed(args.seed)

    # ---- data ----------------------------------------------------------------
    var_registry = select(all_var, branches=args.branches, steps=args.steps, data_type=args.data_type)
    num_var = len(var_registry)
    if len(var_registry) != 13:  # make_masks, the model and the noise below still assume 12 IN + MO_Omega
        raise ValueError(f"selected {len(var_registry)} variables, but masks/model are fixed to 13")
    _, parameter_table, point_id, file_names = load_data(args.data_folder, param_names=[v.branch for v in var_registry],
                                                         max_files=args.max_files)
    dataset = pMSSM_Dataset(parameter_table, point_id, var_registry, device)
    dataset = dataset.filter_invalid(branches=["MO_Omega"])
    print(f"valid samples: {len(dataset)}")
    train_idx, val_idx, _ = split_indices(len(dataset), (0.8, 0.1, 0.1), seed=args.seed)
    train, val = dataset.subset(train_idx), dataset.subset(val_idx)
    del dataset
    mean, std = train.transform()
    val.transform(mean, std)
    print(f"train: {len(train)}, val: {len(val)}")

    masks = make_masks(device)
    n_val = len(val)
    val_set = (
        val.parameters,
        torch.randint(len(masks), (n_val,), device=device, generator=g),
        torch.rand(n_val, device=device, generator=g),
        torch.randn(n_val, num_var, device=device, generator=g),
    )

    # ---- model / optim -------------------------------------------------------
    model = Encoder_Transformer(args.n_embd, args.n_layers, args.n_heads).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    path = AffineProbPath(scheduler=CondOTScheduler())
    sampler = EpochSampler(train.parameters, args.batch_size, g)

    # ---- train ---------------------------------------------------------------
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}
    running, n_running = 0.0, 0
    best_val, best_state = float("inf"), None
    t0 = time.time()
    model.train()
    for step in range(args.max_iters):
        lr = lr_at(step, args.max_iters, args.warmup_iters, args.learning_rate, args.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        target = sampler.next()
        B = target.shape[0]
        loss = fm_loss(model, path, masks, target,
                       torch.randint(len(masks), (B,), device=device, generator=g),
                       torch.rand(B, device=device, generator=g),
                       torch.randn(B, num_var, device=device, generator=g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running += loss.item()
        n_running += 1

        if (step + 1) % args.eval_interval == 0 or step == args.max_iters - 1:
            val_loss = evaluate(model, path, masks, val_set, args.eval_batch_size)
            train_loss = running / n_running
            running, n_running = 0.0, 0
            for k, v in zip(history, (step + 1, train_loss, val_loss, lr)):
                history[k].append(v)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"step {step + 1}/{args.max_iters} epoch {sampler.epoch}: "
                  f"train {train_loss:.4f}  val {val_loss:.4f}  lr {lr:.2e}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    # ---- save ----------------------------------------------------------------
    run_dir = os.path.join(args.out_dir, args.run_name)  # one folder per run: <out_dir>/<run_name>/<run_name>_*
    os.makedirs(run_dir, exist_ok=True)
    prefix = os.path.join(run_dir, args.run_name)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},  # final
        "best_model_state": best_state,                                     # lowest val loss
        "best_val_loss": best_val,
        "hparams": hparams,
        "mean": mean.cpu(), "std": std.cpu(),
        "variables": [v.branch for v in var_registry],
        "masks": masks.cpu(),
        "file_names": [str(f) for f in file_names],
    }, f"{prefix}_weights.pt")
    np.savez(f"{prefix}_loss_history.npz", **{k: np.asarray(v) for k, v in history.items()})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["step"], history["train_loss"], label="train")
    ax.plot(history["step"], history["val_loss"], label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("flow-matching loss")
    ax.set_yscale("log")
    ax.set_title(f"{args.run_name}  (best val {best_val:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_loss.png", dpi=150)
    print(f"saved {prefix}_weights.pt, {prefix}_loss_history.npz, {prefix}_loss.png")


if __name__ == "__main__":
    main()
