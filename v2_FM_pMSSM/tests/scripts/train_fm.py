"""Flow-matching training of a transformer velocity field on pMSSM data.

Converted from tests/training.ipynb. Differences from the notebook:
  * training batches come from the normalized training split, shuffled each epoch
  * validation uses the whole validation split with fixed noise/times/masks
  * AdamW with linear warmup + cosine decay, gradient clipping

Outputs (prefix = <out_dir>/<name>/<name>, name = fm_<jobid>[_<run_name>], out_dir defaults to tests/checkpoints):
  <prefix>_weights.pt       dict with model_state, hparams, mean, std, train_files, val_files, test_files, ...,
                            checkpoints: {step: model_state} at the checkpoint steps (n_checkpoints log-spaced
                            steps from first_checkpoint to max_iters, e.g. 100, 1k, 10k, 100k)
  <prefix>_loss_history.npz arrays: step, train_loss, val_loss, lr; per-mask validation losses (at the
                            checkpoint steps): mask_step, mask_loss (n_checkpoints, n_masks), mask_names
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

from v2_fm_pmssm.data import load_data, pMSSM_Dataset, select, all_var
from v2_fm_pmssm.training import (Encoder_Transformer, MASK_TYPES, make_mask_sampler, eval_masks, EpochSampler,
                                  fm_loss, evaluate, lr_at, test_loss)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML file with hyperparameters; command-line arguments override it")
    p.add_argument("--data_folder", default="/u/yinya/archive/gen_flat_v2/ntuples")
    p.add_argument("--branches", nargs="+", default=None)  # passed to select(); None = all variables
    p.add_argument("--data_types", nargs="+", default=None) # passed to select() (modality), used only when branches is None
    p.add_argument("--steps", nargs="+", default=None)      # passed to select(), used only when branches and data_types are None
    p.add_argument("--mask_type", default="by_feature", choices=sorted(MASK_TYPES))  # training and validation masks: by_type (= by_modality, type, modality) or by_feature (= feature)
    p.add_argument("--max_files", type=int, default=None)  # limits the training and the validation files
    p.add_argument("--split_file", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                       "configs", "gen_flat_v2_split_files.json"))  # names of the test and val files
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"))  # tests/checkpoints
    p.add_argument("--run_name", default=None)  # optional; the folder is fm_<jobid>_<run_name> (fm_<jobid> without it)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--warmup_frac", type=float, default=0.02)   # fraction of max_iters
    p.add_argument("--warmup_iters", type=int, default=None)    # optional absolute override of warmup_frac
    p.add_argument("--eval_interval", type=int, default=1_000)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--n_checkpoints", type=int, default=4)       # weights saved and per-mask losses evaluated at n_checkpoints
    p.add_argument("--first_checkpoint", type=int, default=100)  # log-spaced steps from first_checkpoint to max_iters
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--use_adaptive_rmsnorm", action=argparse.BooleanOptionalAction, default=True)  # --no-use_adaptive_rmsnorm turns it off
    p.add_argument("--feature_embedding", default="shared", choices=["shared", "per_feature"])  # shared: one value layer + variable-id embedding; per_feature: own linear layer per variable
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
    if args.mask_type not in MASK_TYPES:  # argparse checks choices only on the command line, not for YAML values
        raise ValueError(f"unknown mask_type {args.mask_type!r}, choose from {sorted(MASK_TYPES)}")
    args.mask_type = MASK_TYPES[args.mask_type]  # save the canonical name, e.g. modality -> by_type
    if args.feature_embedding not in ("shared", "per_feature"):  # same: YAML values are not checked by argparse
        raise ValueError(f"unknown feature_embedding {args.feature_embedding!r}, choose 'shared' or 'per_feature'")
    if args.warmup_iters is None:
        args.warmup_iters = int(args.warmup_frac * args.max_iters)
    if args.warmup_iters >= args.max_iters:
        raise ValueError(f"warmup_iters ({args.warmup_iters}) must be smaller than max_iters ({args.max_iters})")
    if not 0 < args.first_checkpoint < args.max_iters:
        raise ValueError(f"first_checkpoint ({args.first_checkpoint}) must be between 0 and max_iters ({args.max_iters})")
    # log-spaced, the last one is max_iters; e.g. 100, 1000, 10000, 100000
    checkpoint_steps = sorted(set(round(s) for s in np.geomspace(args.first_checkpoint, args.max_iters, args.n_checkpoints)))
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    args.run_name = f"fm_{job_id}" if args.run_name is None else f"fm_{job_id}_{args.run_name}"
    hparams = vars(args)
    print(json.dumps(hparams, indent=2))
    run_dir = os.path.join(args.out_dir, args.run_name)  # one folder per run: <out_dir>/<run_name>/<run_name>_*
    os.makedirs(run_dir, exist_ok=True)
    print(f"outputs go to: {run_dir}", flush=True)
    slurm_log = os.environ.get("FM_SLURM_LOG")  # set by tests/slurm/train_fm.sbatch
    if slurm_log and os.path.exists(slurm_log):
        try:  # a move on the same filesystem keeps Slurm writing to the moved file
            os.replace(slurm_log, os.path.join(run_dir, f"{args.run_name}.out"))
        except OSError as e:  # e.g. out_dir on another filesystem: leave the log in logs/
            print(f"could not move the log {slurm_log}: {e}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(args.seed)
    g = torch.Generator(device=device).manual_seed(args.seed)

    # ---- data ----------------------------------------------------------------
    var_registry = select(all_var, branches=args.branches, data_types=args.data_types, steps=args.steps)
    num_var = len(var_registry)
    param_names = [v.branch for v in var_registry]
    with open(args.split_file) as f:
        splits = json.load(f)  # {"test": [file names], "val": [file names]}
    # training: every file except the test and validation files
    _, parameter_table, point_id, train_files = load_data(args.data_folder, param_names=param_names, max_files=args.max_files,
                                                          exclude=splits["test"] + splits["val"])
    train = pMSSM_Dataset(parameter_table, point_id, var_registry, device).filter_invalid(branches=['MO_Omega'])
    # validation: the fixed validation files
    val_files = [os.path.join(args.data_folder, f) for f in splits["val"]]
    _, parameter_table, point_id, _ = load_data(exact_paths=val_files, param_names=param_names, max_files=args.max_files)
    val = pMSSM_Dataset(parameter_table, point_id, var_registry, device).filter_invalid(branches=['MO_Omega'])
    del parameter_table, point_id
    mean, std = train.transform()
    val.transform(mean, std)
    print(f"train: {len(train)}, val: {len(val)}")

    # training and validation masks follow mask_type; the validation masks are drawn once,
    # so every evaluation uses the same masks
    mask_sampler = make_mask_sampler(args.mask_type, var_registry, seed=args.seed, device=device)
    n_val = len(val)
    val_set = (
        val.parameters,
        make_mask_sampler(args.mask_type, var_registry, seed=args.seed, device=device)(n_val),
        torch.rand(n_val, device=device, generator=g),
        torch.randn(n_val, num_var, device=device, generator=g),
    )
    # per-mask validation losses (fixed masks, see eval_masks) and saved weights, at checkpoint_steps
    val_masks = eval_masks(var_registry)              # later an hparam
    checkpoints = {}                                   # {step: model_state}
    mask_history = {"mask_step": [], "mask_loss": []}  # separate from history, its zip stays unchanged

    # ---- model / optim -------------------------------------------------------
    model = Encoder_Transformer(args.n_embd, args.n_layers, args.n_heads, variables=var_registry,
                                use_adaptive_rmsnorm=args.use_adaptive_rmsnorm,
                                feature_embedding=args.feature_embedding).to(device)
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
        loss = fm_loss(model, path, mask_sampler(B), target,
                       torch.rand(B, device=device, generator=g),
                       torch.randn(B, num_var, device=device, generator=g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running += loss.item()
        n_running += 1

        if step + 1 in checkpoint_steps:
            checkpoints[step + 1] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.eval()
            mask_history["mask_loss"].append([test_loss(model, val.parameters, known, loss_on, seed=args.seed,
                                                        eval_bs=args.eval_batch_size).mean().item()
                                              for known, loss_on in val_masks.values()])
            model.train()
            mask_history["mask_step"].append(step + 1)
            print(f"  checkpoint step {step + 1}:  "
                  + "  ".join(f"{n} {l:.4f}" for n, l in zip(val_masks, mask_history["mask_loss"][-1])), flush=True)

        if (step + 1) % args.eval_interval == 0 or step == args.max_iters - 1:
            val_loss = evaluate(model, path, val_set, args.eval_batch_size)
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
    prefix = os.path.join(run_dir, args.run_name)
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},  # final
        "best_model_state": best_state,                                     # lowest val loss
        "best_val_loss": best_val,
        "checkpoints": checkpoints,                                         # {step: model_state} at checkpoint_steps
        "hparams": hparams,
        "mean": mean.cpu(), "std": std.cpu(),
        "variables": [v.branch for v in var_registry],
        "variable_ids": [v.id for v in var_registry],
        "train_files": [str(f) for f in train_files],  # training files (full paths)
        "val_files": splits["val"],                  # validation file names
        "test_files": splits["test"],                # test file names, used by eval_fm.ipynb
    }, f"{prefix}_weights.pt")
    np.savez(f"{prefix}_loss_history.npz", **{k: np.asarray(v) for k, v in history.items()},
             **{k: np.asarray(v) for k, v in mask_history.items()}, mask_names=np.array(list(val_masks)))

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
