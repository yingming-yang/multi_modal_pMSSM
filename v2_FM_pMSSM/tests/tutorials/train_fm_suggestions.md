# Suggestions for train_fm.py / training.ipynb

These are the changes from the earlier, larger version of `train_fm.py`, written up
so they can be adopted one at a time. Most important first. None of them is applied yet.

## Training correctness

1. **Shuffle the training data every epoch.**
   `split_indices` returns sorted indices, so `sample_data` walks through the
   training set in file order. Every batch then comes from one or two neighbouring
   files. A fresh `torch.randperm` at the start of each epoch removes that ordering.

2. **Give the validation batch its own index.**
   `sample_data(batch_size, val)` uses the same global `index` as training.
   After building `val_batch`, `index` is already 1000, so training starts at row
   1000 of `train` and skips the first 1000 rows in the first epoch.
   A separate counter per dataset, or slicing `val[:batch_size]` directly, fixes this.

3. **Validate on the whole validation set.**
   The validation loss comes from one batch of 1000 points with random masks.
   It is noisy and depends on which masks were drawn. Looping over all of `val`
   in large batches, with noise, times and masks drawn once with a fixed seed,
   gives a stable number that is comparable between steps and runs.

4. **Average the training loss over each eval interval.**
   The logged training loss is the loss of a single batch, so the curve is noisy.
   Keeping a running sum between evaluations gives a smoother, more honest curve.

5. **Learning-rate warmup and decay.**
   For a 16-layer transformer, a linear warmup over about 2000 steps followed by
   cosine decay to about 1e-5 is usually more stable than a constant rate.
   It also usually reaches a lower final loss.

6. **AdamW and gradient clipping.**
   `torch.optim.AdamW(..., weight_decay=0.01)` and
   `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `opt.step()`
   guard against occasional loss spikes in deep models.

7. **Fix the random seed.**
   `torch.manual_seed(42)` at the top makes runs reproducible. The data split is
   already seeded, but the masks, times and noise are not.

## Model code

8. **Take the device from the input in `forward`.**
   `forward` uses the global `device`. Using `idx.device` instead makes the model
   work wherever its input lives, for example when it is loaded in another script.

9. **Update the `forward` docstring.**
   It still says shape `(B, 3)` and `known` of shape `(2,)`. The real shapes are
   `(B, 14)` for the input and `(13,)` or `(B, 13)` for `known`.

10. **`dropout` is defined but never used.**
    Either pass it to `Encoder` as `attn_dropout` and `ff_dropout`, or remove it.

## Saving and reproducibility

11. **Save the best-validation weights as well as the final ones.**
    Keep a copy of `state_dict()` whenever the validation loss improves.

12. **Save more than the weights.**
    Put the hyperparameters, `mean`, `std` and the list of loaded files into
    the checkpoint dict. `mean` and `std` are needed to undo the normalization
    when sampling, and the file list is needed to reproduce the exact dataset.

13. **Record the step number in the loss history.**
    The plot uses `arange(n) * eval_interval` as the x axis. The last entry is
    logged at `max_iters - 1`, which is not a multiple of `eval_interval`, so its
    x position is slightly off. Saving the step alongside the losses avoids this.

14. **Nicer loss plot.**
    A log y axis, axis labels and a train/val legend make the plot readable.

## Workflow

15. **Command-line arguments for hyperparameters.**
    With `argparse`, hyperparameters can be changed at submission time,
    for example `sbatch train_fm.sbatch --max_iters 200000`, without editing the script.

16. **Slurm partition and time limit.**
    Raven moves every job asking for 1 to 3 GPUs to the shared `gpu1` partition,
    whatever `--partition` says. Jobs with a time limit of 2 hours or less can
    also use the `gpu-bw` nodes for backfill, which usually shortens the wait.
    So set `--time=02:00:00` if the run fits. For quick tests, the `gpudev`
    partition has a 15-minute limit and almost no queue:
    `sbatch --partition=gpudev --time=00:15:00 train_fm.sbatch`

17. **Unused imports.**
    `ODESolver`, `ModelWrapper`, `odeint`, `skld` and `F` are not used yet.
    Keep them if they are meant for sampling later.

## Hyperparameter values changed in train_fm.py

These are the only value changes from the notebook. They are marked `# CHANGED:` in the script.

| Setting | Notebook | train_fm.py | Reason |
|---|---|---|---|
| data | 20 files | all 499 files | about 3M points after the Omega filter, so a 12M-parameter model does not overfit |
| learning_rate | 1e-3 | 3e-4 | 1e-3 without warmup is risky for 16 layers |
| max_iters | 120 | 50000 | about 17 epochs over the full training set |
| eval_interval | 10 | 500 | 100 points on the loss curve |
