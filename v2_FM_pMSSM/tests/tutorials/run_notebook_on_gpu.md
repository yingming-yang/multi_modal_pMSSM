# Running notebook cells on a Raven GPU node

The login node has no GPU and is shared by everyone. MPCDF asks users not to run long
computations there. This guide shows how to get a GPU for a short time and run
`tests/training.ipynb` on it, in three ways:

- **Option A: run the notebook headless.** The notebook runs from top to bottom and its outputs, including plots, are saved into a copy you open in VS Code. This is the most robust option.
- **Option B: connect VS Code to a Jupyter server on the GPU node.** You keep clicking through cells as usual in VS Code, but the kernel runs on the GPU.
- **Option C: Jupyter in the browser through MPCDF's Remote Visualization Service (RVS).** You get an interactive JupyterLab session with one A100 for up to 8 hours, without writing any Slurm commands.

Options B and C are interactive. Use them to debug. Use Option A, or a batch script, for long runs.

**The `interactive` partition has no GPU.** On Raven, `--partition=interactive` runs on `raven05`
and `raven06`, which are CPU-only nodes. The partitions with A100 GPUs are `gpudev` (15 minutes),
`gpu1` (up to 24 hours, shared node, 1 to 3 GPUs) and `rvs` (used by Option C).

All commands assume the project directory:

```bash
cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
```

---

## 1. Get an interactive shell on a GPU node

```bash
srun --partition=gpudev --gres=gpu:a100:1 --cpus-per-task=18 --mem=60G \
     --time=00:15:00 --pty bash
```

- `gpudev` is a single A100 node reserved for short tests. It has a 15-minute limit and usually starts within minutes.
- The prompt changes to a node name like `yinya@ravg1002`. Everything you type now runs on that node.
- When the 15 minutes are up, the shell and everything running in it are killed.

Check that you got a GPU:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
pixi run python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

`pixi run` works here because the node has CUDA. On the login node it fails with a
`__cuda` error. There, use `.pixi/envs/default/bin/python` directly instead.

When you are done, type `exit` to give the GPU back.

**Need more than 15 minutes?** Leave out `--partition=gpudev` and set a longer `--time`,
for example `--time=01:00:00`. Slurm then puts the job in the shared `gpu1` partition,
where the wait can be hours or days. Check the expected start without submitting:

```bash
sbatch --test-only --gres=gpu:a100:1 --cpus-per-task=18 --mem=60G --time=01:00:00 --wrap="true"
```

Keep `--cpus-per-task=18 --mem=60G` for one GPU. Without them Slurm asks for more than a
quarter of the node's memory and rejects the job.

---

## 2. Option A: run the notebook headless with nbconvert

### 2.1 Mark the cells you do not want to run

The training cells (the `Encoder_Transformer()` instance, `sample_data`, the validation batch,
the training loop and the training-loss plot) are not needed when you load trained weights.
Give these cells the tag `skip`:

- In VS Code, click the `...` menu on the cell, choose **Add Cell Tag**, and type `skip`.
  If the menu entry is missing, install the extension **Jupyter Cell Tags** (`ms-toolsai.vscode-jupyter-cell-tags`).
- Or tag them by index from the shell. Cell indices start at 0. Check them first, because they change when you add or delete cells:

```bash
.pixi/envs/default/bin/python - <<'EOF'
import json
path = "tests/training.ipynb"
skip = [12, 13, 14, 16, 17]          # cell indices to skip, adjust to your notebook
nb = json.load(open(path))
for i, c in enumerate(nb["cells"]):
    print(i, "".join(c["source"])[:60].replace("\n", " | "))
for i in skip:
    tags = nb["cells"][i]["metadata"].setdefault("tags", [])
    if "skip" not in tags:
        tags.append("skip")
json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
EOF
```

Save the notebook in VS Code before running this, and reload it afterwards.
Otherwise VS Code may overwrite the tags with its unsaved copy.

### 2.2 Run it on the GPU node

Inside the `srun` shell from step 1:

```bash
cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests
pixi run jupyter nbconvert --to notebook --execute training.ipynb \
    --output training_executed.ipynb --output-dir training \
    --ExecutePreprocessor.timeout=-1 \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip"]'
```

- The notebook runs from top to bottom with the `python3` kernel from the pixi environment.
- The result, with all printed output and plots, is written to `tests/training/training_executed.ipynb`. Your original notebook is not changed.
- Cells tagged `skip` are left out of the result entirely.
- `--ExecutePreprocessor.timeout=-1` removes the per-cell time limit. The default limit is 30 seconds, which is too short for sampling.
- If a cell raises an error, the run stops there. Add `--allow-errors` to keep going and see the error inside the result notebook.

### 2.3 Look at the result

Open `tests/training/training_executed.ipynb` in VS Code. The plots are stored in the file, so
you do not need a running kernel to see them.

### Things to watch

- **Paths are relative to the notebook's folder.** The notebook lives in `tests/`, so a cell that loads `fm_31911102_weights.pt` looks in `tests/`. If the weights are in `tests/training/`, set `run_name = "training/fm_31911102"`.
- **Plan for 15 minutes.** Loading 20 files takes well under a minute. With the GPU, sampling 5000 points per case should take about a minute in total. Loading all 499 files takes 5 to 8 minutes on its own.
- **Run unattended with sbatch instead.** Put the `nbconvert` command in a batch script, like `train_fm.sbatch`, and add `#SBATCH --partition=gpudev` and `#SBATCH --time=00:15:00`.

---

## 3. Option B: connect VS Code to a Jupyter server on the GPU node

This keeps the normal notebook workflow: you run cells one by one and inspect variables.
The kernel lives on the GPU node, so it dies when the job ends, and all variables are lost.

### 3.1 Start a Jupyter server on the GPU node

Start the allocation inside `tmux` on the login node. If VS Code or your SSH connection drops,
the GPU job and the Jupyter server keep running. Without `tmux`, a disconnect ends the `srun`
shell and the job with it.

```bash
tmux new -s gpu                # later: reattach with  tmux attach -t gpu
```

Inside `tmux`, get a GPU shell. Pick one of the two lines:

```bash
# quick: gpudev, 15 minutes, usually starts within minutes
srun --partition=gpudev --gres=gpu:a100:1 --cpus-per-task=18 --mem=60G --time=00:15:00 --pty bash

# longer: gpu1, here 2 hours, but the wait in the queue can be long
srun --gres=gpu:a100:1 --cpus-per-task=18 --mem=60G --time=02:00:00 --pty bash
```

Then, on the GPU node:

```bash
cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
pixi run jupyter server --no-browser --ip=$(hostname) --port=8888
```

If port 8888 is taken, Jupyter picks the next free port. Use the URL that Jupyter prints.
To leave `tmux` running in the background, press `Ctrl-b` and then `d`.

It prints a URL with a token, for example:

```
http://ravg1002:8888/?token=3f9c...e1
```

Leave this shell, or the `tmux` session, running. Stopping it stops the server.

### 3.2 Point VS Code at it

1. Open `tests/training.ipynb`.
2. Click the kernel name in the top right, then **Select Another Kernel...**, then **Existing Jupyter Server...**.
3. Paste the URL including the token, give it a name, and pick the **Python 3** kernel.
4. Run cells as usual. Check with `torch.cuda.is_available()` and `device` in the first cells.

When the job ends, switch the kernel back to the local pixi environment. To give the GPU back
early, stop the server with `Ctrl-c`, then type `exit` in the GPU shell.

**Connection not fully tested.** The VS Code server runs on the login node, so it has to reach
the Jupyter port on the compute node over Raven's internal network. The login node does resolve
and reach the GPU nodes (tested with `ravg1002`, port 22), but port 8888 has not been tried.
If VS Code cannot connect, use Option C or Option A.

---

## 4. Option C: Jupyter on a GPU through RVS (in the browser)

MPCDF runs JupyterLab sessions on Raven's GPU nodes through the web portal
<https://rvs.mpcdf.mpg.de>. The session type **Jupyter for machine learning** gives you one A100,
18 cores and up to 512 GB of shared memory, for up to 8 hours. You work in the browser, not in
VS Code. The notebook files are the same ones on `/u`, so edits show up in both places.

### 4.1 One-time setup

1. Register the pixi environment as a Jupyter kernel. Run this on the login node, using the
   environment's Python directly, because `pixi run` fails on the login node:

   ```bash
   cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
   .pixi/envs/default/bin/python -m ipykernel install --user \
       --name v2_fm_pmssm --display-name "v2_FM_pMSSM (pixi)"
   ```

   This writes a kernel spec to `~/.local/share/jupyter/kernels/v2_fm_pmssm/` that points to
   `.pixi/envs/default/bin/python`. Every Jupyter in your home directory, including RVS, can then
   use it. Rerun it if you delete and recreate the pixi environment.
2. Log in at <https://rvs.mpcdf.mpg.de> with your MPCDF (Kerberos) username and password.
3. Choose **Initialize Remote Visualization** for Raven and set a Jupyter password. You need to
   do this only once per cluster.

### 4.2 Start a session

1. On the portal, choose **Submit new session**. Select Raven, the session type
   **Jupyter for machine learning**, and a run time of up to 8 hours. Choose **Lab** as the interface.
2. The job goes into the `rvs` queue. You get an email when the session is ready.
3. Choose **Connect to session** and enter your Jupyter password.
4. In JupyterLab, open `workspace/multi_modal_pMSSM/v2_FM_pMSSM/tests/eval_fm.ipynb`. Click the
   kernel name in the top right and select **v2_FM_pMSSM (pixi)**.
5. Check in the first cell that `torch.cuda.is_available()` is `True`.

The plain **Jupyter** session type has no GPU. Choose the machine-learning one.

When you are done, stop the session from the portal so the GPU is released before the time limit.

### Things to watch

- **Relative paths** are resolved from the notebook's folder, as in VS Code. `training/{run}/...`
  therefore still points to `tests/training/`.
- **Save before switching.** If the same notebook is open in VS Code and in the browser, whichever
  saves last overwrites the other. Close it in one of them.
- **Import errors** in the pixi kernel can come from modules that the RVS session loads by default,
  such as `anaconda/3`, because they set `PYTHONPATH`. Run `import sys; print(sys.executable, sys.path)`
  in a cell. If paths outside `.pixi/envs/default` come first, check `$HOME/.jupyter/modules.conf`.
- MPCDF's documentation, including how to start sessions from the command line:
  <https://docs.mpcdf.mpg.de/doc/visualization/index.html>

---

## 5. Useful commands

```bash
squeue --me                    # your jobs and their state
scancel <jobid>                # cancel a job
sacct -j <jobid> -o JobID,State,Elapsed,MaxRSS   # what a finished job used
sbatch --test-only --partition=gpudev --gres=gpu:a100:1 --cpus-per-task=18 --mem=60G --time=00:15:00 --wrap=true  # see how long the current waiting time is
```

## 6. Making sampling faster

These matter regardless of where the notebook runs:

- **Lower `n_samples`** while developing, for example to 1000. Cost scales linearly with it.
- **Try `step_size = 0.1`** for the ODE solver. It halves the number of model calls. Compare the plots with `0.05` before trusting it.
- **A smaller model** samples much faster. Width 128 with 6 layers has about a tenth of the parameters of width 256 with 16 layers.
