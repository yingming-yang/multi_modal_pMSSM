# Sampling part of tests/eval_fm.ipynb as a script, to run on a GPU node.
#
# 1. eval_fm.ipynb with sample_mode = "save" writes tests/checkpoints/<run>/sampling_inputs.pt
# 2. this script samples every job in it (all masks) and writes tests/checkpoints/<run>/sampling_outputs.pt
# 3. eval_fm.ipynb with sample_mode = "load" reads the outputs and makes the plots
#
#   cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
#   pixi run python -u tests/scripts/sample_fm.py --run fm_31974634
#
# Sections and names follow eval_fm.ipynb. Changes vs. the notebook are marked with "# CHANGED:".
import argparse
import os
import time
import torch

from v2_fm_pmssm.training import load_run, generate

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="run name, e.g. fm_31974634")
p.add_argument("--input_dir", default="tests/checkpoints/", help="folder with the run folders")
p.add_argument("--seed", type=int, default=0, help="job i is sampled with seed + i")
args = p.parse_args()

# CHANGED: the parameters come from the inputs file written by the notebook, not from a parameter cell
run = args.run
input_dir = args.input_dir
inputs_path = f"{input_dir}{run}/sampling_inputs.pt"
outputs_path = f"{input_dir}{run}/sampling_outputs.pt"
inputs = torch.load(inputs_path)
settings, jobs = inputs['settings'], inputs['jobs']
n_samples = settings['n_samples']
step_size = settings['step_size']
final = settings['final']
print(f"{inputs_path}: {settings}, jobs: {list(jobs)}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
if device.type != "cuda":  # CHANGED: warn, sampling on the CPU is only good for tiny tests
    print("WARNING: no GPU, sampling on the CPU will be slow", flush=True)

# ---- checkpoint (CHANGED: no test set, only the model is needed) ----
vf = load_run(run, input_dir, device, final)["model"]   # rebuilt as trained, weights loaded, eval mode
n_var = vf.n_features

# ---- sampling (CHANGED: resumes after a timeout, saves after every job) ----
results = {}
if os.path.exists(outputs_path):
    outputs = torch.load(outputs_path)
    if outputs['settings'] == settings:
        results = outputs['results']
        print(f"resuming, already done: {list(results)}", flush=True)
    else:
        print(f"{outputs_path} has other settings {outputs['settings']}, starting over", flush=True)

for i, (name, job) in enumerate(jobs.items()):
    if name in results:
        continue
    t0 = time.time()
    # one seed per job, so a resumed run gives the same samples; all columns (n, n_var), see training/evaluation.py
    results[name] = generate(vf, **job, step_size=step_size, wanted=list(range(n_var)), seed=args.seed + i)
    torch.save({'settings': settings, 'results': results}, outputs_path + ".tmp")
    os.replace(outputs_path + ".tmp", outputs_path)  # a job killed mid-write cannot leave a broken file
    print(f"{name}: {tuple(results[name].shape)} in {time.time() - t0:.0f} s, saved {outputs_path}", flush=True)

print(f"all {len(jobs)} jobs done. Set sample_mode = 'load' in eval_fm.ipynb.")
