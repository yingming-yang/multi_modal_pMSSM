# Sampling and test-loss part of tests/compare_fm.ipynb as a script, to run on a GPU node.
#
# 1. compare_fm.ipynb with sample_mode = "save" writes tests/checkpoints/compare/<compare_name>/sampling_inputs.pt
# 2. this script runs every sampling job and every loss job of every run in it and writes
#    tests/checkpoints/compare/<compare_name>/sampling_outputs.pt
# 3. compare_fm.ipynb with sample_mode = "load" reads the outputs and makes the plots
#
#   cd /u/yinya/workspace/multi_modal_pMSSM/v2_FM_pMSSM
#   pixi run python -u tests/scripts/sample_compare.py --compare benchmark_shared_per_feature
import argparse
import os
import time
import torch

from v2_fm_pmssm.training import load_run, generate, test_loss

p = argparse.ArgumentParser()
p.add_argument("--compare", required=True, help="compare_name of compare_fm.ipynb")
p.add_argument("--input_dir", default="tests/checkpoints/", help="folder with the run folders")
p.add_argument("--seed", type=int, default=0, help="sampling job i of a run is sampled with seed + i")
args = p.parse_args()

compare_dir = f"{args.input_dir}compare/{args.compare}/"
inputs_path = f"{compare_dir}sampling_inputs.pt"
outputs_path = f"{compare_dir}sampling_outputs.pt"
inputs = torch.load(inputs_path)
settings, jobs, loss_jobs, test = inputs['settings'], inputs['jobs'], inputs['loss_jobs'], inputs['test']
print(f"{inputs_path}: {settings}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
if device.type != "cuda":
    print("WARNING: no GPU, sampling on the CPU will be slow", flush=True)

# resume after a timeout: keep what is already done if the settings are the same
results, losses = {}, {}
if os.path.exists(outputs_path):
    outputs = torch.load(outputs_path)
    if outputs['settings'] == settings:
        results, losses = outputs['results'], outputs['losses']
        print(f"resuming, already done: { {run: [*results.get(run, {}), *losses.get(run, {})] for run in settings['runs']} }", flush=True)
    else:
        print(f"{outputs_path} has other settings {outputs['settings']}, starting over", flush=True)

def save():
    torch.save({'settings': settings, 'results': results, 'losses': losses}, outputs_path + ".tmp")
    os.replace(outputs_path + ".tmp", outputs_path)  # a job killed mid-write cannot leave a broken file

for run in settings['runs']:
    r = load_run(run, args.input_dir, device, settings['final'])
    results.setdefault(run, {})
    losses.setdefault(run, {})
    for i, (name, job) in enumerate(jobs[run].items()):
        if name in results[run]:
            continue
        t0 = time.time()
        results[run][name] = generate(r["model"], **job, step_size=settings['step_size'], seed=args.seed + i)
        save()
        print(f"{run} {name}: {tuple(results[run][name].shape)} in {time.time() - t0:.0f} s", flush=True)
    for m, job in loss_jobs[run].items():   # job: known (+ loss_on), keyword arguments of test_loss
        if m in losses[run]:
            continue
        t0 = time.time()
        losses[run][m] = test_loss(r["model"], test[run], **job, seed=settings['loss_seed'])
        save()
        print(f"{run} loss [{m}]: {losses[run][m].mean():.4f} in {time.time() - t0:.0f} s", flush=True)

print(f"all jobs done, saved {outputs_path}. Set sample_mode = 'load' in compare_fm.ipynb.")
