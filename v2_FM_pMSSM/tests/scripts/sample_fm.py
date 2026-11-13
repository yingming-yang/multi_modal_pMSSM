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

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from v2_fm_pmssm.training import Encoder_Transformer

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="run name, e.g. fm_31974634")
p.add_argument("--seed", type=int, default=0, help="job i is sampled with seed + i")
args = p.parse_args()

# CHANGED: the parameters come from the inputs file written by the notebook, not from a parameter cell
run = args.run
input_dir = "tests/checkpoints/"
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
checkpoint = torch.load(f"{input_dir}{run}/{run}_weights.pt", map_location=device, weights_only=False)
hparams = checkpoint["hparams"]

vf = Encoder_Transformer(hparams["n_embd"], hparams["n_layers"], hparams["n_heads"]).to(device)
vf.load_state_dict(checkpoint["model_state"] if final else checkpoint["best_model_state"])
vf.eval()

# ---- notebook: sampling setup ------------------------------------------------
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        xt = torch.cat([x, torch.full((len(x), 1), t.item(), device=x.device)], dim=-1)
        return self.model(xt, **extras)

solver = ODESolver(velocity_model=WrappedModel(vf.eval()))  # create an ODESolver class

@torch.no_grad()
def generate(known, x_known=None, n=n_samples, chunk=50000, time_grid=None, return_intermediates=False):
    '''
    known:   (13,) bool, True = variable is fixed to its value in x_known
    x_known: (n, 13) normalized values; only the known columns are used
    returns: (n, 13) normalized samples, or (len(time_grid), n, 13) with return_intermediates
    '''
    known = torch.as_tensor(known, dtype=torch.bool, device=device)
    noise = torch.randn(n, 13, device=device)
    x_init = noise if x_known is None else torch.where(known, x_known.to(device), noise)
    T = torch.tensor([0.0, 1.0], device=device) if time_grid is None else time_grid
    out = [solver.sample(time_grid=T, x_init=x_init[s:s + chunk], method='midpoint', step_size=step_size,
                         return_intermediates=return_intermediates, known=known)
           for s in range(0, n, chunk)]
    return torch.cat(out, dim=-2)

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
    torch.manual_seed(args.seed + i)  # one seed per job, so a resumed run gives the same samples
    t0 = time.time()
    results[name] = generate(**job).cpu()
    torch.save({'settings': settings, 'results': results}, outputs_path + ".tmp")
    os.replace(outputs_path + ".tmp", outputs_path)  # a job killed mid-write cannot leave a broken file
    print(f"{name}: {tuple(results[name].shape)} in {time.time() - t0:.0f} s, saved {outputs_path}", flush=True)

print(f"all {len(jobs)} jobs done. Set sample_mode = 'load' in eval_fm.ipynb.")
