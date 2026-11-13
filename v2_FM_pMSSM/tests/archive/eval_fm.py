# Sampling and evaluation part of tests/training.ipynb as a script.
# Loads a checkpoint written by train_fm_suggested.py, rebuilds the same test set,
# generates samples for the three masks and saves the figures as <run>_<name>.png.
#
#   pixi run python tests/training/eval_fm.py --run tests/training/fm_<jobid>
#
# Changes vs. the notebook cells are marked with "# CHANGED:".
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # CHANGED: no display on the compute node
import matplotlib.pyplot as plt
import torch
import corner

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from v2_fm_pmssm.data import load_data, pMSSM_Dataset, select, all_var, split_indices
from v2_fm_pmssm.training import Encoder_Transformer, make_masks

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="prefix of the training outputs, e.g. tests/training/fm_12345678")
p.add_argument("--n_samples", type=int, default=5000)
p.add_argument("--step_size", type=float, default=0.05)
p.add_argument("--final", action="store_true", help="use the final weights instead of the best-validation weights")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

def save_fig(fig, name):  # CHANGED: save instead of plt.show()
    fig.savefig(f"{args.run}/{name}.png", dpi=120)
    plt.close(fig)
    print(f"saved {args.run}/{name}.png", flush=True)

# ---- checkpoint and test set (same files, filter and split as in training) ----
ckpt = torch.load(f"{args.run}_weights.pt", map_location=device, weights_only=False)
hp = ckpt["hparams"]
mean, std = ckpt["mean"].to(device), ckpt["std"].to(device)

parameter, parameter_table, point_id, _ = load_data(exact_paths=ckpt["file_names"])
var_registry = select(all_var)
dataset = pMSSM_Dataset(parameter_table, point_id, var_registry, device)
dataset = dataset.filter_invalid(branches=['MO_Omega'])
train_idx, val_idx, test_idx = split_indices(len(dataset), (0.8, 0.1, 0.1), seed=hp["seed"])
test = dataset.subset(test_idx)
del dataset
test.transform(mean, std)
print(f"test points: {len(test)}")

vf = Encoder_Transformer(hp["n_embd"], hp["n_layers"], hp["n_heads"]).to(device)
vf.load_state_dict(ckpt["model_state"] if args.final else ckpt["best_model_state"])
vf.eval()
masks = make_masks(device)

# ---- notebook: sampling setup ------------------------------------------------
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        xt = torch.cat([x, torch.full((len(x), 1), t.item(), device=x.device)], dim=-1)
        return self.model(xt, **extras)

solver = ODESolver(velocity_model=WrappedModel(vf.eval()))  # create an ODESolver class
step_size = args.step_size                     # step size for ode solver
n_samples = min(args.n_samples, len(test))     # number of test points used in the plots

names = [v.branch for v in var_registry]
i_omega = names.index('MO_Omega')

@torch.no_grad()
def generate(known, x_known=None, n=n_samples, chunk=5000, time_grid=None, return_intermediates=False):
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

def to_physical(x_norm, like=test):
    '''Reverse the transform with the dataset class: copy `like` (keeps mean/std), swap in x_norm, call inverse().'''
    out = like.subset(np.arange(len(x_norm)))
    out.parameters = x_norm.to(out.parameters.device).clone()
    out.inverse()
    return out.parameters.cpu().numpy()

# Test points used as conditions (random subset, the split indices are sorted by file)
sel = torch.randperm(len(test), generator=torch.Generator().manual_seed(0))[:n_samples].to(test.parameters.device)
test_x = test.parameters[sel]          # normalized
truth_all = to_physical(test.parameters)  # whole test set in physical units
truth = truth_all[sel.cpu().numpy()]      # the selected points in physical units

def _plot_values(x, j):
    '''Column j for plotting; MO_Omega is shown as log10 since it spans orders of magnitude.'''
    return (np.log10(x[:, j]), 'log10(MO_Omega)') if j == i_omega else (x[:, j], names[j])

def plot_marginals(samples, name, cols=None, ncols=4, bins=50):  # CHANGED: name of the saved figure
    '''samples: {label: (n, 13) physical array}. Grid of 1D histograms as in dataset_inspection.ipynb;
    the binning range comes from the first entry (0.5-99.5 percentile).'''
    cols = list(range(len(names))) if cols is None else list(cols)
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False, constrained_layout=True)
    for ax, j in zip(axes.flat, cols):
        ref, title = _plot_values(next(iter(samples.values())), j)
        rng = np.nanpercentile(ref[np.isfinite(ref)], [0.5, 99.5])
        for label, x in samples.items():
            v, _ = _plot_values(x, j)
            ax.hist(v[np.isfinite(v)], bins=bins, range=rng, density=True, histtype='step', label=label)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('Density')
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    save_fig(fig, name)

def plot_corner(samples, name, cols=None):  # CHANGED: name of the saved figure
    '''Overlaid corner plots, one color per entry of samples.'''
    cols = list(range(len(names))) if cols is None else list(cols)
    prep = {k: np.column_stack([_plot_values(x, j)[0] for j in cols]) for k, x in samples.items()}
    labels = [_plot_values(next(iter(samples.values())), j)[1] for j in cols]
    ref = next(iter(prep.values()))
    rng = [tuple(np.nanpercentile(ref[:, i], [0.5, 99.5])) for i in range(len(cols))]
    fig = None
    for c, (label, x) in enumerate(prep.items()):
        x = x[np.isfinite(x).all(axis=1)]
        fig = corner.corner(x, fig=fig, color=f'C{c}', labels=labels, range=rng, bins=30,
                            hist_kwargs={'density': True}, plot_datapoints=False, plot_density=False,
                            levels=(0.68, 0.95), label_kwargs={'fontsize': 8})
    handles = [plt.Line2D([], [], color=f'C{c}', label=label) for c, label in enumerate(prep)]
    fig.legend(handles=handles, loc='upper right', fontsize=14)
    save_fig(fig, name)

# ---- 1. Joint distribution p(inputs, Omega) ---------------------------------
gen_joint = to_physical(generate(masks[2]))
plot_marginals({'test data': truth, 'generated': gen_joint}, 'joint_marginals')
plot_corner({'test data': truth, 'generated': gen_joint}, 'joint_corner')

T = torch.linspace(0.0, 1.0, 21, device=device)
xint = generate(masks[2], n=1000, time_grid=T, return_intermediates=True).cpu().numpy()
a, b = names.index('IN_M_1'), i_omega

f, axarr = plt.subplots(3, 7, figsize=(14, 6), sharex=True, sharey=True)
for idx, ax in enumerate(axarr.ravel()):
    ax.scatter(xint[idx][:, a], xint[idx][:, b], s=1)
    ax.set_title(f't = {T[idx].item():.2f}', fontsize=8)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
f.supxlabel('IN_M_1 (normalized)')
f.supylabel('log10(MO_Omega) (normalized)')
save_fig(f, 'joint_flow')

# ---- 2. Relic density from the inputs p(Omega | inputs) ---------------------
gen_omega = to_physical(generate(masks[0], x_known=test_x))
t_om, g_om = truth[:, i_omega], gen_omega[:, i_omega]
m1 = np.abs(truth[:, names.index('IN_M_1')])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

ax = axes[0]
ax.scatter(t_om, g_om, s=1, alpha=0.5)
lims = [min(t_om.min(), g_om.min()), max(t_om.max(), g_om.max())]
ax.plot(lims, lims, 'r--', lw=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('true MO_Omega'); ax.set_ylabel('generated MO_Omega')

ax = axes[1]
ax.hist(np.log10(g_om / t_om), bins=60)
ax.set_xlabel('log10(generated / true)'); ax.set_ylabel('Count')

ax = axes[2]  # as in dataset_inspection.ipynb, with |M_1| in place of m_chi_10
ax.scatter(m1, t_om, s=1, label='test data')
ax.scatter(m1, g_om, s=1, alpha=0.5, label='generated')
ax.axvline(91.19/2, color='red', linestyle='--', label=r'$m_Z/2$')
ax.axvline(125.25/2, color='blue', linestyle='--', label=r'$m_h/2$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1e1, 1e4); ax.set_ylim(1e-4, 1e1)
ax.set_xlabel('|IN_M_1|'); ax.set_ylabel('MO_Omega')
ax.legend(markerscale=5, fontsize=8)
save_fig(fig, 'omega_given_inputs')

print(f"median |log10(generated/true)| = {np.median(np.abs(np.log10(g_om / t_om))):.3f}")

# ---- 3. Inputs from the relic density p(inputs | Omega) ---------------------
gen_inputs = to_physical(generate(masks[1], x_known=test_x))
plot_marginals({'test data': truth, 'generated': gen_inputs}, 'inputs_given_omega', cols=range(12))

omega_0 = 0.12
x_fixed = torch.zeros(n_samples, 13, device=device)
x_fixed[:, i_omega] = (np.log10(omega_0) - test.mean[i_omega].item()) / test.std[i_omega].item()  # same transform as the data
gen_fixed = to_physical(generate(masks[1], x_known=x_fixed))

band = truth_all[np.abs(np.log10(truth_all[:, i_omega]) - np.log10(omega_0)) < 0.05]
print(f"test points with |log10(Omega) - log10({omega_0})| < 0.05: {len(band)}")
plot_marginals({f'test data, |dlog10 Omega| < 0.05': band, f'generated, Omega = {omega_0}': gen_fixed},
               'inputs_given_omega_0.12', cols=range(12))
