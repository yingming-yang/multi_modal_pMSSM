"""Evaluation helpers shared by the evaluation notebooks and the sampling scripts:
load a trained run, sample from it, compute the flow-matching test loss for one mask,
and the conversion / plotting helpers of compare_fm.ipynb."""
import corner
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from ..data.dataset import all_var, select, transform, inverse_transform
from .transformer import Encoder_Transformer


def load_run(run, input_dir="tests/checkpoints/", device="cpu", final=True):
    """Load <input_dir>/<run>/<run>_weights.pt and rebuild the model as it was trained.
    final=True: weights after the last step; False: weights with the lowest validation loss.
    Returns a dict: name, checkpoint, hparams, variables (list[Variable]), names (their branch names),
    mean, std, model (in eval mode)."""
    checkpoint = torch.load(f"{input_dir}{run}/{run}_weights.pt", map_location=device, weights_only=False)
    hparams = checkpoint["hparams"]
    model = Encoder_Transformer(hparams["n_embd"], hparams["n_layers"], hparams["n_heads"],
                                variable_ids=checkpoint["variable_ids"],
                                use_adaptive_rmsnorm=hparams.get("use_adaptive_rmsnorm", True),
                                feature_embedding=hparams.get("feature_embedding", "shared")).to(device)
    model.load_state_dict(checkpoint["model_state"] if final else checkpoint["best_model_state"])
    model.eval()
    return dict(name=run, checkpoint=checkpoint, hparams=hparams, model=model,
                variables=select(all_var, branches=checkpoint["variables"]), names=list(checkpoint["variables"]),
                mean=checkpoint["mean"].to(device), std=checkpoint["std"].to(device))


class _WrappedModel(ModelWrapper):
    """The ODE solver calls model(x, t); our model takes [x, t] as one input."""
    def forward(self, x, t, **extras):
        xt = torch.cat([x, torch.full((len(x), 1), t.item(), device=x.device)], dim=-1)
        return self.model(xt, **extras)


@torch.no_grad()
def generate(model, known, x_known=None, n=None, step_size=0.05, wanted=None, chunk=50000, seed=None):
    """Sample n points with the variables in `known` fixed to their values in x_known.
    known:   (num_variables,) bool, True = conditioned on
    x_known: (n, num_variables) or (1, num_variables) normalized values (the same condition for all n);
             only the known columns are used. None if nothing is known.
    n:       number of samples; default len(x_known)
    wanted:  column indices to return; default all unknown columns. Unknown columns that are not
             wanted are still generated (the model needs every token) and dropped, i.e. marginalized.
    Returns (n, len(wanted)) normalized samples on the CPU."""
    device = next(model.parameters()).device
    known = torch.as_tensor(known, dtype=torch.bool, device=device)
    n = len(x_known) if n is None else n
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    noise = torch.randn(n, model.n_features, device=device, generator=g)
    x_init = noise if x_known is None else torch.where(known, x_known.to(device), noise)
    solver = ODESolver(velocity_model=_WrappedModel(model))
    time_grid = torch.tensor([0.0, 1.0], device=device)
    out = torch.cat([solver.sample(time_grid=time_grid, x_init=x_init[s:s + chunk], method='midpoint',
                                   step_size=step_size, known=known)
                     for s in range(0, n, chunk)])
    wanted = (~known).nonzero().flatten() if wanted is None else torch.as_tensor(wanted, device=device)
    return out[:, wanted].cpu()


@torch.no_grad()
def test_loss(model, data, known, loss_on=None, seed=0, eval_bs=8192):
    """Flow-matching loss of every test point for one mask: (N,) on the CPU, each the mean over that
    point's unknown variables (as in fm_loss). Its mean is the mask's test loss.
    data:    (N, num_variables) normalized test points
    known:   (num_variables,) bool, True = conditioned on
    loss_on: (num_variables,) bool, average only over the unknown variables marked here
             (e.g. the variables another model also has); default all unknown variables
    Times and noise come from `seed`, so every model with the same variables sees the same draws
    (paired comparison)."""
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    n, d = data.shape
    times = torch.rand(n, device=device, generator=g)
    noise = torch.randn(n, d, device=device, generator=g)
    path = AffineProbPath(scheduler=CondOTScheduler())
    known = torch.as_tensor(known, dtype=torch.bool, device=device)
    loss_on = ~known if loss_on is None else ~known & torch.as_tensor(loss_on, dtype=torch.bool, device=device)
    out = []
    for s in range(0, n, eval_bs):
        x1, t, x0 = data[s:s + eval_bs].to(device), times[s:s + eval_bs], noise[s:s + eval_bs]
        kn = known.expand(len(x1), -1)
        ps = path.sample(x_0=x0, x_1=x1, t=t)
        v = model(torch.column_stack([torch.where(kn, x1, ps.x_t), t]), known=kn)
        out.append(((v - ps.dx_t) ** 2 * loss_on).sum(1) / loss_on.sum())
    return torch.cat(out).cpu()


# ---- helpers for compare_fm.ipynb ---------------------------------------------------------------
# A run r is the dict from load_run, plus r["names"] (variable names) and, for make_job,
# r["test"] (normalized test points of the run).

def to_norm(r, values):
    """{name: physical value} -> (1, D) normalized row of run r; the other columns are 0 (not used)"""
    device = r["mean"].device
    x = torch.ones(1, len(r["names"]), device=device)   # 1 is harmless for log10
    for n, v in values.items():
        x[0, r["names"].index(n)] = v
    x = transform(x, r["variables"], r["mean"], r["std"])[0]
    keep = torch.tensor([n in values for n in r["names"]], device=device)
    return torch.where(keep, x, torch.zeros_like(x))


def to_physical(r, x_norm, names):
    """normalized samples (n, len(names)) of run r -> numpy array in physical units.
    Uses inverse_transform from dataset.py on the selected columns (works on a copy)."""
    j = [r["names"].index(n) for n in names]
    x = x_norm.to(r["mean"].device).clone()   # inverse_transform works in place
    return inverse_transform(x, [r["variables"][k] for k in j], r["mean"][j], r["std"][j]).cpu().numpy()


def make_job(r, sel, known_from_test=(), known_fixed=None, wanted=()):
    """Sampling job for run r (keyword arguments of generate), or None if r lacks one of the variables
    or nothing is wanted.
    sel:             indices of the test points used as conditions (one sample per point)
    known_from_test: variables fixed to the values of those test points
    known_fixed:     {name: physical value}, the same for every sample
    wanted:          variables to keep (all other unknown variables are generated and dropped)"""
    known_fixed = known_fixed or {}
    if not wanted or not all(n in r["names"] for n in [*known_from_test, *known_fixed, *wanted]):
        return None
    known = torch.tensor([n in known_from_test or n in known_fixed for n in r["names"]])
    if known_from_test:
        x_known = r["test"][torch.as_tensor(sel, device=r["test"].device)].clone()
        if known_fixed:
            cols = [r["names"].index(n) for n in known_fixed]
            x_known[:, cols] = to_norm(r, known_fixed)[:, cols]
    elif known_fixed:
        x_known = to_norm(r, known_fixed)   # the same condition for every sample
    else:
        x_known = None
    return dict(known=known, x_known=x_known, wanted=[r["names"].index(n) for n in wanted], n=len(sel))


def _plot_values(x, name):
    """One column for plotting; MO_Omega as log10 since it spans orders of magnitude."""
    return (np.log10(x), 'log10(MO_Omega)') if name == "MO_Omega" else (x, name)


def _union(samples):
    return list(dict.fromkeys(n for value in samples.values() for n in value[-1]))


def plot_marginals(samples, variables=None, ncols=4, bins=50):
    """samples: {label: (array (n, k) in physical units, list of the k variable names)}.
    One histogram per variable (default: every variable of any set); a set is drawn if it has the variable.
    The binning range comes from the first set with the variable (0.5-99.5 percentile). Returns the figure."""
    variables = _union(samples) if variables is None else list(variables)
    nrows = (len(variables) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False, constrained_layout=True)
    for ax, name in zip(axes.flat, variables):
        rng, title = None, name
        for c, (label, (x, names)) in enumerate(samples.items()):
            if name not in names:
                continue
            v, title = _plot_values(x[:, names.index(name)], name)
            v = v[np.isfinite(v)]
            rng = np.percentile(v, [0.5, 99.5]) if rng is None else rng
            ax.hist(v, bins=bins, range=rng, density=True, histtype='step', color=f'C{c}', label=label)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('Density')
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    return fig


def plot_corner(samples, variables=None):
    """Overlaid corner plots, one color per set (same colors as plot_marginals).
    variables default: the variables every set has. Sets without all variables are left out. Returns the figure."""
    if variables is None:
        variables = [n for n in _union(samples) if all(n in names for _, names in samples.values())]
    colors = {k: f'C{c}' for c, k in enumerate(samples)}
    prep = {k: np.column_stack([_plot_values(x[:, names.index(n)], n)[0] for n in variables])
            for k, (x, names) in samples.items() if all(n in names for n in variables)}
    labels = [_plot_values(np.ones(1), n)[1] for n in variables]
    ref = next(iter(prep.values()))
    rng = [tuple(np.percentile(ref[np.isfinite(ref[:, i]), i], [0.5, 99.5])) for i in range(len(variables))]
    fig = None
    for k, x in prep.items():
        x = x[np.isfinite(x).all(axis=1)]
        fig = corner.corner(x, fig=fig, color=colors[k], labels=labels, range=rng, bins=30,
                            hist_kwargs={'density': True}, plot_datapoints=False, plot_density=False,
                            levels=(0.68, 0.95), label_kwargs={'fontsize': 8})
    fig.legend(handles=[plt.Line2D([], [], color=colors[k], label=k) for k in prep], loc='upper right', fontsize=14)
    return fig


def plot_pred_vs_truth(pairs, variables=None):
    """pairs: {label: (generated (n, k), truth (n, k), list of the k names)}, physical units.
    One row per variable (default: every variable of any pair), one column per label, with the y = x line.
    Returns the figure."""
    variables = _union(pairs) if variables is None else list(variables)
    labels = list(pairs)
    fig, axes = plt.subplots(len(variables), len(labels), figsize=(4 * len(labels), 3.6 * len(variables)),
                             squeeze=False, constrained_layout=True)
    for i, name in enumerate(variables):
        for j, label in enumerate(labels):
            gen, true, names = pairs[label]
            ax = axes[i, j]
            if name not in names:
                ax.set_visible(False)
                continue
            g, t = gen[:, names.index(name)], true[:, names.index(name)]
            ax.scatter(t, g, s=1, alpha=0.3)
            lims = [min(t.min(), g.min()), max(t.max(), g.max())]
            ax.plot(lims, lims, 'r--', lw=1)
            if name == "MO_Omega":
                ax.set_xscale('log'); ax.set_yscale('log')
                err = f"median |log10(gen/true)| = {np.median(np.abs(np.log10(g / t))):.3f}"
            else:
                err = f"median |gen - true| = {np.median(np.abs(g - t)):.3g}"
            ax.set_title(f"{label}\n{err}", fontsize=8)
            ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"generated {name}")
    return fig
