"""Generates a small synthetic pMSSM-like ntuple set for smoke-testing
v1.1.2_bidir_norm_flow.py end-to-end (data loading -> training -> checkpoint).

Not real physics: MO_Omega is a simple bump function of two of the "theory
parameters" plus noise, just so the flow has an actual signal to learn and
the evaluation notebook has something meaningful to plot. One branch is held
constant to also exercise load_data's constant-column filtering.
"""
import os

import numpy as np
import uproot

rng = np.random.default_rng(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "ntuples")
os.makedirs(OUT_DIR, exist_ok=True)

JOB_ID = 99999
NUM_FILES = 4
EVENTS_PER_FILE = 500
UNPHYSICAL_FRACTION = 0.05  # fraction of events with MO_Omega == -1, filtered by load_data


def make_file(idx: int, num_events: int):
    m1 = rng.uniform(100, 2000, num_events).astype(np.float32)
    m2 = rng.uniform(100, 2000, num_events).astype(np.float32)
    mu = rng.uniform(-2000, 2000, num_events).astype(np.float32)
    tanb = rng.uniform(1, 60, num_events).astype(np.float32)
    a_term = rng.uniform(-3000, 3000, num_events).astype(np.float32)
    const_branch = np.full(num_events, 42.0, dtype=np.float32)

    bump = np.exp(-((m1 - 1000.0) / 800.0) ** 2) * np.exp(-((m2 - 1000.0) / 800.0) ** 2)
    mo_omega = (0.12 * bump + rng.normal(0.0, 0.01, num_events)).astype(np.float32)
    mo_omega = np.clip(mo_omega, 1e-4, None)

    unphysical = rng.random(num_events) < UNPHYSICAL_FRACTION
    mo_omega[unphysical] = -1.0

    branches = {
        "IN_M1": m1,
        "IN_M2": m2,
        "IN_mu": mu,
        "IN_tanb": tanb,
        "IN_Aterm": a_term,
        "IN_const": const_branch,
        "MO_Omega": mo_omega,
    }

    path = os.path.join(OUT_DIR, f"ntuple.{JOB_ID}.{idx}.root")
    with uproot.recreate(path) as f:
        # mktree+extend writes a classic TTree; assigning the dict directly
        # (f["susy"] = branches) writes an RNTuple instead, which load_data's
        # tree.arrays(..., library="pd") cannot convert to a DataFrame.
        f.mktree("susy", {name: arr.dtype for name, arr in branches.items()})
        f["susy"].extend(branches)
    return path


if __name__ == "__main__":
    for i in range(NUM_FILES):
        path = make_file(i, EVENTS_PER_FILE)
        print(f"wrote {path} ({EVENTS_PER_FILE} events)")
    total = NUM_FILES * EVENTS_PER_FILE
    print(f"Total events: {total}  (~{int(total * UNPHYSICAL_FRACTION)} unphysical, filtered by load_data)")
