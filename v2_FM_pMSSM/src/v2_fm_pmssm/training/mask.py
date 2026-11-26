import torch

def input_omega_mask(device):
    # 0: inputs known -> predict Omega; 1: Omega known -> predict inputs; 2: unconditional
    return torch.tensor([
        [True] * 12 + [False],
        [False] * 12 + [True],
        [False] * 13,
    ], device=device)


class MaskSampler:
    """Mother class. sampler(length) returns a (length, num_variables) bool tensor on `device`,
    sampler() returns a single (num_variables,) mask.
    True = known (conditioned on), False = unknown (generated), as in input_omega_mask.
    Each sampler has its own random generator, so the same seed always gives the same masks,
    no matter what other random numbers are drawn elsewhere. The masks are drawn on the CPU
    and then moved to `device`, so they are also the same on CPU and GPU."""

    def __init__(self, seed=0, device="cpu"):
        self.generator = torch.Generator().manual_seed(seed)
        self.device = device

    def __call__(self, length=None):
        raise NotImplementedError  # each subclass defines how the masks are drawn


class PresetMaskSampler(MaskSampler):
    """Picks rows of a preset list of masks, uniformly or with probabilities p.
    masks: (num_masks, num_variables) bool tensor, e.g. input_omega_mask("cpu")
    p:     list of probabilities, one per mask, e.g. [0.2, 0.2, 0.6]; None = uniform"""

    def __init__(self, masks, p=None, seed=0, device="cpu"):
        super().__init__(seed, device)
        self.masks = torch.as_tensor(masks, dtype=torch.bool).cpu()
        if p is None:
            p = [1 / len(self.masks)] * len(self.masks)
        self.p = torch.as_tensor(p, dtype=torch.float)
        if len(self.p) != len(self.masks):
            raise ValueError(f"got {len(self.p)} probabilities for {len(self.masks)} masks")
        if (self.p < 0).any() or not torch.isclose(self.p.sum(), torch.tensor(1.0)):
            raise ValueError(f"p must be non-negative and add up to 1, got {p}")

    def __call__(self, length=None):
        n = 1 if length is None else length
        rows = torch.multinomial(self.p, n, replacement=True, generator=self.generator)
        known = self.masks[rows].to(self.device)  # indexing with a tensor makes a copy, self.masks stays unchanged
        return known[0] if length is None else known


class ModalityMaskSampler(MaskSampler):
    """Each modality (Variable.data_type) is known with probability 1/2, all its variables together.
    Masks where every modality is known are drawn again (nothing would be left to generate).
    variables: list[Variable], in column order"""

    def __init__(self, variables, seed=0, device="cpu"):
        super().__init__(seed, device)
        # sorted: a fixed order of modalities, so the same seed gives the same masks in every run
        modalities = sorted({v.data_type for v in variables})
        self.num_modalities = len(modalities)
        self.modality_of = torch.tensor([modalities.index(v.data_type) for v in variables])  # column -> modality

    def __call__(self, length=None):
        n = 1 if length is None else length
        known_modality = torch.rand(n, self.num_modalities, generator=self.generator) < 0.5
        all_known = known_modality.all(dim=1)
        while all_known.any():  # draw again only the rows where every modality is known
            known_modality[all_known] = torch.rand(int(all_known.sum()), self.num_modalities,
                                                   generator=self.generator) < 0.5
            all_known = known_modality.all(dim=1)
        known = known_modality[:, self.modality_of].to(self.device)  # copy each modality's value to its variables
        return known[0] if length is None else known


class FeatureMaskSampler(MaskSampler):
    """Each variable is known with probability 1/2, independently.
    Masks where every variable is known are drawn again (nothing would be left to generate)."""

    def __init__(self, num_variables, seed=0, device="cpu"):
        super().__init__(seed, device)
        self.num_variables = num_variables

    def __call__(self, length=None):
        n = 1 if length is None else length
        known = torch.rand(n, self.num_variables, generator=self.generator) < 0.5
        all_known = known.all(dim=1)
        while all_known.any():  # draw again only the rows where every variable is known
            known[all_known] = torch.rand(int(all_known.sum()), self.num_variables,
                                          generator=self.generator) < 0.5
            all_known = known.all(dim=1)
        known = known.to(self.device)
        return known[0] if length is None else known


# every accepted spelling of mask_type -> its canonical name
MASK_TYPES = {"by_type": "by_type", "by_modality": "by_type", "type": "by_type", "modality": "by_type",
              "by_feature": "by_feature", "feature": "by_feature"}


def make_mask_sampler(mask_type, variables, seed=0, device="cpu"):
    """Training mask sampler for a mask_type:
    by_type (= by_modality, type, modality): ModalityMaskSampler, whole modalities known together
    by_feature (= feature):                   FeatureMaskSampler, every variable on its own"""
    if mask_type not in MASK_TYPES:
        raise ValueError(f"unknown mask_type {mask_type!r}, choose from {sorted(MASK_TYPES)}")
    if MASK_TYPES[mask_type] == "by_type":
        return ModalityMaskSampler(variables, seed, device)
    return FeatureMaskSampler(len(variables), seed, device)


def modality_masks(variables):
    """Every by-modality mask for these variables: {name: (num_variables,) bool, True = known}.
    Each modality (Variable.data_type) is either known or generated; "all known" is left out.
    name = "known: A, B | gen: C, D" (modalities sorted), so the same mask has the same name
    in every model with the same modalities."""
    modalities = sorted({v.data_type for v in variables})
    masks = {}
    for code in range(2 ** len(modalities) - 1):  # the last code would be "all known"
        known_mod = [m for k, m in enumerate(modalities) if code >> k & 1]
        gen_mod = [m for m in modalities if m not in known_mod]
        name = f"known: {', '.join(known_mod) or '-'} | gen: {', '.join(gen_mod)}"
        masks[name] = torch.tensor([v.data_type in known_mod for v in variables])
    return masks


def eval_masks(variables, input_type="theory_parameter"):
    """Fixed masks for the per-mask validation losses: {name: (known, loss_on)}, both (num_variables,) bool.
    joint:               nothing known, loss on every variable
    inputs->M:           inputs (input_type) known, loss on modality M only
    M->inputs:           M known, loss on the inputs only
    observables->inputs: every non-input variable known, loss on the inputs
                         (left out with only one observable modality: then it equals M->inputs)
    In inputs->M and M->inputs the other modalities are generated (unknown) but not in the loss."""
    data_type = [v.data_type for v in variables]
    is_input = torch.tensor([t == input_type for t in data_type])
    observables = sorted(set(data_type) - {input_type})
    nothing = torch.zeros(len(variables), dtype=torch.bool)
    masks = {"joint": (nothing, ~nothing)}
    for m in observables:
        is_m = torch.tensor([t == m for t in data_type])
        masks[f"inputs->{m}"] = (is_input, is_m)
    for m in observables:
        is_m = torch.tensor([t == m for t in data_type])
        masks[f"{m}->inputs"] = (is_m, is_input)
    if len(observables) > 1:
        masks["observables->inputs"] = (~is_input, is_input)
    return masks
