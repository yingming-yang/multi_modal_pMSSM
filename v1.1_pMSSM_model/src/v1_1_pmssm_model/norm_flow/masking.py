"""Arbitrary-conditional flow over a joint (parameters, MO_Omega) vector.

Instead of a fixed direction (params | omega), every dimension of the joint
vector `v = concat(parameters, mo_omega)` can be either "known" (conditioning)
or "unknown" (to be predicted), decided per-example by a binary mask. Each
flow layer only transforms the currently-unknown dims that match its fixed
alternating parity; known dims - and unknown dims not selected by a given
layer - pass through as identity, conditioned on via
`context = (mask ⊙ v, mask)` fed to the transform net. Stacking layers with
alternating parity lets unknown dims interact with each other, the same way
alternating masks let coupling layers mix a fixed set of dims.

Reference: Li, Akbari & Song, "Flow Models for Arbitrary Conditional
Likelihoods" (ACFlow), ICML 2020.
"""
import math

import torch
from torch import nn

import glasflow.nflows as nflows
import glasflow.nflows.nn.nets as nflows_nets
from glasflow.nflows.transforms import splines
from glasflow.nflows.transforms.base import CompositeTransform, Transform


def sample_conditioning_mask(
    batch_size: int,
    total_dim: int,
    device,
    min_unknown: int = 1,
    canonical_masks: torch.Tensor = None,
    canonical_prob: float = 0.0,
):
    """Samples one binary "unknown" mask per example (1 = predict, 0 = given).

    With probability `canonical_prob`, a row of `canonical_masks` is used
    instead of a fully random mask, so training explicitly covers specific
    directions of interest (e.g. all-parameters-unknown / omega-unknown)
    alongside the general arbitrary-conditioning objective.
    """
    mask = (torch.rand(batch_size, total_dim, device=device) < 0.5).float()

    num_unknown = mask.sum(dim=-1)
    too_few = num_unknown < min_unknown
    while too_few.any():
        mask[too_few] = (
            torch.rand(int(too_few.sum()), total_dim, device=device) < 0.5
        ).float()
        num_unknown = mask.sum(dim=-1)
        too_few = num_unknown < min_unknown

    if canonical_masks is not None and canonical_prob > 0.0:
        use_canonical = torch.rand(batch_size, device=device) < canonical_prob
        num_canonical = int(use_canonical.sum())
        if num_canonical > 0:
            rows = torch.randint(0, canonical_masks.shape[0], (num_canonical,), device=device)
            mask[use_canonical] = canonical_masks.to(device)[rows]

    return mask


class MaskedRQCouplingStep(Transform):
    """One arbitrary-conditional coupling layer over the full joint vector.

    Only dims where `unknown_mask * parity == 1` are transformed; every other
    dim (originally known, or unknown but not selected this layer) is passed
    through unchanged and contributes zero log-det, so `unknown_mask` alone
    (no separate context tensor) fully determines the conditioning.
    """

    def __init__(
        self,
        total_dim: int,
        parity_even: bool,
        hidden_dim: int = 128,
        num_transform_blocks: int = 2,
        batch_norm: bool = False,
        num_bins: int = 8,
        tail_bound: float = 1.0,
    ):
        super().__init__()
        self.total_dim = total_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.dim_multiplier = num_bins * 3 - 1  # linear tails, see PiecewiseRationalQuadraticCouplingTransform

        self.register_buffer(
            "parity_mask", nflows.utils.create_alternating_binary_mask(total_dim, even=parity_even).float()
        )

        self.transform_net = nflows_nets.ResidualNet(
            in_features=total_dim,
            out_features=total_dim * self.dim_multiplier,
            hidden_features=hidden_dim,
            context_features=total_dim,  # unknown_mask
            num_blocks=num_transform_blocks,
            activation=nn.ELU(),
            use_batch_norm=batch_norm,
        )

    def _active_mask(self, unknown_mask):
        return unknown_mask * self.parity_mask

    def _spline_params(self, given_values, unknown_mask):
        transform_params = self.transform_net(given_values, context=unknown_mask)
        transform_params = transform_params.reshape(-1, self.total_dim, self.dim_multiplier)

        # Scale down widths/heights before the spline's softmax, same as
        # PiecewiseRationalQuadraticCouplingTransform, for stable init.
        scale = math.sqrt(self.transform_net.hidden_features)
        widths = transform_params[..., : self.num_bins] / scale
        heights = transform_params[..., self.num_bins : 2 * self.num_bins] / scale
        derivatives = transform_params[..., 2 * self.num_bins :]
        return widths, heights, derivatives

    def _transform(self, inputs, context, inverse):
        unknown_mask = context
        active_mask = self._active_mask(unknown_mask)

        given_values = inputs * (1.0 - active_mask)
        widths, heights, derivatives = self._spline_params(given_values, unknown_mask)

        spline_outputs, spline_logabsdet = splines.unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        outputs = torch.where(active_mask.bool(), spline_outputs, inputs)
        logabsdet = (spline_logabsdet * active_mask).sum(dim=-1)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._transform(inputs, context, inverse=False)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context, inverse=True)


def build_arbitrary_conditioning_transform(
    num_flow_steps: int,
    total_dim: int,
    hidden_dim: int = 128,
    num_transform_blocks: int = 2,
    batch_norm: bool = False,
    num_bins: int = 8,
    tail_bound: float = 1.0,
):
    """Stacks alternating-parity `MaskedRQCouplingStep`s into one transform.

    No permutation/linear-mixing layers are used between steps: those would
    need to permute `unknown_mask` identically, which `CompositeTransform`
    does not support (it passes one fixed `context` to every layer). Parity
    alternation across `num_flow_steps` layers is what gives unknown dims the
    chance to condition on and be conditioned on by each other instead.
    """
    return CompositeTransform(
        [
            MaskedRQCouplingStep(
                total_dim=total_dim,
                parity_even=(i % 2 == 0),
                hidden_dim=hidden_dim,
                num_transform_blocks=num_transform_blocks,
                batch_norm=batch_norm,
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            for i in range(num_flow_steps)
        ]
    )


class MaskedFlow(nn.Module):
    """Wraps an arbitrary-conditioning transform with a masked log_prob/sample.

    Not a `glasflow.nflows.flows.Flow`: that class assumes every dim of the
    base distribution is modeled, but here only the "unknown" dims of `v`
    should contribute to the base density - known dims pass through the
    transform as literal identity and must be excluded from `log_prob`, and
    seeded (not sampled) in `sample`.
    """

    def __init__(self, transform, total_dim: int):
        super().__init__()
        self.transform = transform
        self.total_dim = total_dim

    def log_prob(self, v, unknown_mask):
        z, logabsdet = self.transform(v, context=unknown_mask)
        log_pz_per_dim = -0.5 * z**2 - 0.5 * math.log(2 * math.pi)
        log_pz = (log_pz_per_dim * unknown_mask).sum(dim=-1)
        return log_pz + logabsdet

    @torch.no_grad()
    def sample(self, v_known, unknown_mask, num_samples: int = 1):
        """Draws `num_samples` completions of the unknown dims of `v_known`.

        `v_known` may hold arbitrary placeholder values at unknown positions
        (they are discarded); its known-position values are reproduced
        exactly in the output, by construction of the identity pass-through.
        """
        batch_size = v_known.shape[0]
        v_known = v_known.repeat_interleave(num_samples, dim=0)
        unknown_mask = unknown_mask.repeat_interleave(num_samples, dim=0)

        z_unknown = torch.randn_like(v_known)
        z = v_known * (1.0 - unknown_mask) + z_unknown * unknown_mask

        v_generated, _ = self.transform.inverse(z, context=unknown_mask)
        return v_generated.reshape(batch_size, num_samples, self.total_dim)
