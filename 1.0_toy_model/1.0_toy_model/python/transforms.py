"This module is for transform functions"

import torch
from glasflow.nflows import transforms
import glasflow.nflows as nflows
import glasflow.nflows.nn.nets as nflows_nets

# THe Code is adapted from examples in the [neural spline flow repository](https://github.com/bayesiains/nsf).
# 
# First, we define a function that creates the basic MAF or RQS-coupling transform:


def create_base_transform(
    i: int,
    param_dim: int,
    context_dim: int = None,
    hidden_dim: int = 512,
    num_transform_blocks: int = 2,
    batch_norm: bool = False,
    base_transform_type: str = "rq-coupling",
):

    activation_fn = torch.nn.ELU()

    if base_transform_type == "maf":
        return transforms.MaskedAffineAutoregressiveTransform(
            param_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_blocks=num_transform_blocks,
            activation=activation_fn,
            use_batch_norm=batch_norm,
        )
    elif base_transform_type == "rq-coupling":
        if param_dim == 1:
            mask = torch.tensor([1], dtype=torch.uint8)
        else:
            mask = nflows.utils.create_alternating_binary_mask(
                param_dim, even=(i % 2 == 0)
            )
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=(
                lambda in_features, out_features: nflows_nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,
                    num_blocks=num_transform_blocks,
                    activation=activation_fn,
                    use_batch_norm=batch_norm,
                )
            ),
            num_bins=8,
            tails="linear",
            tail_bound=1.0,
            apply_unconditional_transform=False,
        )

    else:
        raise ValueError


# Next, we need to define a transform the randomly permutes the dimensions to ensure that all dimensions interact sufficiently with each other:

def create_linear_transform(param_dim: int):
    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


# Finally, we can define the `create_transform` function which composes a list of `num_flow_steps` transforms where each transform consists of a linear transform (to permute dimensions) and a base transform (MAF or RQ-coupling).
# We start and finish the transforms with a linear transform that ensures that the flow outputs samples that have the correct dimensionality (`param_dim`).

def create_transform(
    num_flow_steps: int, 
    param_dim: int, 
    context_dim: int, 
    base_transform_kwargs: dict
):
    return transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    create_linear_transform(param_dim),
                    create_base_transform(
                        i, param_dim, context_dim=context_dim, **base_transform_kwargs
                    ),
                ]
            )
            for i in range(num_flow_steps)
        ]
        + [create_linear_transform(param_dim)]
    )


# The entire flow model consists of a standard normal base distribution, followed by the flow transform.