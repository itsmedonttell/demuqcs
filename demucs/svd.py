# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Ways to make the model stronger."""
import random
import torch
import typing as tp


def power_iteration(m: torch.Tensor, niters: int = 1, bs: int = 1) -> torch.Tensor:
    """This is the power method. batch size is used to try multiple starting point in parallel."""
    assert m.dim() == 2
    assert m.shape[0] == m.shape[1]
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype)

    for _ in range(niters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True)
        b = n / (1e-10 + norm)

    return norm.mean()


# We need a shared RNG to make sure all the distributed worker will skip the penalty together,
# as otherwise we wouldn't get any speed up.
penalty_rng: random.Random = random.Random(1234)


def svd_penalty(model: torch.nn.Module, min_size: float = 0.1, dim: int = 1, niters: int = 2, powm: bool = False, convtr: bool = True,
                proba: float = 1, conv_only: bool = False, exact: bool = False, bs: int = 1) -> float:
    """
    Penalty on the largest singular value for a layer.
    Args:
        - model: model to penalize
        - min_size: minimum size in MB of a layer to penalize.
        - dim: projection dimension for the svd_lowrank. Higher is better but slower.
        - niters: number of iterations in the algorithm used by svd_lowrank.
        - powm: use power method instead of lowrank SVD, my own experience
            is that it is both slower and less stable.
        - convtr: when True, differentiate between Conv and Transposed Conv.
            this is kept for compatibility with older experiments.
        - proba: probability to apply the penalty.
        - conv_only: only apply to conv and conv transposed, not LSTM
            (might not be reliable for other models than Demucs).
        - exact: use exact SVD (slow but useful at validation).
        - bs: batch_size for power method.
    """
    total: float = 0.0
    if penalty_rng.random() > proba:
        return 0.0

    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if p.numel() / 2**18 < min_size:
                continue
            p_tensor: torch.Tensor = p  # Keep original as p, work with p_tensor for reshaping
            if convtr:
                if isinstance(m, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
                    if p.dim() in [3, 4]:
                        p_tensor = p.transpose(0, 1).contiguous()
            if p_tensor.dim() == 3:
                p_tensor = p_tensor.view(len(p_tensor), -1)
            elif p_tensor.dim() == 4:
                p_tensor = p_tensor.view(len(p_tensor), -1)
            elif p_tensor.dim() == 1:
                continue
            elif conv_only:
                continue
            assert p_tensor.dim() == 2, (name, p_tensor.shape)
            if exact:
                estimate = torch.svd(p_tensor, compute_uv=False)[1].pow(2).max()
            elif powm:
                a, b = p_tensor.shape
                if a < b:
                    n = p_tensor.mm(p_tensor.t())
                else:
                    n = p_tensor.t().mm(p_tensor)
                estimate = power_iteration(n, niters, bs)
            else:
                estimate = torch.svd_lowrank(p_tensor, dim, niters)[1][0].pow(2)
            total += estimate.item()
    return total / proba
