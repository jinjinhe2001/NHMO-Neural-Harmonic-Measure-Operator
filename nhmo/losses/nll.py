"""L_NLL: maximum likelihood of WoS hits under the learned kernel.

Per v2 plan §6.1:
    L_NLL = - (1/B) Σ_i (1/K_i) Σ_k  log K_θ(p_i, ζ*_{i,k})

where K_θ is the NORMALIZED kernel. This module is a pure reduction; the
caller (LossRegistry in registry.py) is responsible for computing
`log_K_at_hits` from the kernel's log_kernel() output and the surface-
sample logsumexp normalization.

C3 / A1: no h anywhere in this file. WoS HITS (ζ* coordinates) are
boundary points where Brownian walks terminated — purely geometric, not
boundary values.
"""
from __future__ import annotations

import torch
from torch import Tensor


def l_nll(
    log_K_at_hits: Tensor,   # (B, K_hits) — NORMALIZED log K at WoS-hit coordinates
    hit_mask: Tensor,        # (B, K_hits) bool, True = valid hit
) -> Tensor:                 # scalar
    """Masked-mean negative log-likelihood.

    Even K_i = 1 (single WoS hit per query) is valid — this is the major
    advantage of NLL over KL and the reason v2 can train with very small
    per-query hit counts.
    """
    mask_f = hit_mask.to(log_K_at_hits.dtype)
    n_valid = mask_f.sum()
    if n_valid.item() == 0:
        # Defensive: shouldn't occur in practice.
        return torch.zeros((), device=log_K_at_hits.device, dtype=log_K_at_hits.dtype)
    # -log K at valid entries, zero elsewhere; divide by count of valid.
    neg_log_K = -(log_K_at_hits * mask_f)
    return neg_log_K.sum() / n_valid
