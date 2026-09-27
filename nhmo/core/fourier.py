"""NeRF-style Fourier positional features for inputs in [-1, 1]^d.

Per v2 plan section 4.3:
  - bands = 10 for p (query point)
  - bands = 10 for ζ (surface point)
  - bands = 4  for ζ_normal (normals, less oscillatory)

Input range convention:
  All coordinates MUST be in [-1, 1]^d (± range_tol tolerance). NeRF's
  f_k = π · 2^k ladder is calibrated for magnitude ~1; inputs outside
  [-1, 1] mean every frequency is an octave off and indicate a
  preprocessing bug. `forward()` asserts the range on every call.

  nhmo/geometry/surface.py, interior.py, and the (deferred) preprocessing
  own the responsibility of normalizing all sampled points into this range.

Implementation notes:
  - No nn.Linear inside this module — projection to d_model is the kernel
    head's responsibility. This keeps FourierFeatures reusable across
    different embedding dimensions.
  - num_bands is REQUIRED (no default). kernel.py instantiates three
    separate instances because the three input streams have different
    band counts per §4.3.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class FourierFeatures(nn.Module):
    def __init__(
        self,
        num_bands: int,
        input_dim: int = 3,
        include_input: bool = True,
        freq_base: float = 2.0,
        scale: float = math.pi,
        range_tol: float = 0.05,
    ) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        self.num_bands = num_bands
        self.input_dim = input_dim
        self.include_input = include_input
        self.range_tol = range_tol
        freqs = scale * (freq_base ** torch.arange(num_bands, dtype=torch.float32))
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.num_bands * self.input_dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., input_dim) → (..., output_dim).

        Asserts x ∈ [-1 - range_tol, 1 + range_tol]^input_dim.
        """
        max_abs = x.detach().abs().max().item() if x.numel() > 0 else 0.0
        if max_abs > 1.0 + self.range_tol:
            raise AssertionError(
                f"FourierFeatures input must be in [-1, 1]^{self.input_dim} "
                f"(± eps={self.range_tol}); got max |x| = {max_abs:.4f}. "
                f"Check the preprocessing normalization."
            )

        xf = x.unsqueeze(-1) * self.freqs  # (..., input_dim, num_bands)
        sin = torch.sin(xf).flatten(-2)    # (..., input_dim * num_bands)
        cos = torch.cos(xf).flatten(-2)
        if self.include_input:
            return torch.cat([x, sin, cos], dim=-1)
        return torch.cat([sin, cos], dim=-1)
