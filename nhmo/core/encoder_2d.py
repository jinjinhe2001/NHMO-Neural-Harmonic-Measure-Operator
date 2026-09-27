"""2D Transolver-style encoder — Phase 7.0 parallel of `encoder.py`.

Same architecture as 3D (slice-attention + per-point Fourier + type
embedding + transformer over slice tokens). Dimension-specific changes:

  - Surface/interior points are (B, N, 2) instead of (B, N, 3).
  - When `use_normals=True`, `input_dim = 4` (point + normal), not 6.
  - Otherwise `input_dim = 2`.

Everything below the Fourier feature extractor is dimension-agnostic
and reuses the 3D-tested logic by importing `ShapeEncoder`,
`REQUIRED_ENCODER_CFG_KEYS`, and `_FORBIDDEN_SHAPE_CTX_KEYS` from
`encoder.py`.

Architectural invariants (mirror encoder.py):
  - C1 + A6: rejects shape_ctx keys sdf_grid / mesh / faces / tets.
  - C2 + A2: no parameter dimension scales with N_surface / N_interior.
  - A9: standalone nn.Module; HarmonicMeasureField2D owns it via
    composition (Phase 7.0 wiring in kernel_2d.py's __init__).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from nhmo.core.fourier import FourierFeatures
from nhmo.core.kernel import ShapeLatent
from nhmo.core.encoder import (
    REQUIRED_ENCODER_CFG_KEYS,
    ShapeEncoder,
    _FORBIDDEN_SHAPE_CTX_KEYS,
)


class TransolverEncoder2D(ShapeEncoder):
    def __init__(self, encoder_cfg: dict) -> None:
        super().__init__()

        for key in REQUIRED_ENCODER_CFG_KEYS:
            if key not in encoder_cfg:
                raise KeyError(
                    f"encoder_cfg missing required key: {key!r}. "
                    f"All required keys: {REQUIRED_ENCODER_CFG_KEYS}."
                )

        self.encoder_cfg = dict(encoder_cfg)

        d_model = int(encoder_cfg["d_model"])
        n_slices = int(encoder_cfg["n_slices"])
        n_layers = int(encoder_cfg["n_layers"])
        n_heads = int(encoder_cfg["n_heads"])
        mlp_ratio = int(encoder_cfg["mlp_ratio"])
        dropout = float(encoder_cfg["dropout"])
        pre_norm = bool(encoder_cfg["pre_norm"])
        use_normals = bool(encoder_cfg["use_normals"])
        fourier_bands = int(encoder_cfg["fourier_bands"])

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_slices = n_slices
        self.use_normals = use_normals

        # 2D per-point input: [point, normal] = 4 dims when use_normals, else 2.
        input_dim = 4 if use_normals else 2
        self.fourier = FourierFeatures(num_bands=fourier_bands, input_dim=input_dim)
        self.input_proj = nn.Linear(self.fourier.output_dim, d_model)

        # Type embedding: 0 = surface, 1 = interior (§5.5).
        self.type_embed = nn.Embedding(2, d_model)

        # Slice projection: per-point → M slice logits. Softmax over M.
        self.slice_proj = nn.Linear(d_model, n_slices)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=pre_norm,
        )
        self.slice_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, shape_ctx: dict) -> ShapeLatent:
        for bad in _FORBIDDEN_SHAPE_CTX_KEYS:
            if bad in shape_ctx:
                raise ValueError(
                    f"TransolverEncoder2D must not receive {bad!r} "
                    f"(C1 / A6: point-cloud input only). shape_ctx keys: "
                    f"{sorted(shape_ctx.keys())}"
                )

        if "surface_points" not in shape_ctx:
            raise KeyError("shape_ctx missing required key 'surface_points'")
        surface_points = shape_ctx["surface_points"]                  # (B, N_s, 2)
        if surface_points.shape[-1] != 2:
            raise ValueError(
                f"TransolverEncoder2D requires 2D surface_points; got last dim "
                f"{surface_points.shape[-1]}. Use TransolverEncoder (3D) for 3D inputs."
            )
        B, N_s, _ = surface_points.shape

        if self.use_normals:
            if "surface_normals" not in shape_ctx:
                raise KeyError(
                    "shape_ctx missing 'surface_normals' (required when use_normals=True)"
                )
            surface_normals = shape_ctx["surface_normals"]
            surface_feat = torch.cat([surface_points, surface_normals], dim=-1)   # (B, N_s, 4)
        else:
            surface_feat = surface_points                                         # (B, N_s, 2)

        surface_tokens = self._embed_points(surface_feat)             # (B, N_s, d)
        type_surface = self.type_embed(
            torch.zeros(1, dtype=torch.long, device=surface_tokens.device)
        ).view(1, 1, -1)
        surface_tokens = surface_tokens + type_surface

        interior_points = shape_ctx.get("interior_points")
        if interior_points is not None:
            B_i, N_i, _ = interior_points.shape
            if B_i != B:
                raise ValueError(
                    f"batch-size mismatch: surface B={B}, interior B={B_i}"
                )
            if self.use_normals:
                interior_feat = torch.cat(
                    [interior_points, torch.zeros_like(interior_points)], dim=-1
                )
            else:
                interior_feat = interior_points
            interior_tokens = self._embed_points(interior_feat)       # (B, N_i, d)
            type_interior = self.type_embed(
                torch.ones(1, dtype=torch.long, device=interior_tokens.device)
            ).view(1, 1, -1)
            interior_tokens = interior_tokens + type_interior
            tokens = torch.cat([surface_tokens, interior_tokens], dim=1)
        else:
            tokens = surface_tokens

        slice_logits = self.slice_proj(tokens)                        # (B, N, M)
        slice_weights = torch.softmax(slice_logits, dim=-1)           # (B, N, M)
        slice_num = torch.einsum("bnm,bnd->bmd", slice_weights, tokens)
        slice_den = slice_weights.sum(dim=1).unsqueeze(-1) + 1e-6
        slice_features = slice_num / slice_den                         # (B, M, d)

        slice_features = self.slice_transformer(slice_features)

        return ShapeLatent(tokens=slice_features)

    def _embed_points(self, feat: Tensor) -> Tensor:
        B, N, _ = feat.shape
        flat = feat.reshape(B * N, -1)
        fourier = self.fourier(flat)
        proj = self.input_proj(fourier)
        return proj.reshape(B, N, self.d_model)
