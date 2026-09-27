"""Transolver-style shape encoder (§5 of nhmo_v2_rewrite_plan.txt).

Takes a point cloud (surface points + normals + optional interior samples)
and produces M slice-tokens of shape (B, M, d_model). These slice-tokens
form the ShapeLatent consumed by the kernel head.

Architecture (§5.1, single-aggregation variant):
  1. Per-point embedding: concat [point, normal] (6D), Fourier-encode,
     project to d_model. Interior points use [point, 0]; type_embed(1)
     added. Surface points get type_embed(0).
  2. Soft-assign each point to M slices:
         slice_weights[n, m] = softmax_over_m(Linear(tokens[n]))
  3. Slice aggregation (weighted mean, permutation-invariant in N):
         slice_features[m] = Σ_n w[n, m] · tokens[n] / Σ_n w[n, m]
  4. n_layers × standard pre-norm Transformer over the M slice tokens.

Architectural invariants:
  - C1 + A6: no SDF grid, no mesh connectivity. forward() rejects the
    shape_ctx keys sdf_grid / mesh / faces / tets with ValueError.
  - C2 + A2: no parameter dimension depends on N_surface or N_interior.
    The only N-dependent tensor is the intermediate slice_weights, which
    is an activation, not a parameter. `n_slices = M` is a capacity
    parameter, fixed at train time.
  - A9: the encoder is a standalone nn.Module. HarmonicMeasureField owns
    it via composition; a single encode() call is reused across many
    (p, ζ) queries per shape.

Reference: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on
General Geometries", ICML 2024.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from nhmo.core.fourier import FourierFeatures
from nhmo.core.kernel import ShapeLatent


REQUIRED_ENCODER_CFG_KEYS: tuple[str, ...] = (
    "d_model",
    "n_slices",
    "n_layers",
    "n_heads",
    "mlp_ratio",
    "dropout",
    "pre_norm",
    "use_normals",
    "fourier_bands",
)


# shape_ctx keys that are explicitly FORBIDDEN (A6 enforcement).
_FORBIDDEN_SHAPE_CTX_KEYS: tuple[str, ...] = ("sdf_grid", "mesh", "faces", "tets")


class ShapeEncoder(ABC, nn.Module):
    """Abstract base: shape_ctx dict → ShapeLatent."""

    @abstractmethod
    def forward(self, shape_ctx: dict) -> ShapeLatent: ...


class TransolverEncoder(ShapeEncoder):
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

        # Per-point input: concat [point, normal] → 6D when use_normals else 3D.
        input_dim = 6 if use_normals else 3
        self.fourier = FourierFeatures(num_bands=fourier_bands, input_dim=input_dim)
        self.input_proj = nn.Linear(self.fourier.output_dim, d_model)

        # Type embedding: 0 = surface, 1 = interior. §5.5.
        self.type_embed = nn.Embedding(2, d_model)

        # Slice projection: per-point → M slice logits. Softmax over M.
        self.slice_proj = nn.Linear(d_model, n_slices)

        # Standard pre-norm transformer over the M slice tokens.
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
        # -------- A6 / C1: reject grid / mesh inputs
        for bad in _FORBIDDEN_SHAPE_CTX_KEYS:
            if bad in shape_ctx:
                raise ValueError(
                    f"TransolverEncoder must not receive {bad!r} "
                    f"(§5.2 / A6: point-cloud input only). shape_ctx keys: "
                    f"{sorted(shape_ctx.keys())}"
                )

        if "surface_points" not in shape_ctx:
            raise KeyError("shape_ctx missing required key 'surface_points'")
        surface_points = shape_ctx["surface_points"]  # (B, N_s, 3)
        B, N_s, _ = surface_points.shape

        if self.use_normals:
            if "surface_normals" not in shape_ctx:
                raise KeyError(
                    "shape_ctx missing 'surface_normals' (required when use_normals=True)"
                )
            surface_normals = shape_ctx["surface_normals"]
            surface_feat = torch.cat([surface_points, surface_normals], dim=-1)  # (B, N_s, 6)
        else:
            surface_feat = surface_points  # (B, N_s, 3)

        # Surface tokens + type_embed(0)
        surface_tokens = self._embed_points(surface_feat)  # (B, N_s, d)
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
                # Pad zeros for interior normals (§5.5).
                interior_feat = torch.cat(
                    [interior_points, torch.zeros_like(interior_points)], dim=-1
                )
            else:
                interior_feat = interior_points
            interior_tokens = self._embed_points(interior_feat)  # (B, N_i, d)
            type_interior = self.type_embed(
                torch.ones(1, dtype=torch.long, device=interior_tokens.device)
            ).view(1, 1, -1)
            interior_tokens = interior_tokens + type_interior
            tokens = torch.cat([surface_tokens, interior_tokens], dim=1)  # (B, N_s + N_i, d)
        else:
            tokens = surface_tokens  # (B, N_s, d)

        # -------- Slice aggregation (Transolver §5.1 step 2–3)
        # slice_weights[n, m] = softmax_over_m(Linear(tokens[n]))
        slice_logits = self.slice_proj(tokens)                       # (B, N, M)
        slice_weights = torch.softmax(slice_logits, dim=-1)          # (B, N, M)

        # Weighted mean over N: slice_features[m] = Σ_n w[n, m] · tokens[n] / Σ_n w[n, m]
        slice_num = torch.einsum("bnm,bnd->bmd", slice_weights, tokens)  # (B, M, d)
        slice_den = slice_weights.sum(dim=1).unsqueeze(-1) + 1e-6        # (B, M, 1)
        slice_features = slice_num / slice_den                            # (B, M, d)

        # -------- Self-attention over slice tokens (§5.1 step 4)
        slice_features = self.slice_transformer(slice_features)           # (B, M, d)

        return ShapeLatent(tokens=slice_features)

    def _embed_points(self, feat: Tensor) -> Tensor:
        """(B, N, input_dim) → (B, N, d_model) via Fourier + Linear."""
        B, N, _ = feat.shape
        flat = feat.reshape(B * N, -1)
        fourier = self.fourier(flat)
        proj = self.input_proj(fourier)
        return proj.reshape(B, N, self.d_model)
