"""SDF-aware 2D shape encoder — Phase 7.0+ rebuild for 2D MNIST.

v1's working architecture used a CNN on the SDF grid plus ISAB on the
boundary point cloud, then cross-attention. v2 (`encoder_2d.py`) only
sees the boundary point cloud. The 5x rel-L2 gap on 2D MNIST (v2 ~0.17 vs
v1 ~0.036 on h=x) traces to that missing inductive bias for interior
geometry. This module restores the SDF stem.

Architecture:
  1. SDFStemCNN: (B, 1, G, G) -> (B, C, G/4, G/4). Three conv blocks with
     two 2x2 maxpools. For G=64 this produces a 16x16 = 256 spatial token
     grid.
  2. Spatial positional embedding (learned, one per spatial cell).
  3. Boundary projection: per-point [point, normal] -> Fourier -> Linear
     -> token (B, N_s, d). Type embed 0.
  4. Optional interior tokens (if interior_points present): per-point ->
     Fourier -> Linear, type embed 1.
  5. Concat [spatial_tokens, surface_tokens, interior_tokens] -> Transformer
     self-attention layers (n_layers).

Returns a ShapeLatent with the full union as tokens.

The slice aggregation step from `TransolverEncoder2D` is dropped; spatial
locality is preserved end-to-end as in v1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from nhmo.core.fourier import FourierFeatures
from nhmo.core.kernel import ShapeLatent
from nhmo.core.encoder import ShapeEncoder


REQUIRED_SDF_ENCODER_CFG_KEYS: tuple[str, ...] = (
    "d_model",
    "n_layers",
    "n_heads",
    "mlp_ratio",
    "dropout",
    "pre_norm",
    "use_normals",
    "fourier_bands",
    "sdf_grid_size",
    "sdf_stem_channels",
)


class SDFStemCNN(nn.Module):
    """3-conv-block CNN. (B, 1, G, G) -> (B, out_channels, G/4, G/4).

    Ported from v1 `legacy/v1/nhm/models/transformer.py:SDFStemCNN`,
    with channel widths preserved (32 -> 48 -> out_channels).
    """

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, sdf: Tensor) -> Tensor:
        if sdf.dim() == 3:
            sdf = sdf.unsqueeze(1)
        return self.conv(sdf)


class EncoderWithSDF2D(ShapeEncoder):
    def __init__(self, encoder_cfg: dict) -> None:
        super().__init__()

        for key in REQUIRED_SDF_ENCODER_CFG_KEYS:
            if key not in encoder_cfg:
                raise KeyError(
                    f"encoder_cfg missing required key: {key!r}. "
                    f"All required keys: {REQUIRED_SDF_ENCODER_CFG_KEYS}."
                )

        self.encoder_cfg = dict(encoder_cfg)

        d_model = int(encoder_cfg["d_model"])
        n_layers = int(encoder_cfg["n_layers"])
        n_heads = int(encoder_cfg["n_heads"])
        mlp_ratio = int(encoder_cfg["mlp_ratio"])
        dropout = float(encoder_cfg["dropout"])
        pre_norm = bool(encoder_cfg["pre_norm"])
        use_normals = bool(encoder_cfg["use_normals"])
        fourier_bands = int(encoder_cfg["fourier_bands"])
        sdf_grid_size = int(encoder_cfg["sdf_grid_size"])
        sdf_stem_channels = int(encoder_cfg["sdf_stem_channels"])

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if sdf_grid_size % 4 != 0:
            raise ValueError(
                f"sdf_grid_size ({sdf_grid_size}) must be divisible by 4 "
                f"(stem applies two 2x2 maxpools)."
            )

        self.d_model = d_model
        self.use_normals = use_normals
        self.spatial_h = sdf_grid_size // 4
        self.spatial_w = sdf_grid_size // 4
        n_spatial = self.spatial_h * self.spatial_w

        self.sdf_stem = SDFStemCNN(out_channels=sdf_stem_channels)
        self.sdf_proj = nn.Linear(sdf_stem_channels, d_model)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, n_spatial, d_model) * 0.02
        )

        input_dim = 4 if use_normals else 2
        self.fourier = FourierFeatures(num_bands=fourier_bands, input_dim=input_dim)
        self.input_proj = nn.Linear(self.fourier.output_dim, d_model)

        self.type_embed = nn.Embedding(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=pre_norm,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.sdf_proj, self.input_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, shape_ctx: dict) -> ShapeLatent:
        if "sdf_grid" not in shape_ctx:
            raise KeyError(
                "EncoderWithSDF2D requires 'sdf_grid' in shape_ctx "
                "(B, G, G) or (B, 1, G, G)."
            )
        if "surface_points" not in shape_ctx:
            raise KeyError("shape_ctx missing required key 'surface_points'")

        sdf_grid = shape_ctx["sdf_grid"]
        if sdf_grid.dim() == 3:
            sdf_grid = sdf_grid.unsqueeze(1)
        if sdf_grid.shape[1] != 1:
            raise ValueError(
                f"sdf_grid must have 1 channel; got shape {tuple(sdf_grid.shape)}"
            )
        B = sdf_grid.shape[0]

        spatial_feat = self.sdf_stem(sdf_grid)
        spatial_feat = spatial_feat.flatten(2).transpose(1, 2)
        spatial_tokens = self.sdf_proj(spatial_feat) + self.spatial_pos_embed
        spatial_tokens = spatial_tokens + self.type_embed(
            torch.full((1,), 2, dtype=torch.long, device=spatial_tokens.device)
        ).view(1, 1, -1)

        surface_points = shape_ctx["surface_points"]
        if surface_points.shape[-1] != 2:
            raise ValueError(
                f"surface_points last dim must be 2; got {surface_points.shape[-1]}"
            )
        if self.use_normals:
            surface_normals = shape_ctx.get("surface_normals")
            if surface_normals is None:
                raise KeyError("use_normals=True requires 'surface_normals' in shape_ctx")
            surface_feat = torch.cat([surface_points, surface_normals], dim=-1)
        else:
            surface_feat = surface_points
        surface_tokens = self._embed_points(surface_feat)
        surface_tokens = surface_tokens + self.type_embed(
            torch.zeros(1, dtype=torch.long, device=surface_tokens.device)
        ).view(1, 1, -1)

        all_tokens = [spatial_tokens, surface_tokens]

        interior_points = shape_ctx.get("interior_points")
        if interior_points is not None:
            if self.use_normals:
                interior_feat = torch.cat(
                    [interior_points, torch.zeros_like(interior_points)], dim=-1
                )
            else:
                interior_feat = interior_points
            interior_tokens = self._embed_points(interior_feat)
            interior_tokens = interior_tokens + self.type_embed(
                torch.ones(1, dtype=torch.long, device=interior_tokens.device)
            ).view(1, 1, -1)
            all_tokens.append(interior_tokens)

        tokens = torch.cat(all_tokens, dim=1)
        tokens = self.transformer(tokens)
        return ShapeLatent(tokens=tokens)

    def _embed_points(self, feat: Tensor) -> Tensor:
        B, N, _ = feat.shape
        flat = feat.reshape(B * N, -1)
        fourier = self.fourier(flat)
        proj = self.input_proj(fourier)
        return proj.reshape(B, N, self.d_model)
