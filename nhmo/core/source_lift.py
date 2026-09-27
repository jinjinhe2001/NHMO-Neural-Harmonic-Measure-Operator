"""Source-lifting head — amortized Poisson particular solution.

Story: u(p) = v_φ(p; Ω, f) + ⟨h, K_θ(p, ·; Ω)⟩
       \\____________/      \\______________/
        zero-BC particular   harmonic correction
        solution (this file)  via existing K_θ

Key design — ZERO-BC GAUGE: we constrain v_φ|∂Ω ≡ 0 by construction so the
boundary integral above can use the RAW Dirichlet data h, not h − v_φ|∂Ω.
This kills the Newtonian-decomp catastrophic cancellation we saw earlier.

Implementation: v_φ(p) = max(0, −SDF(p)) · ṽ(p, Ω, f), where SDF is the
signed-distance-to-∂Ω (negative inside Ω in our convention) and ṽ is the
unconstrained network output. On ∂Ω, SDF = 0 ⇒ v_φ = 0. ✓

ṽ is produced by a small cross-attention head: query = Fourier(p), keys/
values = (shape_latent ∪ source_tokens), where source_tokens summarize
the spatial pattern of f via a Transolver-style slice aggregator over
points (q_i, f(q_i)).

This is amortized: ONE forward pass per query at inference, no per-query
Monte Carlo, no volumetric Green's evaluation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nhmo.core.fourier import FourierFeatures
from nhmo.core.kernel import ShapeLatent, _CrossAttnBlock


class SourceTokenEncoder(nn.Module):
    """Transolver-style slice aggregator over (q, f(q)) point cloud.

    Input:
        f_pos:   (B, N, 3)  — positions where f is sampled (e.g., v_tet)
        f_value: (B, N)     — scalar source value at each position
    Output:
        source_tokens: (B, M, d_model) — M slice tokens summarizing f
    """

    def __init__(
        self,
        d_model: int = 128,
        n_slices: int = 32,
        fourier_bands: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        space_dim: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_slices = n_slices
        self.space_dim = space_dim
        self.fourier_q = FourierFeatures(num_bands=fourier_bands, input_dim=space_dim)
        # Per-point token: concat[Fourier(pos), f_value] → linear → d_model
        self.proj_in = nn.Linear(self.fourier_q.output_dim + 1, d_model)
        # Soft slice assignment
        self.slice_proj = nn.Linear(d_model, n_slices)
        # Self-attention over the M slice tokens
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * mlp_ratio, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(self, f_pos: Tensor, f_value: Tensor) -> Tensor:
        B, N, _ = f_pos.shape
        # Per-point embedding
        feat_pos = self.fourier_q(f_pos.reshape(B * N, self.space_dim)).reshape(B, N, -1)
        f_v = f_value.unsqueeze(-1)                                    # (B, N, 1)
        tokens = self.proj_in(torch.cat([feat_pos, f_v], dim=-1))      # (B, N, d)
        # Soft slice assignment
        slice_logits = self.slice_proj(tokens)                          # (B, N, M)
        slice_weights = F.softmax(slice_logits, dim=-1)                 # over M
        # Aggregate: slice_features[b, m, :] = Σ_n w[b, n, m] · tokens[b, n, :]
        # Normalize by sum_n w[b, n, m] to keep magnitudes bounded.
        # einsum: tokens (B, N, d), slice_weights (B, N, M) -> (B, M, d)
        sum_w = slice_weights.sum(dim=1, keepdim=False).clamp(min=1e-6)  # (B, M)
        slices = torch.einsum("bnm,bnd->bmd", slice_weights, tokens)    # (B, M, d)
        slices = slices / sum_w.unsqueeze(-1)
        for layer in self.layers:
            slices = layer(slices)
        return slices


class SourceLiftHead(nn.Module):
    """Maps (p, shape_latent, source_tokens, sdf_at_p) → v_φ(p) with v_φ|∂Ω = 0.

    Architecture mirrors HarmonicMeasureField's kernel head: cross-attention
    from Fourier(p) to (shape_latent ∪ source_tokens), then MLP head to a
    scalar, then SDF-mask for zero-BC gauge.

    The mask `(-sdf).clamp(min=0)` returns 0 on ∂Ω (sdf=0) and grows linearly
    with depth into Ω. We apply softplus(...) on the unconstrained ṽ output
    if we want strictly nonneg v, but here we allow any sign — the source
    contribution can be either sign depending on f.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_cross_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        pre_norm: bool = True,
        fourier_bands_p: int = 10,
        space_dim: int = 3,
        use_uh_cond: bool = False,
        gauge_kind: str = 'linear',  # 'linear' = (-sdf).clamp(min=0); 'tanh' = tanh(-sdf/eps)
        gauge_eps: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.space_dim = space_dim
        self.use_uh_cond = use_uh_cond
        self.gauge_kind = gauge_kind
        self.gauge_eps = gauge_eps
        self.fourier_p = FourierFeatures(num_bands=fourier_bands_p, input_dim=space_dim)
        self.proj_p = nn.Linear(self.fourier_p.output_dim, d_model)
        if use_uh_cond:
            # Embed scalar u_h(p) into d_model conditioning vector.
            self.uh_proj = nn.Linear(1, d_model)
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, mlp_ratio, dropout, pre_norm)
            for _ in range(n_cross_layers)
        ])
        # Head: optionally augment with u_h scalar before final MLP.
        head_in = d_model + (1 if use_uh_cond else 0)
        self.head_mlp = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        p: Tensor,                  # (B, 3) — query points (one per batch entry)
        shape_latent_tokens: Tensor,    # (B, M_shape, d) — from K_θ encoder
        source_tokens: Tensor,           # (B, M_source, d) — from SourceTokenEncoder
        sdf_at_p: Tensor,                # (B,) — signed distance, negative inside Ω
        u_h_at_p: Tensor | None = None,  # (B,) — kernel boundary integral at p (optional cond)
    ) -> Tensor:                          # (B,) — v_φ(p)
        B = p.shape[0]
        p_feat = self.fourier_p(p)
        p_token = self.proj_p(p_feat).unsqueeze(1)                      # (B, 1, d)
        if self.use_uh_cond:
            if u_h_at_p is None:
                raise ValueError("use_uh_cond=True requires u_h_at_p in forward()")
            uh_token = self.uh_proj(u_h_at_p.unsqueeze(-1)).unsqueeze(1)  # (B, 1, d)
            p_token = p_token + uh_token
        context = torch.cat([shape_latent_tokens, source_tokens], dim=1)
        x = p_token
        for layer in self.cross_layers:
            x = layer(x, context)
        x_flat = x.squeeze(1)                                            # (B, d)
        if self.use_uh_cond:
            x_flat = torch.cat([x_flat, u_h_at_p.unsqueeze(-1)], dim=-1)
        tilde_v = self.head_mlp(x_flat).squeeze(-1)                      # (B,)
        # Zero-BC gauge: v_φ = gauge(p) · ṽ. Two options:
        #   linear: depth = max(0, -sdf) — original; ramps slowly inside Ω
        #   tanh:   depth = tanh(max(0,-sdf)/eps) — saturates fast; preserves interior signal
        if self.gauge_kind == 'tanh':
            depth = torch.tanh((-sdf_at_p).clamp(min=0.0) / self.gauge_eps)
        else:
            depth = (-sdf_at_p).clamp(min=0.0)
        return depth * tilde_v


class CnnSourceFieldEncoder(nn.Module):
    """CNN encoder over the full 2D source field f(x,y) on a regular grid.

    Mirrors v1's SDFStemCNN pattern, but consumes the source field instead of
    the SDF. Provides spectrally rich representation of f for the lift —
    addresses the failure mode where high-frequency Poisson sources (sin_cos,
    gaussian) are under-represented by 512 sampled-point tokens.

    Input: (B, 1, G, G) source field (or (B, G, G), broadcast)
    Output: (B, S, d_model) spatial tokens with positional embedding
    """

    def __init__(
        self,
        d_model: int = 128,
        grid_size: int = 64,
        stem_channels: int = 64,
    ) -> None:
        super().__init__()
        if grid_size % 4 != 0:
            raise ValueError(f"grid_size ({grid_size}) must be divisible by 4")
        self.d_model = d_model
        self.spatial_h = grid_size // 4
        self.spatial_w = grid_size // 4
        n_spatial = self.spatial_h * self.spatial_w
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, stem_channels, 3, padding=1),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )
        self.proj = nn.Linear(stem_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_spatial, d_model) * 0.02)

    def forward(self, source_grid: Tensor) -> Tensor:
        if source_grid.dim() == 3:
            source_grid = source_grid.unsqueeze(1)
        feat = self.stem(source_grid)                        # (B, C, h, w)
        feat = feat.flatten(2).transpose(1, 2)               # (B, S, C)
        return self.proj(feat) + self.pos_embed              # (B, S, d)


class PoissonLiftModule(nn.Module):
    """Full Poisson-lift module composing source encoder + lift head.

    This module is JOINT to a frozen K_θ via shared shape_latent (computed
    by the K_θ encoder upstream). At training time we receive the
    shape_latent already encoded; we don't re-encode the geometry.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_source_slices: int = 32,
        n_source_layers: int = 2,
        n_cross_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        fourier_bands_q: int = 8,
        fourier_bands_p: int = 10,
        space_dim: int = 3,
        use_cnn_source: bool = False,
        cnn_grid_size: int = 64,
        cnn_stem_channels: int = 64,
        use_uh_cond: bool = False,
        gauge_kind: str = 'linear',
        gauge_eps: float = 0.05,
    ) -> None:
        super().__init__()
        self.space_dim = space_dim
        self.use_cnn_source = use_cnn_source
        self.use_uh_cond = use_uh_cond
        self.source_enc = SourceTokenEncoder(
            d_model=d_model, n_slices=n_source_slices,
            fourier_bands=fourier_bands_q, n_layers=n_source_layers,
            n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout,
            space_dim=space_dim,
        )
        if use_cnn_source:
            self.cnn_source_enc = CnnSourceFieldEncoder(
                d_model=d_model, grid_size=cnn_grid_size,
                stem_channels=cnn_stem_channels,
            )
        self.lift_head = SourceLiftHead(
            d_model=d_model, n_cross_layers=n_cross_layers,
            n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout,
            fourier_bands_p=fourier_bands_p,
            space_dim=space_dim,
            use_uh_cond=use_uh_cond,
            gauge_kind=gauge_kind,
            gauge_eps=gauge_eps,
        )

    def encode_source(
        self,
        f_pos: Tensor,
        f_value: Tensor,
        source_grid: Tensor | None = None,
    ) -> Tensor:
        """Compute source_tokens once per (Ω, f). Reusable across queries.

        If `use_cnn_source` is True and `source_grid` is given, concatenate the
        spatial CNN tokens to the point-cloud slice tokens.
        """
        slice_tokens = self.source_enc(f_pos, f_value)
        if self.use_cnn_source:
            if source_grid is None:
                raise ValueError(
                    "use_cnn_source=True requires `source_grid` arg in encode_source()"
                )
            spatial_tokens = self.cnn_source_enc(source_grid)
            return torch.cat([slice_tokens, spatial_tokens], dim=1)
        return slice_tokens

    def forward(
        self,
        p: Tensor,
        shape_latent_tokens: Tensor,
        source_tokens: Tensor,
        sdf_at_p: Tensor,
        u_h_at_p: Tensor | None = None,
    ) -> Tensor:
        return self.lift_head(p, shape_latent_tokens, source_tokens, sdf_at_p, u_h_at_p)
