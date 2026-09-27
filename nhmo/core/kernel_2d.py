"""2D harmonic measure field — Phase 7.0 parallel of `kernel.py`.

Exposes `HarmonicMeasureField2D`, the 2D analog of `HarmonicMeasureField`.
Reuses the 3D module's structural building blocks (`_CrossAttnBlock`,
`ShapeLatent`, `REQUIRED_KERNEL_HEAD_CFG_KEYS`, `EllipticMeasureField`)
because they are all dimension-agnostic; only the Fourier feature
extractors and the `p` / `ζ` / `ζ_normal` tensor shapes change.

Design reasoning (why a parallel file, not a `dim=` parameter on the 3D
class):
  - Phase 7.0 plan called for a parallel entrypoint to keep the 3D
    code's `.view(-1, 3)` assumptions untouched while we bring up the
    2D pipeline. A single dimension-parameterized class risks breaking
    Phase 5/6 tests during 2D debugging.
  - The API is byte-for-byte identical across 2D / 3D. Downstream
    LossRegistry and Trainer code receives `(log_kernel, kernel, solve,
    encode)` and doesn't care about `dim`.
  - After Phase 7.0 ships green, a unification pass MAY later fold 2D
    and 3D into one class with `dim` as a config — that's a refactor,
    not a greenfield.

Invariants shared verbatim with the 3D module (PHILOSOPHY.md §0):
  - C2 / A2: no parameter tensor scales with N_surface.
  - C3 / A1: `log_kernel` / `kernel` NEVER take `h`; `solve` takes
    `h_values` only at the post-hoc integration step.
  - A4: no softmax over ζ anywhere (logsumexp-based normalization).
  - A10: no ζ ↔ ζ attention. Cross-attention only.
  - A9: encoder is a separate module (Phase 3's `TransolverEncoder2D`
    in Phase 7.0's `nhmo/core/encoder_2d.py`; see forthcoming file).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from nhmo.core.fourier import FourierFeatures
from nhmo.core.integration import post_hoc_solve
from nhmo.core.kernel import (
    EllipticMeasureField,
    REQUIRED_KERNEL_HEAD_CFG_KEYS,
    ShapeLatent,
    _CrossAttnBlock,
)


class HarmonicMeasureField2D(EllipticMeasureField):
    """2D parallel of `HarmonicMeasureField`.

    API is identical to the 3D version; every tensor that was `(B, 3)`
    in 3D is `(B, 2)` here, and `(B, N, 3)` becomes `(B, N, 2)`.
    `ShapeLatent` is reused directly — encoder produces
    `(B, M, d_model)` tokens regardless of domain dimension.

    Phase 7.0 ships with no 2D encoder yet; tests pass `encoder_cfg={}`
    and construct `ShapeLatent` manually, exercising only the kernel
    head. Full encoder wiring lands in `nhmo/core/encoder_2d.py`.
    """

    def __init__(
        self,
        encoder_cfg: dict,
        kernel_head_cfg: dict,
        normalize: str,
    ) -> None:
        super().__init__()
        if normalize not in {"soft", "hard", "none"}:
            raise ValueError(
                f"normalize must be one of soft/hard/none, got {normalize!r}"
            )
        for key in REQUIRED_KERNEL_HEAD_CFG_KEYS:
            if key not in kernel_head_cfg:
                raise KeyError(
                    f"kernel_head_cfg missing required key: {key!r}. "
                    f"All required keys: {REQUIRED_KERNEL_HEAD_CFG_KEYS}."
                )

        self.encoder_cfg = encoder_cfg
        self.kernel_head_cfg = dict(kernel_head_cfg)
        self.normalize = normalize
        # Phase 7.1.1 Fix 3: soft tanh cap on raw log_K̃ to prevent
        # delta-collapse under soft normalization. Same defaults and
        # semantics as the 3D HarmonicMeasureField. Without this cap,
        # the 2D path delta-collapsed during overnight MNIST training
        # (NLL drove to -1e6 by step 30 k).
        _cap = kernel_head_cfg.get("log_K_max", None)
        self.log_K_max: float | None = None if _cap is None else float(_cap)

        # The 2D encoder is Phase 7.0's `nhmo/core/encoder_2d.py`. Gate the
        # construction behind encoder_cfg non-empty so tests can build
        # ShapeLatent manually. Until that file exists, requesting the
        # encoder raises NotImplementedError rather than ImportError (the
        # latter would read as a broken install).
        if encoder_cfg:
            enc_type = encoder_cfg.get("type", "transolver")
            if enc_type == "transolver":
                from nhmo.core.encoder_2d import TransolverEncoder2D
                self._encoder = TransolverEncoder2D(
                    {k: v for k, v in encoder_cfg.items() if k != "type"}
                )
            elif enc_type == "sdf_cnn":
                from nhmo.core.encoder_2d_sdf import EncoderWithSDF2D
                self._encoder = EncoderWithSDF2D(
                    {k: v for k, v in encoder_cfg.items() if k != "type"}
                )
            else:
                raise ValueError(
                    f"unknown encoder type: {enc_type!r} "
                    f"(supported: 'transolver', 'sdf_cnn')"
                )
        else:
            self._encoder = None

        d_model = int(kernel_head_cfg["d_model"])
        n_heads = int(kernel_head_cfg["n_heads"])
        n_cross_layers = int(kernel_head_cfg["n_cross_layers"])
        mlp_ratio = int(kernel_head_cfg["mlp_ratio"])
        dropout = float(kernel_head_cfg["dropout"])
        pre_norm = bool(kernel_head_cfg["pre_norm"])
        bands_p = int(kernel_head_cfg["fourier_bands_p"])
        bands_zeta = int(kernel_head_cfg["fourier_bands_zeta"])
        bands_normal = int(kernel_head_cfg["fourier_bands_normal"])
        self.use_distance = bool(kernel_head_cfg.get("use_distance", False))

        self.d_model = d_model

        # 2D Fourier features. Separate instances per stream (§4.3).
        self.fourier_p = FourierFeatures(num_bands=bands_p, input_dim=2)
        self.fourier_zeta = FourierFeatures(num_bands=bands_zeta, input_dim=2)
        self.fourier_normal = FourierFeatures(num_bands=bands_normal, input_dim=2)

        self.proj_p = nn.Linear(self.fourier_p.output_dim, d_model)
        self.proj_zeta = nn.Linear(self.fourier_zeta.output_dim, d_model)
        self.proj_normal = nn.Linear(self.fourier_normal.output_dim, d_model)

        if self.use_distance:
            self.fourier_delta = FourierFeatures(num_bands=bands_zeta, input_dim=2)
            self.proj_delta = nn.Linear(self.fourier_delta.output_dim + 1, d_model)

        # Cross-attention stack: ζ queries, (shape_latent ∪ p_token) kv.
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, mlp_ratio, dropout, pre_norm)
            for _ in range(n_cross_layers)
        ])

        # Head — same pattern as 3D.
        self.head_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    # ---------------------------------------------------------------- encode
    def encode(self, shape_ctx: dict) -> ShapeLatent:
        if self._encoder is None:
            raise ValueError(
                "HarmonicMeasureField2D was constructed with encoder_cfg={}; "
                "encode(shape_ctx) is unavailable. Either construct with a "
                "non-empty encoder_cfg once encoder_2d.py ships, or build "
                "ShapeLatent manually and call log_kernel / kernel / solve."
            )
        return self._encoder(shape_ctx)

    # ------------------------------------------------------------ log_kernel
    def log_kernel(
        self,
        p: Tensor,                 # (B, 2)
        zeta: Tensor,              # (B, N, 2)
        zeta_normal: Tensor,       # (B, N, 2)
        shape_latent: ShapeLatent,
    ) -> Tensor:                   # (B, N)
        B, N, _ = zeta.shape

        p_feat = self.fourier_p(p)                     # (B, fourier_p_dim)
        p_token = self.proj_p(p_feat).unsqueeze(1)     # (B, 1, d)

        z_flat = zeta.reshape(B * N, 2)
        n_flat = zeta_normal.reshape(B * N, 2)
        z_feat = self.fourier_zeta(z_flat)             # (B*N, fourier_zeta_dim)
        n_feat = self.fourier_normal(n_flat)           # (B*N, fourier_normal_dim)
        z_proj = self.proj_zeta(z_feat).reshape(B, N, self.d_model)
        n_proj = self.proj_normal(n_feat).reshape(B, N, self.d_model)
        zeta_tokens = z_proj + n_proj                  # (B, N, d)

        if self.use_distance:
            delta = p.unsqueeze(1) - zeta                          # (B, N, 2)
            d_norm = delta.norm(dim=-1, keepdim=True)              # (B, N, 1)
            delta_flat = (delta * 0.5).reshape(B * N, 2)
            delta_feat = self.fourier_delta(delta_flat).reshape(B, N, -1)
            d_token = self.proj_delta(torch.cat([delta_feat, d_norm], dim=-1))
            zeta_tokens = zeta_tokens + d_token

        context = torch.cat([shape_latent.tokens, p_token], dim=1)  # (B, M+1, d)

        x = zeta_tokens
        for layer in self.cross_layers:
            x = layer(x, context)

        log_K_tilde = self.head_mlp(x).squeeze(-1)     # (B, N)
        # Phase 7.1.1 soft cap (mirrors 3D HarmonicMeasureField).
        if self.log_K_max is not None:
            cap = self.log_K_max
            log_K_tilde = cap * torch.tanh(log_K_tilde / cap)
        return log_K_tilde

    # ----------------------------------------------------------------- kernel
    def kernel(
        self,
        p: Tensor,
        zeta: Tensor,
        zeta_normal: Tensor,
        shape_latent: ShapeLatent,
        surface_area_weights: Tensor | None = None,
    ) -> Tensor:
        log_K_tilde = self.log_kernel(p, zeta, zeta_normal, shape_latent)
        if self.normalize == "none":
            return torch.exp(log_K_tilde)
        if self.normalize == "soft":
            return torch.exp(log_K_tilde)
        if self.normalize == "hard":
            if surface_area_weights is None:
                raise ValueError(
                    "kernel() with normalize='hard' requires surface_area_weights; "
                    "got None."
                )
            log_w = torch.log(surface_area_weights + 1e-30)
            log_Z = torch.logsumexp(log_K_tilde + log_w, dim=-1, keepdim=True)
            log_K = log_K_tilde - log_Z
            return torch.exp(log_K)
        raise AssertionError(f"unreachable normalize: {self.normalize}")

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        p: Tensor,
        zeta: Tensor,
        zeta_normal: Tensor,
        surface_area_weights: Tensor,
        h_values: Tensor,
        shape_latent: ShapeLatent,
    ) -> Tensor:                   # (B,)
        # C3 enforcement mirrors the 3D path: no h_values ever reach
        # self.kernel. The AST test in tests/test_philosophy.py covers
        # kernel.py; a 2D equivalent test goes in tests/test_kernel_2d.py.
        K = self.kernel(
            p, zeta, zeta_normal, shape_latent,
            surface_area_weights=surface_area_weights,
        )
        return post_hoc_solve(K, surface_area_weights, h_values)

    # ----------------------------------------------------------------- sample
    def sample(
        self,
        p: Tensor,
        shape_latent: ShapeLatent,
        n_samples: int,
    ) -> Tensor:
        raise NotImplementedError(
            "2D importance sampling not needed for the Phase 7.0 loss set. "
            "API slot retained for parity with the 3D class."
        )
