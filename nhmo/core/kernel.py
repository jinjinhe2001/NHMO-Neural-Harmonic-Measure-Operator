"""Harmonic measure field — the core v2 module.

Learns K_θ(p, ζ; Ω), the DENSITY of the harmonic measure ω_p(dζ) with
respect to the surface measure σ on ∂Ω:

    ω(p, dζ) = K_θ(p, ζ) · dσ(ζ)

For the Laplacian, this density is exactly the Poisson kernel — hence
variable names `K`, `log_K`, `log_K_tilde` throughout refer specifically
to the Poisson-kernel density. The class is named `HarmonicMeasureField`
to emphasize that the learned object is the measure ω_p itself (the
fundamental physical quantity that NHMO stands for), not just the
Laplacian Poisson-kernel formula. Future Robin/Neumann specializations
will subclass `EllipticMeasureField` (defined below) with their own
operator-specific density.

Trained and used WITHOUT ever seeing the boundary condition h
(commitment C3 in PHILOSOPHY.md). Solution for any h is computed by
post-hoc integration:

    u(p) = Σ_j  w_j · K_θ(p, ζ_j) · h(ζ_j)       ζ_j ~ σ(∂Ω)

Architectural invariants (PHILOSOPHY.md §0, v2 plan §4, §12):
  - C2 / A2: no tensor in the head has a parameter dimension sized to
    N_surface. The kernel head's output shape (B, N) is dynamic in N.
  - C3 / A1: log_kernel and kernel NEVER take h as input; solve takes
    h_values ONLY at its quadrature step, via post_hoc_solve. An AST
    static test in tests/test_philosophy.py enforces this.
  - A4: no softmax over ζ anywhere. Normalization (for `normalize="hard"`)
    is logsumexp-based, divisive by a scalar Z — not a softmax.
  - A10: the head has NO ζ↔ζ attention. The only attention is
    ζ-as-query ← (shape_latent ∪ p_token)-as-keys/values — cross, never
    self. A unit test in tests/test_kernel.py (test_symmetry_in_zeta)
    enforces permutation invariance in the ζ batch dimension.
  - A9: the encoder is a separate module (Phase 3). HarmonicMeasureField
    accepts a precomputed ShapeLatent so encoding can be cached across
    many (p, ζ) queries per shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nhmo.core.fourier import FourierFeatures
from nhmo.core.integration import post_hoc_solve


REQUIRED_KERNEL_HEAD_CFG_KEYS: tuple[str, ...] = (
    "n_cross_layers",
    "d_model",
    "n_heads",
    "mlp_ratio",
    "dropout",
    "pre_norm",
    "fourier_bands_p",
    "fourier_bands_zeta",
    "fourier_bands_normal",
)


@dataclass
class ShapeLatent:
    """Opaque container for encoded shape context.

    Phase 3's encoder produces this. Tests construct it directly from a
    random tensor so the kernel head can be exercised without the encoder.
    """
    tokens: Tensor          # (B, M, d_model)


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block: queries from x, keys/values from context.

    NO ζ↔ζ attention (A10 guarantee). x (the ζ tokens) is only the query;
    keys and values come exclusively from the context (shape_latent ∪
    p_token). Each output token x_j depends on (x_j, context), never on
    other x_k.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.pre_norm = pre_norm
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # x: (B, N, d)        — queries (ζ tokens)
        # context: (B, M+1, d) — keys+values (shape_latent + p_token)
        if self.pre_norm:
            q = self.norm_q(x)
            kv = self.norm_kv(context)
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        else:
            attn_out, _ = self.attn(x, context, context, need_weights=False)
            x = self.norm_q(x + attn_out)
            x = self.norm2(x + self.mlp(x))
        return x


class EllipticMeasureField(nn.Module):
    """Abstract base for elliptic-operator measure fields.

    Phase 7+ Robin/Neumann BVPs will subclass this with their own measure
    densities. Phase 7.0 ships only the Laplacian specialization —
    `HarmonicMeasureField` below. Leaving this as an empty marker class
    now so that the class hierarchy matches the project's intended scope
    ("Neural Harmonic Measure Operator" is the Laplacian instance of a
    broader "Neural Elliptic Measure Operator" family).

    Subclasses are expected to expose:
      - `log_kernel(p, zeta, zeta_normal, shape_latent) -> (B, N)`
      - `kernel(p, zeta, zeta_normal, shape_latent, surface_area_weights)`
      - `normalize: str in {"soft", "hard", "none"}`
    """
    pass


class HarmonicMeasureField(EllipticMeasureField):
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
        # Phase 7.1.1: soft cap on raw log_K_tilde to stabilize soft
        # normalization. None = no cap (backward compat). MCB-recommended
        # value is 15 — log(e^15) ≈ 3.3e6, above the 10^3-10^6 Poisson
        # kernel dynamic range quoted in PHILOSOPHY.md §0 so the cap never
        # binds on correct physics, but prevents numerical log_Z explosion
        # during early training.
        _cap = kernel_head_cfg.get("log_K_max", None)
        self.log_K_max: float | None = None if _cap is None else float(_cap)

        # Build the encoder submodule iff encoder_cfg is supplied.
        # A9: encoder is a distinct nn.Module, reused across many (p, ζ)
        # queries per shape. Phase-2 tests pass encoder_cfg={} and build
        # ShapeLatent manually; the encoder is never constructed in that path.
        if encoder_cfg:
            # Deferred import to avoid the circular kernel ↔ encoder edge at
            # module load time. encoder.py imports ShapeLatent from this file.
            from nhmo.core.encoder import TransolverEncoder
            self._encoder = TransolverEncoder(encoder_cfg)
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

        self.d_model = d_model

        # Three separate Fourier instances per v2 plan §4.3. No weight sharing.
        self.fourier_p = FourierFeatures(num_bands=bands_p, input_dim=3)
        self.fourier_zeta = FourierFeatures(num_bands=bands_zeta, input_dim=3)
        self.fourier_normal = FourierFeatures(num_bands=bands_normal, input_dim=3)

        self.proj_p = nn.Linear(self.fourier_p.output_dim, d_model)
        self.proj_zeta = nn.Linear(self.fourier_zeta.output_dim, d_model)
        self.proj_normal = nn.Linear(self.fourier_normal.output_dim, d_model)

        # Cross-attention stack: ζ queries, (shape_latent ∪ p_token) keys+values.
        # A10 guarantee: no self-attention over ζ anywhere in this module.
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, mlp_ratio, dropout, pre_norm)
            for _ in range(n_cross_layers)
        ])

        # Head: LN → Linear → GELU → Linear → scalar per ζ token.
        self.head_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    # ----------------------------------------------------------------- encode
    def encode(self, shape_ctx: dict) -> ShapeLatent:
        """Delegate to the configured encoder. encoder_cfg must be non-empty
        at construction time for encode() to work.
        """
        if self._encoder is None:
            raise ValueError(
                "HarmonicMeasureField was constructed with encoder_cfg={}; "
                "encode(shape_ctx) is unavailable. Either construct with a "
                "non-empty encoder_cfg, or build ShapeLatent manually and "
                "call log_kernel / kernel / solve directly."
            )
        return self._encoder(shape_ctx)

    # ------------------------------------------------------------- log_kernel
    def log_kernel(
        self,
        p: Tensor,                 # (B, 3)
        zeta: Tensor,              # (B, N, 3)
        zeta_normal: Tensor,       # (B, N, 3)
        shape_latent: ShapeLatent,
    ) -> Tensor:                   # (B, N) log K̃ (raw, unnormalized)
        B, N, _ = zeta.shape

        # Query token: Fourier(p) → Linear → (B, 1, d)
        p_feat = self.fourier_p(p)                     # (B, fourier_p_dim)
        p_token = self.proj_p(p_feat).unsqueeze(1)     # (B, 1, d)

        # ζ tokens: Fourier(ζ) + Fourier(normal), each projected, then ADDED
        # per v2 plan §4.2 ("Fourier(ζ_normal) → linear ADDED to it").
        z_flat = zeta.reshape(B * N, 3)
        n_flat = zeta_normal.reshape(B * N, 3)
        z_feat = self.fourier_zeta(z_flat)             # (B*N, fourier_zeta_dim)
        n_feat = self.fourier_normal(n_flat)           # (B*N, fourier_normal_dim)
        z_proj = self.proj_zeta(z_feat).reshape(B, N, self.d_model)
        n_proj = self.proj_normal(n_feat).reshape(B, N, self.d_model)
        zeta_tokens = z_proj + n_proj                  # (B, N, d)

        # Context = shape_latent ∪ p_token. Keys and values for cross-attn.
        context = torch.cat([shape_latent.tokens, p_token], dim=1)  # (B, M+1, d)

        # Cross-attend ζ ← context. No ζ ↔ ζ attention.
        x = zeta_tokens
        for layer in self.cross_layers:
            x = layer(x, context)

        # MLP head: (B, N, d) → (B, N, 1) → squeeze → (B, N)
        log_K_tilde = self.head_mlp(x).squeeze(-1)
        # Phase 7.1.1 soft cap: smooth tanh saturation at ±log_K_max.
        # For |log_K_tilde| ≲ 0.3·log_K_max this is ~identity; above that
        # it saturates asymptotically, keeping logsumexp-derived log_Z
        # bounded regardless of MLP output range.
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
    ) -> Tensor:                   # (B, N) K (normalized per self.normalize)
        log_K_tilde = self.log_kernel(p, zeta, zeta_normal, shape_latent)

        if self.normalize == "none":
            return torch.exp(log_K_tilde)
        if self.normalize == "soft":
            # Normalization penalty is applied as L_Z in Phase 4; the raw
            # exp(log K̃) is returned here.
            return torch.exp(log_K_tilde)
        if self.normalize == "hard":
            if surface_area_weights is None:
                raise ValueError(
                    "kernel() with normalize='hard' requires surface_area_weights; "
                    "got None."
                )
            # log Z = log Σ_j w_j · exp(log K̃_j); computed via logsumexp
            # to stay stable when log K̃ blows up near ∂Ω (§4.5).
            log_w = torch.log(surface_area_weights + 1e-30)
            log_Z = torch.logsumexp(log_K_tilde + log_w, dim=-1, keepdim=True)
            log_K = log_K_tilde - log_Z                # (B, N)
            return torch.exp(log_K)                    # one exp, at the end
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
        # C3 enforcement boundary (A1): the following call must NEVER pass
        # h_values into self.kernel. The AST test in tests/test_philosophy.py
        # verifies this.
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
            "Phase 4 (importance sampling for L_MV). v2 plan §6.2's 'elegant' "
            "martingale formulation may make this unnecessary; the API slot "
            "stays reserved until Phase 4 decides."
        )
