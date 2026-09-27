"""Composable loss registry with configurable weights + ablation toggles (§6.5).

Contract:
  - Four loss components: nll, mv, bl, z.
  - Each weighted by λ_nll / λ_mv / λ_bl / λ_z.
  - Setting a λ to None OR omitting it → disables the component (skip its
    forward; emit 0.0 as its value in the output dict for logging
    consistency).
  - Returns a dict with keys {"nll", "mv", "bl", "z", "total"} every call,
    even when components are disabled.

Config validation:
  - Loss WEIGHTS (lambda_*): loose (None or missing = disable).
  - Component-specific SUB-SECTION hyperparameters: strict. If λ_mv != None,
    cfg["mean_value"] must contain S, r_frac_min, r_frac_max,
    r_floor_rel_bbox, sample_log_uniform. If λ_bl != None,
    cfg["boundary_limit"] must contain epsilon, gamma, delta. Missing any
    key raises KeyError naming the missing key.

L_Z gating by model.normalize:
  - "hard" → L_Z ≡ 0 (kernel is exactly normalized; see core/kernel.py).
  - "soft" → L_Z = l_z_soft(log K̃, w).
  - "none" → L_Z ≡ 0 (debug mode).

Expected batch keys (constructed by Phase 5's trainer):
  shape_latent            : ShapeLatent       (precomputed via encoder.encode)
  p_nll                   : (B, 3)            query points for NLL
  zeta_hits               : (B, K_max, 3)     WoS hit coords
  hit_normals             : (B, K_max, 3)
  hit_mask                : (B, K_max) bool
  p_mv                    : (B, 3)            interior points for MV
  sdf_at_p_mv             : (B,)              distance to ∂Ω
  zeta_bl_0               : (B, 3)            boundary point for BL target
  normal_bl_0             : (B, 3)
  zeta_surface            : (B, N, 3)         shared surface quadrature
  normal_surface          : (B, N, 3)
  surface_area_weights    : (B, N)
"""
from __future__ import annotations

import torch
from torch import Tensor

from nhmo.losses.analytic import l_analytic
from nhmo.losses.boundary_limit import l_bl
from nhmo.losses.mean_value import l_mv
from nhmo.losses.nll import l_nll
from nhmo.losses.normalization import l_z_soft


REQUIRED_MV_SUBCFG_KEYS: tuple[str, ...] = (
    "S", "r_frac_min", "r_frac_max", "r_floor_rel_bbox", "sample_log_uniform",
)
REQUIRED_BL_SUBCFG_KEYS: tuple[str, ...] = (
    "epsilon", "gamma", "delta",
)


class LossRegistry:
    def __init__(self, cfg: dict) -> None:
        self.lambda_nll = cfg.get("lambda_nll", None)
        self.lambda_mv = cfg.get("lambda_mv", None)
        self.lambda_bl = cfg.get("lambda_bl", None)
        self.lambda_z = cfg.get("lambda_z", None)
        self.lambda_analytic = cfg.get("lambda_analytic", None)  # Phase 6 §8.2
        # L_Z Huber delta (Phase 7.1.1 stability fix). None = quadratic
        # (backward compat). Recommended for MCB: 1.0.
        self.z_huber_delta = cfg.get("z_huber_delta", None)

        # Strict validation for sub-section hyperparameters — only when the
        # corresponding weight is enabled. Phase 4's ablation configs can
        # disable a component entirely without supplying its sub-section.
        if self.lambda_mv is not None:
            mv_cfg = cfg.get("mean_value")
            if mv_cfg is None:
                raise KeyError(
                    "losses.mean_value sub-cfg is required when lambda_mv is set"
                )
            for k in REQUIRED_MV_SUBCFG_KEYS:
                if k not in mv_cfg:
                    raise KeyError(
                        f"losses.mean_value missing required key: {k!r}. "
                        f"All required: {REQUIRED_MV_SUBCFG_KEYS}."
                    )
            self.mv_cfg = dict(mv_cfg)
        else:
            self.mv_cfg = None

        if self.lambda_bl is not None:
            bl_cfg = cfg.get("boundary_limit")
            if bl_cfg is None:
                raise KeyError(
                    "losses.boundary_limit sub-cfg is required when lambda_bl is set"
                )
            for k in REQUIRED_BL_SUBCFG_KEYS:
                if k not in bl_cfg:
                    raise KeyError(
                        f"losses.boundary_limit missing required key: {k!r}. "
                        f"All required: {REQUIRED_BL_SUBCFG_KEYS}."
                    )
            self.bl_cfg = dict(bl_cfg)
        else:
            self.bl_cfg = None

        self.cfg = dict(cfg)

    def __call__(self, batch: dict, model) -> dict[str, Tensor]:
        # Reference tensor for device / dtype / zero construction.
        any_param = next(model.parameters())
        zero = 0.0 * any_param.sum()  # graph-connected zero
        out: dict[str, Tensor] = {"nll": zero, "mv": zero, "bl": zero, "z": zero, "analytic": zero}

        # --- Shared forwards reused across components ---
        # NLL, L_Z (soft), and L_analytic all need log K̃ over the surface
        # samples at p_nll. Compute once and share.
        shape_latent = batch["shape_latent"]
        zeta_surf = batch["zeta_surface"]
        normal_surf = batch["normal_surface"]
        w_surf = batch["surface_area_weights"]

        log_K_tilde_surf_nll: Tensor | None = None  # at p_nll
        log_Z_nll: Tensor | None = None
        need_surface_forward = (
            self.lambda_nll is not None or self.lambda_analytic is not None
        )
        if need_surface_forward:
            log_K_tilde_surf_nll = model.log_kernel(
                batch["p_nll"], zeta_surf, normal_surf, shape_latent,
            )                                                 # (B, N)
            log_w = torch.log(w_surf + 1e-30)
            log_Z_nll = torch.logsumexp(
                log_K_tilde_surf_nll + log_w, dim=-1, keepdim=True,
            )

        if self.lambda_nll is not None:
            log_K_tilde_hits = model.log_kernel(
                batch["p_nll"], batch["zeta_hits"], batch["hit_normals"], shape_latent,
            )                                                 # (B, K_max)
            log_K_hits = log_K_tilde_hits - log_Z_nll         # (B, K_max)
            out["nll"] = l_nll(log_K_hits, batch["hit_mask"])

        # --- L_analytic (Phase 6 §8.2): log-space MSE at surface samples ---
        if self.lambda_analytic is not None:
            if "log_K_true_at_surface" not in batch:
                raise KeyError(
                    "lambda_analytic is set but batch['log_K_true_at_surface'] "
                    "is missing. Use AnalyticBallDataset (or similar) for "
                    "pretraining."
                )
            # Normalize log K_θ at surface samples
            log_K_surf = log_K_tilde_surf_nll - log_Z_nll     # (B, N)
            out["analytic"] = l_analytic(log_K_surf, batch["log_K_true_at_surface"])

        # --- L_Z (soft only; hard and none are ≡ 0) ---
        if self.lambda_z is not None and getattr(model, "normalize", None) == "soft":
            # Reuse log_K_tilde_surf_nll if available; else compute.
            log_K_tilde_surf = log_K_tilde_surf_nll
            if log_K_tilde_surf is None:
                # No p_nll forward was done; use p_mv as a proxy for the
                # normalization check. This is approximate but consistent
                # with §6.4's "L_Z on held-out p" semantics.
                p_for_z = batch.get("p_mv", batch.get("p_nll"))
                log_K_tilde_surf = model.log_kernel(
                    p_for_z, zeta_surf, normal_surf, shape_latent,
                )
            out["z"] = l_z_soft(log_K_tilde_surf, w_surf, huber_delta=self.z_huber_delta)

        # --- L_MV ---
        if self.lambda_mv is not None:
            out["mv"] = l_mv(
                model,
                batch["p_mv"],
                shape_latent,
                zeta_surf,
                normal_surf,
                batch["sdf_at_p_mv"],
                S=int(self.mv_cfg["S"]),
                r_frac_min=float(self.mv_cfg["r_frac_min"]),
                r_frac_max=float(self.mv_cfg["r_frac_max"]),
                r_floor_rel_bbox=float(self.mv_cfg["r_floor_rel_bbox"]),
                sample_log_uniform=bool(self.mv_cfg["sample_log_uniform"]),
                # Phase 7.1.0 fix (H5): always pass weights. l_mv now
                # normalizes K internally regardless of model.normalize.
                surface_area_weights=w_surf,
            )

        # --- L_BL ---
        if self.lambda_bl is not None:
            out["bl"] = l_bl(
                model,
                shape_latent,
                batch["zeta_bl_0"],
                batch["normal_bl_0"],
                zeta_surf,
                normal_surf,
                w_surf,
                epsilon=float(self.bl_cfg["epsilon"]),
                gamma=float(self.bl_cfg["gamma"]),
                delta=float(self.bl_cfg["delta"]),
                sdf_fn=batch.get("sdf_fn"),
            )

        # Total = Σ λ · component.
        total = zero
        if self.lambda_nll is not None:
            total = total + float(self.lambda_nll) * out["nll"]
        if self.lambda_mv is not None:
            total = total + float(self.lambda_mv) * out["mv"]
        if self.lambda_bl is not None:
            total = total + float(self.lambda_bl) * out["bl"]
        if self.lambda_z is not None:
            total = total + float(self.lambda_z) * out["z"]
        if self.lambda_analytic is not None:
            total = total + float(self.lambda_analytic) * out["analytic"]
        out["total"] = total

        return out
