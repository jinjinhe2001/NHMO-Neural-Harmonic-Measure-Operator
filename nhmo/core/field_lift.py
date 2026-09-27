"""Field-to-field 2D Poisson lift — Transolver-style spatial expressiveness.

Replaces the per-point `PoissonLiftModule` (in `source_lift.py`) with a
field-to-field operator: input is the full 2D field of (mask, h, f, u_h),
output is v_phi as a full field. Preserves NHMO decomposition u = u_h + v_phi
but escapes the per-point bottleneck (no spatial coupling, gauge constraint
forcing near-boundary signal to zero, subsampled supervision).

Architecture: thin CNN U-Net over the input stack. ~300k params, similar to
Transolver's 273k.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FieldLift2D(nn.Module):
    """U-Net field-to-field lift.

    Input:
        x: (B, C, H, W) — channels are (mask, h, f, u_h, optionally sdf)
    Output:
        v_phi: (B, H, W) — lift correction at every pixel
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        depth: int = 4,
        gauge_mode: str = 'mask',  # 'mask' = multiply by mask; 'none' = unconstrained
    ) -> None:
        super().__init__()
        self.gauge_mode = gauge_mode
        self.in_channels = in_channels

        # Encoder
        chs = [base_channels * (2 ** i) for i in range(depth)]
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, chs[0], 3, padding=1),
            nn.GroupNorm(8, chs[0]),
            nn.GELU(),
        )
        self.down_blocks = nn.ModuleList()
        for i in range(depth - 1):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1, stride=2),
                nn.GroupNorm(8, chs[i + 1]),
                nn.GELU(),
                nn.Conv2d(chs[i + 1], chs[i + 1], 3, padding=1),
                nn.GroupNorm(8, chs[i + 1]),
                nn.GELU(),
            ))

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(chs[-1], chs[-1], 3, padding=1),
            nn.GroupNorm(8, chs[-1]),
            nn.GELU(),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(chs[i], chs[i - 1], 4, stride=2, padding=1),
                nn.GroupNorm(8, chs[i - 1]),
                nn.GELU(),
                nn.Conv2d(chs[i - 1] * 2, chs[i - 1], 3, padding=1),  # cat skip
                nn.GroupNorm(8, chs[i - 1]),
                nn.GELU(),
            ))

        self.out_conv = nn.Conv2d(chs[0], 1, 3, padding=1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: (B, C, H, W) input field stack (mask, h, f, u_h, ...)
        mask: (B, H, W) interior mask
        returns v_phi: (B, H, W)
        """
        h = self.in_conv(x)
        skips = [h]
        for blk in self.down_blocks:
            h = blk(h)
            skips.append(h)
        h = self.mid(h)
        # Decoder with skip connections
        for i, blk in enumerate(self.up_blocks):
            # Upsample
            up_layer = blk[0:3]  # ConvTranspose + GN + GELU
            h = up_layer(h)
            # Skip from corresponding encoder level
            skip = skips[len(self.up_blocks) - i - 1]
            # Resize skip if shape mismatch (rounding)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            # Conv + GN + GELU
            conv_layer = blk[3:]
            h = conv_layer(h)
        v = self.out_conv(h).squeeze(1)              # (B, H, W)
        if self.gauge_mode == 'mask':
            v = v * mask
        return v
