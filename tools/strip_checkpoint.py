"""Strip a training checkpoint down to what inference needs.

Keeps the model weights and the configuration needed to rebuild the model,
drops optimizer and RNG state, replaces machine-specific absolute paths in the
stored config by relative names, and verifies that every kept tensor is
bitwise identical to the source checkpoint. The output contains only tensors
and plain Python types, so it also loads with `torch.load(..., weights_only=True)`.

    python tools/strip_checkpoint.py SRC DST [--kind kernel3d|lift3d|kernel2d|lift2d|rhead2d]
                                             [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

KEEP = {
    "kernel3d": ["step", "model_state_dict", "cfg", "nhmo_version"],
    "kernel2d": ["step", "model_state_dict", "cfg"],
    "lift3d": ["step", "lift_state_dict", "lift_cfg", "category", "use_local_baseline", "kernel_ckpt"],
    "lift2d": ["step", "lift_state_dict", "lift_cfg", "y_std", "kernel_ckpt"],
    "rhead2d": ["step", "r_state_dict", "r_cfg", "y_std", "frozen_mf_ckpt"],
}
STATE_KEYS = ("model_state_dict", "lift_state_dict", "r_state_dict")


def _sanitize(obj):
    """Replace absolute machine paths inside the stored config by relative names."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("/") and ("/lp-dev/" in obj or "/home/" in obj):
        parts = Path(obj).parts
        if "runs" in parts:
            return "/".join(parts[parts.index("runs"):])
        return Path(obj).name
    return obj


def strip(src: str, dst: str, kind: str, overrides: dict | None = None) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = {}
    for k in KEEP[kind]:
        if k not in ck:
            continue
        v = ck[k]
        if k in STATE_KEYS:
            v = {n: t.detach().clone().contiguous() for n, t in v.items()}
        elif k in ("cfg", "lift_cfg", "r_cfg"):
            v = _sanitize(v)
        elif isinstance(v, str):
            v = _sanitize(v)
        out[k] = v
    for k, v in (overrides or {}).items():
        out[k] = v
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)

    # verify: reload with weights_only=True and compare every tensor bitwise
    back = torch.load(dst, map_location="cpu", weights_only=True)
    n = 0
    for k in STATE_KEYS:
        if k in ck:
            assert set(back[k]) == set(ck[k]), f"{k}: key mismatch"
            for name, t in ck[k].items():
                assert torch.equal(back[k][name], t), f"{k}.{name} differs"
                n += 1
    return {"src": src, "dst": dst, "kind": kind, "tensors_checked": n,
            "src_bytes": os.path.getsize(src), "dst_bytes": os.path.getsize(dst),
            "dropped_keys": sorted(set(ck) - set(out))}


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("src")
    pa.add_argument("dst")
    pa.add_argument("--kind", required=True, choices=sorted(KEEP))
    pa.add_argument("--set", nargs="*", default=[], help="key=value (value parsed as JSON if possible)")
    a = pa.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            ov[k] = json.loads(v)
        except json.JSONDecodeError:
            ov[k] = v
    print(json.dumps(strip(a.src, a.dst, a.kind, ov), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
