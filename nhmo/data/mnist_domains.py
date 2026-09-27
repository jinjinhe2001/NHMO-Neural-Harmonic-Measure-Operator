"""MNIST-based 2D domains for Phase 7.0 §5.1 experiments.

Pipeline per MNIST digit:
  1. Read image from idx files (28×28 uint8 grayscale).
  2. Binarize at threshold 0.5 (post-normalization to [0, 1]).
  3. Extract foreground contours via skimage.measure.find_contours.
  4. Normalize pixel coords (0..27) → model-frame [-1, 1]² (no stretching;
     center preserved).
  5. Simplify each polyline by uniform-arc-length resampling to a fixed
     number of vertices per contour.
  6. Cache the resulting (outer_polyline, inner_polylines) on disk.

Domain definition for each digit:
    Ω = [-1, 1]²  \\  (interior of digit contours)

The boundary ∂Ω consists of the OUTER SQUARE and all digit contour
polylines together. The kernel head sees all these boundary points
uniformly; what makes the domain multiply connected is that digit
contours form internal holes.

MNIST idx binary format (Yann LeCun):
  magic (4 bytes), n (4 bytes), [dims...], pixel bytes.
  Little-endian vs big-endian: idx files use BIG-endian.
"""
from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


# ------------------------------------------------------- idx file reader

def _read_idx(path: Path) -> np.ndarray:
    """Read an idx-format file (.gz or uncompressed). Returns a numpy
    array of the natural dtype (uint8 for images/labels)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic not in (2049, 2051):
            raise ValueError(
                f"{path}: unsupported idx magic number {magic} "
                "(expected 2049=labels or 2051=images)."
            )
        n, = struct.unpack(">I", f.read(4))
        if magic == 2049:                           # labels
            return np.frombuffer(f.read(n), dtype=np.uint8)
        # Images
        rows, = struct.unpack(">I", f.read(4))
        cols, = struct.unpack(">I", f.read(4))
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def load_mnist_split(
    raw_dir: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images + labels for split ∈ {"train", "test"}.

    Looks for train-images-idx3-ubyte.gz / train-labels-idx1-ubyte.gz
    for "train", t10k-* for "test". Uncompressed variants also OK.
    """
    raw_dir = Path(raw_dir)
    prefix = "train" if split == "train" else "t10k"
    images_name = f"{prefix}-images-idx3-ubyte"
    labels_name = f"{prefix}-labels-idx1-ubyte"
    images_path = raw_dir / f"{images_name}.gz"
    labels_path = raw_dir / f"{labels_name}.gz"
    if not images_path.exists():
        images_path = raw_dir / images_name
    if not labels_path.exists():
        labels_path = raw_dir / labels_name
    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"MNIST {split} files not found under {raw_dir}; looked for "
            f"{images_name}(.gz) and {labels_name}(.gz)."
        )
    images = _read_idx(images_path)                 # (N, 28, 28) uint8
    labels = _read_idx(labels_path)                 # (N,) uint8
    return images, labels


# ------------------------------------------------------ contour extraction

def binarize_digit(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """uint8 (0..255) digit → bool (True = foreground)."""
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image; got {img.dtype}")
    return (img.astype(np.float32) / 255.0) > threshold


def _resample_polyline(polyline: np.ndarray, n: int) -> np.ndarray:
    """Uniform arc-length resample of a closed 2D polyline to n vertices.

    polyline: (V, 2). Returns (n, 2).
    """
    if polyline.shape[0] < 2:
        raise ValueError(f"polyline has < 2 vertices: {polyline.shape}")
    # Ensure closure (last vertex == first) for arc-length cumulation
    closed = np.vstack([polyline, polyline[:1]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=-1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    if total <= 0.0:
        # Degenerate polyline — return the first vertex n times
        return np.broadcast_to(polyline[0], (n, 2)).copy()
    target_s = np.linspace(0.0, total, n + 1)[:-1]    # n points, skip endpoint dup
    out = np.empty((n, 2), dtype=polyline.dtype)
    for i, s in enumerate(target_s):
        # Find segment containing arc-length s
        idx = int(np.searchsorted(cumlen, s, side="right") - 1)
        idx = max(0, min(idx, closed.shape[0] - 2))
        seg_start = closed[idx]
        seg_end = closed[idx + 1]
        seg_len = seg_lens[idx]
        if seg_len <= 0:
            out[i] = seg_start
        else:
            t = (s - cumlen[idx]) / seg_len
            out[i] = seg_start + t * (seg_end - seg_start)
    return out


def extract_digit_contours(
    binary_img: np.ndarray,
    n_vertices_per_contour: int = 50,
) -> List[np.ndarray]:
    """Return a list of simplified polylines enclosing foreground blobs.

    Each polyline is (n_vertices_per_contour, 2) in PIXEL coords
    (row, col) order matching skimage.measure.find_contours.
    Caller is responsible for normalizing to model frame.
    """
    try:
        from skimage.measure import find_contours
    except ImportError as exc:
        raise ImportError(
            "nhmo.data.mnist_domains.extract_digit_contours requires "
            "scikit-image (skimage). Install via `pip install scikit-image`."
        ) from exc

    contours_raw = find_contours(binary_img.astype(np.float32), level=0.5)
    # skimage returns points in (row, col) order. Keep as-is for now; caller
    # converts to (x, y) model frame.
    out = []
    for c in contours_raw:
        if c.shape[0] < 3:
            continue
        simplified = _resample_polyline(c, n_vertices_per_contour)
        out.append(simplified.astype(np.float32))
    return out


def pixel_to_model_frame(
    polyline_pixel: np.ndarray,
    image_shape: tuple[int, int] = (28, 28),
    margin: float = 0.2,
) -> np.ndarray:
    """Map a (V, 2) polyline in pixel (row, col) to model frame [-1, 1]².

    Ordering:
      - Input is (row, col); row = y-from-top; col = x.
      - Model frame: +x to the right, +y UP (image row increases downward →
        flip y).
      - Fit the digit into [-(1 - margin), (1 - margin)]² with a small
        margin so the digit doesn't touch the outer boundary.

    Args:
        polyline_pixel: (V, 2) with columns [row, col].
        image_shape: (H, W).
        margin: fraction of [-1, 1]² reserved as buffer outside the digit.
                (default 0.2 → digit occupies [-0.8, 0.8]².)
    Returns:
        (V, 2) polyline in (x, y) model-frame coords.
    """
    H, W = image_shape
    row = polyline_pixel[:, 0]                # y-from-top
    col = polyline_pixel[:, 1]                # x-from-left
    # Map [0, W-1] → [-1+margin, 1-margin]
    scale = (1.0 - margin)
    x = (col / (W - 1)) * 2.0 * scale - scale
    y_flipped = (row / (H - 1)) * 2.0 * scale - scale
    y = -y_flipped                            # flip to +y up
    return np.stack([x, y], axis=-1).astype(np.float32)


# ------------------------------------------------------ domain record

@dataclass
class MnistDigitDomain:
    """One MNIST-derived 2D domain: the unit square with digit holes."""
    digit_index: int
    label: int
    digit_contours: List[np.ndarray]       # each (V, 2) in model frame
    outer_square: np.ndarray               # (V, 2) — unit square [-1,1]² polyline

    @property
    def all_contours(self) -> List[np.ndarray]:
        """All boundary contours: outer square plus digit interior contours."""
        return [self.outer_square] + list(self.digit_contours)


def _unit_square_polyline(n_per_edge: int = 20) -> np.ndarray:
    """CCW unit square [-1, 1]² with n_per_edge samples along each side."""
    t = np.linspace(-1.0, 1.0, n_per_edge + 1, dtype=np.float32)[:-1]
    bottom = np.stack([t, -np.ones_like(t)], axis=-1)
    right  = np.stack([np.ones_like(t), t], axis=-1)
    top    = np.stack([-t, np.ones_like(t)], axis=-1)
    left   = np.stack([-np.ones_like(t), -t], axis=-1)
    return np.concatenate([bottom, right, top, left], axis=0)


def mnist_digit_to_domain(
    img: np.ndarray,
    digit_index: int,
    label: int,
    n_vertices_per_contour: int = 50,
    image_shape: tuple[int, int] = (28, 28),
    margin: float = 0.2,
    square_samples_per_edge: int = 20,
) -> MnistDigitDomain:
    """One-stop pipeline: uint8 28×28 → MnistDigitDomain."""
    bin_img = binarize_digit(img)
    contours_pixel = extract_digit_contours(bin_img, n_vertices_per_contour)
    contours_model = [
        pixel_to_model_frame(c, image_shape=image_shape, margin=margin)
        for c in contours_pixel
    ]
    outer = _unit_square_polyline(square_samples_per_edge)
    return MnistDigitDomain(
        digit_index=digit_index,
        label=int(label),
        digit_contours=contours_model,
        outer_square=outer,
    )
