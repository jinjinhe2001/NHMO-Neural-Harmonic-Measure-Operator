"""Generate the 2D MNIST-domain PDE benchmark with parametric boundary families.

Domain: Omega = [-1, 1]^2 minus the digit interior, from an MNIST training image
upsampled to a 256x256 mask. Per shape, --k-lap Laplace problems (random boundary
family and coefficients) and --k-poi Poisson problems (random source family,
boundary family and coefficients). Boundary families: poly3, trig1, trig2,
exp_mix; sources: f0_sin_cos, f1_poly, f2_gaussian, f3_asym. Coefficients are
U[-1, 1] for train/test and U[1, 2] for test_ood.

Reference solutions: 5-point finite differences on the 256x256 pixel grid
(sparse direct solve); models resample to 128x128 at train/eval time.

Output (per shape directory <k>_idx<MNIST index>): mask.npy, boundary.npy and one
npz per problem with keys h, u_true (and f for Poisson), bc_family, bc_coeffs,
source_family; e.g. laplace_poly3_a-0.32_b+0.71_c+0.05.npz.

Reproducing the benchmark of the paper. The original generator seeded each
split with `seed + hash(split_name) % 10000`, and Python randomizes str hashes
per process, so the offsets were not recorded. We recovered them by replaying
the coefficient draws against the problem names on disk: train 4409,
test 8984, test_ood 8077 (with --seed 0). These are the defaults of
--split-seeds, so

    python tools/gen_mnist_pde_data_param_bc.py --mnist-dir data/mnist/raw \\
        --out data/mnist_pde_2d_paramBC --seed 0
    python tools/filter_param_bc.py data/mnist_pde_2d_paramBC data/mnist_pde_2d_paramBC_lf

regenerates the benchmark. The filter step keeps the poly3 / exp_mix families
used for the headline numbers (991 / 50 / 50 shapes, 8005 / 408 / 397 problems).
"""
from __future__ import annotations

import argparse
import gzip
import os
import struct
import time
from pathlib import Path

import numpy as np


# ============================================================================
# MNIST loading
# ============================================================================
def load_mnist_images(path: Path, n: int) -> np.ndarray:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rb") as fh:
        magic, num = struct.unpack(">II", fh.read(8))
        rows, cols = struct.unpack(">II", fh.read(8))
        buf = fh.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


# ============================================================================
# Domain mask (same as gen_mnist_pde_data.py)
# ============================================================================
def make_domain_mask(digit_28: np.ndarray, R: int, threshold: float = 0.5) -> np.ndarray:
    from scipy.ndimage import zoom
    img = digit_28.astype(np.float32) / 255.0
    img = zoom(img, R / 28.0, order=1)
    img = img[:R, :R]
    if img.shape != (R, R):
        out = np.zeros((R, R), dtype=np.float32)
        out[: img.shape[0], : img.shape[1]] = img
        img = out
    digit_int = img > threshold
    domain = ~digit_int
    domain[0, :] = False
    domain[-1, :] = False
    domain[:, 0] = False
    domain[:, -1] = False
    return domain


def boundary_mask(domain_mask: np.ndarray) -> np.ndarray:
    R = domain_mask.shape[0]
    bd = np.zeros_like(domain_mask)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nei = np.roll(domain_mask, shift=(di, dj), axis=(0, 1))
        nei[0 if di > 0 else -1 if di < 0 else slice(None), :] = False
        nei[:, 0 if dj > 0 else -1 if dj < 0 else slice(None)] = False
        bd |= (~domain_mask & nei)
    return bd


# ============================================================================
# Parametric BC families
# ============================================================================
def _basis_field(R: int, kind: str):
    """Return basis field arrays for the given BC family kind.

    Each family is a linear combination h = sum_i a_i * basis_i(x, y).
    Returns (basis_arrays, coeff_names) where basis_arrays is a list of
    (R, R) ndarray and coeff_names is the list of coefficient names.
    """
    xs = np.linspace(-1.0, 1.0, R, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, R, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    if kind == "poly3":
        # u = a (x³ - 3xy²) + b (y³ - 3x²y) + c x²
        # First two are harmonic; c x² breaks harmonicity for Laplace use,
        # but we use this family for BCs only — interior solution is found
        # numerically.
        return [
            X ** 3 - 3 * X * Y ** 2,
            Y ** 3 - 3 * X ** 2 * Y,
            X ** 2,
        ], ["a", "b", "c"]

    if kind == "trig1":
        # h = a sin(πx) cos(πy) + b sin(πy) cos(πx)
        return [
            np.sin(np.pi * X) * np.cos(np.pi * Y),
            np.sin(np.pi * Y) * np.cos(np.pi * X),
        ], ["a", "b"]

    if kind == "trig2":
        # h = a cos(2πx) + b sin(2πy) + c xy
        return [
            np.cos(2 * np.pi * X),
            np.sin(2 * np.pi * Y),
            X * Y,
        ], ["a", "b", "c"]

    if kind == "exp_mix":
        # h = a exp(0.5x) cos(0.5y) + b xy² + c y
        return [
            np.exp(0.5 * X) * np.cos(0.5 * Y),
            X * Y ** 2,
            Y,
        ], ["a", "b", "c"]

    raise ValueError(f"unknown BC family: {kind}")


PARAM_BC_FAMILIES = ["poly3", "trig1", "trig2", "exp_mix"]


def sample_bc_field(R: int, kind: str, rng: np.random.RandomState,
                    coeff_low: float = -1.0, coeff_high: float = 1.0):
    """Sample a parametric BC field. Returns (h_field, coeffs_dict)."""
    basis, names = _basis_field(R, kind)
    coeffs = rng.uniform(coeff_low, coeff_high, size=len(basis)).astype(np.float32)
    h = sum(c * b for c, b in zip(coeffs, basis)).astype(np.float32)
    return h, dict(zip(names, coeffs.tolist()))


# ============================================================================
# Source families (same as gen_mnist_pde_data.py)
# ============================================================================
def poisson_sources(R: int):
    xs = np.linspace(-1.0, 1.0, R)
    ys = np.linspace(-1.0, 1.0, R)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return [
        ("f0_sin_cos",  (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)).astype(np.float32)),
        ("f1_poly",     ((1 + 2 * X * X) * (1 + 2 * Y * Y)).astype(np.float32)),
        ("f2_gaussian", np.exp(-3.0 * (X * X + Y * Y)).astype(np.float32)),
        ("f3_asym",     ((X * X + Y * Y) * np.sign(X)).astype(np.float32)),
    ]


# ============================================================================
# Sparse Poisson solver (same as gen_mnist_pde_data.py)
# ============================================================================
def solve_poisson_5pt(domain_mask: np.ndarray, f_full: np.ndarray, h_full: np.ndarray) -> np.ndarray:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    R = domain_mask.shape[0]
    interior_idx = np.argwhere(domain_mask)
    n_int = len(interior_idx)
    if n_int == 0:
        return np.zeros_like(f_full)
    idx_lookup = -np.ones((R, R), dtype=np.int64)
    for k, (i, j) in enumerate(interior_idx):
        idx_lookup[i, j] = k

    h2 = (2.0 / R) ** 2
    A = lil_matrix((n_int, n_int))
    rhs = np.zeros(n_int, dtype=np.float64)

    for k, (i, j) in enumerate(interior_idx):
        A[k, k] = -4.0 / h2
        rhs[k] = float(f_full[i, j])
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < R and 0 <= nj < R and domain_mask[ni, nj]:
                A[k, idx_lookup[ni, nj]] += 1.0 / h2
            else:
                rhs[k] -= float(h_full[ni, nj]) / h2 if 0 <= ni < R and 0 <= nj < R else 0.0

    u_int = spsolve(csr_matrix(A), rhs)
    u_full = h_full.astype(np.float64).copy()
    for k, (i, j) in enumerate(interior_idx):
        u_full[i, j] = u_int[k]
    return u_full.astype(np.float32)


# ============================================================================
# Per-shape generation with parametric BC
# ============================================================================
def _coeffs_to_problem_name(prefix: str, coeffs: dict) -> str:
    """Build a deterministic problem name from coefficient dict.

    e.g. ('laplace_poly3', {'a': 0.32, 'b': -0.71, 'c': 0.05})
       -> 'laplace_poly3_a0.32_b-0.71_c0.05'
    """
    coeff_str = "_".join(f"{k}{v:+.2f}" for k, v in coeffs.items())
    return f"{prefix}_{coeff_str}"


def gen_one_shape_param(digit_28, label, R, out_dir, k_lap, k_poi,
                         rng, coeff_low=-1.0, coeff_high=1.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_mask = make_domain_mask(digit_28, R)
    bd = boundary_mask(domain_mask)
    np.save(out_dir / "mask.npy", domain_mask)
    np.save(out_dir / "boundary.npy", bd)

    sources = poisson_sources(R)

    # Laplace problems: pick K_lap random (family, coeffs) draws.
    for k in range(k_lap):
        family = PARAM_BC_FAMILIES[rng.randint(len(PARAM_BC_FAMILIES))]
        h, coeffs = sample_bc_field(R, family, rng, coeff_low, coeff_high)
        # For Laplace, u_true is solved with f=0 (gives true harmonic
        # extension of h on the boundary; for non-harmonic h-fields this
        # gives the harmonic interpolant, which is the standard Laplace
        # answer).
        u_true = solve_poisson_5pt(domain_mask, np.zeros_like(h), h)
        prob_name = _coeffs_to_problem_name(f"laplace_{family}", coeffs)
        np.savez_compressed(out_dir / f"{prob_name}.npz",
                            h=h, u_true=u_true,
                            bc_family=family,
                            bc_coeffs=np.array(list(coeffs.values()), dtype=np.float32))

    # Poisson problems: pick K_poi random (source, family, coeffs) draws.
    for k in range(k_poi):
        sname, f = sources[rng.randint(len(sources))]
        family = PARAM_BC_FAMILIES[rng.randint(len(PARAM_BC_FAMILIES))]
        h, coeffs = sample_bc_field(R, family, rng, coeff_low, coeff_high)
        u_true = solve_poisson_5pt(domain_mask, f, h)
        prob_name = _coeffs_to_problem_name(f"poisson_{sname}_{family}", coeffs)
        np.savez_compressed(out_dir / f"{prob_name}.npz",
                            h=h, f=f, u_true=u_true,
                            bc_family=family,
                            bc_coeffs=np.array(list(coeffs.values()), dtype=np.float32),
                            source_family=sname)

    return {"label": int(label), "n_interior": int(domain_mask.sum())}


# ============================================================================
# Main
# ============================================================================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mnist-dir", default=os.environ.get("MNIST_DIR", "data/mnist/raw"),
                    help="directory with train-images-idx3-ubyte(.gz)")
    pa.add_argument("--out", required=True)
    pa.add_argument("--n-train-shapes", type=int, default=1000)
    pa.add_argument("--n-test-shapes", type=int, default=50,
                    help="In-distribution test shapes (same coeff range as train)")
    pa.add_argument("--n-test-ood-shapes", type=int, default=50,
                    help="OOD test shapes with extrapolated coefficient range")
    pa.add_argument("--k-lap", type=int, default=8,
                    help="# Laplace problems per shape")
    pa.add_argument("--k-poi", type=int, default=8,
                    help="# Poisson problems per shape")
    pa.add_argument("--resolution", type=int, default=256)
    pa.add_argument("--coeff-train-low", type=float, default=-1.0)
    pa.add_argument("--coeff-train-high", type=float, default=1.0)
    pa.add_argument("--coeff-ood-low", type=float, default=1.0)
    pa.add_argument("--coeff-ood-high", type=float, default=2.0)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--split-seeds", type=str, default="4409,8984,8077",
                    help="per-split RNG offsets for train,test,test_ood (added to --seed); "
                         "the defaults reproduce the paper benchmark")
    pa.add_argument("--splits", type=str, default="train,test,test_ood",
                    help="comma-separated subset of splits to generate")
    pa.add_argument("--max-shapes-per-split", type=int, default=-1,
                    help="generate only the first N shapes of each split (for checks)")
    args = pa.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images_path = Path(args.mnist_dir) / "train-images-idx3-ubyte.gz"
    labels_path = Path(args.mnist_dir) / "train-labels-idx1-ubyte.gz"
    if not images_path.exists():
        images_path = Path(args.mnist_dir) / "train-images-idx3-ubyte"
        labels_path = Path(args.mnist_dir) / "train-labels-idx1-ubyte"

    n_total = args.n_train_shapes + args.n_test_shapes + args.n_test_ood_shapes
    images = load_mnist_images(images_path, n_total + 100)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(images))[:n_total]
    train_idx = perm[: args.n_train_shapes]
    test_idx = perm[args.n_train_shapes: args.n_train_shapes + args.n_test_shapes]
    test_ood_idx = perm[args.n_train_shapes + args.n_test_shapes:
                         args.n_train_shapes + args.n_test_shapes + args.n_test_ood_shapes]

    print(f"[gen-paramBC] n_train={len(train_idx)} n_test={len(test_idx)} "
          f"n_test_ood={len(test_ood_idx)} k_lap={args.k_lap} k_poi={args.k_poi}", flush=True)
    print(f"[gen-paramBC] BC families: {PARAM_BC_FAMILIES}", flush=True)
    print(f"[gen-paramBC] coeff range train/test: [{args.coeff_train_low}, "
          f"{args.coeff_train_high}], OOD: [{args.coeff_ood_low}, {args.coeff_ood_high}]", flush=True)

    t0 = time.time()
    splits = [
        ("train", train_idx, args.coeff_train_low, args.coeff_train_high),
        ("test", test_idx, args.coeff_train_low, args.coeff_train_high),
        ("test_ood", test_ood_idx, args.coeff_ood_low, args.coeff_ood_high),
    ]
    offsets = dict(zip(["train", "test", "test_ood"], [int(x) for x in args.split_seeds.split(",")]))
    wanted = set(args.splits.split(","))
    for split_name, idxs, c_low, c_high in splits:
        if split_name not in wanted:
            continue
        # Per-split RNG so coefficients don't bleed across splits.
        split_rng = np.random.RandomState(args.seed + offsets[split_name])
        for k, i in enumerate(idxs):
            if 0 <= args.max_shapes_per_split <= k:
                break
            shape_dir = out / split_name / f"{k:04d}_idx{int(i):05d}"
            shape_dir.mkdir(parents=True, exist_ok=True)
            stat = gen_one_shape_param(images[i], int(i), args.resolution, shape_dir,
                                        args.k_lap, args.k_poi, split_rng, c_low, c_high)
            elapsed = time.time() - t0
            if (k + 1) % 50 == 0 or k == 0:
                print(f"[{split_name} {k+1}/{len(idxs)}] idx={int(i)} "
                      f"interior={stat['n_interior']} ({elapsed:.1f}s)", flush=True)
        print(f"[gen-paramBC] {split_name} DONE: {len(idxs)} shapes, "
              f"{int(time.time()-t0)}s elapsed", flush=True)

    print(f"[gen-paramBC] all DONE in {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
