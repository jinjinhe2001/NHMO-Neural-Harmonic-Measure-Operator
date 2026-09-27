"""Concatenate KDE-target shards (tools/precompute_omega_gt_2d.py or tools/gen_mask_corpus.py).

Shards are concatenated in the order given on the command line.

    python tools/merge_omega_gt.py data/omega_gt_5k.pt shard0.pt shard1.pt shard2.pt
"""
import argparse

import torch

STACKED = ('surface_points', 'surface_normals', 'surface_weights', 'sdf_grids', 'queries', 'omega_gt')


def main() -> int:
    pa = argparse.ArgumentParser(description='Concatenate omega_gt shards along the shape axis.')
    pa.add_argument('out')
    pa.add_argument('shards', nargs='+')
    a = pa.parse_args()
    parts = [torch.load(s, map_location='cpu', weights_only=False) for s in a.shards]
    merged = {k: torch.cat([p[k] for p in parts], dim=0) for k in STACKED}
    # list-valued per-shape fields present in every shard (shape_ids, mnist_indices, ...)
    for k, v in parts[0].items():
        if isinstance(v, list) and all(isinstance(p.get(k), list) for p in parts):
            merged[k] = sum((p[k] for p in parts), [])
    cfg = dict(parts[0].get('config', {}))
    cfg['n_shapes_total'] = len(merged['shape_ids'])
    cfg['shards'] = a.shards
    merged['config'] = cfg
    torch.save(merged, a.out)
    print(f'wrote {a.out}: {len(merged["shape_ids"])} shapes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
