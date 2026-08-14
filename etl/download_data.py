"""Fetch + build all source data into ./data.

Two sources:
  1. REAL      -- the ONNX operator catalog (Apache-2.0), downloaded from the
                  onnx/onnx repo and parsed into data/onnx/operators.json.
  2. SYNTHETIC -- the edge-AI hardware fleet, generated deterministically from
                  a seed into data/fleet/fleet.json.

Both are regenerable, so ./data is gitignored. Same seed => same graph.
"""
from __future__ import annotations

import argparse

from etl import generate as gen
from etl import onnx_catalog


def download_all(seed: int = gen.DEFAULT_SEED, scale: float = 1.0,
                 force: bool = False) -> None:
    ops = onnx_catalog.build(force=force)
    print(f"[1/2] ONNX operator catalog: {len(ops)} operators "
          f"-> {onnx_catalog.ONNX_DIR / 'operators.json'}  (real, Apache-2.0)")

    fleet = gen.generate(seed=seed, scale=scale, operators=ops)
    path = gen.write(fleet)
    print(f"[2/2] synthetic fleet: {fleet.node_count:,} nodes, "
          f"{fleet.edge_count:,} edges -> {path}  (seed={seed}, scale={scale})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=gen.DEFAULT_SEED)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="Re-download ONNX docs.")
    args = ap.parse_args()
    download_all(seed=args.seed, scale=args.scale, force=args.force)
