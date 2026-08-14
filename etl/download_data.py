"""Fetch + build all source data into ./data.

Three upstreams:

  1. REAL -- ONNX operator catalog (Apache-2.0, onnx/onnx). The operator
     vocabulary the whole graph is indexed on.
  2. REAL -- ONNX Runtime kernel registrations (MIT, microsoft/onnxruntime).
     Which operators each execution provider actually implements.
  3. REAL -- MLPerf Tiny v1.2 results (Apache-2.0, mlcommons). Measured
     throughput, accuracy and energy on real commercial edge hardware.

  4. SYNTHETIC -- the edge-AI hardware fleet, generated deterministically from
     a seed, providing the scale and density the real sources don't have.

Everything is regenerable, so ./data is gitignored. Same seed => same graph.
"""
from __future__ import annotations

import argparse

from etl import generate as gen
from etl import mlperf_tiny, onnx_catalog, ort_kernels


def download_all(seed: int = gen.DEFAULT_SEED, scale: float = 1.0,
                 force: bool = False) -> None:
    ops = onnx_catalog.build(force=force)
    print(f"[1/4] REAL  ONNX operator catalog     : {len(ops):5d} operators "
          f"(Apache-2.0, onnx/onnx)")

    kernels = ort_kernels.build(force=force)
    eps = len({k.execution_provider for k in kernels})
    print(f"[2/4] REAL  ONNX Runtime kernels      : {len(kernels):5d} registrations "
          f"across {eps} execution providers (MIT)")

    results = mlperf_tiny.build(force=force)
    print(f"[3/4] REAL  MLPerf Tiny {mlperf_tiny.ROUND} results  : {len(results):5d} measured "
          f"submissions on real hardware (Apache-2.0)")

    fleet = gen.generate(seed=seed, scale=scale, operators=ops)
    path = gen.write(fleet)
    print(f"[4/4] SYNTH generated fleet           : {fleet.node_count:5d} nodes, "
          f"{fleet.edge_count} edges (seed={seed}, scale={scale})")
    print(f"      -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=gen.DEFAULT_SEED)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="Re-download upstream sources.")
    args = ap.parse_args()
    download_all(seed=args.seed, scale=args.scale, force=args.force)
