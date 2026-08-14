"""Build the REAL subgraph from public sources.

Two upstreams, both permissively licensed, both giving facts that are checkable
against their source:

  * ONNX Runtime kernel registrations (MIT) -- which operators each execution
    provider actually implements, at which opsets. Real answers to the same
    coverage question the synthetic fleet models at larger scale.
  * MLPerf Tiny v1.2 results (Apache-2.0) -- 73 measured submissions on real
    commercial edge hardware, with throughput, accuracy and energy.

Every node this module emits carries `provenance: "real"` and a `source`
naming the upstream, so a query can always separate measured from generated.
"""
from __future__ import annotations

from etl import mlperf_tiny as tiny
from etl import ort_kernels as ort
from etl.generate import Fleet
from etl.onnx_catalog import Operator, categorize

SRC_ONNX = "onnx"
SRC_ORT = "onnxruntime"
SRC_TINY = f"mlperf-tiny-{tiny.ROUND}"

# ONNX Runtime execution provider -> Accelerator node metadata.
ORT_DEVICES = {
    "ort-cpu": ("ONNX Runtime CPU EP", "CPU"),
    "ort-cuda": ("ONNX Runtime CUDA EP", "GPU-CUDA"),
    "ort-dml": ("ONNX Runtime DirectML EP", "GPU-DirectML"),
    "ort-rocm": ("ONNX Runtime ROCm EP", "GPU-ROCm"),
    "ort-trt": ("ONNX Runtime TensorRT EP", "GPU-TensorRT"),
}


def add_ort_layer(fleet: Fleet, kernels: list[ort.OrtKernel],
                  known_operators: set[str]) -> None:
    """Real ONNX Runtime kernels -> Runtime, Accelerator, Operator, Kernel."""
    fleet.add_nodes("Runtime", [{
        "id": "runtime:ort", "name": "ONNX Runtime", "version": "main",
        "format": "onnx",
    }], provenance="real", source=SRC_ORT)

    used_devices = sorted({k.device_id for k in kernels})
    fleet.add_nodes("Accelerator", [{
        "id": f"accel:{dev}",
        "name": ORT_DEVICES.get(dev, (dev, "Unknown"))[0],
        "kind": ORT_DEVICES.get(dev, (dev, "Unknown"))[1],
    } for dev in used_devices], provenance="real", source=SRC_ORT)

    for dev in used_devices:
        fleet.add_edge("Runtime", "runtime:ort", "TARGETS", "Accelerator", f"accel:{dev}")

    # Operators outside the parsed ai.onnx catalog (com.microsoft, ai.onnx.ml,
    # ...) need Operator nodes of their own, or their kernels would dangle.
    extra: dict[str, dict] = {}
    for k in kernels:
        op_id = f"op:{k.domain}:{k.operator}".lower()
        if op_id in known_operators or op_id in extra:
            continue
        extra[op_id] = {
            "id": op_id, "name": k.operator, "domain": k.domain,
            "since_version": k.opset_min,
            "version_count": 1,
            "category": categorize(k.operator),
            "is_control_flow": 0,
        }
    if extra:
        fleet.add_nodes("Operator", list(extra.values()),
                        provenance="real", source=SRC_ORT)

    fleet.add_nodes("Kernel", [{
        "id": k.id,
        "name": f"{k.operator}@{k.execution_provider}",
        "execution_provider": k.execution_provider,
        "opset_spec": k.opset_spec,
        "opset_min": k.opset_min,
        "opset_max": k.opset_max,
        "type_count": k.type_count,
        "is_fallback": 1 if k.device_id == "ort-cpu" else 0,
    } for k in kernels], provenance="real", source=SRC_ORT)

    for k in kernels:
        fleet.add_edge("Kernel", k.id, "IMPLEMENTS", "Operator",
                       f"op:{k.domain}:{k.operator}".lower())
        fleet.add_edge("Kernel", k.id, "RUNS_ON", "Accelerator", f"accel:{k.device_id}")
        fleet.add_edge("Kernel", k.id, "PROVIDED_BY", "Runtime", "runtime:ort")


def add_mlperf_layer(fleet: Fleet, results: list[tiny.TinyResult]) -> None:
    """Real MLPerf Tiny submissions -> Vendor, Board, SoC, Accelerator,
    Runtime, BenchmarkTask, Model, Deployment."""
    vendors, boards, socs, accels, runtimes = {}, {}, {}, {}, {}

    for r in results:
        vid = f"vendor:{tiny.slug(r.organization)}"
        vendors.setdefault(vid, {"id": vid, "name": r.organization, "country": ""})

        bid = f"board:{tiny.slug(r.board_name)}"
        boards.setdefault(bid, {
            "id": bid, "name": r.board_name, "system_desc": r.system_desc,
            "availability": r.availability, "hardware_notes": r.hardware_notes[:180],
            "_vendor": vid,
        })

        if r.host_processor:
            sid = f"soc:{tiny.slug(r.host_processor)}"
            socs.setdefault(sid, {
                "id": sid, "name": r.host_processor,
                "cpu_frequency": r.host_frequency[:120], "_vendor": vid,
            })
            boards[bid].setdefault("_soc", sid)

        if r.accelerator:
            aid = f"accel:{tiny.slug(r.accelerator)}"
            accels.setdefault(aid, {"id": aid, "name": r.accelerator, "kind": "NPU"})

        if r.inference_framework and r.inference_framework not in ("N/A", "None"):
            rid = f"runtime:{tiny.slug(r.inference_framework)}"
            runtimes.setdefault(rid, {
                "id": rid, "name": r.inference_framework, "version": "",
                "format": "", 
            })

    fleet.add_nodes("Vendor", list(vendors.values()), provenance="real", source=SRC_TINY)
    fleet.add_nodes("SoC", [{k: v for k, v in s.items() if not k.startswith("_")}
                            for s in socs.values()], provenance="real", source=SRC_TINY)
    fleet.add_nodes("Accelerator", list(accels.values()), provenance="real", source=SRC_TINY)
    fleet.add_nodes("Runtime", list(runtimes.values()), provenance="real", source=SRC_TINY)
    fleet.add_nodes("Board", [{k: v for k, v in b.items() if not k.startswith("_")}
                              for b in boards.values()], provenance="real", source=SRC_TINY)

    for s in socs.values():
        fleet.add_edge("SoC", s["id"], "MADE_BY", "Vendor", s["_vendor"])
    for b in boards.values():
        fleet.add_edge("Board", b["id"], "MADE_BY", "Vendor", b["_vendor"])
        if b.get("_soc"):
            fleet.add_edge("Board", b["id"], "HAS_SOC", "SoC", b["_soc"])

    # Benchmark tasks + their reference models
    tasks, models = [], []
    for key, (name, dataset, metric, target) in tiny.TASKS.items():
        tasks.append({"id": f"task:{key}", "name": name, "code": key,
                      "dataset": dataset, "metric": metric, "quality_target": target})
        models.append({"id": f"model:mlperf-{key}", "name": f"MLPerf Tiny {name}",
                       "family": "mlperf-tiny-reference", "task": name})
    fleet.add_nodes("BenchmarkTask", tasks, provenance="real", source=SRC_TINY)
    fleet.add_nodes("Model", models, provenance="real", source=SRC_TINY)
    for key in tiny.TASKS:
        fleet.add_edge("Model", f"model:mlperf-{key}", "SOLVES",
                       "BenchmarkTask", f"task:{key}")

    deployments = []
    for r in results:
        row = {
            "id": r.id,
            "round": r.round,
            "division": r.division,
            "availability": r.availability,
            "throughput_inf_s": r.throughput_inf_s,
            "accuracy": r.accuracy,
        }
        if r.power_uj_per_inf:
            row["energy_uj_per_inf"] = r.power_uj_per_inf
        deployments.append(row)
    fleet.add_nodes("Deployment", deployments, provenance="real", source=SRC_TINY)

    for r in results:
        fleet.add_edge("Deployment", r.id, "ON_BOARD", "Board",
                       f"board:{tiny.slug(r.board_name)}")
        fleet.add_edge("Deployment", r.id, "MEASURES", "Model",
                       f"model:mlperf-{r.task}")
        if r.accelerator:
            fleet.add_edge("Deployment", r.id, "USES_ACCELERATOR", "Accelerator",
                           f"accel:{tiny.slug(r.accelerator)}")
        if r.inference_framework and r.inference_framework not in ("N/A", "None"):
            fleet.add_edge("Deployment", r.id, "VIA_RUNTIME", "Runtime",
                           f"runtime:{tiny.slug(r.inference_framework)}")


def add_onnx_operators(fleet: Fleet, operators: list[Operator]) -> None:
    """Add the real ONNX operator catalog if the fleet does not already have it.

    The synthetic generator also adds these (it indexes models on them), so when
    both layers are loaded this is a no-op. On a `--layers real` load the fleet
    starts empty, and without this every kernel's IMPLEMENTS edge would dangle.
    """
    existing = {o["id"] for o in fleet.nodes.get("Operator", [])}
    missing = [o for o in operators if o.id not in existing]
    if not missing:
        return
    fleet.add_nodes("Operator", [{
        "id": o.id, "name": o.name, "domain": o.domain,
        "since_version": o.since_version, "version_count": o.version_count,
        "category": o.category, "is_control_flow": 1 if o.is_control_flow else 0,
    } for o in missing], provenance="real", source=SRC_ONNX)


def build_real(fleet: Fleet, operators: list[Operator]) -> Fleet:
    """Attach every real layer to an existing (possibly empty) fleet."""
    add_onnx_operators(fleet, operators)
    known = {o["id"] for o in fleet.nodes.get("Operator", [])}
    add_ort_layer(fleet, ort.load_cached(), known)
    add_mlperf_layer(fleet, tiny.load_cached())
    return fleet
