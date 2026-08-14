"""Parse ONNX Runtime's real operator-kernel registration matrix.

This is REAL data: which operators each ONNX Runtime execution provider
actually implements, at which opset ranges, for which tensor types. It is the
ground truth behind the whole "does this operator have a kernel on my target"
question -- the same question the synthetic fleet models at larger scale.

Source : https://github.com/microsoft/onnxruntime -- docs/OperatorKernels.md
         (generated from the registered kernels by tools/python/gen_opkernel_doc.py)
License: MIT

Parsed shape:
    ## Operators implemented by <EP>ExecutionProvider
    | Op Name | Parameters | OpSet Version | Types Supported |
    |Abs|*in* X:**T**|13+|**T** = tensor(double), ...|
    |||[6, 12]|...                 <- continuation row: same op, older opset

Rows with an empty Op Name are continuation rows carrying an additional opset
registration for the operator named above them.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORT_DIR = DATA_DIR / "onnxruntime"
KERNELS_URL = ("https://raw.githubusercontent.com/microsoft/onnxruntime/main/"
               "docs/OperatorKernels.md")

# Execution provider -> the device class it runs on. Used to attach real
# kernels to a real Accelerator node.
EP_DEVICE = {
    "CPUExecutionProvider": ("ort-cpu", "CPU", "x86-64 / arm64 CPU"),
    "CUDAExecutionProvider": ("ort-cuda", "GPU-CUDA", "NVIDIA CUDA GPU"),
    "DmlExecutionProvider": ("ort-dml", "GPU-DirectML", "DirectML GPU"),
    "ROCMExecutionProvider": ("ort-rocm", "GPU-ROCm", "AMD ROCm GPU"),
    "TensorrtExecutionProvider": ("ort-trt", "GPU-TensorRT", "NVIDIA TensorRT"),
}


@dataclass(frozen=True)
class OrtKernel:
    id: str
    operator: str
    domain: str
    execution_provider: str
    device_id: str
    opset_spec: str
    opset_min: int
    opset_max: int          # 999 == open-ended ("N+")
    type_count: int


def _parse_opset(spec: str) -> tuple[int, int]:
    """'13+' -> (13, 999); '[6, 12]' -> (6, 12); '13' -> (13, 13)."""
    spec = spec.strip()
    m = re.fullmatch(r"\[(\d+),\s*(\d+)\]", spec)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d+)\+", spec)
    if m:
        return int(m.group(1)), 999
    m = re.fullmatch(r"(\d+)", spec)
    if m:
        return int(m.group(1)), int(m.group(1))
    return -1, -1


def parse_kernels(markdown: str) -> list[OrtKernel]:
    kernels: list[OrtKernel] = []
    provider: str | None = None
    domain = "ai.onnx"
    current_op: str | None = None
    seen: set[str] = set()

    for raw in markdown.splitlines():
        line = raw.strip()

        m = re.match(r"^## Operators implemented by (\w+)$", line)
        if m:
            provider, domain, current_op = m.group(1), "ai.onnx", None
            continue
        if provider is None or not line.startswith("|"):
            continue

        m = re.match(r"^\|\*\*Operator Domain:\*\*\s*\*([^*]+)\*", line)
        if m:
            domain = m.group(1).strip()
            current_op = None
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= {"-"}:
            continue
        if cells[0] in ("Op Name",):
            continue

        op_name = cells[0]
        if op_name:
            current_op = op_name
        elif current_op is None:
            continue

        opset_spec = cells[2]
        opset_min, opset_max = _parse_opset(opset_spec)
        if opset_min < 0:
            continue

        types = cells[3] if len(cells) > 3 else ""
        type_count = len(re.findall(r"tensor\(", types))

        kid = f"ortk:{provider}:{domain}:{current_op}:{opset_min}-{opset_max}".lower()
        if kid in seen:
            continue
        seen.add(kid)
        device_id = EP_DEVICE.get(provider, (f"ort-{provider.lower()}", "Unknown", ""))[0]
        kernels.append(OrtKernel(
            id=kid, operator=current_op, domain=domain,
            execution_provider=provider, device_id=device_id,
            opset_spec=opset_spec, opset_min=opset_min, opset_max=opset_max,
            type_count=type_count,
        ))
    return kernels


def download(force: bool = False) -> Path:
    import requests

    ORT_DIR.mkdir(parents=True, exist_ok=True)
    path = ORT_DIR / "OperatorKernels.md"
    if path.exists() and not force:
        return path
    resp = requests.get(KERNELS_URL, timeout=180)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return path


def build(force: bool = False) -> list[OrtKernel]:
    path = download(force=force)
    kernels = parse_kernels(path.read_text(encoding="utf-8"))
    (ORT_DIR / "kernels.json").write_text(json.dumps({
        "source": KERNELS_URL,
        "license": "MIT",
        "kernel_count": len(kernels),
        "kernels": [asdict(k) for k in kernels],
    }, indent=1), encoding="utf-8")
    return kernels


def load_cached() -> list[OrtKernel]:
    path = ORT_DIR / "kernels.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m etl.download_data`.")
    return [OrtKernel(**k) for k in json.loads(path.read_text(encoding="utf-8"))["kernels"]]


if __name__ == "__main__":
    ks = build()
    from collections import Counter
    print(f"[ort] {len(ks)} kernel registrations")
    for ep, n in Counter(k.execution_provider for k in ks).most_common():
        ops = len({k.operator for k in ks if k.execution_provider == ep})
        print(f"      {ep:28s} {n:5d} registrations over {ops:4d} distinct operators")
    print(f"      domains: {Counter(k.domain for k in ks)}")
