"""Parse the real MLPerf Tiny v1.2 inference results.

REAL data: 73 submitted measurements of four TinyML benchmarks on real
commercial hardware, with real throughput, accuracy and (where submitted)
energy per inference.

Source : https://github.com/mlcommons/tiny_results_v1.2 -- summary.csv
License: Apache-2.0
Round  : MLPerf Tiny v1.2 (closed division)

MLPerf Tiny benchmark tasks:
    ad   anomaly detection        (ToyADMOS/DCASE, AUC)
    ic   image classification     (CIFAR-10, top-1)
    kws  keyword spotting         (Speech Commands, top-1)
    vww  visual wake words        (VWW/COCO, top-1)
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TINY_DIR = DATA_DIR / "mlperf-tiny"
ROUND = "v1.2"
SUMMARY_URL = (f"https://raw.githubusercontent.com/mlcommons/tiny_results_{ROUND}/"
               "main/summary.csv")

TASKS = {
    "ad":  ("Anomaly Detection", "ToyADMOS / DCASE2020", "AUC", 0.85),
    "ic":  ("Image Classification", "CIFAR-10", "top-1 accuracy", 85.0),
    "kws": ("Keyword Spotting", "Speech Commands v2", "top-1 accuracy", 90.0),
    "vww": ("Visual Wake Words", "VWW / MSCOCO", "top-1 accuracy", 80.0),
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-") or "unknown"


@dataclass(frozen=True)
class TinyResult:
    id: str
    organization: str
    board_name: str
    system_desc: str
    task: str
    availability: str
    division: str
    throughput_inf_s: float | None
    accuracy: float | None
    has_power: bool
    power_uj_per_inf: float | None
    host_processor: str
    host_frequency: str
    accelerator: str
    inference_framework: str
    software_libraries: str
    hardware_notes: str
    round: str


def _num(value: str) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out


def parse_summary(text: str) -> list[TinyResult]:
    results: list[TinyResult] = []
    for i, row in enumerate(csv.DictReader(text.splitlines())):
        org = (row.get("Organization") or "").strip()
        task = (row.get("MlperfModel") or "").strip().lower()
        if not org or task not in TASKS:
            continue
        has_power = (row.get("HasPower") or "").strip().lower() == "true"
        power = _num(row.get("Power"))
        results.append(TinyResult(
            id=f"tiny:{ROUND}:{i:04d}",
            organization=org,
            board_name=(row.get("BoardName") or "").strip() or "unspecified",
            system_desc=(row.get("SystemDesc") or "").strip(),
            task=task,
            availability=(row.get("Availability") or "").strip(),
            division=(row.get("Division") or "").strip(),
            throughput_inf_s=_num(row.get("Result")),
            accuracy=_num(row.get("Accuracy")),
            has_power=has_power,
            power_uj_per_inf=power if (has_power and power) else None,
            host_processor=(row.get("HostProcessorModelName") or "").strip(),
            host_frequency=(row.get("HostProcessorFrequency") or "").strip(),
            accelerator=(row.get("AcceleratorModelName") or "").strip(),
            inference_framework=(row.get("InferenceFramework") or "").strip(),
            software_libraries=(row.get("SoftwareLibraries") or "").strip(),
            hardware_notes=(row.get("HardwareNotes") or "").strip(),
            round=ROUND,
        ))
    return results


def download(force: bool = False) -> Path:
    import requests

    TINY_DIR.mkdir(parents=True, exist_ok=True)
    path = TINY_DIR / "summary.csv"
    if path.exists() and not force:
        return path
    resp = requests.get(SUMMARY_URL, timeout=180)
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return path


def build(force: bool = False) -> list[TinyResult]:
    path = download(force=force)
    results = parse_summary(path.read_text(encoding="utf-8"))
    (TINY_DIR / "results.json").write_text(json.dumps({
        "source": SUMMARY_URL,
        "license": "Apache-2.0",
        "round": ROUND,
        "result_count": len(results),
        "results": [asdict(r) for r in results],
    }, indent=1), encoding="utf-8")
    return results


def load_cached() -> list[TinyResult]:
    path = TINY_DIR / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m etl.download_data`.")
    return [TinyResult(**r) for r in json.loads(path.read_text(encoding="utf-8"))["results"]]


if __name__ == "__main__":
    rs = build()
    from collections import Counter
    print(f"[mlperf-tiny] {len(rs)} results, round {ROUND}")
    print(f"      orgs        : {len({r.organization for r in rs})}")
    print(f"      boards      : {len({r.board_name for r in rs})}")
    print(f"      accelerators: {sorted({r.accelerator for r in rs if r.accelerator})}")
    print(f"      frameworks  : {len({r.inference_framework for r in rs if r.inference_framework})}")
    print(f"      with power  : {sum(1 for r in rs if r.power_uj_per_inf)}")
    print(f"      per task    : {dict(Counter(r.task for r in rs))}")
