"""Deterministic synthetic generator for the Edge AI deployment fleet.

Everything this module emits is SYNTHETIC. Vendor, board and SoC names are
deliberately fictional so no number here can be mistaken for a claim about a
real product. The only real data in this KG is the ONNX operator catalog
(see `etl/onnx_catalog.py`).

Why synthetic: the interesting questions in edge-AI deployment are structural
("which operators have no kernel on this accelerator, and what does the CPU
fallback cost me?"). Those need dense, complete coverage across a hardware
fleet -- which no public dataset provides, and which cannot be honestly faked
using real product names.

Determinism: everything derives from `--seed` (default 20260814), so the same
seed reproduces the same graph byte-for-byte.

Metrics are *derived*, not invented -- see `docs/data-provenance.md` for the
cost model and its stated limits.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from etl.onnx_catalog import Operator, load_cached

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FLEET_PATH = DATA_DIR / "fleet" / "fleet.json"
DEFAULT_SEED = 20260814

# --------------------------------------------------------------------------
# Fictional vendors. Not real companies.
# --------------------------------------------------------------------------
VENDORS = [
    ("Corvid Silicon", "IN"), ("Nimbus Micro", "US"), ("Tessera Labs", "DE"),
    ("Akshara Semi", "IN"), ("Halcyon Devices", "JP"), ("Vermilion Systems", "US"),
    ("Kestrel Embedded", "UK"), ("Suvarna Microsystems", "IN"),
]

# Accelerator archetypes: (kind, op categories it can run, opset ceiling,
#   int8 GOPS range, sram KB range, relative energy per op)
ACCEL_ARCHETYPES = [
    ("MCU-CPU", {"elementwise", "activation", "shape", "reduction", "tensor",
                 "matmul", "convolution", "spatial", "normalization", "recurrent",
                 "quantization", "signal", "attention", "stochastic",
                 "loss"}, 99,
     (0.5, 3.0), (64, 512), 1.00),
    ("DSP", {"elementwise", "activation", "signal", "reduction", "matmul",
             "convolution", "spatial", "shape"}, 17, (8.0, 40.0), (256, 2048), 0.42),
    ("NPU-Lite", {"convolution", "matmul", "activation", "spatial", "normalization",
                  "elementwise", "quantization"}, 13, (30.0, 120.0), (512, 4096), 0.16),
    ("NPU-Pro", {"convolution", "matmul", "activation", "spatial", "normalization",
                 "elementwise", "quantization", "reduction", "attention", "shape"}, 19,
     (150.0, 900.0), (2048, 16384), 0.11),
    ("GPU-Embedded", {"convolution", "matmul", "activation", "spatial", "normalization",
                      "elementwise", "quantization", "reduction", "attention", "shape",
                      "recurrent", "tensor"}, 21, (400.0, 2400.0), (4096, 32768), 0.30),
]

RUNTIMES = [
    # (name, version, model format, accelerator kinds it can target)
    ("TFLite Micro", "1.4", "tflite", {"MCU-CPU", "DSP", "NPU-Lite"}),
    ("ONNX Runtime", "1.20", "onnx", {"MCU-CPU", "DSP", "NPU-Lite", "NPU-Pro", "GPU-Embedded"}),
    ("ExecuTorch", "0.5", "pte", {"MCU-CPU", "NPU-Lite", "NPU-Pro", "GPU-Embedded"}),
    ("TVM microTVM", "0.17", "tar", {"MCU-CPU", "DSP", "NPU-Lite", "NPU-Pro"}),
    ("CMSIS-NN", "6.0", "c-array", {"MCU-CPU"}),
    ("Vendor SDK", "3.1", "blob", {"DSP", "NPU-Lite", "NPU-Pro"}),
    ("OpenVINO", "2025.1", "ir", {"MCU-CPU", "GPU-Embedded", "NPU-Pro"}),
]

# Model architecture families -> operator categories they draw from, and a
# rough MAC budget. Task assignment comes from CLINICAL_TASKS below.
MODEL_FAMILIES = [
    ("tiny-cnn",    ["convolution", "activation", "spatial", "normalization", "elementwise", "shape"], (0.4, 12.0)),
    ("depthwise-cnn", ["convolution", "activation", "spatial", "normalization", "elementwise", "shape", "reduction"], (2.0, 45.0)),
    ("resnet-1d",   ["convolution", "activation", "normalization", "elementwise", "reduction", "shape"], (5.0, 90.0)),
    ("lstm-seq",    ["recurrent", "matmul", "activation", "elementwise", "shape"], (1.0, 30.0)),
    ("gru-seq",     ["recurrent", "matmul", "activation", "elementwise", "shape"], (0.8, 22.0)),
    ("tiny-transformer", ["attention", "matmul", "normalization", "activation", "elementwise", "shape", "reduction"], (12.0, 240.0)),
    ("spectro-cnn", ["signal", "convolution", "activation", "spatial", "normalization", "elementwise"], (3.0, 60.0)),
    ("mlp-features", ["matmul", "activation", "elementwise", "shape"], (0.1, 4.0)),
]

PRECISIONS = [
    # (name, size multiplier vs fp32, throughput multiplier, accuracy delta)
    ("fp32", 1.00, 1.00, 0.000),
    ("fp16", 0.50, 1.70, -0.002),
    ("int8", 0.25, 3.20, -0.011),
    ("int4", 0.14, 4.80, -0.037),
]

SENSORS = [
    ("ECG 3-lead", "ecg", 500, 3, 16), ("ECG 12-lead", "ecg", 1000, 12, 16),
    ("PPG wrist", "ppg", 128, 2, 14), ("PPG fingertip", "ppg", 256, 1, 14),
    ("EEG 8-channel", "eeg", 256, 8, 24), ("EEG 32-channel", "eeg", 512, 32, 24),
    ("EMG surface", "emg", 2000, 4, 16), ("IMU 6-axis", "imu", 200, 6, 16),
    ("IMU 9-axis", "imu", 400, 9, 16), ("Contact mic", "audio", 16000, 1, 16),
    ("Chest mic array", "audio", 8000, 4, 16), ("Thermistor array", "temp", 4, 8, 12),
    ("SpO2 optical", "spo2", 64, 2, 14), ("Bio-impedance", "bioz", 100, 4, 16),
]

# Signal-processing stages. `ops` are the ONNX operator categories the stage
# maps onto once it is lowered into the graph.
SIGNAL_STAGES = [
    ("DC removal", "filter", 20, 0.4, ["elementwise", "reduction"]),
    ("Bandpass 0.5-40Hz", "filter", 200, 3.1, ["elementwise", "convolution"]),
    ("Notch 50Hz", "filter", 200, 1.8, ["elementwise", "convolution"]),
    ("Baseline wander removal", "filter", 400, 4.2, ["elementwise", "convolution", "reduction"]),
    ("Resample", "transform", 100, 1.2, ["spatial", "shape"]),
    ("Windowing (Hann)", "transform", 64, 0.3, ["signal", "elementwise"]),
    ("STFT", "transform", 256, 12.5, ["signal", "matmul"]),
    ("Mel filterbank", "transform", 256, 6.0, ["signal", "matmul"]),
    ("Wavelet decomposition", "transform", 512, 18.0, ["convolution", "elementwise"]),
    ("R-peak detection", "feature", 800, 2.6, ["reduction", "shape", "elementwise"]),
    ("HRV features", "feature", 30000, 0.9, ["reduction", "elementwise"]),
    ("Spectral entropy", "feature", 512, 3.4, ["reduction", "elementwise", "signal"]),
    ("Z-score normalize", "transform", 100, 0.5, ["normalization", "reduction"]),
    ("Artifact rejection", "filter", 1000, 5.5, ["reduction", "elementwise", "shape"]),
    ("Downsample decimate", "transform", 100, 0.7, ["spatial", "shape"]),
    ("Envelope extraction", "feature", 300, 2.0, ["elementwise", "convolution"]),
]

# (task, category, sensor modalities, latency budget ms, min sensitivity)
CLINICAL_TASKS = [
    ("Atrial fibrillation detection", "cardiac", ["ecg", "ppg"], 1000, 0.95),
    ("Ventricular arrhythmia alarm", "cardiac", ["ecg"], 250, 0.98),
    ("QRS morphology classification", "cardiac", ["ecg"], 500, 0.92),
    ("Heart-rate variability scoring", "cardiac", ["ecg", "ppg"], 5000, 0.85),
    ("Blood-pressure surrogate estimation", "cardiac", ["ppg", "bioz"], 2000, 0.80),
    ("Seizure onset detection", "neuro", ["eeg"], 300, 0.97),
    ("Sleep-stage classification", "neuro", ["eeg", "imu"], 30000, 0.86),
    ("Cognitive-load estimation", "neuro", ["eeg", "ppg"], 4000, 0.78),
    ("Tremor quantification", "neuro", ["imu", "emg"], 1000, 0.88),
    ("Gait-anomaly detection", "mobility", ["imu"], 2000, 0.84),
    ("Fall detection", "mobility", ["imu"], 200, 0.96),
    ("Freezing-of-gait prediction", "mobility", ["imu", "emg"], 500, 0.90),
    ("Cough event classification", "respiratory", ["audio"], 1000, 0.89),
    ("Wheeze detection", "respiratory", ["audio"], 1500, 0.91),
    ("Respiratory-rate estimation", "respiratory", ["audio", "bioz", "imu"], 5000, 0.87),
    ("Apnea-event detection", "respiratory", ["spo2", "audio"], 10000, 0.94),
    ("Hypoxemia early warning", "respiratory", ["spo2", "ppg"], 3000, 0.95),
    ("Muscle-fatigue estimation", "musculoskeletal", ["emg"], 2000, 0.82),
]

DATASETS = [
    ("MIT-BIH Arrhythmia", "PhysioNet", 47, 24, "ODC-BY 1.0"),
    ("PTB-XL ECG", "PhysioNet", 18869, 5300, "CC-BY 4.0"),
    ("CinC Challenge 2017", "PhysioNet", 8528, 240, "ODC-BY 1.0"),
    ("CHB-MIT Scalp EEG", "PhysioNet", 24, 980, "ODC-BY 1.0"),
    ("Sleep-EDF Expanded", "PhysioNet", 197, 3800, "ODC-BY 1.0"),
    ("TUH EEG Corpus", "Temple University", 14000, 27000, "Custom research"),
    ("MobiAct", "TEI of Crete", 67, 92, "Research use"),
    ("UCI HAR", "UCI ML Repository", 30, 25, "CC-BY 4.0"),
    ("Daphnet FoG", "UCI ML Repository", 10, 35, "CC-BY 4.0"),
    ("ICBHI Respiratory", "ICBHI Challenge", 126, 5, "Research use"),
    ("Coswara", "IISc Bangalore", 2000, 60, "CC-BY 4.0"),
    ("BIDMC PPG and Respiration", "PhysioNet", 53, 8, "ODC-BY 1.0"),
]

CERTIFICATIONS = [
    ("IEC 62304 Class A", "IEC", "A"), ("IEC 62304 Class B", "IEC", "B"),
    ("IEC 62304 Class C", "IEC", "C"), ("ISO 13485", "ISO", "QMS"),
    ("FDA 510(k) Class II", "FDA", "II"), ("EU MDR Class IIa", "EU", "IIa"),
]

FORM_FACTORS = ["wearable-band", "patch", "chest-module", "handheld",
                "bedside-module", "implant-adjacent", "m.2-module", "som"]


def _rid(prefix: str, n: int) -> str:
    return f"{prefix}:{n:05d}"


@dataclass
class Fleet:
    """The complete generated graph, as plain dicts ready for the loader."""
    seed: int
    scale: float
    nodes: dict[str, list[dict]] = field(default_factory=dict)
    edges: list[tuple] = field(default_factory=list)

    def add_nodes(self, label: str, rows: list[dict]) -> None:
        self.nodes.setdefault(label, []).extend(rows)

    def add_edge(self, src_label, src_id, rel, tgt_label, tgt_id, props=None) -> None:
        self.edges.append((src_label, src_id, rel, tgt_label, tgt_id, props))

    @property
    def node_count(self) -> int:
        return sum(len(v) for v in self.nodes.values())

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def generate(seed: int = DEFAULT_SEED, scale: float = 1.0,
             operators: list[Operator] | None = None) -> Fleet:
    rng = random.Random(seed)
    ops = operators if operators is not None else load_cached()
    fleet = Fleet(seed=seed, scale=scale)

    def n(base: int) -> int:
        return max(1, int(round(base * scale)))

    # ---------------- Vendors ----------------
    vendors = []
    for i, (name, country) in enumerate(VENDORS):
        vid = _rid("vendor", i)
        vendors.append({"id": vid, "name": name, "country": country})
    fleet.add_nodes("Vendor", vendors)

    # ---------------- Operators (REAL) ----------------
    op_rows = [{
        "id": o.id, "name": o.name, "domain": o.domain,
        "since_version": o.since_version, "version_count": o.version_count,
        "category": o.category, "is_control_flow": 1 if o.is_control_flow else 0,
    } for o in ops]
    fleet.add_nodes("Operator", op_rows)
    ops_by_cat: dict[str, list[Operator]] = {}
    for o in ops:
        ops_by_cat.setdefault(o.category, []).append(o)

    # ---------------- Certifications ----------------
    certs = [{"id": _rid("cert", i), "name": nm, "body": body, "class": cls}
             for i, (nm, body, cls) in enumerate(CERTIFICATIONS)]
    fleet.add_nodes("Certification", certs)

    # ---------------- Datasets ----------------
    datasets = [{"id": _rid("dataset", i), "name": nm, "source": src,
                 "subjects": subj, "hours": hrs, "license": lic}
                for i, (nm, src, subj, hrs, lic) in enumerate(DATASETS)]
    fleet.add_nodes("Dataset", datasets)

    # ---------------- Sensors ----------------
    sensors = [{"id": _rid("sensor", i), "name": nm, "modality": mod,
                "sample_rate_hz": sr, "channels": ch, "adc_bits": bits}
               for i, (nm, mod, sr, ch, bits) in enumerate(SENSORS)]
    fleet.add_nodes("Sensor", sensors)

    # ---------------- Signal stages ----------------
    stages = []
    for i, (nm, kind, win, kmacs, cats) in enumerate(SIGNAL_STAGES):
        stages.append({"id": _rid("stage", i), "name": nm, "kind": kind,
                       "window_ms": win, "cost_kmacs": kmacs, "_cats": cats})
    fleet.add_nodes("SignalStage", [{k: v for k, v in s.items() if not k.startswith("_")}
                                    for s in stages])
    for s in stages:
        for cat in s["_cats"]:
            for o in rng.sample(ops_by_cat.get(cat, []),
                                min(2, len(ops_by_cat.get(cat, [])))):
                fleet.add_edge("SignalStage", s["id"], "USES_OPERATOR", "Operator", o.id, None)

    # ---------------- Runtimes ----------------
    runtimes = []
    for i, (nm, ver, fmt, kinds) in enumerate(RUNTIMES):
        runtimes.append({"id": _rid("runtime", i), "name": nm, "version": ver,
                         "format": fmt, "_kinds": kinds})
    fleet.add_nodes("Runtime", [{k: v for k, v in r.items() if not k.startswith("_")}
                                for r in runtimes])

    # ---------------- SoCs + Accelerators ----------------
    socs, accelerators = [], []
    for i in range(n(40)):
        vendor = rng.choice(vendors)
        soc = {
            "id": _rid("soc", i),
            "name": f"{vendor['name'].split()[0][:3].upper()}-S{100 + i}",
            "process_nm": rng.choice([7, 12, 16, 22, 28, 40, 55]),
            "cpu_arch": rng.choice(["cortex-m55", "cortex-m85", "cortex-a53",
                                    "cortex-a78", "riscv-rv32imc", "riscv-rv64gc"]),
            "cpu_mhz": rng.choice([80, 200, 400, 800, 1200, 1800, 2200]),
            "cores": rng.choice([1, 1, 2, 4, 4, 8]),
            "_vendor": vendor["id"],
        }
        socs.append(soc)

        # Every SoC has a CPU (the universal fallback), plus 0-2 accelerators.
        kinds = ["MCU-CPU"]
        extra = rng.choices([0, 1, 1, 2], k=1)[0]
        kinds += rng.sample([k[0] for k in ACCEL_ARCHETYPES if k[0] != "MCU-CPU"],
                            min(extra, 4))
        for kind in kinds:
            arch = next(a for a in ACCEL_ARCHETYPES if a[0] == kind)
            _, cats, opset_ceiling, gops_rng, sram_rng, energy = arch
            aid = _rid("accel", len(accelerators))
            accelerators.append({
                "id": aid,
                "name": f"{soc['name']}-{kind}",
                "kind": kind,
                "gops_int8": round(rng.uniform(*gops_rng), 2),
                "sram_kb": rng.choice([s for s in (64, 128, 256, 512, 1024, 2048, 4096,
                                                   8192, 16384, 32768)
                                       if sram_rng[0] <= s <= sram_rng[1]] or [sram_rng[0]]),
                "clock_mhz": rng.choice([100, 200, 400, 600, 800, 1000, 1400]),
                "opset_ceiling": opset_ceiling,
                "energy_factor": energy,
                "_cats": cats, "_soc": soc["id"],
            })
    fleet.add_nodes("SoC", [{k: v for k, v in s.items() if not k.startswith("_")} for s in socs])
    fleet.add_nodes("Accelerator", [{k: v for k, v in a.items() if not k.startswith("_")}
                                    for a in accelerators])
    for s in socs:
        fleet.add_edge("SoC", s["id"], "MADE_BY", "Vendor", s["_vendor"], None)
    for a in accelerators:
        fleet.add_edge("SoC", a["_soc"], "HAS_ACCELERATOR", "Accelerator", a["id"], None)

    accel_by_soc: dict[str, list[dict]] = {}
    for a in accelerators:
        accel_by_soc.setdefault(a["_soc"], []).append(a)

    # ---------------- Runtime -> Accelerator ----------------
    for rt in runtimes:
        for a in accelerators:
            if a["kind"] in rt["_kinds"]:
                fleet.add_edge("Runtime", rt["id"], "TARGETS", "Accelerator", a["id"], None)

    # ---------------- Kernels ----------------
    # A Kernel is a concrete (operator, accelerator, runtime) implementation.
    # It exists only when the accelerator's archetype covers the operator's
    # category AND the operator's opset is within the accelerator's ceiling.
    # The gaps this leaves are the whole point of the KG.
    kernels = []
    for a in accelerators:
        rts = [r for r in runtimes if a["kind"] in r["_kinds"]]
        if not rts:
            continue
        chosen_rts = rng.sample(rts, min(len(rts), rng.choice([1, 2, 2, 3])))
        for rt in chosen_rts:
            for o in ops:
                if o.category not in a["_cats"]:
                    continue
                if o.since_version > a["opset_ceiling"]:
                    continue
                if o.is_control_flow and a["kind"] != "MCU-CPU":
                    continue
                # A few operators are simply missing from any given vendor's
                # kernel library -- the realistic, annoying case.
                if a["kind"] != "MCU-CPU" and rng.random() < 0.18:
                    continue
                kid = _rid("kernel", len(kernels))
                kernels.append({
                    "id": kid,
                    "name": f"{o.name}@{a['name']}/{rt['name']}",
                    "efficiency": round(rng.uniform(0.35, 0.98), 3),
                    "is_fallback": 1 if a["kind"] == "MCU-CPU" else 0,
                    "_op": o.id, "_accel": a["id"], "_rt": rt["id"],
                })
    fleet.add_nodes("Kernel", [{k: v for k, v in k2.items() if not k.startswith("_")}
                               for k2 in kernels])
    for k in kernels:
        fleet.add_edge("Kernel", k["id"], "IMPLEMENTS", "Operator", k["_op"], None)
        fleet.add_edge("Kernel", k["id"], "RUNS_ON", "Accelerator", k["_accel"], None)
        fleet.add_edge("Kernel", k["id"], "PROVIDED_BY", "Runtime", k["_rt"], None)

    # ---------------- Boards ----------------
    boards = []
    for i in range(n(120)):
        soc = rng.choice(socs)
        vendor = rng.choice(vendors)
        power = rng.choice([15, 30, 60, 120, 250, 500, 1200, 2500, 5000])
        boards.append({
            "id": _rid("board", i),
            "name": f"{vendor['name'].split()[0][:2].upper()}{rng.choice('XKNVR')}-{200 + i}",
            "form_factor": rng.choice(FORM_FACTORS),
            "price_usd": round(rng.uniform(9, 420), 2),
            "power_budget_mw": power,
            "ram_kb": rng.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]),
            "flash_kb": rng.choice([256, 512, 1024, 2048, 4096, 8192, 16384, 65536]),
            "year": rng.choice([2022, 2023, 2023, 2024, 2024, 2025, 2026]),
            "battery_powered": 1 if power <= 250 else 0,
            "_soc": soc["id"], "_vendor": vendor["id"],
        })
    fleet.add_nodes("Board", [{k: v for k, v in b.items() if not k.startswith("_")}
                              for b in boards])
    for b in boards:
        fleet.add_edge("Board", b["id"], "HAS_SOC", "SoC", b["_soc"], None)
        fleet.add_edge("Board", b["id"], "MADE_BY", "Vendor", b["_vendor"], None)
        for c in rng.sample(certs, rng.choice([0, 1, 1, 2])):
            fleet.add_edge("Board", b["id"], "CERTIFIED_FOR", "Certification", c["id"], None)

    # ---------------- Clinical tasks / sensors / pipelines ----------------
    tasks = []
    for i, (nm, cat, mods, budget, sens) in enumerate(CLINICAL_TASKS):
        tid = _rid("task", i)
        tasks.append({"id": tid, "name": nm, "category": cat,
                      "latency_budget_ms": budget, "min_sensitivity": sens,
                      "_mods": mods})
        for s in sensors:
            if s["modality"] in mods:
                fleet.add_edge("ClinicalTask", tid, "REQUIRES_SENSOR", "Sensor", s["id"], None)
        for c in rng.sample(certs, rng.choice([1, 1, 2])):
            fleet.add_edge("ClinicalTask", tid, "GOVERNED_BY", "Certification", c["id"], None)
    fleet.add_nodes("ClinicalTask", [{k: v for k, v in t.items() if not k.startswith("_")}
                                     for t in tasks])

    # Sensor -> stage chains
    for s in sensors:
        chain = rng.sample(stages, rng.choice([3, 4, 4, 5]))
        fleet.add_edge("Sensor", s["id"], "FEEDS", "SignalStage", chain[0]["id"], None)
        for a, b in zip(chain, chain[1:]):
            fleet.add_edge("SignalStage", a["id"], "NEXT_STAGE", "SignalStage", b["id"], None)
        s["_chain"] = chain

    # ---------------- Models + variants ----------------
    models, variants = [], []
    # Never generate fewer models than clinical tasks, so every task is
    # covered at any scale factor and task-anchored queries stay non-empty.
    model_total = max(n(60), len(tasks))
    for i in range(model_total):
        fam, cats, mac_rng = rng.choice(MODEL_FAMILIES)
        # Cover every clinical task at least once before assigning at random,
        # so task-anchored queries are never empty on a small scale factor.
        task = tasks[i] if i < len(tasks) else rng.choice(tasks)
        macs_m = round(rng.uniform(*mac_rng), 2)
        params_k = round(macs_m * rng.uniform(8, 60), 1)
        mid = _rid("model", i)
        models.append({"id": mid, "name": f"{fam}-{task['category']}-{i:03d}",
                       "family": fam, "task": task["name"], "params_k": params_k,
                       "macs_m": macs_m, "_cats": cats, "_task": task["id"]})

        # Operators this model uses, drawn from its family's categories.
        used = []
        for cat in cats:
            pool = ops_by_cat.get(cat, [])
            if not pool:
                continue
            for o in rng.sample(pool, min(len(pool), rng.choice([2, 3, 3, 4]))):
                used.append(o)
        for o in dict.fromkeys(used):
            fleet.add_edge("Model", mid, "USES_OPERATOR", "Operator", o.id,
                           {"count": rng.randint(1, 24)})
        models[-1]["_ops"] = list(dict.fromkeys(used))

        fleet.add_edge("Model", mid, "SOLVES", "ClinicalTask", task["id"], None)
        for d in rng.sample(datasets, rng.choice([1, 1, 2])):
            fleet.add_edge("Model", mid, "TRAINED_ON", "Dataset", d["id"], None)

        # Pipeline that feeds this model
        cand_sensors = [s for s in sensors if s["modality"] in task["_mods"]]
        if cand_sensors:
            src = rng.choice(cand_sensors)
            fleet.add_edge("SignalStage", src["_chain"][-1]["id"], "PRECEDES", "Model", mid, None)

        base_acc = rng.uniform(0.78, 0.985)
        fp32_size_kb = params_k * 4.0
        for pname, size_mult, thr_mult, acc_delta in PRECISIONS:
            vid = _rid("variant", len(variants))
            variants.append({
                "id": vid,
                "name": f"{models[-1]['name']}-{pname}",
                "precision": pname,
                "size_kb": round(fp32_size_kb * size_mult, 2),
                "accuracy": round(min(0.999, base_acc + acc_delta), 4),
                "format": rng.choice(["onnx", "tflite", "pte"]),
                "_model": mid, "_thr": thr_mult,
            })
            fleet.add_edge("ModelVariant", vid, "VARIANT_OF", "Model", mid, None)

    fleet.add_nodes("Model", [{k: v for k, v in m.items() if not k.startswith("_")}
                              for m in models])
    fleet.add_nodes("ModelVariant", [{k: v for k, v in v2.items() if not k.startswith("_")}
                                     for v2 in variants])

    # ---------------- Deployments ----------------
    # A Deployment is a measured (variant, board, runtime) triple. Metrics are
    # derived from the cost model in docs/data-provenance.md.
    models_by_id = {m["id"]: m for m in models}
    accel_by_id = {a["id"]: a for a in accelerators}
    boards_by_id = {b["id"]: b for b in boards}
    soc_by_id = {s["id"]: s for s in socs}

    # kernel coverage lookup: (accel_id, runtime_id) -> set(op_id)
    coverage: dict[tuple[str, str], set[str]] = {}
    for k in kernels:
        coverage.setdefault((k["_accel"], k["_rt"]), set()).add(k["_op"])

    deployments = []
    per_variant = max(1, int(round(6 * scale)))
    for v in variants:
        model = models_by_id[v["_model"]]
        for b in rng.sample(boards, min(len(boards), per_variant)):
            accels = accel_by_soc.get(b["_soc"], [])
            if not accels:
                continue
            # Pick the strongest accelerator on the board and a runtime that
            # can target it.
            accel = max(accels, key=lambda a: a["gops_int8"])
            rts = [r for r in runtimes if accel["kind"] in r["_kinds"]
                   and (accel["id"], r["id"]) in coverage]
            if not rts:
                continue
            rt = rng.choice(rts)
            cpu = next((a for a in accels if a["kind"] == "MCU-CPU"), accel)

            covered = coverage.get((accel["id"], rt["id"]), set())
            model_ops = model["_ops"]
            fallback_ops = [o for o in model_ops if o.id not in covered]
            frac_fb = len(fallback_ops) / max(1, len(model_ops))

            # Cost model: MACs split between accelerator and CPU fallback.
            macs = model["macs_m"] * 1e6
            thr = v["_thr"]
            acc_ops = accel["gops_int8"] * 1e9 * thr
            cpu_ops = cpu["gops_int8"] * 1e9 * thr
            t_acc = (macs * 2 * (1 - frac_fb)) / max(acc_ops, 1.0)
            t_cpu = (macs * 2 * frac_fb) / max(cpu_ops, 1.0)
            latency_ms = round((t_acc + t_cpu) * 1000 * rng.uniform(1.05, 1.45), 3)

            energy_mj = ((t_acc * accel["energy_factor"] + t_cpu * cpu["energy_factor"])
                         * b["power_budget_mw"])
            power_mw = round(min(b["power_budget_mw"] * rng.uniform(0.35, 0.98),
                                 b["power_budget_mw"]), 2)
            memory_kb = round(v["size_kb"] * rng.uniform(1.15, 1.9), 2)
            fits = 1 if (memory_kb <= b["ram_kb"] and v["size_kb"] <= b["flash_kb"]) else 0

            did = _rid("deploy", len(deployments))
            deployments.append({
                "id": did,
                "latency_ms": latency_ms,
                "power_mw": power_mw,
                "energy_mj": round(energy_mj, 4),
                "memory_kb": memory_kb,
                "fallback_op_count": len(fallback_ops),
                "fallback_fraction": round(frac_fb, 4),
                "accelerator_kind": accel["kind"],
                "fits": fits,
                "_variant": v["id"], "_board": b["id"], "_rt": rt["id"],
                "_accel": accel["id"],
            })
    fleet.add_nodes("Deployment", [{k: v2 for k, v2 in d.items() if not k.startswith("_")}
                                   for d in deployments])
    for d in deployments:
        fleet.add_edge("Deployment", d["id"], "OF_VARIANT", "ModelVariant", d["_variant"], None)
        fleet.add_edge("Deployment", d["id"], "ON_BOARD", "Board", d["_board"], None)
        fleet.add_edge("Deployment", d["id"], "VIA_RUNTIME", "Runtime", d["_rt"], None)
        fleet.add_edge("Deployment", d["id"], "USES_ACCELERATOR", "Accelerator", d["_accel"], None)

    return fleet


def write(fleet: Fleet, path: Path = FLEET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "seed": fleet.seed, "scale": fleet.scale,
        "node_count": fleet.node_count, "edge_count": fleet.edge_count,
        "nodes": fleet.nodes, "edges": fleet.edges,
    }), encoding="utf-8")
    return path


def load(path: Path = FLEET_PATH) -> Fleet:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python -m etl.download_data` first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fleet = Fleet(seed=payload["seed"], scale=payload["scale"])
    fleet.nodes = payload["nodes"]
    fleet.edges = [tuple(e) for e in payload["edges"]]
    return fleet


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the synthetic edge-AI fleet.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    f = generate(seed=args.seed, scale=args.scale)
    p = write(f)
    print(f"[generate] seed={f.seed} scale={f.scale} "
          f"nodes={f.node_count:,} edges={f.edge_count:,} -> {p}")
    for label, rows in sorted(f.nodes.items(), key=lambda kv: -len(kv[1])):
        print(f"           {label:16s} {len(rows):>7,}")
