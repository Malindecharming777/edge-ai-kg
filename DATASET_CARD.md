---
license: apache-2.0
pretty_name: Edge AI Deployment Knowledge Graph
tags:
  - knowledge-graph
  - edge-ai
  - tinyml
  - onnx
  - hardware
  - model-deployment
  - quantization
  - biosignal
  - synthetic
language:
  - en
size_categories:
  - 10K<n<100K
---

# Dataset Card for `edge-ai-kg`

A property-graph dataset of **edge-AI deployment**: what neural-network
operators a model needs, which kernels a given accelerator actually implements,
and what it costs when those two sets do not line up.

> ⚠️ **This dataset is part real, part synthetic, and the two must not be
> conflated.** Every node carries a `provenance` property (`"real"` |
> `"synthetic"`) and a `source`. Read
> [Real vs. synthetic](#real-vs-synthetic) before using or quoting anything.

**25,145 nodes · 76,291 edges · 16 node labels · 22 edge types**
**1,030 real nodes from 3 public sources · 24,115 generated**

---

## Dataset Summary

Deploying a model onto custom edge silicon fails in a specific, unglamorous
way: one operator has no kernel on the NPU, silently falls back to the CPU, and
the latency budget is gone. Answering *which* operator, on *which* board, under
*which* runtime is a multi-hop join across three normally-disconnected domains —
a hardware fleet, a kernel library, and a model's operator surface.

This dataset connects those three domains into a single graph so those
questions are traversals rather than scripts:

- **Hardware spine** — `Vendor → SoC → Accelerator`, `Board`, `Runtime`
- **Model spine** — `Model → Operator`, `Kernel → {Operator, Accelerator, Runtime}`, `ModelVariant`
- **Clinical spine** — `Sensor → SignalStage → Model → ClinicalTask`, `Dataset`, `Certification`

The spines meet at **`Operator`** (a model's operator surface versus a kernel
library's coverage) and **`Deployment`** (one variant landed on one board).

## Supported Tasks

| Task | Description |
|---|---|
| Graph query benchmarking | 12-query catalog with recorded questions, timings and expected shapes |
| Multi-hop retrieval / GraphRAG | Dense, typed, semantically meaningful multi-hop paths over a technical domain |
| Anti-join / negation evaluation | "Which operator has *no* kernel here" — coverage-gap reasoning |
| Impact analysis | Blast radius of removing a single node (a dropped kernel) |
| Constraint satisfaction | Feasibility under joint latency / power / memory / certification constraints |
| Text-to-Cypher / NL query | Natural questions with unambiguous graph answers |

## Languages

English (identifiers, names and labels).

---

## Real vs. synthetic

This is the most important section of this card.

### Real — three public sources

| Source | License | Contributes | Real facts |
|---|---|---|---|
| [onnx/onnx](https://github.com/onnx/onnx) `docs/Operators.md` | Apache-2.0 | 205 `Operator` | name, domain, opset version, revision count |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) `docs/OperatorKernels.md` | MIT | 734 `Kernel`, 3 `Accelerator`, 170 extra `Operator` | which operator each execution provider implements, at which opset range, over how many tensor types |
| [mlcommons/tiny_results_v1.2](https://github.com/mlcommons/tiny_results_v1.2) `summary.csv` | Apache-2.0 | 73 `Deployment`, 14 `Board`, 12 `SoC`, 7 `Vendor`, 5 `Runtime`, 3 `Accelerator`, 4 `BenchmarkTask`, 4 `Model` | measured throughput, accuracy, energy per inference on real commercial hardware |

**ONNX Runtime kernel coverage** (verifiable against the upstream doc):

| Execution provider | Registrations | Distinct operators |
|---|---:|---:|
| `CPUExecutionProvider` | 293 | 293 |
| `CUDAExecutionProvider` | 236 | 236 |
| `DmlExecutionProvider` | 205 | 205 |

**MLPerf Tiny v1.2** covers 7 organisations (Qualcomm, Renesas,
STMicroelectronics, Syntiant, Bosch, Skymizer, and one independent submitter)
across 14 boards and the four TinyML tasks — anomaly detection, image
classification, keyword spotting, visual wake words. 18 of the 73 submissions
include measured energy.

The only fields we add on top of these are `category` and `is_control_flow` on
`Operator` — a coarse grouping defined in `etl/onnx_catalog.py`, **not** part of
the ONNX standard.

### Synthetic

**Everything else**, generated deterministically by `etl/generate.py`.

Vendor, board and SoC names (`Corvid Silicon`, `Nimbus Micro`, `Tessera Labs`,
`Akshara Semi`, `Halcyon Devices`, `Vermilion Systems`, `Kestrel Embedded`,
`Suvarna Microsystems`) are **deliberately fictional**. This is a design
constraint, not an oversight: attaching invented latency and power figures to
real part numbers would produce a dataset that looks authoritative and is not.

**Names that are real, attached to synthetic facts:**

- **Runtimes** — TFLite Micro, ONNX Runtime, ExecuTorch, TVM microTVM,
  CMSIS-NN, OpenVINO, "Vendor SDK". Real projects; their operator coverage,
  versions and accelerator support here are invented.
- **Datasets** — MIT-BIH, PTB-XL, CHB-MIT, Sleep-EDF, TUH EEG, MobiAct,
  UCI HAR, Daphnet FoG, ICBHI, Coswara, BIDMC. Real corpora, and the
  `subjects` / `hours` / `license` fields are approximately right — but
  **no model here was trained on them and no accuracy figure was measured on
  them.** The `TRAINED_ON` edges are fabricated.
- **Certifications** — IEC 62304, ISO 13485, FDA 510(k), EU MDR are real
  regimes; which synthetic board holds which is invented.

## How the numbers were produced

`Deployment` metrics are **derived**, not sampled, so the graph stays
internally consistent — a board missing a kernel genuinely pays for it:

1. A model's operators are split into those the target accelerator has a kernel
   for and those it does not; the latter share is `fallback_fraction`.
2. Work is `2 × MACs`, split by that share between accelerator and CPU.
3. Each unit runs at `gops_int8 × precision_multiplier`
   (fp16 1.7×, int8 3.2×, int4 4.8× relative to fp32).
4. `latency_ms` is the sum, scaled by a 1.05–1.45 overhead factor.
5. `energy_mj` weights each unit's time by its archetype's `energy_factor`
   (NPU-Pro 0.11, NPU-Lite 0.16, GPU-Embedded 0.30, DSP 0.42, MCU-CPU 1.00).
6. `fits` is true when the working set fits both board RAM and flash.

**What the model ignores:** memory bandwidth, DMA setup, layer fusion, cache
behaviour, thermal throttling, and per-kernel quality beyond one `efficiency`
scalar. It is good enough to make *structural* questions behave sensibly. It is
**not a performance simulator.**

---

## Dataset Structure

### Node labels

| Label | Count | Fields |
|---|---:|---|
| `Kernel` | 22,578 | id, name, efficiency, is_fallback |
| `Deployment` | 1,513 | id, latency_ms, power_mw, energy_mj, memory_kb, fallback_op_count, fallback_fraction, accelerator_kind, fits |
| `ModelVariant` | 240 | id, name, precision, size_kb, accuracy, format |
| `Operator` | 375 | id, name, domain, since_version, version_count, category, is_control_flow |
| `Board` | 134 | id, name, form_factor, price_usd, power_budget_mw, ram_kb, flash_kb, year, battery_powered |
| `Accelerator` | 91 | id, name, kind, gops_int8, sram_kb, clock_mhz, opset_ceiling, energy_factor |
| `Model` | 64 | id, name, family, task, params_k, macs_m |
| `SoC` | 52 | id, name, process_nm, cpu_arch, cpu_mhz, cores |
| `ClinicalTask` | 18 | id, name, category, latency_budget_ms, min_sensitivity |
| `SignalStage` | 16 | id, name, kind, window_ms, cost_kmacs |
| `Sensor` | 14 | id, name, modality, sample_rate_hz, channels, adc_bits |
| `Dataset` | 12 | id, name, source, subjects, hours, license |
| `Vendor` | 15 | id, name, country |
| `Runtime` | 13 | id, name, version, format |
| `Certification` | 6 | id, name, body, class |
| `BenchmarkTask` | 4 | id, name, code, dataset, metric, quality_target |

Every node has a globally unique `id` of the form `<prefix>:<5-digit>`.

### Edge types

| Edge | From → To | Count |
|---|---|---:|
| `IMPLEMENTS` | Kernel → Operator | 22,578 |
| `RUNS_ON` | Kernel → Accelerator | 22,578 |
| `PROVIDED_BY` | Kernel → Runtime | 22,578 |
| `OF_VARIANT` | Deployment → ModelVariant | 1,440 |
| `ON_BOARD` | Deployment → Board | 1,513 |
| `VIA_RUNTIME` | Deployment → Runtime | 1,500 |
| `USES_ACCELERATOR` | Deployment → Accelerator | 1,451 |
| `USES_OPERATOR` | Model → Operator `{count}`, SignalStage → Operator | 1,069 |
| `TARGETS` | Runtime → Accelerator | 429 |
| `VARIANT_OF` | ModelVariant → Model | 240 |
| `MADE_BY` | SoC → Vendor, Board → Vendor | 186 |
| `HAS_SOC` | Board → SoC | 134 |
| `CERTIFIED_FOR` | Board → Certification | 104 |
| `HAS_ACCELERATOR` | SoC → Accelerator | 85 |
| `TRAINED_ON` | Model → Dataset | 82 |
| `SOLVES` | Model → ClinicalTask \| BenchmarkTask | 64 |
| `PRECEDES` | SignalStage → Model | 60 |
| `REQUIRES_SENSOR` | ClinicalTask → Sensor | 51 |
| `NEXT_STAGE` | SignalStage → SignalStage | 40 |
| `GOVERNED_BY` | ClinicalTask → Certification | 22 |
| `FEEDS` | Sensor → SignalStage | 14 |

### Accelerator archetypes

`kind` determines which operator categories a unit can run and its opset
ceiling. **These constraints are what create the coverage gaps** the dataset
exists to expose.

| Kind | Opset ceiling | Covers | int8 GOPS | Energy factor |
|---|---:|---|---|---:|
| `MCU-CPU` | 99 | every category (universal fallback) | 0.5–3 | 1.00 |
| `DSP` | 17 | signal, elementwise, conv, matmul, spatial | 8–40 | 0.42 |
| `NPU-Lite` | 13 | conv, matmul, activation, spatial, quantization | 30–120 | 0.16 |
| `NPU-Pro` | 19 | + reduction, attention, shape | 150–900 | 0.11 |
| `GPU-Embedded` | 21 | + recurrent, tensor | 400–2400 | 0.30 |

Every SoC carries an `MCU-CPU`, so *something* can always run. The question the
graph answers is never "can it run" but **"is it ever accelerated, and what does
the fallback cost".**

### Files

Both artifacts are regenerable and therefore gitignored; `data/` is rebuilt with
one command.

| Path | Contents |
|---|---|
| `data/onnx/Operators.md` | Upstream ONNX operator document (cached verbatim) |
| `data/onnx/operators.json` | Parsed operator catalog + provenance |
| `data/onnxruntime/OperatorKernels.md` | Upstream ONNX Runtime kernel document |
| `data/onnxruntime/kernels.json` | Parsed kernel registrations + provenance |
| `data/mlperf-tiny/summary.csv` | Upstream MLPerf Tiny v1.2 summary |
| `data/mlperf-tiny/results.json` | Parsed submissions + provenance |
| `data/fleet/fleet.json` | Generated nodes and edges |

### Prebuilt snapshot

A `.sgsnap` of the full graph (~970 KB gzipped) is published at
[`samyama-graph` releases, `kg-snapshots-v9`](https://github.com/samyama-ai/samyama-graph/releases/tag/kg-snapshots-v9)
and imports in under a second. All 16 catalog queries are verified against the
imported snapshot.

---

## Dataset Creation

### Curation rationale

Built to make the structural questions in edge-AI deployment answerable in one
query. Those questions need **dense, complete coverage** of a hardware fleet
crossed with a kernel library — no public dataset provides that, and it cannot
be honestly assembled from real product names. So the fleet is synthetic and
the questions are real.

### Generation

```bash
python -m etl.download_data --seed 20260814 --scale 1.0
```

`--seed` fully determines the **synthetic** layer: same seed and scale reproduce
the same nodes, edges and ids byte-for-byte. `--scale` multiplies fleet size
(`1.0` ≈ 24.1K synthetic nodes). The generator guarantees that every
`ClinicalTask` has at least one `Model` at any scale.

The **real** layer is not scaled or seeded — it is whatever the upstream
sources say. It is therefore reproducible only up to the upstream state at
fetch time: `onnx/onnx` and `microsoft/onnxruntime` are fetched from `main` and
will drift, while `mlcommons/tiny_results_v1.2` is a frozen published round and
will not. Cached copies live under `data/` so a given build is re-loadable
even after upstream moves.

Load the layers independently:

```bash
python -m etl.loader --layers real        # 1,235 nodes / 2,466 edges, all real
python -m etl.loader --layers synthetic   # generated fleet only
python -m etl.loader                      # both (default)
```

### Annotations

None. There are no human labels or judgements in this dataset.

### Personal and sensitive information

**None.** No personal data, no patient records, no real device telemetry. The
biosignal modalities (ECG, EEG, PPG, EMG) are schema-level concepts only —
**this dataset contains no physiological recordings whatsoever.**

---

## Considerations for Using the Data

### Intended uses

- Benchmarking graph engines on multi-hop joins, anti-joins and aggregations
- Developing and demonstrating deployment-feasibility tooling
- Teaching graph modelling of a hardware/software/regulatory domain
- Text-to-Cypher and GraphRAG evaluation where ground truth is checkable

### Out-of-scope uses

**Filter on `provenance` first.** Almost every restriction below applies to the
synthetic layer only; the real layer is citable within the terms of its
upstream sources.

**Do not** use the **synthetic** layer to:

- **select hardware, or estimate real latency, power or cost** — the numbers are generated;
- **train a model that predicts inference performance** — it would learn this cost model, not physics;
- **compare real vendors, SoCs or NPUs** — those vendors do not exist;
- **make clinical, safety or regulatory claims** — `ClinicalTask` sensitivity thresholds and `Certification` mappings are illustrative;
- **cite accuracy figures against MIT-BIH, PTB-XL, CHB-MIT or any named corpus** — no model here was trained or evaluated on them.

**For the real layer**, the honest caveats are narrower but still real:

- MLPerf Tiny results are **closed-division submissions under specific
  conditions**; cite them as MLPerf results, with the round (v1.2), not as
  general device benchmarks. MLCommons owns the MLPerf trademark and has its
  own rules for citing results — follow those, not this card.
- ONNX Runtime kernel coverage is a **snapshot of `main`** at fetch time, not a
  release. It moves. Re-fetch before drawing conclusions.
- Absence of a kernel in the table does not always mean an operator cannot run —
  ORT can decompose or fall back in ways the registration table does not show.

### Known limitations

1. **Heavily skewed to `Kernel`** — 22,578 of 25,145 nodes (90%) are kernels, and 3 edge types carry 89% of edges. Realistic (kernel libraries *are* the bulk), but it means whole-graph statistics are dominated by one label.
2. **The cost model is the ground truth**, so any model trained on it recovers the model, not reality.
3. **Operator categories are heuristic** — regex over operator names with a short override table; some assignments are debatable.
4. **Uniform random structure** — real fleets cluster (vendors reuse IP, boards share SoC families). Sampling here is close to uniform, so the graph has less community structure than a real one.
5. **No temporal dimension** — `Board.year` exists but nothing evolves; there is no kernel-library version history.
6. **Modest scale** — 24K nodes at `scale=1.0`. Use `--scale` for larger, but the topology stays statistically the same.

### Biases

The domain framing carries choices worth naming: the clinical tasks are
weighted toward cardiac and neuro applications; sensor modalities reflect
wearable and bedside monitoring rather than imaging; certifications are
IEC/ISO/FDA/EU only, with no other national regime represented. Vendor
countries were assigned for flavour and carry no meaning.

---

## Validation

Because the engine this was built against has query-shape bugs that return
**plausible but wrong rows**, `tests/test_correctness.py` validates query
*results* against ground truth recomputed in Python — not merely that queries
run. 50 tests cover catalog parsing, fleet invariants (id uniqueness, edge
endpoint integrity, `int8 == fp32 / 4`, opset ceilings respected, no internal
fields leaking into properties) and query correctness.

Two tests exist purely as bug canaries: an fp32/int8 size-ratio check that
detects cartesian products, and one asserting every catalog query uses a single
sort key *and* actually returns sorted rows. See `docs/engine-notes.md`.

---

## Licensing

- **This dataset and its generator**: Apache-2.0.
- **ONNX operator catalog**: Apache-2.0, © the ONNX project contributors.
- **ONNX Runtime kernel registrations**: MIT, © Microsoft Corporation.
- **MLPerf Tiny v1.2 results**: Apache-2.0, © MLCommons. "MLPerf" is a trademark of MLCommons; results are reproduced from the public v1.2 closed-division summary and should be cited per MLCommons' own policy.
- Upstream source documents are cached under `data/` at build time rather than vendored into the repo.
- Real project and corpus **names** appear as labels under nominative use. No affiliation with or endorsement by any named project, vendor or standards body is implied.

## Citation

```bibtex
@misc{edge_ai_kg_2026,
  title  = {Edge AI Deployment Knowledge Graph},
  author = {Samyama},
  year   = {2026},
  note   = {Synthetic edge-AI hardware/kernel/model graph with a real ONNX
            operator catalog. Generated with seed 20260814.},
  howpublished = {\url{https://git.samyama.ai/Samyama.ai/edge-ai-kg}}
}
```

## Contact

Samyama — https://git.samyama.ai/Samyama.ai/edge-ai-kg
