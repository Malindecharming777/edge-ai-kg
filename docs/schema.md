# Edge AI KG -- schema

15 node labels, 21 edge types. At `--scale 1.0`: **24,115 nodes, 73,825 edges**
(seed `20260814`). See [`data-provenance.md`](data-provenance.md) for what is
real and what is synthetic, and [`engine-notes.md`](engine-notes.md) for the
v1.7.0 behaviours the queries work around.

The graph answers one shape of question: **can this model run on this silicon,
and what does it cost me when it can't?**

## Node labels

| Label | Count | Key fields |
|---|---:|---|
| `Kernel` | 21,844 | id, name, efficiency, is_fallback |
| `Deployment` | 1,440 | id, latency_ms, power_mw, energy_mj, memory_kb, fallback_op_count, fallback_fraction, accelerator_kind, fits |
| `ModelVariant` | 240 | id, name, precision, size_kb, accuracy, format |
| `Operator` | 205 | id, name, domain, since_version, version_count, category, is_control_flow |
| `Board` | 120 | id, name, form_factor, price_usd, power_budget_mw, ram_kb, flash_kb, year, battery_powered |
| `Accelerator` | 85 | id, name, kind, gops_int8, sram_kb, clock_mhz, opset_ceiling, energy_factor |
| `Model` | 60 | id, name, family, task, params_k, macs_m |
| `SoC` | 40 | id, name, process_nm, cpu_arch, cpu_mhz, cores |
| `ClinicalTask` | 18 | id, name, category, latency_budget_ms, min_sensitivity |
| `SignalStage` | 16 | id, name, kind, window_ms, cost_kmacs |
| `Sensor` | 14 | id, name, modality, sample_rate_hz, channels, adc_bits |
| `Dataset` | 12 | id, name, source, subjects, hours, license |
| `Vendor` | 8 | id, name, country |
| `Runtime` | 7 | id, name, version, format |
| `Certification` | 6 | id, name, body, class |

Every node carries a unique `id`; `schema/edge_ai_kg.cypher` indexes it per
label. Uniqueness is a loader invariant -- this engine does not parse
`CREATE CONSTRAINT`.

## Edge types

| Edge | From -> To | Count | Meaning |
|---|---|---:|---|
| `IMPLEMENTS` | Kernel -> Operator | 21,844 | this kernel implements this operator |
| `RUNS_ON` | Kernel -> Accelerator | 21,844 | on this compute unit |
| `PROVIDED_BY` | Kernel -> Runtime | 21,844 | shipped by this runtime |
| `OF_VARIANT` | Deployment -> ModelVariant | 1,440 | what was deployed |
| `ON_BOARD` | Deployment -> Board | 1,440 | where |
| `VIA_RUNTIME` | Deployment -> Runtime | 1,440 | through which runtime |
| `USES_ACCELERATOR` | Deployment -> Accelerator | 1,440 | on which compute unit |
| `USES_OPERATOR` | Model -> Operator `{count}` | 1,069 | model's operator surface |
| `TARGETS` | Runtime -> Accelerator | 426 | runtime can target this unit |
| `VARIANT_OF` | ModelVariant -> Model | 240 | fp32 / fp16 / int8 / int4 |
| `MADE_BY` | Board\|SoC -> Vendor | 160 | supply chain |
| `HAS_SOC` | Board -> SoC | 120 | board's chip |
| `CERTIFIED_FOR` | Board -> Certification | 104 | regulatory posture |
| `HAS_ACCELERATOR` | SoC -> Accelerator | 85 | chip's compute units |
| `TRAINED_ON` | Model -> Dataset | 82 | provenance |
| `SOLVES` | Model -> ClinicalTask | 60 | clinical purpose |
| `PRECEDES` | SignalStage -> Model | 60 | pipeline feeds model |
| `REQUIRES_SENSOR` | ClinicalTask -> Sensor | 51 | required modality |
| `NEXT_STAGE` | SignalStage -> SignalStage | 40 | DSP chain |
| `GOVERNED_BY` | ClinicalTask -> Certification | 22 | regulatory requirement |
| `FEEDS` | Sensor -> SignalStage | 14 | front of the pipeline |

## The two spines

**Hardware spine** -- what silicon can do:

```
Vendor <- SoC <- Board
           |
           +-> Accelerator <- RUNS_ON - Kernel - IMPLEMENTS -> Operator
                    ^                      |
                    |                 PROVIDED_BY
                 TARGETS                   v
                    +-------------------- Runtime
```

**Clinical spine** -- what the device has to do:

```
Sensor -> SignalStage -> ... -> SignalStage -> Model -> ClinicalTask
                                                |            |
                                          ModelVariant   Certification
                                                |
                                           Deployment -> Board
```

The two spines meet at `Operator` (a model's operator surface vs. a kernel
library's coverage) and at `Deployment` (a measured landing of one variant on
one board). Those two joins are where every interesting query lives.

## Accelerator archetypes

`kind` drives which operator categories a unit can run and its opset ceiling --
this is what creates the coverage gaps the queries hunt for.

| Kind | Opset ceiling | Covers | int8 GOPS | Energy factor |
|---|---:|---|---|---:|
| `MCU-CPU` | 99 | everything (universal fallback) | 0.5-3 | 1.00 |
| `DSP` | 17 | signal, elementwise, conv, matmul, spatial | 8-40 | 0.42 |
| `NPU-Lite` | 13 | conv, matmul, activation, spatial, quant | 30-120 | 0.16 |
| `NPU-Pro` | 19 | + reduction, attention, shape | 150-900 | 0.11 |
| `GPU-Embedded` | 21 | + recurrent, tensor | 400-2400 | 0.30 |

Every SoC has an `MCU-CPU`, so *something* can always run -- the question the
graph answers is never "can it run" but "is it ever accelerated, and what does
the fallback cost".
