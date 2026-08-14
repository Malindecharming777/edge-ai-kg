# Edge AI Deployment Knowledge Graph

**25,145 nodes. 76,291 edges. Boards, kernels and neural networks in one graph — so you can ask what actually runs on your silicon.**

Real ONNX + ONNX Runtime + MLPerf Tiny data, plus a generated fleet for scale. Every node is stamped `real` or `synthetic`.

> Part of the **Samyama** ecosystem — loaded into and queried via the graph engine at [samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph).
> This repo holds the loader, the generator and the query catalog for the KG.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License"></a>

![Edge AI KG — 16 questions answered](demo/edgeai-questions.gif)

*All 16 [catalog queries](benchmarks/queries.py) run end to end — each question, the Cypher it becomes, and the answer. The last four run on real ONNX Runtime and MLPerf Tiny data. Long-form: the whole run in one image, nothing scrolled off. Re-record with [`scripts/record_gif.sh`](scripts/record_gif.sh).*

---

## The question this exists to answer

Deploying a model onto custom edge hardware fails in a specific, boring way:
**one operator has no kernel on your NPU, silently falls back to the CPU, and
your latency budget is gone.** Finding out which operator, on which board,
under which runtime, is a graph traversal — a model's operator surface joined
against a kernel library joined against a hardware fleet.

```cypher
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.name = "depthwise-cnn-neuro-043"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.kind = "NPU-Lite"
WITH op.name AS operator, op.category AS category,
     op.since_version AS opset, count(k) AS kernels
WHERE kernels = 0
RETURN operator, category, opset
ORDER BY category
```

```
operator            category       opset
PRelu               activation     16
HardSwish           activation     22
ThresholdedRelu     activation     22
Col2Im              convolution    18
Cos                 elementwise    22
LayerNormalization  normalization  17
RMSNormalization    normalization  23
ReduceL1            reduction      18
...                                          17 rows in 24 ms
```

Seventeen operators on the CPU instead of the NPU — including `PRelu` and
`LayerNormalization`, which you would have assumed were accelerated.
**One query, no ETL.**

(The projection goes through `WITH` and sorts on a single key deliberately —
see [engine notes](docs/engine-notes.md).)
Flatten this into JSON and it becomes a script you maintain forever.

---

## What is in it

Two spines that meet in the middle:

```
Vendor <- SoC <- Board                        Sensor -> SignalStage -> ... -> Model
           |                                                                   |
           +-> Accelerator <- Kernel -> Operator <---- USES_OPERATOR -----------+
                    ^            |                                             |
                 TARGETS    PROVIDED_BY                                  ModelVariant
                    +--------- Runtime                                         |
                                                                          Deployment -> Board
```

**Hardware**: 15 vendors, 52 SoCs, 91 accelerators (MCU-CPU / DSP / NPU-Lite /
NPU-Pro / GPU-Embedded / real CPU+CUDA+DirectML), 134 boards, 13 runtimes.
**Software**: 375 operators, 22,578 kernels, 64 models, 240 quantized variants,
1,513 deployments (73 of them real MLPerf Tiny measurements).
**Clinical**: 14 biosignal sensors, 16 DSP stages, 18 clinical tasks, 4 MLPerf
benchmark tasks, 12 datasets, 6 certifications.

Full detail in [`docs/schema.md`](docs/schema.md).

## Data: what's real, what's synthetic

| Source | License | What it contributes |
|---|---|---|
| [onnx/onnx](https://github.com/onnx/onnx) | Apache-2.0 | **205 real operators** — names, domains, opset versions |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | **734 real kernel registrations** across CPU / CUDA / DirectML execution providers |
| [mlcommons/tiny_results_v1.2](https://github.com/mlcommons/tiny_results_v1.2) | Apache-2.0 | **73 measured submissions** — real boards from Qualcomm, Renesas, ST, Syntiant, Bosch, with real throughput, accuracy and energy |
| generated | — | **The fleet**: 120 boards, 85 accelerators, 21,844 kernels, 1,440 deployments. Vendor and board names deliberately fictional (`Corvid Silicon`, `Tessera Labs`, …) |

**1,030 nodes are real; 24,115 are generated.** The split is queryable, not just
documented — every node carries `provenance` and `source`:

```bash
python -m etl.loader --layers real        # public-source subgraph only
```

No **generated** number is a claim about any real product — attaching invented
latency figures to real part numbers would produce a dataset that looks
authoritative and isn't. The real layer, by contrast, is checkable line by line
against its upstream sources, and `tests/test_real_layer.py` does exactly that. Deployment metrics are *derived* from a documented cost model rather
than drawn at random, so a board missing a kernel really does pay for it.
Read the **[dataset card](DATASET_CARD.md)** — which covers intended and
out-of-scope uses, known limitations and biases — plus
[`docs/data-provenance.md`](docs/data-provenance.md) for the cost model, before
quoting anything.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m etl.download_data     # fetch 3 public sources + generate the fleet
python -m demo.demo             # narrated walkthrough, in-process, no server
```

`demo.demo` runs the engine **embedded** — no server, no Docker, nothing to
start. To use a running server instead:

```bash
# from the samyama-graph checkout
./target/release/samyama --http-port 8080

python -m etl.loader --url http://127.0.0.1:8080        # ~13s for 25K/76K
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080
python -m mcp_server.server                             # expose over MCP
pytest                                                  # 50 tests
```

Scale the fleet with `--scale` (`1.0` ≈ 24K nodes) and change the world with
`--seed`. Same seed, same graph, every time.

## Load it without building it

A prebuilt `.sgsnap` snapshot of the full graph (25,145 nodes / 76,291 edges,
both layers, ~970 KB gzipped) is published on the engine repo's releases, so you
can skip the ETL entirely:

```bash
# 1. start the engine (from a samyama-graph checkout)
./target/release/samyama --http-port 8080

# 2. fetch and import the snapshot
curl -fL -o edge-ai-kg.sgsnap \
  https://github.com/samyama-ai/samyama-graph/releases/download/kg-snapshots-v9/edge-ai-kg.sgsnap
curl -X POST -F "file=@edge-ai-kg.sgsnap" http://127.0.0.1:8080/api/snapshot/import

# 3. ask it something
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080
```

Import takes well under a second. All 16 catalog queries are verified to work
against the imported snapshot, not just against a freshly-loaded graph.

Export your own after any change:

```bash
curl -X POST -o edge-ai-kg.sgsnap http://127.0.0.1:8080/api/snapshot/export
```

## The query catalog

16 queries in [`benchmarks/queries.py`](benchmarks/queries.py), each recording
the question it answers and why it's awkward without a graph. All 16 return
rows; median 14 ms, slowest 73 ms. **EA13–EA16 run entirely on real data**, so
their answers can be checked against the upstream sources.

| id | Question |
|---|---|
| EA01 | Which operators fall back to CPU for this model on this accelerator? |
| EA02 | Which operators have the fewest kernels fleet-wide? |
| EA03 | Which boards meet this clinical task's latency budget? |
| EA04 | What does int8 quantization unlock that fp32 can't fit? |
| EA05 | How much of the ONNX surface does each accelerator class cover? |
| EA06 | If a vendor drops one kernel, what breaks? |
| EA07 | Trace electrode → DSP pipeline → model → board |
| EA08 | Which runtime gives the widest coverage per accelerator? |
| EA09 | Which battery-powered boards are certified for regulated tasks? |
| EA10 | What does CPU fallback actually cost in latency? |
| EA11 | Which models are CPU-only no matter which board you pick? |
| EA12 | How concentrated is the fleet on one silicon vendor? |
| **EA13** | **REAL:** which ai.onnx operators does ONNX Runtime implement on CPU but not CUDA? |
| **EA14** | **REAL:** MLPerf Tiny v1.2 throughput leaders per benchmark task |
| **EA15** | **REAL:** which operators are registered on only one execution provider? |
| **EA16** | **REAL vs SYNTHETIC:** what is measured and what is generated |

## Engine notes

This KG is built against **Samyama Graph v1.7.0 (OSS)**. Building it surfaced
several engine behaviours that the loader and queries work around — including
two that **silently return wrong rows** rather than erroring:

- a bound variable re-used in a later `MATCH` is not always joined, producing a cartesian product ([#360](https://github.com/samyama-ai/samyama-graph/issues/360));
- `RETURN DISTINCT` is a no-op ([#361](https://github.com/samyama-ai/samyama-graph/issues/361));
- `ORDER BY` on a `RETURN` alias is dropped, and only the first sort key applies ([#362](https://github.com/samyama-ai/samyama-graph/issues/362));
- aggregating a bare node variable returns N rows of `1` ([#363](https://github.com/samyama-ai/samyama-graph/issues/363));
- `DETACH DELETE` doesn't clear property columns, so deleted values resurrect ([#364](https://github.com/samyama-ai/samyama-graph/issues/364));
- `min()` mis-compares an int sentinel against float values ([#365](https://github.com/samyama-ai/samyama-graph/issues/365));
- the tenant/graph argument is ignored on the OSS HTTP path ([#366](https://github.com/samyama-ai/samyama-graph/issues/366));
- negated pattern predicates and `CREATE CONSTRAINT` don't parse ([#367](https://github.com/samyama-ai/samyama-graph/issues/367)).

All nine are filed upstream — tracking issue [samyama-graph#368](https://github.com/samyama-ai/samyama-graph/issues/368).

Each is documented with a minimal reproduction and the workaround used in
[`docs/engine-notes.md`](docs/engine-notes.md). Because of these,
[`tests/test_correctness.py`](tests/test_correctness.py) validates query
**results** against ground truth computed in Python — a query that runs and
returns plausible rows is not evidence that it is right.

## Structure

```
etl/          onnx_catalog.py, ort_kernels.py, mlperf_tiny.py, real_layer.py (real)
              generate.py (synthetic) + loader.py
schema/       edge_ai_kg.cypher — indexes and documented relationship shapes
benchmarks/   the 12-query catalog + runner
mcp_server/   7 MCP tools shaped around deployment questions
demo/         two walkthroughs (question-driven + 6-beat story) + recorded gif
scripts/      record_gif.sh — long-form demo recording
docs/         schema, data provenance, engine notes
DATASET_CARD.md  HF-style card: structure, provenance, intended + out-of-scope uses
tests/        50 tests: parsing, fleet + real-layer invariants, query correctness
```

## License

Apache-2.0. The ONNX operator catalog is Apache-2.0 from the ONNX project;
everything else is generated.
