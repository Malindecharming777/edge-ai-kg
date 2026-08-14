# Edge AI Deployment Knowledge Graph

**24,115 nodes. 73,825 edges. Boards, kernels and neural networks in one graph — so you can ask what actually runs on your silicon.**

> Part of the **Samyama** ecosystem — loaded into and queried via the graph engine at [samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph).
> This repo holds the loader, the generator and the query catalog for the KG.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License"></a>

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
WITH op, count(k) AS kernels
WHERE kernels = 0
RETURN op.name AS operator, op.category AS category, op.since_version AS opset
ORDER BY category, operator
```

```
operator            category       opset
Col2Im              convolution    18
HardSwish           activation     22
PRelu               activation     16
LayerNormalization  normalization  17
RMSNormalization    normalization  23
AveragePool         spatial        22
Resize              spatial        19
...                                          17 rows in 44 ms
```

Seventeen operators on the CPU instead of the NPU — including `AveragePool` and
`LayerNormalization`, which you would have assumed were accelerated.
**One query, no ETL.**
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

**Hardware**: 8 vendors, 40 SoCs, 86 accelerators (MCU-CPU / DSP / NPU-Lite /
NPU-Pro / GPU-Embedded), 120 boards, 7 runtimes.
**Software**: 205 real ONNX operators, 21,548 kernels, 60 models, 240 quantized
variants, 1,440 measured deployments.
**Clinical**: 14 biosignal sensors, 16 DSP stages, 18 clinical tasks, 12
datasets, 6 certifications.

Full detail in [`docs/schema.md`](docs/schema.md).

## Data: what's real, what's synthetic

| | |
|---|---|
| **Real** | The **ONNX operator catalog** — 205 operators with domains and opset versions, parsed from [onnx/onnx](https://github.com/onnx/onnx) (Apache-2.0). |
| **Synthetic** | **Everything else**, generated deterministically from a seed. Vendor, board and SoC names are deliberately fictional (`Corvid Silicon`, `Tessera Labs`, …). |

No number here is a claim about any real product — attaching invented latency
figures to real part numbers would produce a dataset that looks authoritative
and isn't. Deployment metrics are *derived* from a documented cost model rather
than drawn at random, so a board missing a kernel really does pay for it.
Read [`docs/data-provenance.md`](docs/data-provenance.md) before quoting
anything.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m etl.download_data     # fetch ONNX catalog + generate the fleet
python -m demo.demo             # narrated walkthrough, in-process, no server
```

`demo.demo` runs the engine **embedded** — no server, no Docker, nothing to
start. To use a running server instead:

```bash
# from the samyama-graph checkout
./target/release/samyama --http-port 8080

python -m etl.loader --url http://127.0.0.1:8080        # ~25s for 24K/74K
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080
python -m mcp_server.server                             # expose over MCP
pytest                                                  # 33 tests
```

Scale the fleet with `--scale` (`1.0` ≈ 24K nodes) and change the world with
`--seed`. Same seed, same graph, every time.

## The query catalog

12 queries in [`benchmarks/queries.py`](benchmarks/queries.py), each recording
the question it answers and why it's awkward without a graph. All 12 return
rows; median 26 ms, slowest 117 ms.

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

## Engine notes

This KG is built against **Samyama Graph v1.7.0 (OSS)**. Building it surfaced
several engine behaviours that the loader and queries work around — including
two that **silently return wrong rows** rather than erroring:

- a bound variable re-used in a second `MATCH` clause is not always joined, producing a cartesian product;
- `RETURN DISTINCT` is a no-op;
- `min()` mis-compares an int sentinel against float values;
- negated pattern predicates and `CREATE CONSTRAINT` don't parse;
- the tenant/graph argument is ignored on the OSS HTTP path.

Each is documented with a minimal reproduction and the workaround used in
[`docs/engine-notes.md`](docs/engine-notes.md). Because of these,
[`tests/test_correctness.py`](tests/test_correctness.py) validates query
**results** against ground truth computed in Python — a query that runs and
returns plausible rows is not evidence that it is right.

## Structure

```
etl/          onnx_catalog.py (real data) + generate.py (synthetic) + loader.py
schema/       edge_ai_kg.cypher — indexes and documented relationship shapes
benchmarks/   the 12-query catalog + runner
mcp_server/   7 MCP tools shaped around deployment questions
demo/         narrated walkthrough
docs/         schema, data provenance, engine notes
tests/        33 tests: parsing, fleet invariants, query correctness
```

## License

Apache-2.0. The ONNX operator catalog is Apache-2.0 from the ONNX project;
everything else is generated.
