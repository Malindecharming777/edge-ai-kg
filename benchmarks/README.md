# Benchmarks

12 queries covering the questions an edge-AI team asks while landing a model on
custom silicon. Each entry in [`queries.py`](queries.py) carries the plain
question, the Cypher, and a `why_graph` note explaining why the question is
awkward in a relational or document store.

```bash
python -m benchmarks.run_benchmark --url http://127.0.0.1:8080   # all 12
python -m benchmarks.run_benchmark --only EA01 --rows 20         # one
python -m benchmarks.run_benchmark --json-out results.json       # machine-readable
```

Omit `--url` to run against an in-process embedded engine.

## Current results

Samyama Graph v1.7.0 OSS, 24,115 nodes / 73,825 edges, server on localhost:

**12/12 queries return rows, 0 empty, 0 failed. Median 26 ms, slowest 117 ms.**

The slow end (EA08 at ~117 ms, EA11 at ~101 ms) are the queries that touch all
21,548 kernels; the anti-joins that anchor on a single model run in 30-40 ms.

## A warning about "it returned rows"

Three of the engine behaviours in [`../docs/engine-notes.md`](../docs/engine-notes.md)
return **wrong rows rather than errors**. EA04 originally returned 12
confident-looking rows that paired one model's fp32 variant with another
model's int8 variant. Treat a green run as necessary, not sufficient —
[`../tests/test_correctness.py`](../tests/test_correctness.py) is what actually
checks the answers.
