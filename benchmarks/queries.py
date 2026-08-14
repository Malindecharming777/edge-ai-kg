"""Benchmark / demo query catalog for the Edge AI deployment KG.

Each entry is a question an edge-AI team actually asks while trying to land a
model on custom silicon. The `why_graph` field records why the question is
awkward in a relational or document store -- most of these are variable-depth
joins across hardware, kernel-library and model structure, or anti-joins
("which operator has NO kernel here").

Engine notes:
  * `NOT (pattern)` does not parse on this build; anti-joins are written as
    `OPTIONAL MATCH ... WITH ... count(x) AS n ... WHERE n = 0`.
  * String literals must be double-quoted.
"""
from __future__ import annotations

QUERIES: list[dict] = [
    {
        "id": "EA01",
        "title": "CPU fallback audit for one model on one accelerator",
        "question": ("Which operators in this model have no kernel on this "
                     "accelerator, and therefore silently fall back to the CPU?"),
        "why_graph": ("The anti-join is over a 3-hop path "
                      "(Model->Operator<-Kernel->Accelerator). In SQL this is a "
                      "NOT EXISTS over a join of four tables, re-written per "
                      "accelerator; here it is one pattern."),
        "cypher": """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.id = "model:00000"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.id = "accel:00001"
WITH op, count(k) AS kernels
WHERE kernels = 0
WITH op.name AS operator, op.category AS category, op.since_version AS opset
RETURN operator, category, opset
ORDER BY category
""",
    },
    {
        "id": "EA02",
        "title": "Fleet-wide operator coverage gaps",
        "question": ("Across the whole accelerator fleet, which operators used "
                     "by our models have the fewest kernel implementations?"),
        "why_graph": ("Ranks a join fan-out; the answer tells you which "
                      "operator to avoid at architecture-design time."),
        "cypher": """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WITH op, count(DISTINCT m) AS models_using
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op)
WITH op, models_using, count(k) AS kernel_count
RETURN op.name AS operator, op.category AS category,
       models_using, kernel_count
ORDER BY kernel_count ASC
LIMIT 15
""",
    },
    {
        "id": "EA03",
        "title": "Boards that meet a clinical task's latency budget",
        "question": ("For a given clinical task, which boards run a variant of "
                     "a solving model inside the task's latency budget?"),
        "why_graph": ("Six hops: Task<-Model<-Variant<-Deployment->Board, with a "
                      "predicate that compares a measured value against a "
                      "budget stored on the task."),
        "cypher": """
MATCH (t:ClinicalTask)<-[:SOLVES]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE t.name = "Fall detection" AND d.fits = 1
  AND d.latency_ms < t.latency_budget_ms
WITH b.name AS board, b.form_factor AS form, b.power_budget_mw AS power_mw,
     v.precision AS precision, d.latency_ms AS latency_ms,
     d.fallback_op_count AS fallback_ops
RETURN board, form, power_mw, precision, latency_ms, fallback_ops
ORDER BY latency_ms ASC
LIMIT 12
""",
    },
    {
        "id": "EA04",
        "title": "What quantization unlocks",
        "question": ("Which board/model pairs do not fit at fp32 but do fit "
                     "once quantized to int8?"),
        "why_graph": ("Self-join across two variants of the same model on the "
                      "same board -- the classic 'what changed' question."),
        # IMPORTANT: written as a single linear pattern + conditional aggregation
        # rather than as a self-join. Both the two-MATCH form AND the
        # comma-separated single-MATCH form return a cartesian product on
        # v1.7.0 at this scale -- see docs/engine-notes.md item 1. The
        # invariant `int8_kb == fp32_kb / 4` makes the breakage detectable, and
        # tests/test_correctness.py asserts it.
        "cypher": """
MATCH (m:Model)<-[:VARIANT_OF]-(v:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)
      -[:ON_BOARD]->(b:Board)
WITH m.name AS model, b.name AS board, b.ram_kb AS board_ram_kb,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb,
     min(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN d.latency_ms ELSE 999999.0 END) AS int8_latency_ms
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, board_ram_kb, fp32_kb, int8_kb, int8_latency_ms
ORDER BY fp32_kb DESC
LIMIT 12
""",
    },
    {
        "id": "EA05",
        "title": "Kernel coverage by accelerator archetype",
        "question": ("How much of the ONNX operator surface does each class of "
                     "accelerator actually implement?"),
        "why_graph": "Aggregation over a 2-hop path, grouped by a node property.",
        "cypher": """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WITH a.kind AS accelerator_kind, count(DISTINCT op) AS operators_covered,
     count(DISTINCT a) AS accelerators
RETURN accelerator_kind, accelerators, operators_covered
ORDER BY operators_covered DESC
""",
    },
    {
        "id": "EA06",
        "title": "Blast radius of losing one kernel",
        "question": ("If a vendor drops the kernel for this operator, which "
                     "deployments regress to CPU fallback?"),
        "why_graph": ("Impact analysis -- reachability from one node out to "
                      "every affected deployment. This is the question that is "
                      "genuinely painful without a graph."),
        "cypher": """
MATCH (op:Operator)<-[:USES_OPERATOR]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE op.name = "Conv" AND d.fits = 1
RETURN op.name AS operator, count(DISTINCT d) AS deployments_at_risk,
       count(DISTINCT b) AS boards_affected, count(DISTINCT m) AS models_affected
""",
    },
    {
        "id": "EA07",
        "title": "End-to-end on-device path: sensor to board",
        "question": ("Trace one complete on-device path: sensor, signal "
                     "pipeline, model, quantized variant, deployment, board."),
        "why_graph": ("A path query. The whole point of the KG -- the physical "
                      "chain from electrode to silicon is a path, not a table."),
        "cypher": """
MATCH (s:Sensor)-[:FEEDS]->(st:SignalStage)-[:NEXT_STAGE*0..3]->(last:SignalStage)
      -[:PRECEDES]->(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE v.precision = "int8" AND d.fits = 1
WITH s.name AS sensor, s.sample_rate_hz AS hz, last.name AS final_stage,
     m.name AS model, b.name AS board, d.latency_ms AS latency_ms
RETURN sensor, hz, final_stage, model, board, latency_ms
ORDER BY latency_ms ASC
LIMIT 10
""",
    },
    {
        "id": "EA08",
        "title": "Best runtime per accelerator",
        "question": ("For each accelerator, which runtime gives the widest "
                     "operator coverage?"),
        "why_graph": ("Groups a 3-way relationship (kernel joins operator, "
                      "accelerator and runtime) that has no natural table."),
        "cypher": """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:PROVIDED_BY]->(r:Runtime)
MATCH (k)-[:IMPLEMENTS]->(op:Operator)
WITH a.kind AS accelerator_kind, r.name AS runtime,
     count(DISTINCT op) AS operators
WITH accelerator_kind, runtime, operators
RETURN accelerator_kind, runtime, operators
ORDER BY operators DESC
LIMIT 20
""",
    },
    {
        "id": "EA09",
        "title": "Battery-powered boards for regulated tasks",
        "question": ("Which battery-powered boards are certified for the same "
                     "standard a regulated clinical task demands?"),
        "why_graph": ("Joins two independent subgraphs (regulatory and "
                      "hardware) through a shared certification node."),
        "cypher": """
MATCH (t:ClinicalTask)-[:GOVERNED_BY]->(c:Certification)<-[:CERTIFIED_FOR]-(b:Board)
WHERE b.battery_powered = 1
WITH c.name AS certification, t.name AS task, b.name AS board,
     b.power_budget_mw AS power_mw, b.form_factor AS form
RETURN certification, task, board, power_mw, form
ORDER BY power_mw ASC
LIMIT 15
""",
    },
    {
        "id": "EA10",
        "title": "Fallback cost distribution by accelerator kind",
        "question": ("How much latency does CPU fallback actually cost, per "
                     "accelerator class?"),
        "why_graph": ("Correlates a structural property (missing kernels) with "
                      "a measured one (latency) in a single pass."),
        "cypher": """
MATCH (d:Deployment)
WHERE d.fits = 1
RETURN d.accelerator_kind AS accelerator_kind,
       count(d) AS deployments,
       avg(d.fallback_fraction) AS avg_fallback_fraction,
       avg(d.latency_ms) AS avg_latency_ms,
       max(d.latency_ms) AS worst_latency_ms
ORDER BY avg_fallback_fraction DESC
""",
    },
    {
        "id": "EA11",
        "title": "Models whose operators no accelerator can fully run",
        "question": ("Which models depend on operators that NO accelerator in "
                     "the fleet implements -- i.e. CPU-only no matter what "
                     "board you pick?"),
        "why_graph": ("Fleet-wide anti-join. Every SoC has a CPU that runs "
                      "everything, so the question that matters is not 'is it "
                      "runnable' but 'is it ever *accelerated*'. Catches the "
                      "architecture mistake before tape-out, not after."),
        "cypher": """
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.kind <> "MCU-CPU"
WITH m, op, count(k) AS accel_kernels
WHERE accel_kernels = 0
RETURN m.name AS model, m.family AS family,
       count(op) AS cpu_only_ops, collect(op.name) AS operators
ORDER BY cpu_only_ops DESC
LIMIT 10
""",
    },
    {
        "id": "EA12",
        "title": "Vendor concentration in feasible deployments",
        "question": ("If we shipped every deployment that fits, how much of "
                     "the fleet would sit on a single silicon vendor?"),
        "why_graph": ("Supply-chain concentration is a 4-hop rollup "
                      "(Deployment->Board->SoC->Vendor)."),
        "cypher": """
MATCH (d:Deployment)-[:ON_BOARD]->(b:Board)-[:HAS_SOC]->(s:SoC)
      -[:MADE_BY]->(vendor:Vendor)
WHERE d.fits = 1
RETURN vendor.name AS vendor, vendor.country AS country,
       count(DISTINCT b) AS boards, count(d) AS deployments
ORDER BY deployments DESC
""",
    },
]

BY_ID = {q["id"]: q for q in QUERIES}
