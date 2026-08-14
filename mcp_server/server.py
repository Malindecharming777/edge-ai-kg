"""MCP server exposing the Edge AI deployment KG.

Run:  python -m mcp_server.server

Tools are deliberately shaped around the questions an edge-AI team asks while
trying to land a model on custom silicon, rather than exposing raw Cypher.
`run_cypher` is available as an escape hatch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastmcp import FastMCP

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
_graph_cfg = _config.get("graph", {})
GRAPH = _graph_cfg.get("graph", "default")
URL = _graph_cfg.get("url")

mcp = FastMCP("edge-ai-kg")
_client = None


def client():
    global _client
    if _client is None:
        from samyama import SamyamaClient
        _client = SamyamaClient.connect(URL) if URL else SamyamaClient.embedded()
    return _client


def _rows(cypher: str) -> list[dict[str, Any]]:
    result = client().query(cypher.strip(), GRAPH)
    cols = list(result.columns)
    return [dict(zip(cols, rec)) for rec in result.records]


def _q(value: str) -> str:
    """Quote a string for Cypher. This engine wants double quotes and has no
    escape syntax, so quotes and newlines are stripped."""
    cleaned = str(value).replace('"', "").replace("\\", "").replace("\n", " ")
    return f'"{cleaned}"'


@mcp.tool()
def fallback_audit(model_name: str, accelerator_kind: str = "NPU-Lite") -> list[dict]:
    """Operators in a model that have NO kernel on a given accelerator class,
    and therefore fall back to the CPU. The core deployment question."""
    return _rows(f"""
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.name = {_q(model_name)}
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.kind = {_q(accelerator_kind)}
WITH op, count(k) AS kernels
WHERE kernels = 0
WITH op.name AS operator, op.category AS category, op.since_version AS opset
RETURN operator, category, opset
ORDER BY category
""")


@mcp.tool()
def boards_for_task(task_name: str, precision: str = "int8",
                    limit: int = 15) -> list[dict]:
    """Boards that run a model for this clinical task inside its latency budget."""
    return _rows(f"""
MATCH (t:ClinicalTask)<-[:SOLVES]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE t.name = {_q(task_name)} AND v.precision = {_q(precision)}
  AND d.fits = 1 AND d.latency_ms < t.latency_budget_ms
WITH b.name AS board, b.form_factor AS form_factor,
     b.power_budget_mw AS power_mw, m.name AS model,
     d.latency_ms AS latency_ms, d.fallback_op_count AS fallback_ops
RETURN board, form_factor, power_mw, model, latency_ms, fallback_ops
ORDER BY latency_ms ASC
LIMIT {int(limit)}
""")


@mcp.tool()
def kernel_blast_radius(operator_name: str) -> list[dict]:
    """If this operator lost its kernel, how many deployments, boards and models
    would regress to CPU fallback?"""
    return _rows(f"""
MATCH (op:Operator)<-[:USES_OPERATOR]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE op.name = {_q(operator_name)} AND d.fits = 1
RETURN op.name AS operator, count(DISTINCT d) AS deployments_at_risk,
       count(DISTINCT b) AS boards_affected, count(DISTINCT m) AS models_affected
""")


@mcp.tool()
def operator_coverage(accelerator_kind: str | None = None) -> list[dict]:
    """How much of the ONNX operator surface each accelerator class implements."""
    where = f"WHERE a.kind = {_q(accelerator_kind)}" if accelerator_kind else ""
    return _rows(f"""
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:IMPLEMENTS]->(op:Operator)
{where}
WITH a.kind AS accelerator_kind, count(DISTINCT op) AS operators_covered,
     count(DISTINCT a) AS accelerators
RETURN accelerator_kind, accelerators, operators_covered
ORDER BY operators_covered DESC
""")


@mcp.tool()
def quantization_unlock(limit: int = 15) -> list[dict]:
    """Model/board pairs that do not fit at fp32 but do fit at int8.

    Written as a single linear pattern with conditional aggregation rather than
    a self-join -- the self-join form returns a cartesian product on engine
    v1.7.0 (docs/engine-notes.md item 1).
    """
    return _rows(f"""
MATCH (m:Model)<-[:VARIANT_OF]-(v:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)
      -[:ON_BOARD]->(b:Board)
WITH m.name AS model, b.name AS board,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb,
     min(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN d.latency_ms ELSE 999999.0 END) AS int8_latency_ms
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, fp32_kb, int8_kb, int8_latency_ms
ORDER BY fp32_kb DESC
LIMIT {int(limit)}
""")


@mcp.tool()
def device_path(precision: str = "int8", limit: int = 10) -> list[dict]:
    """Trace complete on-device paths: sensor -> signal pipeline -> model ->
    variant -> deployment -> board."""
    return _rows(f"""
MATCH (s:Sensor)-[:FEEDS]->(st:SignalStage)-[:NEXT_STAGE*0..3]->(last:SignalStage)
      -[:PRECEDES]->(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE v.precision = {_q(precision)} AND d.fits = 1
WITH s.name AS sensor, s.sample_rate_hz AS sample_rate_hz,
     last.name AS final_stage, m.name AS model, b.name AS board,
     d.latency_ms AS latency_ms
RETURN sensor, sample_rate_hz, final_stage, model, board, latency_ms
ORDER BY latency_ms ASC
LIMIT {int(limit)}
""")


@mcp.tool()
def run_cypher(cypher: str) -> list[dict]:
    """Escape hatch: run an arbitrary read query against the KG.

    Engine caveats: `RETURN DISTINCT` is a no-op and negated pattern predicates
    do not parse -- see docs/engine-notes.md.
    """
    return _rows(cypher)


if __name__ == "__main__":
    mcp.run()
