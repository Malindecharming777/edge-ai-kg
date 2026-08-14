"""Narrated walkthrough of the Edge AI deployment KG.

    python -m demo.demo                          # embedded engine, self-contained
    python -m demo.demo --url http://127.0.0.1:8080
    python -m demo.demo --fast                   # no typing delays

The story, in five beats:
  1. what the graph holds
  2. the question that breaks a document store: what falls back to CPU
  3. what that fallback costs, measured
  4. what quantization unlocks
  5. blast radius: one kernel disappears, what breaks
"""
from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
console = Console()
PAUSE = 1.4


def beat(seconds: float | None = None) -> None:
    time.sleep(PAUSE if seconds is None else seconds)


def say(text: str) -> None:
    console.print(f"[bold cyan]>[/bold cyan] {text}")
    beat(0.9)


def run(client, title: str, cypher: str, note: str = "") -> list:
    console.print()
    console.print(Panel(Syntax(cypher.strip(), "cypher", theme="ansi_dark",
                               word_wrap=True),
                        title=f"[bold]{title}[/bold]", border_style="cyan"))
    started = time.perf_counter()
    result = client.query(cypher.strip(), GRAPH)
    elapsed = (time.perf_counter() - started) * 1000

    table = Table(show_header=True, header_style="bold magenta", box=None,
                  pad_edge=False)
    for col in result.columns:
        table.add_column(col)
    for rec in result.records[:8]:
        table.add_row(*[f"{v:,.3f}" if isinstance(v, float) else str(v) for v in rec])
    console.print(table)
    console.print(f"[dim]{len(result.records)} rows in {elapsed:.1f} ms"
                  f"{'  --  ' + note if note else ''}[/dim]")
    beat()
    return result.records


def ensure_loaded(client, scale: float) -> None:
    existing = client.query("MATCH (n) RETURN count(n) AS n", GRAPH).records[0][0]
    if existing > 1000:
        say(f"Using the graph already loaded ({existing:,} nodes).")
        return
    say("Building the graph in-process (no server needed) ...")
    ops = oc.load_cached()
    fleet = gen.generate(scale=scale, operators=ops)
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    say(f"Loaded {fleet.node_count:,} nodes and {fleet.edge_count:,} edges.")


def main() -> None:
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    if args.fast:
        PAUSE = 0.0

    from samyama import SamyamaClient
    client = SamyamaClient.connect(args.url) if args.url else SamyamaClient.embedded()

    console.print()
    console.print(Panel.fit(
        "[bold]Edge AI deployment knowledge graph[/bold]\n"
        "[dim]hardware + kernels + models, in one graph[/dim]",
        border_style="bold cyan"))
    beat()

    ensure_loaded(client, args.scale)

    # ---- Beat 1: what is in here -------------------------------------
    console.print()
    console.rule("[bold]1. What the graph holds[/bold]")
    say("Boards, SoCs and accelerators on one side. Models, operators and "
        "kernels on the other.")
    run(client, "Kernel coverage per accelerator class", """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WITH a.kind AS accelerator_kind, count(DISTINCT op.id) AS operators_covered,
     count(DISTINCT a.id) AS accelerators
RETURN accelerator_kind, accelerators, operators_covered
ORDER BY operators_covered DESC
""", "the ONNX operator list is real; the fleet is synthetic")
    say("Every SoC has a CPU that runs all 205 operators. The NPUs do not. "
        "That gap is the whole problem.")

    # ---- Beat 2: the hero question -----------------------------------
    console.print()
    console.rule("[bold]2. What silently falls back to the CPU[/bold]")
    say("Pick a model and an accelerator. Which operators have no kernel?")
    model = client.query(
        'MATCH (m:Model)-[:USES_OPERATOR]->(o:Operator) WITH m, count(o) AS n '
        'WHERE n > 12 RETURN m.id, m.name ORDER BY n DESC LIMIT 1', GRAPH).records[0]
    run(client, "Operators with no NPU kernel -> CPU fallback", f"""
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.id = "{model[0]}"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.kind = "NPU-Lite"
WITH op, count(k) AS kernels
WHERE kernels = 0
WITH op.name AS operator, op.category AS category, op.since_version AS opset
RETURN operator, category, opset
ORDER BY category
""", f"model: {model[1]}")
    say("This is an anti-join over a 3-hop path. It is the query that is "
        "genuinely awkward without a graph.")

    # ---- Beat 3: what it costs ---------------------------------------
    console.print()
    console.rule("[bold]3. What the fallback costs[/bold]")
    run(client, "Latency vs fallback fraction, by accelerator class", """
MATCH (d:Deployment)
WHERE d.fits = 1
RETURN d.accelerator_kind AS accelerator_kind, count(d) AS deployments,
       avg(d.fallback_fraction) AS avg_fallback_fraction,
       avg(d.latency_ms) AS avg_latency_ms
ORDER BY avg_fallback_fraction DESC
""", "more fallback, more latency -- derived, not invented")

    # ---- Beat 4: quantization ----------------------------------------
    console.print()
    console.rule("[bold]4. What quantization unlocks[/bold]")
    say("Which models do not fit at fp32, but do fit at int8 on the same board?")
    run(client, "fp32 misses, int8 fits", """
MATCH (m:Model)<-[:VARIANT_OF]-(v:ModelVariant)<-[:OF_VARIANT]-(d:Deployment)
      -[:ON_BOARD]->(b:Board)
WITH m.name AS model, b.name AS board, b.ram_kb AS board_ram_kb,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, board_ram_kb, fp32_kb, int8_kb
ORDER BY fp32_kb DESC
LIMIT 8
""")

    # ---- Beat 5: blast radius ----------------------------------------
    console.print()
    console.rule("[bold]5. Blast radius[/bold]")
    say("A vendor drops the Conv kernel from their SDK. What breaks?")
    run(client, "Deployments at risk if one kernel disappears", """
MATCH (op:Operator)<-[:USES_OPERATOR]-(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE op.name = "Conv" AND d.fits = 1
RETURN op.name AS operator, count(DISTINCT d.id) AS deployments_at_risk,
       count(DISTINCT b.id) AS boards_affected, count(DISTINCT m.id) AS models_affected
""", "one hop out from a single node")

    # ---- Beat 6: the full chain --------------------------------------
    console.print()
    console.rule("[bold]6. Electrode to silicon, in one query[/bold]")
    run(client, "Sensor -> DSP pipeline -> model -> board", """
MATCH (s:Sensor)-[:FEEDS]->(st:SignalStage)-[:NEXT_STAGE*0..3]->(last:SignalStage)
      -[:PRECEDES]->(m:Model)<-[:VARIANT_OF]-(v:ModelVariant)
      <-[:OF_VARIANT]-(d:Deployment)-[:ON_BOARD]->(b:Board)
WHERE v.precision = "int8" AND d.fits = 1
WITH s.name AS sensor, last.name AS final_stage, m.name AS model,
     b.name AS board, d.latency_ms AS latency_ms
RETURN sensor, final_stage, model, board, latency_ms
ORDER BY latency_ms ASC
LIMIT 8
""", "variable-length path over the DSP chain")

    console.print()
    console.print(Panel.fit(
        "[bold]The point[/bold]\n"
        "Hardware, kernels and models are one connected structure.\n"
        "Flatten it into JSON and every question above becomes a script.",
        border_style="bold green"))
    console.print()


if __name__ == "__main__":
    main()
