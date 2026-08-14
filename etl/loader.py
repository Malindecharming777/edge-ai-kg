"""Build and load the Edge AI deployment graph into Samyama.

Usage:
    python -m etl.loader                          # embedded engine, default scale
    python -m etl.loader --url http://127.0.0.1:8080
    python -m etl.loader --scale 4 --graph edge_ai
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from etl import generate as gen
from etl.helpers import create_edges, create_nodes

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "edge_ai_kg.cypher"

# Every node label carries a unique `id`; indexing it is what makes the
# edge-creation MATCH ... WHERE lookups cheap.
NODE_LABELS = [
    "Vendor", "SoC", "Accelerator", "Board", "Runtime", "Operator", "Kernel",
    "Model", "ModelVariant", "Sensor", "SignalStage", "ClinicalTask",
    "Dataset", "Certification", "Deployment",
]


def connect(url: str | None):
    from samyama import SamyamaClient
    return SamyamaClient.connect(url) if url else SamyamaClient.embedded()


def apply_schema(client, graph: str) -> int:
    """Run the index statements from schema/edge_ai_kg.cypher."""
    applied = 0
    if not SCHEMA_PATH.exists():
        return 0
    for raw in SCHEMA_PATH.read_text(encoding="utf-8").splitlines():
        stmt = raw.split("//", 1)[0].strip().rstrip(";")
        if not stmt:
            continue
        try:
            client.query(stmt, graph)
            applied += 1
        except Exception as exc:  # index already exists, or unsupported syntax
            click.echo(f"  ! schema stmt skipped ({exc.__class__.__name__}): {stmt[:60]}",
                       err=True)
    return applied


def reset_graph(client, graph: str) -> None:
    try:
        client.query("MATCH (n) DETACH DELETE n", graph)
    except Exception:
        try:
            client.query("MATCH (n) DELETE n", graph)
        except Exception as exc:
            click.echo(f"  ! could not reset graph: {exc}", err=True)


@click.command()
@click.option("--url", default=None,
              help="Samyama server URL, e.g. http://127.0.0.1:8080. Omit for embedded.")
@click.option("--graph", default="default", show_default=True, help="Target graph / tenant.")
@click.option("--seed", type=int, default=gen.DEFAULT_SEED, show_default=True)
@click.option("--scale", type=float, default=1.0, show_default=True,
              help="Fleet size multiplier. 1.0 ~ 24K nodes / 73K edges.")
@click.option("--limit", type=int, default=None,
              help="Cap nodes per label for a fast smoke load.")
@click.option("--regenerate/--use-cached", default=False,
              help="Regenerate the fleet instead of loading data/fleet/fleet.json.")
@click.option("--reset/--no-reset", default=True, show_default=True,
              help="Delete existing nodes in the target graph first.")
def main(url, graph, seed, scale, limit, regenerate, reset):
    started = time.time()

    if regenerate:
        click.echo(f"[1/4] generating fleet (seed={seed}, scale={scale}) ...")
        fleet = gen.generate(seed=seed, scale=scale)
        gen.write(fleet)
    else:
        try:
            fleet = gen.load()
            click.echo(f"[1/4] loaded cached fleet (seed={fleet.seed}, scale={fleet.scale})")
        except FileNotFoundError:
            click.echo(f"[1/4] no cached fleet; generating (seed={seed}, scale={scale}) ...")
            fleet = gen.generate(seed=seed, scale=scale)
            gen.write(fleet)

    nodes = fleet.nodes
    edges = fleet.edges
    if limit:
        kept: dict[str, set[str]] = {}
        trimmed = {}
        for label, rows in nodes.items():
            trimmed[label] = rows[:limit]
            kept[label] = {r["id"] for r in trimmed[label]}
        nodes = trimmed
        edges = [e for e in edges
                 if e[1] in kept.get(e[0], ()) and e[4] in kept.get(e[3], ())]
        click.echo(f"      --limit {limit}: {sum(len(v) for v in nodes.values()):,} nodes, "
                   f"{len(edges):,} edges")

    client = connect(url)
    where = url or "embedded"
    click.echo(f"[2/4] connected ({where}), graph={graph}")
    if reset:
        reset_graph(client, graph)
    applied = apply_schema(client, graph)
    click.echo(f"      schema: {applied} index statements applied")

    click.echo("[3/4] loading nodes ...")
    t0 = time.time()
    total_nodes = 0
    for label in NODE_LABELS:
        rows = nodes.get(label, [])
        if not rows:
            continue
        created = create_nodes(client, graph, label, rows)
        total_nodes += created
        click.echo(f"      {label:15s} {created:>7,}")
    node_secs = time.time() - t0

    click.echo("[4/4] loading edges ...")
    t0 = time.time()
    total_edges = create_edges(client, graph, edges)
    edge_secs = time.time() - t0

    elapsed = time.time() - started
    click.echo("")
    click.echo(f"  nodes {total_nodes:,} in {node_secs:.1f}s "
               f"({total_nodes / max(node_secs, 1e-6):,.0f}/s)")
    click.echo(f"  edges {total_edges:,} in {edge_secs:.1f}s "
               f"({total_edges / max(edge_secs, 1e-6):,.0f}/s)")
    click.echo(f"  total {elapsed:.1f}s")

    try:
        got = client.query("MATCH (n) RETURN count(n) AS n", graph).records[0][0]
        click.echo(f"  verified: {got:,} nodes in graph '{graph}'")
        if got != total_nodes:
            click.echo(f"  ! expected {total_nodes:,}", err=True)
            sys.exit(1)
    except Exception as exc:
        click.echo(f"  ! verification query failed: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
