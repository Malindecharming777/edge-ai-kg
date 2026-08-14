"""Question-driven walkthrough: the 12 catalog questions, asked and answered.

    python -m demo.questions                          # embedded, self-contained
    python -m demo.questions --url http://127.0.0.1:8080
    python -m demo.questions --only EA01 EA06
    python -m demo.questions --fast                   # no pacing (for CI)

Unlike `demo.demo`, which tells a story in six beats, this walks the whole
`benchmarks/queries.py` catalog: each plain-English question, the Cypher it
becomes, the answer, and why the question is awkward without a graph.

This is the script recorded for `demo/edgeai-questions.gif`.
"""
from __future__ import annotations

import argparse
import textwrap
import time

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from benchmarks.queries import QUERIES
from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
console = Console(width=100)
PACE = 1.0
MAX_ROWS = 6


def pause(factor: float = 1.0) -> None:
    if PACE:
        time.sleep(PACE * factor)


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value)
    return text if len(text) <= 44 else text[:41] + "..."


def ensure_loaded(client, scale: float) -> None:
    count = client.query("MATCH (n) RETURN count(n) AS n", GRAPH).records[0][0]
    if count > 1000:
        console.print(f"[dim]Graph already loaded: {count:,} nodes.[/dim]\n")
        return
    console.print("[dim]Building the graph in-process (no server needed)...[/dim]")
    fleet = gen.generate(scale=scale, operators=oc.load_cached())
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    console.print(f"[dim]Loaded {fleet.node_count:,} nodes, "
                  f"{fleet.edge_count:,} edges.[/dim]\n")


def ask(client, query: dict, position: str) -> float:
    question = " ".join(query["question"].split())
    console.print()
    console.rule(f"[bold cyan]{query['id']}[/bold cyan] [dim]{position}[/dim]",
                 align="left", style="cyan")
    console.print()
    for line in textwrap.wrap(question, width=92):
        console.print(f"  [bold white]{line}[/bold white]")
    pause(1.3)

    console.print()
    console.print(Panel(Syntax(query["cypher"].strip(), "cypher",
                               theme="ansi_dark", word_wrap=True),
                        border_style="grey37", padding=(0, 1)))
    pause(0.7)

    started = time.perf_counter()
    result = client.query(query["cypher"].strip(), GRAPH)
    elapsed = (time.perf_counter() - started) * 1000

    table = Table(show_header=True, header_style="bold magenta",
                  box=None, pad_edge=False, padding=(0, 2, 0, 0))
    for column in result.columns:
        table.add_column(column, overflow="fold")
    for record in result.records[:MAX_ROWS]:
        table.add_row(*[fmt(v) for v in record])
    console.print(table)

    more = len(result.records) - MAX_ROWS
    suffix = f"  (+{more} more)" if more > 0 else ""
    console.print(f"[green]{len(result.records)} rows[/green] "
                  f"[dim]in {elapsed:.0f} ms{suffix}[/dim]")
    console.print()
    for line in textwrap.wrap("Why a graph: " + " ".join(query["why_graph"].split()),
                              width=92):
        console.print(f"  [dim italic]{line}[/dim italic]")
    pause(1.6)
    return elapsed


def main() -> None:
    global PACE
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Query ids to run, e.g. --only EA01 EA06")
    args = parser.parse_args()
    if args.fast:
        PACE = 0.0

    from samyama import SamyamaClient
    client = SamyamaClient.connect(args.url) if args.url else SamyamaClient.embedded()

    console.print()
    console.print(Panel.fit(
        "[bold]Edge AI Deployment Knowledge Graph[/bold]\n"
        "[dim]12 questions an edge-AI team actually asks,\n"
        "answered against boards, kernels and models in one graph.[/dim]",
        border_style="bold cyan", padding=(1, 3)))
    pause(1.2)

    ensure_loaded(client, args.scale)

    selected = [q for q in QUERIES if args.only is None or q["id"] in args.only]
    timings = []
    for index, query in enumerate(selected, start=1):
        timings.append(ask(client, query, f"{index} of {len(selected)}"))

    console.print()
    console.rule(style="green")
    median = sorted(timings)[len(timings) // 2] if timings else 0.0
    console.print(
        f"  [bold green]{len(selected)}/{len(selected)} questions answered[/bold green] "
        f"[dim]· median {median:.0f} ms · slowest {max(timings):.0f} ms[/dim]")
    console.print("  [dim]Hardware, kernels and models are one connected structure.[/dim]")
    console.print("  [dim]Flatten it into JSON and every question above becomes a script.[/dim]")
    console.print()


if __name__ == "__main__":
    main()
