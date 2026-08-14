"""Run every catalog query against a loaded Edge AI KG and report timings.

Usage:
    python -m benchmarks.run_benchmark --url http://127.0.0.1:8080 --graph edge_ai
    python -m benchmarks.run_benchmark --only EA01 --rows 20
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import click

from benchmarks.queries import QUERIES


def connect(url: str | None):
    from samyama import SamyamaClient
    return SamyamaClient.connect(url) if url else SamyamaClient.embedded()


def run_one(client, graph: str, query: dict, repeats: int) -> dict:
    timings: list[float] = []
    result = None
    error = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            result = client.query(query["cypher"].strip(), graph)
        except Exception as exc:
            error = str(exc)
            break
        timings.append((time.perf_counter() - t0) * 1000.0)
    return {
        "id": query["id"],
        "title": query["title"],
        "error": error,
        "rows": 0 if result is None else len(result.records),
        "columns": [] if result is None else list(result.columns),
        "records": [] if result is None else result.records,
        "ms_median": round(statistics.median(timings), 2) if timings else None,
        "ms_min": round(min(timings), 2) if timings else None,
    }


@click.command()
@click.option("--url", default=None, help="Samyama server URL. Omit for embedded.")
@click.option("--graph", default="edge_ai", show_default=True)
@click.option("--repeats", default=3, show_default=True)
@click.option("--rows", default=5, show_default=True, help="Result rows to print.")
@click.option("--only", default=None, help="Run a single query id, e.g. EA01.")
@click.option("--json-out", type=click.Path(), default=None,
              help="Write full results to a JSON file.")
def main(url, graph, repeats, rows, only, json_out):
    client = connect(url)
    selected = [q for q in QUERIES if only is None or q["id"] == only]
    if not selected:
        raise SystemExit(f"no query matching {only!r}")

    results, failures = [], 0
    for q in selected:
        res = run_one(client, graph, q, repeats)
        results.append(res)
        status = "FAIL" if res["error"] else ("EMPTY" if res["rows"] == 0 else "ok")
        if res["error"]:
            failures += 1
        click.echo("")
        click.echo(f"[{res['id']}] {res['title']}")
        click.echo(f"      Q: {q['question']}")
        if res["error"]:
            click.echo(f"      !! {res['error'][:200]}")
            continue
        click.echo(f"      {status} -- {res['rows']} rows, "
                   f"median {res['ms_median']} ms (min {res['ms_min']} ms)")
        if res["records"]:
            click.echo(f"      {res['columns']}")
            for rec in res["records"][:rows]:
                click.echo(f"        {rec}")

    ok = sum(1 for r in results if not r["error"] and r["rows"] > 0)
    empty = sum(1 for r in results if not r["error"] and r["rows"] == 0)
    click.echo("")
    click.echo(f"== {ok}/{len(results)} returning rows, {empty} empty, {failures} failed")
    if results and all(r["ms_median"] for r in results if not r["error"]):
        med = [r["ms_median"] for r in results if not r["error"]]
        click.echo(f"== median latency across queries: {statistics.median(med):.2f} ms, "
                   f"max {max(med):.2f} ms")

    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=1, default=str),
                                  encoding="utf-8")
        click.echo(f"== wrote {json_out}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
