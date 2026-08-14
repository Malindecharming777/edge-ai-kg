# {{KG_NAME}} Knowledge Graph

**{{N}} nodes. {{M}} edges. {{ONE_LINE_SCOPE}} from {{K}} sources.**

![{{KG_NAME}} demo](demo/{{KG_SLUG}}.gif)

> Part of the **Samyama** ecosystem — loaded into and queried via the graph engine at [samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph).
> This repo holds the loader and source-data specifics for the KG.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License"></a>

---

We loaded {{SOURCES}} into one graph, then asked:

> *"{{EXAMPLE_QUESTION}}"*

```cypher
MATCH (a:{{NodeA}})-[:{{REL}}]->(b:{{NodeB}})
RETURN b.name, count(a) AS n
ORDER BY n DESC LIMIT 5
```

**One query across every source.** Powered by [Samyama Graph](https://github.com/samyama-ai/samyama-graph).

---

## Demo

A narrated walkthrough on a fast, real subset.

```bash
python -m demo.demo                                                     # run live
asciinema rec --overwrite --cols 92 --rows 32 --idle-time-limit 2.0 \
  -c "bash -c 'python -m demo.demo'" demo/{{KG_SLUG}}.cast              # re-record
agg demo/{{KG_SLUG}}.cast demo/{{KG_SLUG}}.gif                          # convert to gif
```

---

## Schema

**Node labels** -- {{NODE_LABELS}}
**Edge types** -- {{EDGE_TYPES}}
**Data sources** -- {{SOURCES}}

See [`schema/{{KG_SLUG}}_kg.cypher`](schema/{{KG_SLUG}}_kg.cypher) for the full schema.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

python -m etl.download_data          # fetch source data into data/
python -m etl.loader                 # build + load the graph
python -m mcp_server.server          # expose the KG over MCP
pytest                               # run tests
```

## Structure
```
etl/          # downloaders + graph loader
schema/       # cypher schema / ontology
mcp_server/   # MCP server exposing the KG
demo/         # narrated demo (cast + gif)
benchmarks/   # benchmark queries
docs/         # design + source notes
tests/        # pytest
pyproject.toml
```

---
## Using this template
1. Click **Use this template** (this repo is a Gitea template) and name your repo `<domain>-kg`.
2. Find-and-replace the `{{...}}` placeholders (KG_NAME, KG_SLUG, NODE_LABELS, EDGE_TYPES, SOURCES, N, M, K, example query).
3. Rename `schema/template_kg.cypher` -> `schema/<slug>_kg.cypher` and fill in real node/edge definitions.
4. Implement the `etl/` downloaders + loader and the `mcp_server/` tools for your domain.
5. Add the repo to the **samyama-graph** team.

### Placeholder reference
| Placeholder | Meaning |
|-------------|---------|
| `{{KG_NAME}}` | Human name, e.g. "Drug Interactions" |
| `{{KG_SLUG}}` | file/id slug, e.g. "druginteractions" |
| `{{NODE_LABELS}}` / `{{EDGE_TYPES}}` | comma-separated schema |
| `{{SOURCES}}` | data sources + licenses |
| `{{N}}`/`{{M}}`/`{{K}}` | node/edge/source counts |
| `{{EXAMPLE_QUESTION}}` / `{{NodeA}}`/`{{REL}}`/`{{NodeB}}` | hero query |
