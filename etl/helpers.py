"""Shared ETL helpers: id minting, Cypher literal rendering, batched writes.

Engine notes baked in here (learned the hard way, keep them):

* Strings must be double-quoted. Embedded quotes/newlines are stripped rather
  than escaped -- the parser does not accept backslash escapes.
* Floats must never be rendered in scientific notation. `repr(1e-05)` produces
  `1e-05`, which the Cypher parser rejects. `cypher_literal` forces fixed
  notation.
* Integer-valued floats must stay floats where the property is semantically a
  float, because `WHERE n.x > 0.5` against an int-typed property silently
  returns nothing.
* Edge creation matches endpoints with `WHERE`, not inline map properties --
  inline properties do not trigger an index scan on this build.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence


def norm_id(prefix: str, value: str) -> str:
    return f"{prefix}:{value.strip().lower().replace(' ', '_')}"


def _clean(text: Any) -> str:
    return (str(text).replace('"', "").replace("\\", "")
            .replace("\n", " ").replace("\r", " "))


def cypher_literal(value: Any) -> str:
    """Render a Python value as a Cypher literal this engine will accept."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "0.0"
        # Fixed notation only -- the parser rejects 1e-05.
        text = f"{value:.6f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    return f'"{_clean(value)}"'


def props_map(props: dict[str, Any]) -> str:
    inner = ", ".join(f"{k}: {cypher_literal(v)}"
                      for k, v in props.items() if v is not None)
    return "{" + inner + "}"


def chunked(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def create_nodes(client, graph: str, label: str, rows: Sequence[dict],
                 batch: int = 250) -> int:
    """Batch-CREATE nodes. Returns the number created."""
    total = 0
    for group in chunked(rows, batch):
        parts = [f"(:{label} {props_map(r)})" for r in group]
        client.query("CREATE " + ", ".join(parts), graph)
        total += len(group)
    return total


def create_edges(client, graph: str, edges: Sequence[tuple],
                 batch: int = 100) -> int:
    """Batch-CREATE edges.

    Each edge is `(src_label, src_id, rel_type, tgt_label, tgt_id, props|None)`
    and endpoints are looked up by their unique `id` property.
    """
    total = 0
    for group in chunked(edges, batch):
        var_of: dict[tuple[str, str], str] = {}
        match_parts, where_parts, create_parts = [], [], []
        for src_label, src_id, rel, tgt_label, tgt_id, props in group:
            for label, node_id in ((src_label, src_id), (tgt_label, tgt_id)):
                key = (label, node_id)
                if key not in var_of:
                    var = f"v{len(var_of)}"
                    var_of[key] = var
                    match_parts.append(f"({var}:{label})")
                    where_parts.append(f'{var}.id = {cypher_literal(node_id)}')
            prop_part = f" {props_map(props)}" if props else ""
            create_parts.append(
                f"({var_of[(src_label, src_id)]})-[:{rel}{prop_part}]->"
                f"({var_of[(tgt_label, tgt_id)]})"
            )
        client.query(
            f"MATCH {', '.join(match_parts)} "
            f"WHERE {' AND '.join(where_parts)} "
            f"CREATE {', '.join(create_parts)}",
            graph,
        )
        total += len(group)
    return total
