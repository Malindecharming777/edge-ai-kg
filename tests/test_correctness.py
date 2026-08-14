"""Validate catalog queries against ground truth computed in Python.

These tests exist because engine v1.7.0 has query-shape bugs that return
*plausible but wrong* rows (see docs/engine-notes.md). A query that runs and
returns rows is not evidence that it is right, so every structural claim the
demo makes is checked here against the generator's own data.

Runs against an in-process embedded engine, so no server is needed.
"""
from __future__ import annotations

import pytest

from benchmarks.queries import BY_ID
from etl import generate as gen
from etl import onnx_catalog as oc
from etl.helpers import create_edges, create_nodes
from etl.loader import NODE_LABELS

GRAPH = "default"
SCALE = 0.3
SEED = 4242


@pytest.fixture(scope="module")
def loaded():
    """Generate a small fleet, load it into an embedded engine, return both."""
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    fleet = gen.generate(seed=SEED, scale=SCALE, operators=ops)
    # The catalog includes EA13-EA16, which run on the real public-source layer,
    # so the fixture must load both layers or those queries come back empty.
    from etl import real_layer
    real_layer.build_real(fleet, ops)
    try:
        client.query("MATCH (n) DETACH DELETE n", GRAPH)
    except Exception:
        pass
    for label in NODE_LABELS:
        if fleet.nodes.get(label):
            create_nodes(client, GRAPH, label, fleet.nodes[label])
    create_edges(client, GRAPH, fleet.edges)
    return client, fleet


def rows(client, cypher):
    result = client.query(cypher.strip(), GRAPH)
    return list(result.columns), result.records


def index(fleet):
    """Build lookup tables from the fleet for ground-truth computation."""
    out = {
        "node": {label: {r["id"]: r for r in rs} for label, rs in fleet.nodes.items()},
        "out": {},
    }
    for src_label, src_id, rel, tgt_label, tgt_id, props in fleet.edges:
        out["out"].setdefault(rel, []).append((src_id, tgt_id, props))
    return out


# --------------------------------------------------------------------------


def test_graph_loaded_completely(loaded):
    client, fleet = loaded
    _, recs = rows(client, "MATCH (n) RETURN count(n) AS n")
    assert recs[0][0] == fleet.node_count
    _, recs = rows(client, "MATCH ()-[r]->() RETURN count(r) AS n")
    assert recs[0][0] == fleet.edge_count


def test_ea01_fallback_audit_matches_ground_truth(loaded):
    """The hero query: operators with no kernel on a chosen accelerator."""
    client, fleet = loaded
    idx = index(fleet)
    model = fleet.nodes["Model"][0]
    accel = fleet.nodes["Accelerator"][0]

    model_ops = {t for s, t, _ in idx["out"]["USES_OPERATOR"] if s == model["id"]}
    assert model_ops, "model has no operators"
    kernels_on_accel = {k for k, a, _ in idx["out"]["RUNS_ON"] if a == accel["id"]}
    covered = {t for s, t, _ in idx["out"]["IMPLEMENTS"] if s in kernels_on_accel}
    expected = {idx["node"]["Operator"][o]["name"] for o in model_ops - covered}

    cypher = f"""
MATCH (m:Model)-[:USES_OPERATOR]->(op:Operator)
WHERE m.id = "{model['id']}"
OPTIONAL MATCH (k:Kernel)-[:IMPLEMENTS]->(op), (k)-[:RUNS_ON]->(a:Accelerator)
WHERE a.id = "{accel['id']}"
WITH op, count(k) AS kernels
WHERE kernels = 0
RETURN op.name AS operator
"""
    _, recs = rows(client, cypher)
    assert {r[0] for r in recs} == expected


def test_ea04_quantization_unlock_is_not_a_cartesian_product(loaded):
    """int8 size must be exactly a quarter of fp32 size for the SAME model.

    This is the canary for the v1.7.0 self-join bug: a cartesian product pairs
    variants from different models and the ratio drifts away from 4.
    """
    client, _ = loaded
    cols, recs = rows(client, BY_ID["EA04"]["cypher"])
    i_fp32, i_int8 = cols.index("fp32_kb"), cols.index("int8_kb")
    for rec in recs:
        fp32, int8 = rec[i_fp32], rec[i_int8]
        assert int8 > 0
        assert abs(fp32 / int8 - 4.0) < 0.05, (
            f"fp32={fp32} int8={int8} -- variants came from different models"
        )


def test_ea05_operator_coverage_matches_ground_truth(loaded):
    client, fleet = loaded
    idx = index(fleet)
    accel_of_kernel = {k: a for k, a, _ in idx["out"]["RUNS_ON"]}
    op_of_kernel = {k: o for k, o, _ in idx["out"]["IMPLEMENTS"]}
    expected: dict[str, set] = {}
    for kernel, accel_id in accel_of_kernel.items():
        kind = idx["node"]["Accelerator"][accel_id]["kind"]
        expected.setdefault(kind, set()).add(op_of_kernel[kernel])

    cols, recs = rows(client, BY_ID["EA05"]["cypher"])
    got = {r[cols.index("accelerator_kind")]: r[cols.index("operators_covered")]
           for r in recs}
    assert got == {k: len(v) for k, v in expected.items()}


def test_cpu_covers_every_operator(loaded):
    """Every SoC has an MCU-CPU that runs everything -- the fallback premise.

    Scoped to the ONNX catalog: the real ONNX Runtime layer adds operators from
    other domains (com.microsoft, ai.onnx.ml) that the synthetic fleet never
    models, and the synthetic CPU is not expected to cover those.
    """
    client, fleet = loaded
    catalog_ops = [o for o in fleet.nodes["Operator"] if o["source"] == "onnx"]
    _, recs = rows(client, """
MATCH (a:Accelerator)<-[:RUNS_ON]-(k:Kernel)-[:IMPLEMENTS]->(op:Operator)
WHERE a.kind = "MCU-CPU" AND op.source = "onnx"
RETURN count(DISTINCT op.id) AS n
""")
    assert recs[0][0] == len(catalog_ops)


def test_ea06_blast_radius_matches_ground_truth(loaded):
    client, fleet = loaded
    idx = index(fleet)
    op = next(o for o in fleet.nodes["Operator"] if o["name"] == "Conv")

    models = {s for s, t, _ in idx["out"]["USES_OPERATOR"] if t == op["id"]}
    variants = {s for s, t, _ in idx["out"]["VARIANT_OF"] if t in models}
    deploys = {s for s, t, _ in idx["out"]["OF_VARIANT"] if t in variants}
    fitting = {d for d in deploys if idx["node"]["Deployment"][d]["fits"] == 1}

    cols, recs = rows(client, BY_ID["EA06"]["cypher"])
    assert recs, "EA06 returned nothing"
    assert recs[0][cols.index("deployments_at_risk")] == len(fitting)


def test_ea12_vendor_totals_match_ground_truth(loaded):
    client, fleet = loaded
    idx = index(fleet)
    soc_of_board = {b: s for b, s, _ in idx["out"]["HAS_SOC"]}
    vendor_of = {s: v for s, v, _ in idx["out"]["MADE_BY"]}
    board_of_deploy = {d: b for d, b, _ in idx["out"]["ON_BOARD"]}

    expected: dict[str, int] = {}
    for deploy, board in board_of_deploy.items():
        # EA12 filters on `fits`, which only synthetic deployments carry; the
        # real MLPerf submissions have no such notion and are excluded.
        if idx["node"]["Deployment"][deploy].get("fits") != 1:
            continue
        if board not in soc_of_board:
            continue
        vendor = vendor_of[soc_of_board[board]]
        name = idx["node"]["Vendor"][vendor]["name"]
        expected[name] = expected.get(name, 0) + 1

    cols, recs = rows(client, BY_ID["EA12"]["cypher"])
    got = {r[cols.index("vendor")]: r[cols.index("deployments")] for r in recs}
    assert got == expected


def test_every_catalog_query_runs_and_returns_rows(loaded):
    client, _ = loaded
    empty, failed = [], []
    for qid, q in BY_ID.items():
        try:
            _, recs = rows(client, q["cypher"])
            if not recs:
                empty.append(qid)
        except Exception as exc:
            failed.append(f"{qid}: {exc}")
    assert not failed, f"queries failed: {failed}"
    # EA04 needs a model that misses at fp32 but fits at int8 on the same board;
    # at this small scale that combination may legitimately not occur. Its
    # correctness is pinned by test_ea04_shape_is_not_a_cartesian_product below.
    assert not [q for q in empty if q != "EA04"], f"queries returned no rows: {empty}"


def test_ea04_shape_is_not_a_cartesian_product():
    """Purpose-built regression test for the v1.7.0 self-join bug.

    One board carries an fp32 (does not fit) and an int8 (fits) deployment for
    EACH of two models. The correct answer pairs each model with its OWN
    variants -- 2 rows. The buggy join shapes return 4, pairing model A's fp32
    with model B's int8. See docs/engine-notes.md item 1.
    """
    try:
        from samyama import SamyamaClient
        client = SamyamaClient.embedded()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"embedded Samyama engine unavailable: {exc}")

    g = "default"
    for label in ("XBoard", "XModel", "XVariant", "XDeploy"):
        try:
            client.query(f"MATCH (n:{label}) DETACH DELETE n", g)
        except Exception:
            pass

    client.query(
        'CREATE (:XBoard {id: "B1", ram_kb: 1024}), '
        '(:XModel {id: "MA"}), (:XModel {id: "MB"}), '
        '(:XVariant {id: "MA32", precision: "fp32", size_kb: 4000.0}), '
        '(:XVariant {id: "MA8",  precision: "int8", size_kb: 1000.0}), '
        '(:XVariant {id: "MB32", precision: "fp32", size_kb: 8000.0}), '
        '(:XVariant {id: "MB8",  precision: "int8", size_kb: 2000.0}), '
        '(:XDeploy {id: "D1", fits: 0}), (:XDeploy {id: "D2", fits: 1}), '
        '(:XDeploy {id: "D3", fits: 0}), (:XDeploy {id: "D4", fits: 1})', g)
    for deploy, variant in (("D1", "MA32"), ("D2", "MA8"), ("D3", "MB32"), ("D4", "MB8")):
        client.query(f'MATCH (d:XDeploy), (b:XBoard) WHERE d.id = "{deploy}" '
                     f'AND b.id = "B1" CREATE (d)-[:X_ON]->(b)', g)
        client.query(f'MATCH (d:XDeploy), (v:XVariant) WHERE d.id = "{deploy}" '
                     f'AND v.id = "{variant}" CREATE (d)-[:X_OF]->(v)', g)
    for variant, model in (("MA32", "MA"), ("MA8", "MA"), ("MB32", "MB"), ("MB8", "MB")):
        client.query(f'MATCH (v:XVariant), (m:XModel) WHERE v.id = "{variant}" '
                     f'AND m.id = "{model}" CREATE (v)-[:X_VO]->(m)', g)

    result = client.query("""
MATCH (m:XModel)<-[:X_VO]-(v:XVariant)<-[:X_OF]-(d:XDeploy)-[:X_ON]->(b:XBoard)
WITH m.id AS model, b.id AS board,
     sum(CASE WHEN v.precision = "fp32" AND d.fits = 0 THEN 1 ELSE 0 END) AS fp32_misses,
     sum(CASE WHEN v.precision = "int8" AND d.fits = 1 THEN 1 ELSE 0 END) AS int8_hits,
     max(CASE WHEN v.precision = "fp32" THEN v.size_kb ELSE 0 END) AS fp32_kb,
     max(CASE WHEN v.precision = "int8" THEN v.size_kb ELSE 0 END) AS int8_kb
WHERE fp32_misses > 0 AND int8_hits > 0
RETURN model, board, fp32_kb, int8_kb
ORDER BY model
""", g)

    got = {r[0]: (r[2], r[3]) for r in result.records}
    assert got == {"MA": (4000.0, 1000.0), "MB": (8000.0, 2000.0)}, (
        f"expected each model paired with its own variants, got {result.records}"
    )


def test_order_by_is_actually_applied(loaded):
    """ORDER BY on a RETURN-introduced alias is silently ignored on v1.7.0, and
    only the first sort key is honoured. Every catalog query must therefore
    project through WITH and sort on a single key -- assert it really sorts."""
    import re

    client, _ = loaded
    broken = []
    for qid, q in BY_ID.items():
        cypher = q["cypher"].strip()
        match = re.search(r"ORDER BY\s+(.+?)(?:\s+LIMIT|\s*$)", cypher, re.S)
        if not match:
            continue
        keys = [k.strip() for k in match.group(1).split(",")]
        assert len(keys) == 1, f"{qid}: multi-key ORDER BY is not honoured by the engine"
        cols, recs = rows(client, cypher)
        key = keys[0].split()[0]
        descending = keys[0].lower().endswith("desc")
        values = [r[cols.index(key)] for r in recs]
        if values != sorted(values, reverse=descending):
            broken.append(f"{qid} ({key})")
    assert not broken, f"ORDER BY not applied for: {broken}"
