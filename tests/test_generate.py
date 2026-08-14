"""Invariants of the synthetic fleet. These are what the query tests lean on."""
import pytest

from etl import generate as gen
from etl import onnx_catalog as oc


@pytest.fixture(scope="module")
def fleet():
    try:
        ops = oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")
    return gen.generate(seed=1234, scale=0.25, operators=ops)


def test_deterministic_for_a_given_seed(fleet):
    ops = oc.load_cached()
    again = gen.generate(seed=1234, scale=0.25, operators=ops)
    assert again.node_count == fleet.node_count
    assert again.edge_count == fleet.edge_count
    assert again.nodes["Board"] == fleet.nodes["Board"]
    assert again.edges == fleet.edges


def test_different_seeds_differ(fleet):
    other = gen.generate(seed=999, scale=0.25, operators=oc.load_cached())
    assert other.nodes["Board"] != fleet.nodes["Board"]


def test_node_ids_unique_per_label(fleet):
    for label, rows in fleet.nodes.items():
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {label}"


def test_every_edge_endpoint_exists(fleet):
    by_label = {label: {r["id"] for r in rows} for label, rows in fleet.nodes.items()}
    for src_label, src_id, rel, tgt_label, tgt_id, _ in fleet.edges:
        assert src_id in by_label[src_label], f"{rel}: missing source {src_id}"
        assert tgt_id in by_label[tgt_label], f"{rel}: missing target {tgt_id}"


def test_each_variant_belongs_to_exactly_one_model(fleet):
    counts = {}
    for src_label, src_id, rel, _, _, _ in fleet.edges:
        if rel == "VARIANT_OF":
            counts[src_id] = counts.get(src_id, 0) + 1
    assert counts and set(counts.values()) == {1}


def test_int8_is_exactly_a_quarter_of_fp32(fleet):
    """The query tests use this ratio to detect cartesian-product breakage."""
    by_model = {}
    variant_by_id = {v["id"]: v for v in fleet.nodes["ModelVariant"]}
    for src_label, src_id, rel, _, tgt_id, _ in fleet.edges:
        if rel == "VARIANT_OF":
            by_model.setdefault(tgt_id, []).append(variant_by_id[src_id])
    for model_id, variants in by_model.items():
        sizes = {v["precision"]: v["size_kb"] for v in variants}
        assert abs(sizes["fp32"] / 4.0 - sizes["int8"]) < 0.01, model_id


def test_every_clinical_task_has_a_model(fleet):
    solved = {tgt for _, _, rel, _, tgt, _ in fleet.edges if rel == "SOLVES"}
    all_tasks = {t["id"] for t in fleet.nodes["ClinicalTask"]}
    assert all_tasks == solved, "task-anchored queries would return empty"


def test_every_soc_has_a_cpu_fallback(fleet):
    """Every SoC must have an MCU-CPU, or 'falls back to CPU' is meaningless."""
    accel_by_id = {a["id"]: a for a in fleet.nodes["Accelerator"]}
    per_soc = {}
    for src_label, src_id, rel, _, tgt_id, _ in fleet.edges:
        if rel == "HAS_ACCELERATOR":
            per_soc.setdefault(src_id, []).append(accel_by_id[tgt_id]["kind"])
    assert per_soc
    for soc_id, kinds in per_soc.items():
        assert "MCU-CPU" in kinds, soc_id


def test_kernels_respect_opset_ceiling(fleet):
    """A kernel must never exist for an operator above its accelerator's opset."""
    accel = {a["id"]: a for a in fleet.nodes["Accelerator"]}
    op = {o["id"]: o for o in fleet.nodes["Operator"]}
    kernel_op, kernel_accel = {}, {}
    for src_label, src_id, rel, _, tgt_id, _ in fleet.edges:
        if rel == "IMPLEMENTS":
            kernel_op[src_id] = tgt_id
        elif rel == "RUNS_ON":
            kernel_accel[src_id] = tgt_id
    checked = 0
    for kid, op_id in kernel_op.items():
        a = accel[kernel_accel[kid]]
        assert op[op_id]["since_version"] <= a["opset_ceiling"], kid
        checked += 1
    assert checked > 0


def test_fallback_fraction_is_a_fraction(fleet):
    for d in fleet.nodes["Deployment"]:
        assert 0.0 <= d["fallback_fraction"] <= 1.0
        assert d["latency_ms"] > 0
