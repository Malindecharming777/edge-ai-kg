"""The real layer must match its upstream sources exactly.

These are the tests that let the repo claim "real": every count and every
relationship is re-derived from the cached upstream files and compared.
"""
from __future__ import annotations

import pytest

from etl import mlperf_tiny as tiny
from etl import onnx_catalog as oc
from etl import ort_kernels as ort
from etl import real_layer
from etl.generate import Fleet


@pytest.fixture(scope="module")
def sources():
    try:
        return oc.load_cached(), ort.load_cached(), tiny.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


@pytest.fixture(scope="module")
def real(sources):
    ops, _, _ = sources
    return real_layer.build_real(Fleet(seed=0, scale=0.0), ops)


# ---------------------------------------------------------------- ORT ----

def test_ort_opset_spec_parsing():
    assert ort._parse_opset("13+") == (13, 999)
    assert ort._parse_opset("[6, 12]") == (6, 12)
    assert ort._parse_opset("7") == (7, 7)
    assert ort._parse_opset("nonsense") == (-1, -1)


def test_ort_kernels_cover_known_providers(sources):
    _, kernels, _ = sources
    providers = {k.execution_provider for k in kernels}
    assert "CPUExecutionProvider" in providers
    assert len(providers) >= 2, "expected at least CPU and one accelerator provider"


def test_ort_core_operators_registered(sources):
    _, kernels, _ = sources
    cpu = {k.operator for k in kernels
           if k.execution_provider == "CPUExecutionProvider" and k.domain == "ai.onnx"}
    for expected in ("Conv", "MatMul", "Softmax", "Gemm", "LSTM"):
        assert expected in cpu, f"{expected} missing from CPU EP registrations"


def test_ort_opset_ranges_are_sane(sources):
    _, kernels, _ = sources
    for k in kernels:
        assert k.opset_min >= 1
        assert k.opset_max >= k.opset_min


def test_every_real_kernel_has_its_three_edges(real, sources):
    _, kernels, _ = sources
    rels = {}
    for _, src, rel, _, tgt, _ in real.edges:
        rels.setdefault(rel, set()).add(src)
    for rel in ("IMPLEMENTS", "RUNS_ON", "PROVIDED_BY"):
        assert len(rels[rel]) >= len(kernels), f"{rel} missing for some kernels"


def test_real_kernel_operator_targets_all_exist(real):
    """Kernels for com.microsoft / ai.onnx.ml operators must not dangle."""
    operator_ids = {o["id"] for o in real.nodes["Operator"]}
    missing = [tgt for lbl, _, rel, tlbl, tgt, _ in real.edges
               if rel == "IMPLEMENTS" and tgt not in operator_ids]
    assert not missing, f"{len(missing)} IMPLEMENTS edges point at unknown operators"


# ------------------------------------------------------------- MLPerf ----

def test_mlperf_results_parsed(sources):
    _, _, results = sources
    assert len(results) > 20
    assert {r.task for r in results} <= set(tiny.TASKS)


def test_mlperf_throughput_and_accuracy_present(sources):
    _, _, results = sources
    assert all(r.throughput_inf_s and r.throughput_inf_s > 0 for r in results)
    assert any(r.power_uj_per_inf for r in results), "no energy submissions parsed"


def test_mlperf_deployment_count_matches_source(real, sources):
    _, _, results = sources
    real_deploys = [d for d in real.nodes["Deployment"] if d["provenance"] == "real"]
    assert len(real_deploys) == len(results)


def test_mlperf_boards_and_vendors_match_source(real, sources):
    _, _, results = sources
    assert len(real.nodes["Board"]) == len({r.board_name for r in results})
    assert len(real.nodes["Vendor"]) == len({r.organization for r in results})


def test_every_benchmark_task_has_a_model(real):
    solves = {(s, t) for _, s, rel, _, t, _ in real.edges if rel == "SOLVES"}
    tasks = {t["id"] for t in real.nodes["BenchmarkTask"]}
    assert {t for _, t in solves} == tasks


# --------------------------------------------------------- provenance ----

def test_every_real_node_is_stamped_real(real):
    for label, rows in real.nodes.items():
        for row in rows:
            assert row["provenance"] == "real", f"{label} {row['id']} not stamped real"
            assert row["source"], f"{label} {row['id']} has no source"


def test_real_sources_are_the_declared_ones(real):
    sources_seen = {r["source"] for rows in real.nodes.values() for r in rows}
    assert sources_seen <= {real_layer.SRC_ONNX, real_layer.SRC_ORT,
                            real_layer.SRC_TINY}, sources_seen


def test_real_and_synthetic_ids_never_collide(sources):
    """Real ids are prefixed by source; synthetic ids are `<prefix>:<5 digits>`."""
    from etl import generate as gen

    ops, _, _ = sources
    synth = gen.generate(seed=99, scale=0.2, operators=ops)
    real = real_layer.build_real(Fleet(seed=0, scale=0.0), ops)
    for label, rows in real.nodes.items():
        real_ids = {r["id"] for r in rows}
        synth_ids = {r["id"] for r in synth.nodes.get(label, [])}
        # Operator nodes are deliberately shared between layers (same ONNX ids).
        if label == "Operator":
            continue
        assert not (real_ids & synth_ids), f"{label} id collision: {real_ids & synth_ids}"
