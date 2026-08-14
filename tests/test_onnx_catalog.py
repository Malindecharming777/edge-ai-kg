"""The ONNX catalog is the only real data in this KG -- pin its parsing."""
import pytest

from etl import onnx_catalog as oc


@pytest.fixture(scope="module")
def ops():
    try:
        return oc.load_cached()
    except FileNotFoundError:
        pytest.skip("run `python -m etl.download_data` first")


def test_catalog_is_populated(ops):
    assert len(ops) > 150, "ONNX has ~200 operators; parsing likely broke"


def test_core_operators_present(ops):
    names = {o.name for o in ops}
    for expected in ("Conv", "MatMul", "LSTM", "Softmax", "QuantizeLinear", "Gemm"):
        assert expected in names, f"{expected} missing from parsed catalog"


def test_ids_unique(ops):
    ids = [o.id for o in ops]
    assert len(ids) == len(set(ids))


def test_categories_are_sane(ops):
    by_name = {o.name: o for o in ops}
    assert by_name["Conv"].category == "convolution"
    assert by_name["MatMul"].category == "matmul"
    assert by_name["LSTM"].category == "recurrent"
    assert by_name["Softmax"].category == "activation"
    assert by_name["QuantizeLinear"].category == "quantization"


def test_control_flow_flagged(ops):
    by_name = {o.name: o for o in ops}
    assert by_name["Loop"].is_control_flow is True
    assert by_name["Conv"].is_control_flow is False


def test_since_version_is_the_latest(ops):
    # Conv has been revised many times; the parser should keep the highest opset.
    conv = next(o for o in ops if o.name == "Conv")
    assert conv.since_version >= 11
    assert conv.version_count >= 2


def test_categorize_falls_back_to_tensor():
    assert oc.categorize("SomeOperatorThatDoesNotExist") == "tensor"
