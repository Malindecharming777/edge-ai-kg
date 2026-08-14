"""Cypher literal rendering -- these rules are engine workarounds, not style."""
import math

from etl.helpers import chunked, cypher_literal, norm_id, props_map


def test_norm_id():
    assert norm_id("node", "Some Value") == "node:some_value"


def test_floats_never_use_scientific_notation():
    # The engine's parser rejects `1e-05`.
    for value in (1e-5, 1e-9, 0.0000001, 1.5e-8):
        assert "e" not in cypher_literal(value).lower()


def test_float_stays_float():
    # `WHERE n.x > 0.5` silently returns nothing if x was stored as an int.
    assert cypher_literal(2.0) == "2.0"
    assert cypher_literal(3) == "3"


def test_strings_are_double_quoted_and_sanitised():
    assert cypher_literal("hello") == '"hello"'
    assert '"' not in cypher_literal('he said "hi"')[1:-1]
    assert "\n" not in cypher_literal("a\nb")


def test_non_finite_floats_are_safe():
    assert cypher_literal(float("nan")) == "0.0"
    assert cypher_literal(math.inf) == "0.0"


def test_props_map_skips_none():
    out = props_map({"id": "x", "keep": 1, "drop": None})
    assert "drop" not in out and "keep: 1" in out and 'id: "x"' in out


def test_chunked_covers_everything():
    items = list(range(10))
    assert [x for c in chunked(items, 3) for x in c] == items
