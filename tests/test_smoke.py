from etl.helpers import norm_id

def test_norm_id():
    assert norm_id("node", "Some Value") == "node:some_value"
