"""Keep the display tied to its exact observation and original usage receipt."""

import json

from robocasa_common.depth_visualization import record_response


def test_reused_call_is_not_duplicated_and_unknown_usage_is_not_zero(tmp_path):
    """Reusing a response updates the panel once without inventing token counts."""
    policy = tmp_path / "policy"
    call = policy / "call-00001"
    call.mkdir(parents=True)
    response = {"reason": "<script>not executable</script>", "action": [0], "repeat": 2}
    record_response(policy, call, 37, "pixel", response)
    record_response(policy, call, 37, "pixel", response)
    data = json.loads((tmp_path / "visualization.json").read_text())
    assert len(data["calls"]) == 1
    assert data["calls"][0]["usage"] is None
    assert data["calls"][0]["step"] == 37
    viewer = (tmp_path / "visualization.html").read_text()
    assert response["reason"] not in viewer
    assert "textContent=row.response.reason" in viewer


def test_query_and_action_keep_separate_usage_at_same_step(tmp_path):
    """A query adds a separate paid call while cached input remains part of input."""
    policy = tmp_path / "policy"
    for i, queries in enumerate(([{"u": 143, "v": 140}], [])):
        call = policy / f"call-{i:05d}"
        call.mkdir(parents=True)
        usage = {
            "input_tokens": 100,
            "cached_input_tokens": 60,
            "output_tokens": 20,
            "total_tokens": 120,
        }
        (call / "receipt.json").write_text(json.dumps({"usage": usage}))
        record_response(policy, call, 37, "pixel", {"queries": queries}, [{"pixel_depth_m": 0.125}])
    rows = json.loads((tmp_path / "visualization.json").read_text())["calls"]
    assert len(rows) == 2
    assert all(row["step"] == 37 and row["usage"]["total_tokens"] == 120 for row in rows)
    assert rows[0]["response"]["queries"]
    assert rows[1]["query_answers"] == [{"pixel_depth_m": 0.125}]
