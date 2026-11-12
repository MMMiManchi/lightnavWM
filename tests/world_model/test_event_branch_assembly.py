import json

import pytest

from scripts.assemble_event_branches import build


def write_tasks(root, rows):
    root.mkdir()
    (root / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def task(i, sha="source"):
    return {"task_id": str(i), "origin": {"origin_id": "origin", "expected_rgb_sha256": sha}}


def test_partial_origin_replacement_is_rejected_before_reading_futures(tmp_path):
    parent, correction = tmp_path / "parent", tmp_path / "correction"
    write_tasks(parent, [task(i) for i in range(4)])
    write_tasks(correction, [task(0)])
    with pytest.raises(ValueError, match="every branch and repeat"):
        build(parent, tmp_path / "release", [correction])


def test_replacement_must_preserve_reviewed_current_observation(tmp_path):
    parent, correction = tmp_path / "parent", tmp_path / "correction"
    write_tasks(parent, [task(i) for i in range(4)])
    write_tasks(correction, [task(i, "different-source") for i in range(4)])
    with pytest.raises(ValueError, match="changed the reviewed source"):
        build(parent, tmp_path / "release", [correction])
