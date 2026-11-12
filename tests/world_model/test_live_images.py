import json

import numpy as np
import pytest

from experiments.world_model.live_images import PersistentObservationImages


def test_live_image_references_survive_restart_and_refuse_changed_files(tmp_path):
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    images = PersistentObservationImages(tmp_path)
    images.put("ep-1:source", rgb)
    images.put("ep-1:source", rgb)
    assert len(images.index.read_text().splitlines()) == 1
    restarted = PersistentObservationImages(tmp_path)
    np.testing.assert_array_equal(np.asarray(restarted.get("ep-1:source")), rgb)
    with pytest.raises(ValueError, match="pixels cannot change"):
        restarted.put("ep-1:source", np.ones_like(rgb))
    row = json.loads(images.index.read_text())
    (tmp_path / row["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        PersistentObservationImages(tmp_path).get("ep-1:source")
