"""Immutable, restartable observation references for live scene experience."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.world_model.visual_harness import ObservationImages


class PersistentObservationImages(ObservationImages):
    def __init__(self, folder):
        super().__init__()
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = self.folder / "images.jsonl"
        if self.index.exists():
            for line in self.index.read_text().splitlines():
                row = json.loads(line)
                path = (self.folder / row["file"]).resolve()
                if path.parent != self.folder:
                    raise ValueError("observation escaped its admitted image directory")
                self.register_file(row["observation"], path, row["sha256"])

    def put(self, observation, pixels):
        super().put(observation, pixels)
        if observation in self._files:
            return
        filename = hashlib.sha256(observation.encode()).hexdigest() + ".png"
        path = self.folder / filename
        if path.exists():
            raise ValueError("unregistered observation file already exists")
        Image.fromarray(np.asarray(pixels)).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.register_file(observation, path, digest)
        with self.index.open("a") as stream:
            stream.write(json.dumps({"observation": observation, "file": filename, "sha256": digest}) + "\n")
