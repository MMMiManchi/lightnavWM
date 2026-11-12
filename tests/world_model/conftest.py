"""Path setup for the world-model tests.

The implementation package is ``experiments.world_model``. Habitat's package is
imported from ``habitat_server`` without installing it into the client venv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "habitat_server"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)
