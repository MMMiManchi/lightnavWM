"""Collect every remaining train-fit house, then leave the node map to the next step.

Dev houses stay out. Houses already walked in the formal fit run or in
fit_houses_20261103_01 stay out. This does not train.
"""

from __future__ import annotations

from pathlib import Path

from experiments.world_model.branch_exit import new_fit_houses
from experiments.world_model.fit_house_entry import WALKED_SCENES, collect_houses
from experiments.world_model.group_manifest import assign_groups, load_split_rows

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "world_model_data" / "fit_houses_20261103_02"
DATASET = REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed"
EARLIER = REPO / "output" / "world_model_data" / "fit_houses_20261103_01" / "houses"


def remaining_scenes() -> set[str]:
    used = set(WALKED_SCENES)
    if EARLIER.is_dir():
        for house in EARLIER.iterdir():
            if house.is_dir():
                used.add(house.name.rsplit("_", 1)[0])
    return used


def main() -> None:
    _dataset, _digest, rows = load_split_rows(DATASET, "train")
    houses = new_fit_houses(assign_groups(rows), remaining_scenes(), limit=500)
    if not houses:
        raise SystemExit("no new train-fit houses left")
    print(f"collecting {len(houses)} train-fit houses into {OUT}", flush=True)
    collect_houses(OUT, remaining_scenes(), len(houses))


if __name__ == "__main__":
    main()
