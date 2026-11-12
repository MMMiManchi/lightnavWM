import numpy as np

from experiments.world_model.branch_curation import CurationRules, FloorPolygon, assign_region
from experiments.world_model.branch_source_mining import RegionIndex, first_ray_region


def test_fast_region_index_matches_conservative_polygon_annotations():
    polygons = [FloorPolygon(2, "h", 0, ((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3))),
                FloorPolygon(3, "k", 0, ((2, 0), (4, 0), (4, 2), (2, 2)))]
    rng = np.random.default_rng(8)
    points = np.c_[rng.uniform(-1, 5, 100), np.full(100, 0.1), -rng.uniform(-1, 4, 100)]
    labels, clearances = RegionIndex(polygons).assign(points)
    for point, label, distance in zip(points, labels, clearances):
        expected = assign_region(point, polygons, CurationRules())
        assert label == (expected["region_id"] if expected else -1)
        if expected:
            assert abs(distance - expected["clearance_m"]) < 1e-7


def test_an_unconfirmed_intermediate_region_is_not_skipped_for_a_prettier_witness():
    assert first_ray_region([1, 2, 2, 3, 3], [1, 0.01, 0.02, 0.5, 0.6], 1) is None
    assert first_ray_region([1, 2, 2, 3, 3], [1, 0.2, 0.4, 0.5, 0.6], 1) == 2
    assert first_ray_region([1, 1, 1], [1, 1, 1], 1) is None
