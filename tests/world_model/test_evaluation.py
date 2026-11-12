import pytest

from experiments.world_model.evaluation import matched_navigation, scene_shards


def test_one_owner_per_scene_preserves_order_and_every_task():
    tasks = [{"scene": s, "episode_id": i} for i, s in enumerate(["a", "b", "a", "c", "b", "a"])]
    shards = scene_shards(tasks, 2)
    assert sum(map(len, shards)) == len(tasks)
    for scene in ["a", "b", "c"]:
        owners = [i for i, shard in enumerate(shards) if any(t["scene"] == scene for t in shard)]
        assert len(owners) == 1
        assert [t for t in shards[owners[0]] if t["scene"] == scene] == [t for t in tasks if t["scene"] == scene]
    with pytest.raises(ValueError, match="duplicate"):
        scene_shards(tasks + tasks[:1], 2)


def test_report_pairs_same_start_and_counts_improved_and_harmed_controlled_episodes():
    def row(i, success, spl, controlled=0):
        return {"scene": "a", "dataset_episode_id": str(i), "initial_rgb_sha256": str(i),
                "evaluation_only": {"success": success, "spl": spl}, "prediction_controlled_rounds": controlled}
    base = [row(1, 1, .8), row(2, 0, 0)]
    treatment = [row(1, 0, 0, 1), row(2, 1, .6, 2)]
    result = matched_navigation(base, treatment)
    assert result["sr_delta"] == 0 and result["spl_delta"] == pytest.approx(-.1)
    assert result["controlled_success_gained"] == result["controlled_success_lost"] == 1
    with pytest.raises(ValueError, match="identical"):
        matched_navigation(base, treatment[:1])
    treatment[0]["initial_rgb_sha256"] = "different"
    with pytest.raises(ValueError, match="observations"):
        matched_navigation(base, treatment)
