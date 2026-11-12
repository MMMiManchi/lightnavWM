"""Allowlist keys for replaying a chosen set of VLN-CE episodes."""

from types import SimpleNamespace

from lightnav_habitat.vlnce import episode_allow_key, filter_episodes_by_allowlist


def test_allow_key_uses_scene_directory_not_glb_name():
    scene = "data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb"
    assert episode_allow_key(scene, 496) == ("2azQ1b91cZZ", "496")


def test_filter_keeps_only_listed_episodes():
    episodes = [
        SimpleNamespace(scene_id="mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb", episode_id="496"),
        SimpleNamespace(scene_id="mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb", episode_id="1"),
    ]
    kept = filter_episodes_by_allowlist(episodes, {("2azQ1b91cZZ", "496")})
    assert [ep.episode_id for ep in kept] == ["496"]
