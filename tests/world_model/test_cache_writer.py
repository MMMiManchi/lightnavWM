"""The feature writer must fail before it creates a file."""

from __future__ import annotations

from pathlib import Path

import pytest

import experiments.world_model.cache_writer as cache_writer
from experiments.world_model.cache_writer import commit_feature_row, verify_bound_files
from experiments.world_model.feature_cache import (
    AUDITED_PREPROCESSOR,
    SIGLIP_PREPROCESS_SHA256,
    SIGLIP_WEIGHT_SHA256,
    FeatureContractError,
)
from experiments.world_model.jsonutil import loads, sha256_file
from tests.world_model.test_encoder_and_split_audit import _formal_episodes, _manifest, _row

WEIGHT = (
    Path("~/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots")
    / "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    / "model.safetensors"
).expanduser()


@pytest.fixture(autouse=True)
def _cache_root_is_test_local(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_writer, "FEATURE_CACHE_ROOT", tmp_path)


def _commit(tmp_path: Path, row: dict, manifest: dict, destination: Path, weight: Path | None = None):
    episodes, _episode = _formal_episodes()
    commit_feature_row(
        destination,
        row,
        manifest,
        image_root=tmp_path,
        split_episodes=episodes,
        weight_path=weight or Path(manifest["weight_path"]),
    )


def test_audited_preprocessor_file_is_the_hashed_source(tmp_path: Path):
    assert sha256_file(AUDITED_PREPROCESSOR) == SIGLIP_PREPROCESS_SHA256
    other = tmp_path / "preprocessor.json"
    other.write_bytes(AUDITED_PREPROCESSOR.read_bytes())
    with pytest.raises(FeatureContractError, match="preprocessor must be"):
        verify_bound_files(tmp_path / "missing-weight", other)
    dummy = tmp_path / "weights.safetensors"
    dummy.write_bytes(b"not-the-audited-weight")
    with pytest.raises(FeatureContractError, match="weight bytes"):
        verify_bound_files(dummy)


def test_tampered_rows_create_nothing(tmp_path: Path):
    image = tmp_path / "rgb.npy"
    image.write_bytes(b"rgb-bytes")
    episodes, episode = _formal_episodes()
    manifest = _manifest(tmp_path)
    destination = tmp_path / "row.jsonl"
    original = _row(image, episode)

    def refuse(row):
        with pytest.raises(FeatureContractError):
            commit_feature_row(
                destination,
                row,
                manifest,
                image_root=tmp_path,
                split_episodes=episodes,
                weight_path=Path(manifest["weight_path"]),
            )
        assert not destination.exists()

    replaced = _row(image, episode)
    image.write_bytes(b"different-bytes")
    refuse(replaced)
    image.write_bytes(b"rgb-bytes")

    refuse(_row(image, episode, weight_sha256="0" * 64))
    refuse(_row(image, episode, revision="deadbeef"))
    refuse(_row(image, episode, preprocess_sha256="1" * 64))
    refuse(_row(image, episode, feature=[0.0, 1.0]))
    values = [0.0] * 1152
    values[0] = float("nan")
    refuse(_row(image, episode, feature=values))
    values[0] = float("inf")
    refuse(_row(image, episode, feature=values))
    refuse(_row(image, episode, source_run="output/world_model_data/pilot_20261023_train3_r2"))
    refuse(_row(image, episode, source_split="val_unseen"))
    refuse(_row(image, episode, goal_position=[1.0, 2.0, 3.0]))
    refuse(_row(image, episode, diagnostic_only=True))
    refuse(_row(image, episode, source_run="output/world_model_diagnostics/siglip_repro_x"))
    selected = dict(manifest)
    selected["feature_layer_selected"] = "pooler_output"
    with pytest.raises(FeatureContractError, match="feature_layer_selected"):
        commit_feature_row(
            destination,
            original,
            selected,
            image_root=tmp_path,
            split_episodes=episodes,
            weight_path=Path(manifest["weight_path"]),
        )
    assert not destination.exists()
    assert original["feature_sha256"]


def test_existing_diagnostic_and_formal_paths_are_not_overwritten(tmp_path: Path):
    image = tmp_path / "rgb.npy"
    image.write_bytes(b"rgb-bytes")
    _episodes, episode = _formal_episodes()
    manifest = _manifest(tmp_path)
    destination = tmp_path / "kept.jsonl"
    destination.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FeatureContractError, match="cache_exists"):
        _commit(tmp_path, _row(image, episode), manifest, destination)
    assert destination.read_text(encoding="utf-8") == "keep\n"

    formal = tmp_path / "world_model_diagnostics" / "row.jsonl"
    with pytest.raises(FeatureContractError, match="formal_cache"):
        _commit(tmp_path, _row(image, episode), manifest, formal)
    assert not formal.exists()


def test_interrupted_write_removes_the_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    image = tmp_path / "rgb.npy"
    image.write_bytes(b"rgb-bytes")
    _episodes, episode = _formal_episodes()
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        "experiments.world_model.cache_writer.verify_bound_files",
        lambda *_args, **_kwargs: {
            "weight_sha256": SIGLIP_WEIGHT_SHA256,
            "preprocess_sha256": SIGLIP_PREPROCESS_SHA256,
        },
    )

    def boom(_fd, _payload):
        raise OSError("disk full")

    monkeypatch.setattr("experiments.world_model.cache_writer.os.write", boom)
    destination = tmp_path / "partial.jsonl"
    with pytest.raises(OSError, match="disk full"):
        _commit(tmp_path, _row(image, episode), manifest, destination)
    assert not destination.exists()


def test_real_weight_is_hashed_before_the_file_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not WEIGHT.is_file():
        pytest.fail("SigLIP weight is not on disk; the writer success path was not executed")
    image = tmp_path / "rgb.npy"
    image.write_bytes(b"rgb-bytes")
    _episodes, episode = _formal_episodes()
    manifest = _manifest(tmp_path)
    seen: list[Path] = []
    real = verify_bound_files

    def wrapped(weight_path, preprocess_path=AUDITED_PREPROCESSOR):
        seen.append(Path(weight_path))
        assert destination.exists() is False
        return real(weight_path, preprocess_path)

    monkeypatch.setattr("experiments.world_model.cache_writer.verify_bound_files", wrapped)
    destination = tmp_path / "committed.jsonl"
    _commit(tmp_path, _row(image, episode), manifest, destination, weight=WEIGHT)
    assert seen == [WEIGHT]
    body = loads(destination.read_text(encoding="utf-8"))
    assert body["diagnostic_only"] is False
    assert body["feature_layer_selected"] is None
    assert body["weight_sha256"] == SIGLIP_WEIGHT_SHA256
    assert body["preprocess_sha256"] == SIGLIP_PREPROCESS_SHA256
    before = destination.read_bytes()
    with pytest.raises(FeatureContractError, match="cache_exists"):
        _commit(tmp_path, _row(image, episode), manifest, destination, weight=WEIGHT)
    assert destination.read_bytes() == before
