"""Approved transition data and the scene-disjoint split for the world model.

``wm-scene-split/0.1`` hashes the scene id and assigns every route in that scene
to one side. Formal data admission can combine individually approved runs, but
fit and validation rows remain separate at the training API boundary. This
module does not rewrite collected JSON or start training.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, NoReturn

import numpy as np

from experiments.world_model import SCHEMA_VERSION
from experiments.world_model.baselines import action_kind, executed_twist, kinematic_delta
from experiments.world_model.jsonutil import sha256_bytes
from experiments.world_model.pose_math import body_frame_delta
from experiments.world_model.validate import (
    contained_relative,
    create_trusted_regular,
    lexical_components,
    read_trusted_regular,
    snapshot_run,
    validate_snapshot,
)

SCENE_SPLIT_VERSION = "wm-scene-split/0.1"
APPROVAL_SCHEMA = "wm-approval/0.1"
MIN_FIT_SCENES = 8
MIN_FIT_ROUTE_GROUPS = 20
MIN_DEV_SCENES = 3
MIN_DEV_ROUTE_GROUPS = 5
_APPROVAL_FILES = (
    "manifest.json",
    "validation_report.json",
    "episodes.jsonl",
    "observations.jsonl",
    "transitions.jsonl",
    "privileged_labels.jsonl",
)
_MIN_APPROVAL_KEY_BYTES = 16
FORBIDDEN_DIR_NAMES = ("r2r_probe", "go2w_expert", "instruction_triplet_20261021", "pilot_20261023_train3")
MODEL_INPUT_KEYS = ("rgb_t", "rgb_t1", "instruction_text", "executed_twist")
FORBIDDEN_INPUT_KEYS = (
    "goal_position",
    "geodesic_distance_m",
    "reference_path",
    "world_position_xyz_m",
    "success",
    "spl",
    "ndtw",
    "oracle_success",
    "decoded_waypoints",
    "selected_waypoint",
    "client_request",
    "depth",
)


class FormalSplitError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report["reason"])


def scene_partition(scene: str) -> str:
    last = int(hashlib.sha256(str(scene).encode("utf-8")).hexdigest()[-2:], 16)
    return "train_dev" if last % 10 == 0 else "train_fit"


def refuse_forbidden_path(path: Path) -> None:
    """Reject known non-train or aborted directories by a full path component.

    ``pilot_20261023_train3_r2`` is a different component from the aborted
    ``pilot_20261023_train3`` directory, so this exact-name check does not
    block the rerun. Formal training admission uses ``load_accepted_run``,
    which also rejects every ``pilot_`` directory. A manifest string is not
    an approval.
    """
    blocked = sorted(set(Path(path).resolve().parts).intersection(FORBIDDEN_DIR_NAMES))
    if blocked:
        raise FormalSplitError(
            {
                "eligible": False,
                "reason": f"refusing data path that contains {blocked}",
                "blocked": blocked,
            }
        )


def unapproved_collection_component(path: Path) -> str | None:
    """Return a path component that cannot hold formal training data.

    This is independent of the feature-cache writer. A short exact-name list
    is not the rule: every pilot directory, diagnostic directory, and bounded
    acceptance directory matches.
    """
    for part in Path(path).parts:
        name = part.lower()
        if name.startswith("pilot_") or name.startswith("siglip_repro_"):
            return part
        if name.startswith("controlled_trial_"):
            return part
        if "bounded_acceptance" in name or name.startswith("acceptance_"):
            return part
        if name in {"world_model_diagnostics", "r2r_probe", "go2w_expert", "instruction_triplet_20261021"}:
            return part
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def read_recorded_run(run_dir: Path) -> dict[str, Any]:
    """Read one collector directory. This does not admit it for training."""
    run_dir = Path(run_dir)
    refuse_forbidden_path(run_dir)
    if (run_dir / "failure.json").is_file():
        raise FormalSplitError({"eligible": False, "reason": "run has failure.json and is not an accepted set"})
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "validation_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise FormalSplitError({"eligible": False, "reason": "run is missing manifest.json or validation_report.json"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (SCHEMA_VERSION, "wm-transition/0.1"):
        raise FormalSplitError({"eligible": False, "reason": "schema_version is not a recognized collector schema"})
    if (manifest.get("source") or {}).get("split") != "train":
        raise FormalSplitError({"eligible": False, "reason": "source split is not train"})
    if report.get("hard_error_count") != 0:
        raise FormalSplitError({"eligible": False, "reason": "validator reported hard errors"})
    seed_status = (manifest.get("seed") or {}).get("status")
    if seed_status == "verified":
        raise FormalSplitError({"eligible": False, "reason": "seed status verified is not a status this collector can claim"})
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "report": report,
        "episodes": _read_jsonl(run_dir / "episodes.jsonl"),
        "observations": _read_jsonl(run_dir / "observations.jsonl"),
        "transitions": _read_jsonl(run_dir / "transitions.jsonl"),
        "privileged": _read_jsonl(run_dir / "privileged_labels.jsonl"),
        "seed_status": seed_status,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _lexical_within(path: Path, root: Path) -> bool:
    """Compare path components. This does not open or resolve either path."""
    try:
        path_parts = lexical_components(path)
        root_parts = lexical_components(root)
    except ValueError:
        return False
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def _read_frozen_credential(path: Path, *, missing: str) -> bytes:
    """Read a frozen credential through the same no-follow walk as a run."""
    try:
        return read_trusted_regular(path)
    except ValueError as exc:
        text = str(exc)
        if text.startswith("formal input is not frozen"):
            _refuse_training(text)
        if "No such file" in text:
            _refuse_training(missing)
        _refuse_training(f"credential file cannot be opened without following links: {text}")


def _refuse_training(reason: str) -> NoReturn:
    raise FormalSplitError({"eligible": False, "reason": reason})


def approval_mac(key: bytes, payload: dict[str, Any]) -> str:
    """HMAC over the approval fields. The MAC itself is not part of the payload."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def _snapshot_or_refuse(run_dir: Path) -> dict[str, bytes]:
    try:
        return snapshot_run(run_dir, frozen=True)
    except ValueError as exc:
        _refuse_training(str(exc))


def _digests_from_snapshot(files: dict[str, bytes]) -> dict[str, str]:
    """Hash every regular file that ``snapshot_run`` accepted.

    ``wm-approval/0.1`` stays the schema. The signed set is the whole snapshot,
    not only the six JSON files plus ``assets/``. A file added after approval
    changes this map. Symlinks and non-regular files never appear here: the
    snapshot refuses them instead of omitting them.
    """
    missing = [name for name in _APPROVAL_FILES if name not in files]
    if missing:
        _refuse_training(f"approval cannot bind missing file {missing[0]}")
    return {name: sha256_bytes(blob) for name, blob in files.items()}


def _file_digests(run_dir: Path) -> dict[str, str]:
    return _digests_from_snapshot(_snapshot_or_refuse(run_dir))


def _approval_payload(run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": APPROVAL_SCHEMA,
        "run_name": run_dir.name,
        "files": _file_digests(run_dir),
    }


def _read_approval_key(run_dir: Path, key_path: Path | None) -> bytes:
    if key_path is None:
        _refuse_training("approval key is missing")
    key_path = Path(key_path)
    if _lexical_within(key_path, run_dir):
        _refuse_training("approval key inside the run is not a separate credential")
    if _lexical_within(key_path, _repo_root()):
        _refuse_training("approval key inside the repository is not accepted")
    key = _read_frozen_credential(key_path, missing="approval key is missing")
    if len(key) < _MIN_APPROVAL_KEY_BYTES:
        _refuse_training("approval key is too short")
    return key


def issue_run_approval(run_dir: Path, approval_path: Path, approval_key_path: Path) -> None:
    """Write an approval next to a run, never inside it. Writing does not admit the run.

    The run and the key must already have no write bits. This function reads
    them and publishes the approval as mode ``0o444`` only when that final
    path is absent. An existing inode is not removed or replaced. Clearing
    write bits does not stop the owner, a privileged process, or a writable
    descriptor opened earlier. A parent symlink is refused. A bind mount is
    not detected.
    """
    run_dir = Path(run_dir)
    approval_path = Path(approval_path)
    if _lexical_within(approval_path, run_dir):
        _refuse_training("approval record inside the run is not independent")
    key = _read_approval_key(run_dir, approval_key_path)
    payload = _approval_payload(run_dir)
    document = dict(payload)
    document["mac"] = approval_mac(key, payload)
    try:
        create_trusted_regular(
            approval_path,
            (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    except ValueError as exc:
        _refuse_training(str(exc))


def _approval_files_ok(files: Any) -> bool:
    if not isinstance(files, dict) or not files:
        return False
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            return False
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            return False
    return True


def verify_run_approval(
    run_dir: Path,
    approval_path: Path | None,
    approval_key_path: Path | None,
    *,
    files: dict[str, str] | None = None,
) -> None:
    """Refuse a missing, damaged, copied, or stale approval. The manifest string is not read.

    ``files`` is the digest map of the snapshot already read by the caller. When it is
    present, this function does not open the run again.
    """
    run_dir = Path(run_dir)
    if approval_path is None:
        _refuse_training("approval record is missing")
    approval_path = Path(approval_path)
    if _lexical_within(approval_path, run_dir):
        _refuse_training("approval record inside the run is not independent")
    try:
        document = json.loads(
            _read_frozen_credential(approval_path, missing="approval record is missing").decode("utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FormalSplitError({"eligible": False, "reason": "approval record is damaged"}) from exc
    if not isinstance(document, dict):
        _refuse_training("approval record is damaged")
    unknown = sorted(set(document) - {"schema_version", "run_name", "files", "mac"})
    if unknown:
        _refuse_training(f"approval record has unknown fields: {unknown}")
    payload = {
        "schema_version": document.get("schema_version"),
        "run_name": document.get("run_name"),
        "files": document.get("files"),
    }
    if (
        not isinstance(payload["schema_version"], str)
        or not isinstance(payload["run_name"], str)
        or not isinstance(document.get("mac"), str)
        or not _approval_files_ok(payload["files"])
    ):
        _refuse_training("approval record is damaged")
    key = _read_approval_key(run_dir, approval_key_path)
    expected = approval_mac(key, payload)
    if len(document["mac"]) != len(expected) or not hmac.compare_digest(document["mac"], expected):
        _refuse_training("approval record was modified")
    if document.get("schema_version") != APPROVAL_SCHEMA:
        _refuse_training("approval record is damaged")
    if document.get("run_name") != run_dir.name:
        _refuse_training(f"approval run_name does not match {run_dir.name}")
    if files is None:
        files = _file_digests(run_dir)
    if document.get("files") != files:
        _refuse_training("approval digest does not match the run")


def _stored_report_matches(report: dict[str, Any], fresh: dict[str, Any], counts: dict[str, int]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        _refuse_training("validation report schema does not match the run")
    if report.get("hard_error_count") != 0:
        _refuse_training("validation report recorded hard errors")
    for name, actual in counts.items():
        if report.get(name) != actual or fresh.get(name) != actual:
            _refuse_training(f"report {name} count does not match the jsonl")
    if fresh.get("counts") != {"valid": counts["transitions"], "partial": 0, "quarantined": 0}:
        _refuse_training("revalidation counts do not match the transition records")


def _require_training_transitions(transitions: list[dict[str, Any]]) -> None:
    if not transitions:
        _refuse_training("transition record is empty")
    seen: set[str] = set()
    for row in transitions:
        transition_id = row.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id or transition_id in seen:
            _refuse_training(f"duplicate or missing transition id {transition_id!r}")
        seen.add(transition_id)
        status = (row.get("quality") or {}).get("status")
        if status != "valid":
            _refuse_training(
                f"transition {transition_id} quality {status!r} refuses the whole run"
            )


def _jsonl_snapshot(files: dict[str, bytes], name: str) -> list[dict[str, Any]]:
    from experiments.world_model.validate import _read_jsonl_text

    if name not in files:
        _refuse_training(f"approval cannot bind missing file {name}")
    return _read_jsonl_text(name, files[name])


class AdmittedFrame(Mapping[str, Any]):
    """Pixels kept as immutable bytes. Each read returns a new array.

    The retained cost is one byte string per image, the same size as the contiguous
    pixel buffer. ``read_admitted_rgb`` allocates another array of that size and does
    not keep it. Public fields are read-only; changing a returned array does not
    change the byte string. Training must use ``read_admitted_rgb`` and must not open
    ``relative_path``.
    """

    __slots__ = ("_pixel_bytes", "_shape", "_dtype", "_file_sha256", "_relative_path", "_pixel_sha256")
    _KEYS = ("relative_path", "path", "sha256", "shape", "dtype", "pixel_sha256", "pixels")

    def __init__(self, pixel_bytes: bytes, shape: tuple[int, ...], dtype: str, file_sha256: str, relative_path: str):
        digest = hashlib.sha256(pixel_bytes).hexdigest()
        object.__setattr__(self, "_pixel_bytes", pixel_bytes)
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_file_sha256", file_sha256)
        object.__setattr__(self, "_relative_path", relative_path)
        object.__setattr__(self, "_pixel_sha256", digest)

    def __setattr__(self, name: str, value: Any) -> None:
        _refuse_training("admitted frame rejects public modification")

    def __delattr__(self, name: str) -> None:
        _refuse_training("admitted frame rejects public modification")

    def __setitem__(self, key: str, value: Any) -> None:
        _refuse_training("admitted frame rejects public modification")

    def __getitem__(self, key: str) -> Any:
        if key == "pixels":
            return self.copy_pixels()
        if key in ("relative_path", "path"):
            return self._relative_path
        if key == "sha256":
            return self._file_sha256
        if key == "shape":
            return [int(dim) for dim in self._shape]
        if key == "dtype":
            return self._dtype
        if key == "pixel_sha256":
            return self._pixel_sha256
        raise KeyError(key)

    def __iter__(self) -> Any:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def copy_pixels(self) -> np.ndarray:
        digest = hashlib.sha256(self._pixel_bytes).hexdigest()
        if not hmac.compare_digest(digest, self._pixel_sha256):
            _refuse_training("admitted frame was modified")
        try:
            restored = np.frombuffer(self._pixel_bytes, dtype=np.dtype(self._dtype))
            return restored.reshape(self._shape).copy()
        except (TypeError, ValueError):
            _refuse_training("admitted frame was modified")


def _seal_frame(record: dict[str, Any], files: dict[str, bytes]) -> AdmittedFrame:
    relative = contained_relative(record.get("path"))
    blob = files.get(relative)
    if blob is None or sha256_bytes(blob) != record.get("sha256"):
        _refuse_training(f"admitted image bytes do not match {relative}")
    array = np.ascontiguousarray(np.load(io.BytesIO(blob), allow_pickle=False))
    if list(array.shape) != list(record.get("shape")) or str(array.dtype) != str(record.get("dtype")):
        _refuse_training(f"admitted image bytes do not match {relative}")
    return AdmittedFrame(
        array.tobytes(),
        tuple(int(dim) for dim in array.shape),
        str(array.dtype),
        str(record["sha256"]),
        relative,
    )


def _seal_observations(
    observations: list[dict[str, Any]], files: dict[str, bytes]
) -> tuple[list[dict[str, Any]], dict[str, AdmittedFrame]]:
    frames: dict[str, AdmittedFrame] = {}
    sealed: list[dict[str, Any]] = []
    for row in observations:
        copied = dict(row)
        for kind in ("rgb", "depth"):
            record = copied.get(kind)
            if not isinstance(record, dict):
                continue
            frame = _seal_frame(record, files)
            previous = frames.get(frame["relative_path"])
            if previous is not None and previous["sha256"] != frame["sha256"]:
                _refuse_training(f"image path {frame['relative_path']} has conflicting bytes")
            frames[frame["relative_path"]] = frame
            copied[kind] = frame
        sealed.append(copied)
    return sealed, frames


def read_admitted_rgb(frame: Any) -> np.ndarray:
    """Return a new array of the sealed pixels. This does not open a path.

    Future training code must call this function. Opening ``relative_path`` is not
    an admitted read. The returned array is a copy; writing it does not change the
    next call.
    """
    if isinstance(frame, (str, Path)):
        _refuse_training("training input refuses a filesystem path")
    if not isinstance(frame, AdmittedFrame):
        _refuse_training("training input is not an admitted frame")
    return frame.copy_pixels()


def _load_accepted_run(
    run_dir: Path,
    *,
    approval_path: Path | None = None,
    approval_key_path: Path | None = None,
    check_split: bool,
) -> dict[str, Any]:
    """Admit one run as formal training data.

    A manifest field named ``training_use`` does not grant admission. The grant
    is an approval file outside the run, bound to that directory name and to the
    digests of one snapshot. The loader
    parses JSONL, checks image bytes, and seals pixels from that same snapshot.
    It does not leave a path for training to open later. The run, approval, and
    key must have no write bits at the moment each inode is read. That check is
    not a promise that a privileged writer or an already-open writable descriptor
    cannot change the inode. Any transition that is not ``valid`` refuses the
    whole run. Scene-split floors still apply. A future trainer must read pixels
    only through ``read_admitted_rgb``; ``relative_path`` is an audit name, not a
    second copy of the approved image.
    """
    blocked = unapproved_collection_component(run_dir)
    if blocked:
        raise FormalSplitError(
            {
                "eligible": False,
                "reason": f"refusing unapproved collection path {blocked}",
                "blocked": blocked,
            }
        )
    run_dir = Path(run_dir)
    snapshot = _snapshot_or_refuse(run_dir)
    if "failure.json" in snapshot:
        _refuse_training("run has failure.json and is not an accepted set")
    try:
        manifest = json.loads(snapshot["manifest.json"].decode("utf-8"))
        report = json.loads(snapshot["validation_report.json"].decode("utf-8"))
    except KeyError as exc:
        _refuse_training(f"run is missing manifest.json or validation_report.json: {exc}")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FormalSplitError({"eligible": False, "reason": f"revalidation could not read the run: {exc}"}) from exc
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        _refuse_training("manifest or report is damaged")
    if manifest.get("diagnostic_only") is True:
        _refuse_training("diagnostic_only run is not training data")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _refuse_training("schema_version is not the accepted collector schema")
    if (manifest.get("source") or {}).get("split") != "train":
        _refuse_training("source split is not train")
    if (manifest.get("seed") or {}).get("status") == "verified":
        _refuse_training("seed status verified is not a status this collector can claim")
    digests = _digests_from_snapshot(snapshot)
    try:
        fresh = validate_snapshot(snapshot)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FormalSplitError({"eligible": False, "reason": f"revalidation could not read the run: {exc}"}) from exc
    if fresh.get("hard_error_count") != 0:
        codes = ",".join(sorted({item.get("code", "") for item in fresh.get("errors") or []}))
        _refuse_training(f"revalidation reported hard errors: {codes}")
    try:
        episodes = _jsonl_snapshot(snapshot, "episodes.jsonl")
        observations = _jsonl_snapshot(snapshot, "observations.jsonl")
        transitions = _jsonl_snapshot(snapshot, "transitions.jsonl")
        privileged = _jsonl_snapshot(snapshot, "privileged_labels.jsonl")
    except ValueError as exc:
        raise FormalSplitError({"eligible": False, "reason": f"revalidation could not read the run: {exc}"}) from exc
    lengths = {
        "episodes": len(episodes),
        "observations": len(observations),
        "transitions": len(transitions),
        "privileged_labels": len(privileged),
    }
    if report.get("schema_version") != SCHEMA_VERSION:
        _refuse_training("validation report schema does not match the run")
    if report.get("hard_error_count") != 0:
        _refuse_training("validation report recorded hard errors")
    for name, actual in lengths.items():
        if report.get(name) != actual or fresh.get(name) != actual:
            _refuse_training(f"report {name} count does not match the jsonl")
    _require_training_transitions(transitions)
    _stored_report_matches(report, fresh, lengths)
    verify_run_approval(run_dir, approval_path, approval_key_path, files=digests)
    if check_split:
        require_formal_split(episodes)
    observations, frames = _seal_observations(observations, snapshot)
    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "report": report,
        "episodes": episodes,
        "observations": observations,
        "transitions": transitions,
        "privileged": privileged,
        "admitted_frames": frames,
        "seed_status": (manifest.get("seed") or {}).get("status"),
    }


def load_accepted_run(
    run_dir: Path,
    *,
    approval_path: Path | None = None,
    approval_key_path: Path | None = None,
) -> dict[str, Any]:
    """Admit one approved run when it independently meets both split floors."""
    return _load_accepted_run(
        run_dir,
        approval_path=approval_path,
        approval_key_path=approval_key_path,
        check_split=True,
    )


def load_accepted_run_set(
    run_sources: Iterable[tuple[Path, Path, Path]],
) -> dict[str, Any]:
    """Admit approved, single-side runs together after validating combined floors.

    Each tuple is ``(run_dir, approval_path, approval_key_path)``. A run may
    contain only fit or only dev scenes; the combined set must satisfy both
    split floors. Dataset digest, inventory digest, schema and scene-split
    version must match across runs. IDs must be unique across runs. Pixel data
    remains sealed from each run's approved snapshot.
    """
    sources = list(run_sources)
    if not sources:
        _refuse_training("approved run set is empty")

    episodes: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    privileged: list[dict[str, Any]] = []
    admitted_frames: dict[str, AdmittedFrame] = {}
    seen: dict[str, set[str]] = {
        "episode_uid": set(),
        "observation_id": set(),
        "transition_id": set(),
        "label_id": set(),
    }
    run_dirs: list[Path] = []
    collection_identity: tuple[str, str, str, str, str] | None = None

    for run_dir, approval_path, approval_key_path in sources:
        run_dir = Path(run_dir)
        bundle = _load_accepted_run(
            run_dir,
            approval_path=Path(approval_path),
            approval_key_path=Path(approval_key_path),
            check_split=False,
        )
        source = bundle["manifest"].get("source") or {}
        identity_values = (
            source.get("name"),
            source.get("dataset_sha256"),
            source.get("scene_inventory_sha256"),
            source.get("scene_split_version"),
            bundle["manifest"].get("schema_version"),
        )
        if (
            any(not isinstance(value, str) or not value for value in identity_values)
            or source.get("scene_split_version") != SCENE_SPLIT_VERSION
            or any(
                len(str(digest)) != 64
                or any(char not in "0123456789abcdef" for char in str(digest))
                for digest in identity_values[1:3]
            )
        ):
            _refuse_training(f"run is missing a valid dataset/split identity: {run_dir}")
        identity = tuple(str(value) for value in identity_values)
        if collection_identity is not None and identity != collection_identity:
            _refuse_training(f"approved runs have incompatible dataset or split identities: {run_dir}")
        collection_identity = identity
        sides = {scene_partition(str(row["scene"])) for row in bundle["episodes"]}
        if len(sides) != 1:
            _refuse_training(f"run mixes scene partitions: {run_dir}")
        for rows, key in (
            (bundle["episodes"], "episode_uid"),
            (bundle["observations"], "observation_id"),
            (bundle["transitions"], "transition_id"),
            (bundle["privileged"], "label_id"),
        ):
            for row in rows:
                value = row.get(key)
                if not isinstance(value, str) or not value or value in seen[key]:
                    _refuse_training(f"duplicate or missing {key} across approved runs: {value!r}")
                seen[key].add(value)
        episodes.extend(bundle["episodes"])
        observations.extend(bundle["observations"])
        transitions.extend(bundle["transitions"])
        privileged.extend(bundle["privileged"])
        for relative_path, frame in bundle["admitted_frames"].items():
            admitted_frames[f"{run_dir}:{relative_path}"] = frame
        run_dirs.append(run_dir)

    split_report = require_formal_split(episodes)
    return {
        "run_dir": None,
        "run_dirs": run_dirs,
        "admission_mode": "approved_run_set",
        "formal_split": split_report,
        "episodes": episodes,
        "observations": observations,
        "transitions": transitions,
        "privileged": privileged,
        "admitted_frames": admitted_frames,
    }


def assign_scene_split(episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = {}
    assigned = []
    for episode in episodes:
        if episode.get("source_split") != "train":
            raise FormalSplitError({"eligible": False, "reason": f"episode {episode.get('episode_uid')} is not train"})
        scene = str(episode["scene"])
        route = str(episode["route_group_id"])
        grouped.setdefault(route, set()).add(scene)
        assigned.append({**episode, "scene_partition": scene_partition(scene), "scene_split_version": SCENE_SPLIT_VERSION})
    crossed = sorted(route for route, scenes in grouped.items() if len(scenes) != 1)
    if crossed:
        raise FormalSplitError(
            {"eligible": False, "reason": f"route group spans scenes: {crossed[:5]}", "route_scene_conflicts": crossed}
        )
    return assigned


def formal_report(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = assign_scene_split(episodes)
    fit = [row for row in rows if row["scene_partition"] == "train_fit"]
    dev = [row for row in rows if row["scene_partition"] == "train_dev"]
    fit_scenes = {row["scene"] for row in fit}
    dev_scenes = {row["scene"] for row in dev}
    fit_routes = {row["route_group_id"] for row in fit}
    dev_routes = {row["route_group_id"] for row in dev}
    shared_scenes = sorted(fit_scenes & dev_scenes)
    shared_routes = sorted(fit_routes & dev_routes)
    reasons = []
    if shared_scenes:
        reasons.append(f"scenes on both sides: {shared_scenes}")
    if shared_routes:
        reasons.append(f"route groups on both sides: {shared_routes}")
    if len(fit_scenes) < MIN_FIT_SCENES or len(fit_routes) < MIN_FIT_ROUTE_GROUPS:
        reasons.append(
            f"train_fit has {len(fit_scenes)} scenes and {len(fit_routes)} route groups; "
            f"formal training needs {MIN_FIT_SCENES} scenes and {MIN_FIT_ROUTE_GROUPS} route groups"
        )
    if len(dev_scenes) < MIN_DEV_SCENES or len(dev_routes) < MIN_DEV_ROUTE_GROUPS:
        reasons.append(
            f"train_dev has {len(dev_scenes)} scenes and {len(dev_routes)} route groups; "
            f"formal validation needs {MIN_DEV_SCENES} scenes and {MIN_DEV_ROUTE_GROUPS} route groups"
        )
    return {
        "eligible": not reasons,
        "reason": "; ".join(reasons) if reasons else "formal split floors are met",
        "scene_split_version": SCENE_SPLIT_VERSION,
        "train_fit_scenes": len(fit_scenes),
        "train_fit_route_groups": len(fit_routes),
        "train_dev_scenes": len(dev_scenes),
        "train_dev_route_groups": len(dev_routes),
        "train_fit_episodes": len(fit),
        "train_dev_episodes": len(dev),
        "shared_scenes": shared_scenes,
        "shared_routes": shared_routes,
        "episodes": rows,
    }


def require_formal_split(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    report = formal_report(episodes)
    if not report["eligible"]:
        raise FormalSplitError(report)
    return report


def assert_episode_assignment(episodes: Iterable[dict[str, Any]], episode_to_side: dict[str, str]) -> None:
    """Reject an assignment that puts one scene or one route group on both sides."""
    scene_side: dict[str, str] = {}
    route_side: dict[str, str] = {}
    for episode in episodes:
        side = episode_to_side[str(episode["episode_uid"])]
        if side not in ("train_fit", "train_dev"):
            raise FormalSplitError({"eligible": False, "reason": f"invalid side {side}"})
        scene = str(episode["scene"])
        route = str(episode["route_group_id"])
        if scene in scene_side and scene_side[scene] != side:
            raise FormalSplitError({"eligible": False, "reason": f"scene {scene} is on both sides"})
        if route in route_side and route_side[route] != side:
            raise FormalSplitError({"eligible": False, "reason": f"route group {route} is on both sides"})
        scene_side[scene] = side
        route_side[route] = side


def supervised_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per transition. A row that is not valid refuses the whole bundle."""
    observations = {row["observation_id"]: row for row in bundle["observations"]}
    poses = {row["label_id"]: row for row in bundle["privileged"]}
    episodes = {row["episode_uid"]: row for row in bundle["episodes"]}
    rows = []
    for transition in bundle["transitions"]:
        transition_id = transition.get("transition_id")
        status = (transition.get("quality") or {}).get("status")
        if status != "valid":
            _refuse_training(f"transition {transition_id} quality {status!r} refuses the whole run")
        ack = (transition.get("action") or {}).get("execution_ack") or {}
        if ack.get("status") != "executed":
            _refuse_training(f"transition {transition_id} has no executed action")
        try:
            before = observations[transition["obs_before_id"]]
            after = observations[transition["obs_after_id"]]
            pose_before = poses[transition["pose_before_id"]]["pose"]
            pose_after = poses[transition["pose_after_id"]]["pose"]
        except KeyError as exc:
            raise FormalSplitError(
                {"eligible": False, "reason": f"transition {transition_id} references a missing record {exc}"}
            ) from exc
        if before.get("rgb_missing") or after.get("rgb_missing"):
            _refuse_training(f"transition {transition_id} is missing an rgb frame")
        if pose_before.get("world_position_xyz_m") is None or pose_after.get("world_position_xyz_m") is None:
            _refuse_training(f"transition {transition_id} is missing a pose")
        episode = episodes[transition["episode_uid"]]
        outcome = transition["outcome"]
        action = transition.get("action") or {}
        try:
            sim_before = float(before["sim_time_s"])
            sim_after = float(after["sim_time_s"])
            dt_sim = float(action["dt_sim_s"])
            clock_ok = (
                np.isfinite(sim_before) and np.isfinite(sim_after) and np.isfinite(dt_sim)
                and sim_after >= sim_before and abs(dt_sim - (sim_after - sim_before)) <= 1e-9
                and float(action["sim_time_before_s"]) == sim_before
                and float(action["sim_time_after_s"]) == sim_after
            )
        except (KeyError, TypeError, ValueError):
            clock_ok = False
        if not clock_ok:
            _refuse_training(f"transition {transition_id} is missing a consistent measured simulator clock")
        completion = (episode.get("completion") or {}).get("state")
        if outcome["primary_end_source"] == "runner_budget" and completion == "completed":
            _refuse_training(
                f"transition {transition_id} records runner_budget on a trajectory marked completed"
            )
        if bundle.get("admitted_frames") is None:
            rgb_t = before["rgb"]["path"]
            rgb_t1 = after["rgb"]["path"]
        else:
            rgb_t = before["rgb"]
            rgb_t1 = after["rgb"]
            read_admitted_rgb(rgb_t)
            read_admitted_rgb(rgb_t1)
        forward, left, yaw, up = body_frame_delta(
            tuple(pose_before["world_position_xyz_m"]),
            float(pose_before["derived_yaw_rad"]),
            tuple(pose_after["world_position_xyz_m"]),
            float(pose_after["derived_yaw_rad"]),
        )
        twist = executed_twist(transition)
        measured = {"forward_m": forward, "left_m": left, "delta_yaw_rad": yaw, "up_m": up}
        row = {
            "transition_id": transition["transition_id"],
            "episode_uid": transition["episode_uid"],
            "scene": episode["scene"],
            "route_group_id": episode["route_group_id"],
            "recorded_partition": episode["partition"],
            "scene_partition": scene_partition(episode["scene"]),
            "t": transition["t"],
            "model_input": {
                "rgb_t": rgb_t,
                "rgb_t1": rgb_t1,
                "instruction_text": episode["instruction_text"],
                "executed_twist": twist,
            },
            "offline_target": {
                "delta_pose": measured,
                "kinematic_delta": kinematic_delta(transition),
                "yaw_method": pose_before.get("yaw_method"),
                "yaw_disagreement_rad": pose_before.get("yaw_disagreement_rad"),
            },
            "quality": transition["quality"]["status"],
            "primary_end_source": outcome["primary_end_source"],
            "end_sources": list(outcome["end_sources"]),
            "model_stop": transition["prediction"]["model_stop"],
            "inference_called": transition["prediction"]["inference_called"],
            "completion_state": episode["completion"]["state"],
            "action_kind": action_kind(twist),
            "waypoint_times": transition["prediction"]["waypoint_times"],
            "server_override": bool(transition["action"]["server_override"]),
        }
        leaked = FORBIDDEN_INPUT_KEYS
        present = set(row["model_input"])
        if present.intersection(leaked):
            raise FormalSplitError({"eligible": False, "reason": "model input contains a forbidden key"})
        rows.append(row)
    return rows


def admitted_training_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only fit rows from an approved bundle; never return dev rows."""
    if not bundle.get("admitted_frames"):
        _refuse_training(
            "training input was not loaded through load_accepted_run or load_accepted_run_set"
        )
    return [row for row in supervised_rows(bundle) if row["scene_partition"] == "train_fit"]


def admitted_validation_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only scene-held-out dev rows for model selection, not fitting."""
    if not bundle.get("admitted_frames"):
        _refuse_training(
            "validation input was not loaded through load_accepted_run or load_accepted_run_set"
        )
    return [row for row in supervised_rows(bundle) if row["scene_partition"] == "train_dev"]


def end_source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "rows": 0,
        "primary_unknown": 0,
        "runner_budget": 0,
        "model_stop_true": 0,
        "model_stop_null": 0,
        "inference_not_called": 0,
        "budget_labeled_as_model_stop": 0,
    }
    for row in rows:
        counts["rows"] += 1
        if row["primary_end_source"] == "unknown":
            counts["primary_unknown"] += 1
        if row["primary_end_source"] == "runner_budget":
            counts["runner_budget"] += 1
        if row["model_stop"] is True:
            counts["model_stop_true"] += 1
        if row["model_stop"] is None:
            counts["model_stop_null"] += 1
        if row["inference_called"] is False:
            counts["inference_not_called"] += 1
        if row["primary_end_source"] == "runner_budget" and row["model_stop"] is True:
            counts["budget_labeled_as_model_stop"] += 1
    return counts
