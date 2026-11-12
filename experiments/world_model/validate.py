"""Strict checks for one collection run. Isolation is not the same as a hard error.

A quarantined row with an explicit reason is counted and kept. A row that claims
to be valid without an execution ack, a frame, or an honest seed status is a
hard error.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import io
import json
import os
import secrets
import stat
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.world_model import SCHEMA_VERSION
from experiments.world_model.baselines import action_kind, executed_twist
from experiments.world_model.jsonutil import sha256_bytes
from experiments.world_model.online_input import FORBIDDEN_ONLINE_KEYS

TRANSITION_FORBIDDEN = FORBIDDEN_ONLINE_KEYS | {
    "wm_privileged",
    "goals",
    "reference_path",
    "source_geodesic_m",
}


_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def open_relative_nofollow(dir_fd: int, name: str, *, directory: bool) -> int:
    """Open one entry of ``dir_fd`` without following a symlink.

    ``O_NOFOLLOW`` applies to this name only. The caller already holds the parent
    directory fd, so replacing the name with a symlink between an earlier lstat
    and this open fails instead of following the new target.
    """
    flags = _OPEN_FLAGS | (os.O_DIRECTORY if directory else 0)
    return os.open(name, flags, dir_fd=dir_fd)


def open_checked_nofollow(
    dir_fd: int, name: str, *, directory: bool, after_lstat: Any = None
) -> int:
    """Open one entry and require it to be the inode just observed.

    ``after_lstat`` runs after lstat and before the relative open. Production
    passes none. A test passes a callable to force that one interleaving. A
    symlink is rejected by ``O_NOFOLLOW``. A different inode is refused and the
    new fd is closed. Directory link counts are not compared.
    """
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot open without following links: {name}: {exc.strerror}") from exc
    if after_lstat is not None:
        after_lstat()
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"symlink is not an admitted file: {name}")
    if directory and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"refusing non-regular file: {name}")
    if not directory and not stat.S_ISREG(info.st_mode):
        raise ValueError(f"refusing non-regular file: {name}")
    if not directory and info.st_nlink != 1:
        raise ValueError(f"hard link is not an admitted file: {name}")
    try:
        fd = open_relative_nofollow(dir_fd, name, directory=directory)
    except OSError as exc:
        raise ValueError(f"cannot open without following links: {name}: {exc.strerror}") from exc
    try:
        opened = os.fstat(fd)
        same_inode = (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino)
        if directory:
            if not same_inode or not stat.S_ISDIR(opened.st_mode):
                raise ValueError(f"directory entry was replaced before open: {name}")
        elif not same_inode or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError(f"directory entry was replaced before open: {name}")
        return fd
    except Exception:
        os.close(fd)
        raise


def lexical_components(path: Path) -> list[str]:
    """Split a path without resolving symlinks or collapsing ``..``."""
    text = os.fspath(path)
    if "\\" in text:
        raise ValueError(f"cannot be opened without following links: {text}")
    raw = Path(text)
    parts = [part for part in raw.parts if part != os.sep]
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"cannot be opened without following links: {text}")
    if raw.is_absolute():
        return parts
    cwd_parts = [part for part in Path(os.getcwd()).parts if part != os.sep]
    return cwd_parts + parts


def open_trusted(path: Path, *, directory: bool) -> int:
    """Open from the root directory fd, one component at a time.

    Every component is refused when it is a symlink. ``resolve()`` is not used.
    The caller closes the returned fd.
    """
    components = lexical_components(path)
    if not components:
        raise ValueError("cannot be opened without following links: empty path")
    try:
        fd = os.open(os.sep, _OPEN_FLAGS | os.O_DIRECTORY)
    except OSError as exc:
        raise ValueError(f"cannot be opened without following links: {exc.strerror}") from exc
    try:
        for index, name in enumerate(components):
            as_directory = directory or index != len(components) - 1
            try:
                child = open_checked_nofollow(fd, name, directory=as_directory)
            except ValueError as exc:
                raise ValueError(f"cannot be opened without following links: {exc}") from exc
            os.close(fd)
            fd = child
        return fd
    except Exception:
        os.close(fd)
        raise


def require_frozen_fd(fd: int, label: str) -> None:
    """Refuse an inode that still grants write permission through its mode.

    Formal issue and load call this immediately before reading. Clearing the
    write bits is the boundary this process can check. It does not compare
    ``st_size`` or ``st_mtime``, and it does not prove the bytes are immutable:
    the owner, ``CAP_FOWNER``, ``CAP_DAC_OVERRIDE``, or a writable descriptor
    opened before the mode changed can still write the same inode.
    """
    if os.fstat(fd).st_mode & 0o222:
        raise ValueError(f"formal input is not frozen: {label} is writable")


def _read_all(fd: int) -> bytes:
    """Read until EOF. Size and mtime are intentionally not compared."""
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def read_trusted_regular(path: Path) -> bytes:
    """Read one frozen regular file. Parent symlinks are not followed."""
    fd = open_trusted(Path(path), directory=False)
    try:
        require_frozen_fd(fd, os.fspath(path))
        return _read_all(fd)
    finally:
        os.close(fd)


_RENAME_NOREPLACE = 1
_renameat2 = None
_renameat2_lock = threading.Lock()


def rename_noreplace(old_dirfd: int, old_name: str, new_dirfd: int, new_name: str) -> None:
    """Publish ``old_name`` onto ``new_name`` only when the destination is absent.

    Linux ``renameat2(RENAME_NOREPLACE)`` is required. A missing syscall or an
    existing destination fails closed. This function never calls ``rename``.
    """
    global _renameat2
    if _renameat2 is None:
        with _renameat2_lock:
            if _renameat2 is None:
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                if not hasattr(libc, "renameat2"):
                    raise ValueError("atomic no-replace publish is unavailable")
                libc.renameat2.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
                libc.renameat2.restype = ctypes.c_int
                _renameat2 = libc.renameat2
    rc = _renameat2(
        old_dirfd,
        os.fsencode(old_name),
        new_dirfd,
        os.fsencode(new_name),
        _RENAME_NOREPLACE,
    )
    if rc == 0:
        return
    err = ctypes.get_errno()
    if err in {0, errno.ENOSYS, errno.ENOTSUP}:
        raise ValueError("atomic no-replace publish is unavailable")
    raise OSError(err, os.strerror(err), new_name)


def _entry_kind(info: os.stat_result) -> str:
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    if not stat.S_ISREG(info.st_mode):
        return "non-regular"
    if info.st_nlink != 1:
        return "hard link"
    return "file"


def _unlink_matching(dir_fd: int, name: str, file_fd: int) -> None:
    """Remove ``name`` only when it still names the inode behind ``file_fd``."""
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError:
        return
    ours = os.fstat(file_fd)
    if (info.st_dev, info.st_ino) != (ours.st_dev, ours.st_ino):
        return
    if not stat.S_ISREG(info.st_mode):
        return
    os.unlink(name, dir_fd=dir_fd)


def _publish_new_file(parent_fd: int, name: str, data: bytes) -> None:
    """Write ``data`` to a temp file in ``parent_fd``, then no-replace publish it.

    An existing final component of any type is left unchanged. The temp file is
    in the same directory, so the publish stays on one filesystem. Failure
    removes only the temp inode this call created.
    """
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise ValueError(
                f"cannot be opened without following links: {name}: {exc.strerror}"
            ) from exc
    else:
        raise ValueError(f"approval already exists: {_entry_kind(info)} {name}")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    temp = f".{name}.{secrets.token_hex(8)}.partial"
    try:
        file_fd = os.open(temp, flags, 0o444, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"approval temporary file failed: {exc.strerror}") from exc
    published = False
    try:
        remaining = data
        while remaining:
            try:
                count = os.write(file_fd, remaining)
            except OSError as exc:
                raise ValueError(f"approval record write failed: {exc.strerror}") from exc
            if count <= 0:
                raise ValueError("approval record write failed")
            remaining = remaining[count:]
        try:
            os.fsync(file_fd)
        except OSError as exc:
            raise ValueError(f"approval record sync failed: {exc.strerror}") from exc
        os.lseek(file_fd, 0, os.SEEK_SET)
        if _read_all(file_fd) != data:
            raise ValueError("approval record write failed")
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_mode & 0o222
            or opened.st_size != len(data)
        ):
            raise ValueError("formal input is not frozen: approval record is writable")
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise ValueError(
                    f"cannot be opened without following links: {name}: {exc.strerror}"
                ) from exc
        else:
            raise ValueError(f"approval already exists: {name}")
        try:
            rename_noreplace(parent_fd, temp, parent_fd, name)
        except ValueError:
            raise
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise ValueError(f"approval already exists: {name}") from exc
            raise ValueError(f"approval publish failed: {exc.strerror}") from exc
        published = True
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise ValueError(f"approval record sync failed: {exc.strerror}") from exc
    except Exception:
        if not published:
            _unlink_matching(parent_fd, temp, file_fd)
        raise
    finally:
        os.close(file_fd)


def create_trusted_regular(path: Path, data: bytes) -> None:
    """Publish one new regular file without following a symlink in any component.

    The final component must be absent. This does not unlink, truncate, or
    replace an existing file, directory, symlink, hard link, or special file.
    Bind mounts are not detected: a lexical path can still name an inode that
    is also visible somewhere else. Write-bit checks are not immutability.
    """
    components = lexical_components(path)
    if len(components) < 2:
        raise ValueError(f"cannot be opened without following links: {path}")
    name = components[-1]
    fd = os.open(os.sep, _OPEN_FLAGS | os.O_DIRECTORY)
    try:
        for part in components[:-1]:
            try:
                info = os.stat(part, dir_fd=fd, follow_symlinks=False)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise ValueError(
                        f"cannot be opened without following links: {part}: {exc.strerror}"
                    ) from exc
                os.mkdir(part, mode=0o755, dir_fd=fd)
                info = os.stat(part, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(
                    f"cannot be opened without following links: symlink or non-directory {part}"
                )
            child = open_checked_nofollow(fd, part, directory=True)
            os.close(fd)
            fd = child
        _publish_new_file(fd, name, data)
    except Exception:
        os.close(fd)
        raise
    os.close(fd)


def snapshot_run(run_dir: Path, *, frozen: bool = False) -> dict[str, bytes]:
    """Read every regular file through directory fds. Symlinks are not followed.

    The returned bytes are the only contents admission may hash, parse, or decode.
    The run path is opened from the root directory fd, one component at a time.
    A symlink, hard link, fifo, device, or socket is refused before its target is read.
    ``frozen=True`` is the formal issue and load path: each inode must have no
    write bits immediately before it is read. Validation of a live collection
    leaves this off.
    """
    root_fd = open_trusted(Path(run_dir), directory=True)
    files: dict[str, bytes] = {}
    try:
        if frozen:
            require_frozen_fd(root_fd, os.fspath(run_dir))
        _snapshot_dir(root_fd, "", files, frozen=frozen)
    finally:
        os.close(root_fd)
    return files


def _snapshot_dir(dir_fd: int, prefix: str, files: dict[str, bytes], *, frozen: bool) -> None:
    for name in sorted(os.listdir(dir_fd)):
        if name in {".", ".."} or not name or "/" in name or "\\" in name:
            continue
        relative = name if not prefix else f"{prefix}/{name}"
        try:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"cannot inspect {relative}: {exc.strerror}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink is not an admitted file: {relative}")
        if stat.S_ISDIR(info.st_mode):
            try:
                child = open_checked_nofollow(dir_fd, name, directory=True)
            except OSError as exc:
                raise ValueError(
                    f"directory cannot be opened without following links: {relative}: {exc.strerror}"
                ) from exc
            try:
                if frozen:
                    require_frozen_fd(child, relative)
                _snapshot_dir(child, relative, files, frozen=frozen)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"refusing non-regular file: {relative}")
        if info.st_nlink != 1:
            raise ValueError(f"hard link is not an admitted file: {relative}")
        try:
            file_fd = open_checked_nofollow(dir_fd, name, directory=False)
        except OSError as exc:
            raise ValueError(
                f"file cannot be opened without following links: {relative}: {exc.strerror}"
            ) from exc
        try:
            if frozen:
                require_frozen_fd(file_fd, relative)
            files[relative] = _read_all(file_fd)
        finally:
            os.close(file_fd)


def _read_jsonl_text(name: str, blob: bytes | None) -> list[dict[str, Any]]:
    if blob is None:
        return []
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not utf-8") from exc
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}:{line_number} is not json") from exc
    return rows


def contained_relative(path: Any) -> str:
    """Return a relative POSIX path that cannot leave the run directory."""
    if not isinstance(path, str) or not path or path.startswith(("/", "\\")) or "\\" in path:
        raise ValueError(f"image path escapes the run: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"image path escapes the run: {path!r}")
    return "/".join(parts)


def _add(errors: list[dict[str, str]], code: str, detail: str) -> None:
    errors.append({"code": code, "detail": detail})


def _finite_walk(value: Any, where: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_walk(item, f"{where}.{key}", errors)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_walk(item, f"{where}[{index}]", errors)
        return
    if isinstance(value, float) and not np.isfinite(value):
        _add(errors, "non_finite", where)


def _load_array_bytes(blob: bytes | None, record: dict[str, Any] | None) -> None:
    if not record:
        raise FileNotFoundError("missing image record")
    relative = contained_relative(record.get("path"))
    if blob is None:
        raise FileNotFoundError(relative)
    if sha256_bytes(blob) != record.get("sha256"):
        raise ValueError(f"hash mismatch for {relative}")
    array = np.load(io.BytesIO(blob), allow_pickle=False)
    if list(array.shape) != list(record["shape"]):
        raise ValueError(f"shape mismatch for {relative}")
    if str(array.dtype) != str(record["dtype"]):
        raise ValueError(f"dtype mismatch for {relative}")


def _failed_report(code: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "hard_error_count": 1,
        "errors": [{"code": code, "detail": detail}],
        "counts": {"valid": 0, "partial": 0, "quarantined": 0},
        "episodes": 0,
        "observations": 0,
        "transitions": 0,
        "privileged_labels": 0,
    }


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Validate the bytes read by one ``snapshot_run`` pass."""
    try:
        files = snapshot_run(Path(run_dir))
    except ValueError as exc:
        return _failed_report("path_escape", str(exc))
    return validate_snapshot(files)


def validate_snapshot(files: dict[str, bytes]) -> dict[str, Any]:
    """Validate records and image bytes that have already been read."""
    errors: list[dict[str, str]] = []
    partials = sorted(name for name in files if "/" not in name and name.endswith(".jsonl.partial"))
    if partials:
        _add(errors, "run_not_closed", ", ".join(partials))
    manifest = {}
    if "manifest.json" not in files:
        _add(errors, "manifest_missing", "manifest.json")
    else:
        manifest = json.loads(files["manifest.json"].decode("utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            _add(errors, "schema_mismatch", str(manifest.get("schema_version")))
        if (manifest.get("source") or {}).get("split") != "train":
            _add(errors, "split_not_train", str((manifest.get("source") or {}).get("split")))
        seed = manifest.get("seed") or {}
        if seed.get("status") == "verified":
            _add(errors, "seed_claimed_verified", "manifest seed status is verified")
        requested = (seed.get("requested"))
        effective = seed.get("effective") or {}
        if requested is not None and effective.get("requested_copied_as_effective") is True:
            _add(errors, "seed_claimed_verified", "requested seed was copied into effective")
        allow_keys = [tuple(item) for item in manifest.get("allowlist_keys") or []]
        served = [tuple(item) for item in manifest.get("served_keys") or []]
        resets = [tuple(item) for item in manifest.get("reset_keys") or []]
        if not allow_keys or allow_keys != served or served != resets:
            _add(
                errors,
                "allowlist_mismatch",
                f"allow={allow_keys} served={served} reset={resets}",
            )
        if len(allow_keys) != len(set(allow_keys)):
            _add(errors, "duplicate_episode", "allowlist_keys repeat")

    try:
        episodes = _read_jsonl_text("episodes.jsonl", files.get("episodes.jsonl"))
        observations = _read_jsonl_text("observations.jsonl", files.get("observations.jsonl"))
        transitions = _read_jsonl_text("transitions.jsonl", files.get("transitions.jsonl"))
        privileged = _read_jsonl_text("privileged_labels.jsonl", files.get("privileged_labels.jsonl"))
    except ValueError as exc:
        _add(errors, "jsonl_unreadable", str(exc))
        episodes, observations, transitions, privileged = [], [], [], []

    obs_by_id = {}
    for row in observations:
        _finite_walk(row, f"observation:{row.get('observation_id')}", errors)
        oid = row.get("observation_id")
        if oid in obs_by_id:
            _add(errors, "duplicate_observation", str(oid))
        obs_by_id[oid] = row
        if row.get("rgb_missing") and (row.get("rgb") or {}).get("path"):
            _add(errors, "sensor_fallback_marked_present", f"{oid} rgb")
        if row.get("depth_missing") and (row.get("depth") or {}).get("path"):
            _add(errors, "sensor_fallback_marked_present", f"{oid} depth")
        for kind in ("rgb", "depth"):
            record = row.get(kind)
            if not record:
                continue
            try:
                relative = contained_relative(record.get("path"))
                _load_array_bytes(files.get(relative), record)
            except FileNotFoundError:
                _add(errors, "image_missing", f"{oid} {kind}")
            except ValueError as exc:
                code = "path_escape" if "escapes the run" in str(exc) else "image_hash_mismatch"
                _add(errors, code, str(exc))

    referenced = set()
    for row in observations:
        for kind in ("rgb", "depth"):
            record = row.get(kind) or {}
            try:
                if record.get("path"):
                    referenced.add(contained_relative(record.get("path")))
            except ValueError:
                continue
    for relative in sorted(files):
        if relative.startswith("assets/") and relative.endswith(".npy") and relative not in referenced:
            _add(errors, "image_hash_mismatch", f"orphan asset {relative}")

    groups: dict[str, set[str]] = {}
    seen_episodes: set[str] = set()
    for row in episodes:
        episode_uid = row.get("episode_uid")
        if not isinstance(episode_uid, str) or not episode_uid or episode_uid in seen_episodes:
            _add(errors, "duplicate_episode", str(episode_uid))
        else:
            seen_episodes.add(episode_uid)
        _finite_walk(row, f"episode:{row.get('episode_uid')}", errors)
        if row.get("source_split") != "train":
            _add(errors, "split_not_train", str(row.get("episode_uid")))
        if row.get("partition") in ("eval", "val_seen", "val_unseen"):
            _add(errors, "val_contamination", str(row.get("episode_uid")))
        reset = row.get("reset") or {}
        if reset.get("applied_in_reset") is True:
            _add(errors, "seed_reset_claimed_applied", str(row.get("episode_uid")))
        group = row.get("route_group_id")
        if group:
            groups.setdefault(group, set()).add(str(row.get("partition")))
    for group, parts in groups.items():
        if len(parts) != 1:
            _add(errors, "group_partition_conflict", group)

    seen_actions: dict[str, str] = {}
    by_episode: dict[str, list[dict[str, Any]]] = {}
    counts = {"valid": 0, "partial": 0, "quarantined": 0}
    seen_labels: set[str] = set()
    priv: dict[Any, dict[str, Any]] = {}
    for item in privileged:
        label_id = item.get("label_id")
        if not isinstance(label_id, str) or not label_id or label_id in seen_labels:
            _add(errors, "duplicate_label", str(label_id))
            continue
        seen_labels.add(label_id)
        priv[label_id] = item
    for row in transitions:
        _finite_walk(row, f"transition:{row.get('transition_id')}", errors)
        leaked = TRANSITION_FORBIDDEN.intersection(row)
        online = row.get("online_input") or {}
        leaked |= TRANSITION_FORBIDDEN.intersection(online.get("info_keys") or [])
        if leaked:
            _add(errors, "privileged_in_online", f"{row.get('transition_id')} {sorted(leaked)}")
        status = (row.get("quality") or {}).get("status")
        if status not in counts:
            _add(errors, "unknown_quality", str(status))
        else:
            counts[status] += 1
        action_id = (row.get("action") or {}).get("action_id")
        ack = ((row.get("action") or {}).get("execution_ack")) or {}
        if not action_id:
            _add(errors, "action_id_missing", str(row.get("transition_id")))
        elif action_id in seen_actions:
            _add(errors, "action_id_duplicate", action_id)
        else:
            seen_actions[action_id] = str(row.get("transition_id"))
        if ack.get("action_id") and action_id and ack.get("action_id") != action_id:
            _add(errors, "action_id_mismatch", action_id)
        if ack.get("status") not in ("executed", "error"):
            _add(errors, "ack_not_executed", f"{row.get('transition_id')} status={ack.get('status')!r}")
        prediction = row.get("prediction") or {}
        if prediction.get("inference_called") is False:
            stale = any(
                prediction.get(key) not in (None, False)
                for key in (
                    "raw_text",
                    "rvq_tokens",
                    "rvq_codes",
                    "decoded_waypoints",
                    "selected_index",
                    "selected_waypoint",
                )
            )
            if stale or prediction.get("model_stop") is not None or prediction.get("decode_status") != "not_called":
                _add(errors, "stale_prediction", str(row.get("transition_id")))
        if prediction.get("decode_status") == "decode_failed" and prediction.get("model_stop") is True:
            _add(errors, "stop_source_confused", str(row.get("transition_id")))
        if prediction.get("model_stop") is True and prediction.get("decode_status") != "decode_ok":
            _add(errors, "stop_source_confused", str(row.get("transition_id")))
        if status == "valid":
            if ack.get("status") != "executed" or ack.get("executed") is not True:
                _add(errors, "ack_not_executed", str(row.get("transition_id")))
            before = obs_by_id.get(row.get("obs_before_id"))
            after = obs_by_id.get(row.get("obs_after_id"))
            if before is None or after is None:
                _add(errors, "temporal_break", f"{row.get('transition_id')} missing frame")
            else:
                if before.get("episode_uid") != row.get("episode_uid") or after.get("episode_uid") != row.get("episode_uid"):
                    _add(errors, "temporal_break", str(row.get("transition_id")))
                if after.get("step_index") != before.get("step_index", -1) + 1:
                    _add(errors, "temporal_break", f"{row.get('transition_id')} step index")
                if not before.get("rgb") or not after.get("rgb"):
                    _add(errors, "image_missing", str(row.get("transition_id")))
                if before.get("rgb_missing") or after.get("rgb_missing"):
                    _add(errors, "sensor_fallback_marked_present", str(row.get("transition_id")))
            if row.get("episode_uid") not in seen_episodes:
                _add(errors, "reference_mismatch", f"{row.get('transition_id')} unknown episode")
            pose_before = row.get("pose_before_id")
            pose_after = row.get("pose_after_id")
            if pose_before not in priv or pose_after not in priv:
                _add(errors, "pose_missing", str(row.get("transition_id")))
            else:
                pairs = (
                    (pose_before, row.get("obs_before_id")),
                    (pose_after, row.get("obs_after_id")),
                )
                for label_id, observation_id in pairs:
                    pose_row = priv[label_id]
                    pose = pose_row.get("pose") or {}
                    if pose.get("world_position_xyz_m") is None:
                        _add(errors, "pose_missing", label_id)
                    same_obs = pose_row.get("observation_id") == observation_id
                    same_episode = pose_row.get("episode_uid") == row.get("episode_uid")
                    if not same_obs or not same_episode:
                        _add(errors, "reference_mismatch", f"{row.get('transition_id')} {label_id}")
            action = row.get("action") or {}
            final_action = action.get("final_habitat_action")
            if final_action is None:
                _add(errors, "ack_not_executed", f"{row.get('transition_id')} missing final action")
            before_time = before.get("sim_time_s") if before is not None else None
            after_time = after.get("sim_time_s") if after is not None else None
            try:
                dt_sim = float(action["dt_sim_s"])
                times_ok = (
                    before_time is not None
                    and after_time is not None
                    and np.isfinite(float(before_time))
                    and np.isfinite(float(after_time))
                    and np.isfinite(dt_sim)
                    and float(after_time) >= float(before_time)
                    and abs(dt_sim - (float(after_time) - float(before_time))) <= 1e-9
                    and float(action["sim_time_before_s"]) == float(before_time)
                    and float(action["sim_time_after_s"]) == float(after_time)
                )
            except (KeyError, TypeError, ValueError):
                times_ok = False
            if not times_ok:
                _add(errors, "sim_time_mismatch", str(row.get("transition_id")))
            waypoints = prediction.get("decoded_waypoints")
            selected = prediction.get("selected_index")
            if prediction.get("decode_status") == "decode_ok":
                if not isinstance(waypoints, list) or selected is None:
                    _add(errors, "selected_index", str(row.get("transition_id")))
                elif not (0 <= int(selected) < len(waypoints)):
                    _add(errors, "selected_index", str(row.get("transition_id")))
                elif waypoints[int(selected)] != prediction.get("selected_waypoint"):
                    _add(errors, "selected_index", f"{row.get('transition_id')} row mismatch")
            sources = (row.get("outcome") or {}).get("end_sources") or []
            if "decode_fallback" in sources and "model_stop" in sources:
                _add(errors, "stop_source_confused", str(row.get("transition_id")))
            if row.get("obs_after_id") in (online.get("observation_ids") or []):
                _add(errors, "temporal_break", f"{row.get('transition_id')} used the future frame")
        elif status == "quarantined" and not (row.get("quality") or {}).get("errors"):
            _add(errors, "quarantine_without_reason", str(row.get("transition_id")))
        by_episode.setdefault(row.get("episode_uid"), []).append(row)

    for episode_uid, rows in by_episode.items():
        ordered = sorted(rows, key=lambda item: int(item.get("t") or 0))
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.get("quality", {}).get("status") != "valid" or later.get("quality", {}).get("status") != "valid":
                continue
            if int(later["t"]) != int(earlier["t"]) + 1:
                _add(errors, "temporal_break", f"{episode_uid} t gap")
            if later.get("obs_before_id") != earlier.get("obs_after_id"):
                _add(errors, "temporal_break", f"{episode_uid} frame chain")
        episode = next((item for item in episodes if item.get("episode_uid") == episode_uid), {})
        state = (episode.get("completion") or {}).get("state")
        budget = any(
            (row.get("outcome") or {}).get("primary_end_source") == "runner_budget" for row in rows
        )
        if budget and state == "completed":
            _add(errors, "budget_labeled_complete", str(episode_uid))

    action_kinds: Counter[str] = Counter()
    primary_end_sources: Counter[str] = Counter()
    end_source_events: Counter[str] = Counter()
    completion_states: Counter[str] = Counter()
    for episode in episodes:
        state = (episode.get("completion") or {}).get("state") or "unknown"
        completion_states[str(state)] += 1
    for row in transitions:
        outcome = row.get("outcome") or {}
        primary_end_sources[str(outcome.get("primary_end_source") or "unknown")] += 1
        end_source_events.update(str(source) for source in (outcome.get("end_sources") or []))
        if (row.get("quality") or {}).get("status") == "valid":
            try:
                action_kinds[action_kind(executed_twist(row))] += 1
            except (KeyError, TypeError, ValueError):
                action_kinds["unclassified"] += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "hard_error_count": len(errors),
        "errors": errors,
        "counts": counts,
        "episodes": len(episodes),
        "observations": len(observations),
        "transitions": len(transitions),
        "privileged_labels": len(privileged),
        "collection_report": {
            "valid_action_kind_counts": {
                name: action_kinds.get(name, 0)
                for name in ("stop_threshold", "turning", "mostly_forward")
            } | ({"unclassified": action_kinds["unclassified"]} if action_kinds["unclassified"] else {}),
            "primary_end_source_counts": dict(sorted(primary_end_sources.items())),
            "end_source_event_counts": dict(sorted(end_source_events.items())),
            "episode_completion_counts": dict(sorted(completion_states.items())),
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a world-model collection run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = validate_run(Path(args.run_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and report["hard_error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
