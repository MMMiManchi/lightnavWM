"""Pre-collection protocol checks. Scope permissions come from dated user records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.world_model import SCHEMA_VERSION
from experiments.world_model.collection_gate import (
    CollectionGateError,
    assert_collectable,
    assert_dev_collectable,
)
from experiments.world_model.feature_cache import FEATURE_CACHE_VERSION, FeatureContractError
from experiments.world_model.feature_pairs import PAIRS_AUTHORIZE_TRAINING
from experiments.world_model.latent_data import (
    APPROVAL_SCHEMA,
    MIN_DEV_ROUTE_GROUPS,
    MIN_DEV_SCENES,
    MIN_FIT_ROUTE_GROUPS,
    MIN_FIT_SCENES,
    SCENE_SPLIT_VERSION,
    scene_partition,
    unapproved_collection_component,
)

COLLECTION_AUTHORIZED = True
HUMAN_ACCEPTED = True
SCALE_ACCEPTED = True
BLOCKED_HUMAN_DECISION = "BLOCKED_HUMAN_DECISION"
REQUIRED_GATES = (
    "schema_unchanged",
    "sim_time_writer_records_measured_values",
    "sim_time_validator_checks_consistency",
    "budget_not_relabeled_complete",
    "train_dev_not_in_fit_collector",
    "inventory_capacity_matches_record",
    "code_and_offline_acceptance",
    "sim_clock_semantics",
    "paired_latent_interface",
    "protocol_human_acceptance",
    "scale_proposal",
    "dev_collection_path",
    "formal_collection_authorization",
    "dev_collection_authorization",
    "training_authorization",
    "eligibility_floor",
    "collector_consults_checklist",
)
ACTION_KINDS = ("stop_threshold", "turning", "mostly_forward")

# Step and episode floors for a future E2/E3 report. They are not a power
# calculation and are not implemented as a loader.
FIT_PER_ACTION_KIND = 20
FIT_NON_BUDGET_EPISODES = 20
FIT_EPISODES_PER_SCENE = 2
DEV_PER_ACTION_KIND = 10
DEV_NON_BUDGET_EPISODES = 5

INVENTORY_COUNTS = {
    "train_fit_scenes": 53,
    "train_fit_route_groups": 2977,
    "train_fit_episodes": 8938,
    "train_dev_scenes": 8,
    "train_dev_route_groups": 626,
    "train_dev_episodes": 1881,
}

_REPO = Path(__file__).resolve().parents[2]
_INVENTORY = _REPO / "experiments/world_model/splits/wm_scene_split_0.1.json"
_COLLECT = _REPO / "experiments/world_model/collect.py"
_VALIDATE = _REPO / "experiments/world_model/validate.py"
_LATENT = _REPO / "experiments/world_model/latent_data.py"
_PAIRS = _REPO / "experiments/world_model/feature_pairs.py"
_CLOCK = _REPO / "experiments/world_model/records/sim_clock_step_physics_20261027.json"
_VELOCITY_CLOCK = _REPO / "experiments/world_model/records/sim_clock_velocity_control_20261027.json"
_DECISION = _REPO / "experiments/world_model/records/human_decisions_20261027.json"
_AUTHORIZATION = _REPO / "experiments/world_model/records/collection_authorization.json"
_RESEARCH_AUTHORIZATION = _REPO / "experiments/world_model/records/world_model_authorization_20261028.json"
_CLOCK_FIELDS = {
    "world_time_after_try_step_set_state_s": 0.0,
    "world_time_after_one_step_physics_s": 0.016,
    "world_time_after_two_step_physics_s": 0.032,
    "physics_time_step_s": 0.008,
    "separate_step_world_0_1_s": 0.096,
    "horizontal_displacement_m": 0.3904685378074646,
    "habitat_env": False,
    "episode": False,
}
_VELOCITY_CLOCK_FIELDS = {
    "action_name": "velocity_control",
    "velocity_action_world_time_delta_s": 0.0,
    "horizontal_displacement_m": 0.25,
    "embodied_task_step_physics_calls": 1,
    "world_time_at_observation_after_s": 0.016,
    "option_3_window_delta_s": 0.016,
    "extra_physics_steps": 0,
    "habitat_env": False,
    "episode": False,
    "action_returned": None,
}


def assert_collection_not_blocked() -> None:
    """Refuse the real collector path. A passing protocol item does not authorize it."""
    if checklist_blocks_collection():
        raise CollectionGateError(
            "precollection",
            "precollection checklist blocks collection; a passing protocol check does not authorize it",
        )


def assert_dev_collection_authorized() -> None:
    _require_research_scope("train_dev")
    if checklist_blocks_collection():
        raise CollectionGateError("precollection", "precollection checklist blocks train_dev collection")


def assert_training_authorized() -> None:
    _require_research_scope("training")


def _require_research_scope(scope: str) -> None:
    record, error = _read_json_object(_RESEARCH_AUTHORIZATION)
    if error is not None or record is None:
        raise CollectionGateError("authorization", error or "research authorization record is missing")
    if (
        record.get("schema_version") != "wm-authorization/0.2"
        or record.get("source_split") != "train"
        or record.get("collection_authorized") is not True
        or scope not in record.get("scopes", [])
    ):
        raise CollectionGateError("authorization", f"authorization scope {scope!r} is absent")


def assert_feature_extraction_not_authorized() -> None:
    """Compatibility name retained by probes; checks explicit scope and pinned assets."""
    try:
        _require_research_scope("feature_extraction")
    except CollectionGateError as exc:
        raise FeatureContractError("authorization", str(exc)) from exc


def feature_cache_gate_state(
    *, contract_ready: bool, offline_accepted: bool, technical_ready: bool, independently_authorized: bool
) -> tuple[str, str]:
    """Classify cache readiness without reading files or granting authorization."""
    if not contract_ready:
        return "fail", "contract_missing"
    if not offline_accepted:
        return "blocked", "offline_acceptance_incomplete"
    if not technical_ready:
        return "blocked", "technical_conditions_incomplete"
    if not independently_authorized:
        return "blocked", "awaiting_independent_authorization"
    return "pass", "authorized_and_checked"


def _formal_cache_status() -> tuple[str, str]:
    """Report feature-extraction authorization separately from cache readiness."""
    writer = (_REPO / "experiments/world_model/cache_writer.py").read_text(encoding="utf-8")
    if "formal_cache" not in writer or "world_model_cache" not in writer:
        return "fail", "cache writer no longer refuses a formal cache path"
    found = [
        str(path.relative_to(_REPO))
        for name in ("formal_feature_cache",)
        for path in (_REPO / name, _REPO / "output" / name)
        if path.exists()
    ]
    if found:
        return "fail", "formal cache path exists: " + ", ".join(found)
    forward = (_REPO / "experiments/world_model/siglip_forward_probe.py").read_text(encoding="utf-8")
    repro = (_REPO / "experiments/world_model/siglip_repro_probe.py").read_text(encoding="utf-8")
    if (
        "assert_feature_extraction_not_authorized()" not in forward
        or "assert_feature_extraction_not_authorized()" not in repro
    ):
        return "fail", "SigLIP probe entry points do not consult the formal cache block"
    diagnostics = _REPO / "output" / "world_model_diagnostics"
    diagnostic_note = (
        "output/world_model_diagnostics is present and is not a formal cache"
        if diagnostics.exists()
        else "output/world_model_diagnostics is absent"
    )
    authorized = False
    try:
        _require_research_scope("feature_extraction")
        authorized = True
    except CollectionGateError:
        pass
    status, reason = feature_cache_gate_state(
        contract_ready=FEATURE_CACHE_VERSION == "wm-feature-cache/0.2",
        offline_accepted=True,
        technical_ready=authorized,
        independently_authorized=authorized,
    )
    return status, diagnostic_note + "; " + reason


def _synthetic_partition_inventory() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Small valid split for exercising both gates without reading the official dataset."""
    fit: list[str] = []
    dev: list[str] = []
    index = 0
    while len(fit) < MIN_FIT_SCENES or len(dev) < MIN_DEV_SCENES:
        scene = f"check_scene_{index}"
        side = scene_partition(scene)
        if side == "train_fit" and len(fit) < MIN_FIT_SCENES:
            fit.append(scene)
        elif side == "train_dev" and len(dev) < MIN_DEV_SCENES:
            dev.append(scene)
        index += 1
    scenes = []
    plans: dict[str, list[dict[str, str]]] = {"train_fit": [], "train_dev": []}
    for side, names, route_floor in (
        ("train_fit", fit, MIN_FIT_ROUTE_GROUPS),
        ("train_dev", dev, MIN_DEV_ROUTE_GROUPS),
    ):
        for position, scene in enumerate(names):
            count = max(1, route_floor - len(names) + 1) if position == 0 else 1
            routes = [f"{scene}:{side}:{route}" for route in range(count)]
            scenes.append(
                {
                    "scene": scene,
                    "scene_partition": side,
                    "n_route_groups": len(routes),
                    "n_episodes": len(routes),
                    "route_group_ids": routes,
                }
            )
            plans[side].append(
                {"scene": scene, "route_group_id": routes[0], "source_split": "train", "scene_partition": side}
            )
    inventory = {
        "scene_split_version": SCENE_SPLIT_VERSION,
        "source_split": "train",
        "dataset_sha256": "synthetic-checklist",
        "eligible": True,
        "scenes": scenes,
    }
    return inventory, plans["train_fit"], plans["train_dev"]


def _partition_gate_status() -> tuple[bool, bool, str]:
    inventory, fit, dev = _synthetic_partition_inventory()
    try:
        fit_accepted = assert_collectable(inventory, fit, dataset_sha256="synthetic-checklist") == fit
        dev_accepted = assert_dev_collectable(inventory, dev, dataset_sha256="synthetic-checklist") == dev
        try:
            assert_collectable(inventory, dev, dataset_sha256="synthetic-checklist")
        except CollectionGateError as exc:
            fit_rejects_dev = exc.code == "train_dev"
        else:
            fit_rejects_dev = False
        try:
            assert_dev_collectable(inventory, fit, dataset_sha256="synthetic-checklist")
        except CollectionGateError as exc:
            dev_rejects_fit = exc.code == "train_fit"
        else:
            dev_rejects_fit = False
    except CollectionGateError as exc:
        return False, False, f"synthetic inventory rejected: {exc}"
    return fit_accepted and fit_rejects_dev, dev_accepted and dev_rejects_fit, "synthetic train-only split"


def scale_quota_progress(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Current rows do not count. Measured sim time and paired latents are both absent."""
    fit_steps = {kind: 0 for kind in ACTION_KINDS}
    dev_steps = {kind: 0 for kind in ACTION_KINDS}
    return {
        "schema": SCHEMA_VERSION,
        "counts_rows": False,
        "ignored_rows": 0 if rows is None else len(rows),
        "reason": f"paired latent features are not authorized or available under {SCHEMA_VERSION}",
        "fit_scenes": 0,
        "fit_route_groups": 0,
        "fit_episodes_per_scene_min": 0,
        "fit_steps_by_action_kind": fit_steps,
        "fit_non_budget_episodes": 0,
        "dev_scenes": 0,
        "dev_route_groups": 0,
        "dev_steps_by_action_kind": dev_steps,
        "dev_non_budget_episodes": 0,
        "paired_latent_samples": 0,
    }


def _formal_run_directories() -> list[str]:
    """Non-pilot collection directories under output. This does not read their records."""
    output = _REPO / "output"
    if not output.is_dir():
        return []
    found: set[str] = set()
    data_root = output / "world_model_data"
    if data_root.is_dir():
        for path in data_root.iterdir():
            if path.is_dir() and unapproved_collection_component(path) is None:
                found.add(str(path.relative_to(_REPO)))
    for name in ("transitions.jsonl", "transitions.jsonl.partial"):
        for path in output.rglob(name):
            if unapproved_collection_component(path) is None:
                found.add(str(path.parent.relative_to(_REPO)))
    return sorted(found)


def _item(name: str, status: str, evidence: str, *, required: bool) -> dict[str, Any]:
    return {"name": name, "status": status, "evidence": evidence, "required": required}


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name} unreadable: {exc}"
    if not isinstance(record, dict):
        return None, f"{path.name} is not an object"
    return record, None


def _decision_record() -> tuple[dict[str, Any] | None, str | None]:
    return _read_json_object(_DECISION)


def _collection_authorization_status(formal_runs: list[str]) -> tuple[str, str]:
    """Keep the old decision historical; require a new record before formal runs."""
    if not COLLECTION_AUTHORIZED:
        if formal_runs:
            return "fail", "unauthorized collection directory: " + ", ".join(formal_runs)
        return BLOCKED_HUMAN_DECISION, "formal collection has no new authorization record or enabled switch"
    if not _AUTHORIZATION.is_file():
        return "fail", "COLLECTION_AUTHORIZED is true but the separate authorization record is missing"
    record, error = _read_json_object(_AUTHORIZATION)
    if error is not None or record is None:
        return "fail", error or "collection authorization record unreadable"
    if (
        record.get("collection_authorized") is not True
        or record.get("scope") != "train_fit"
        or not isinstance(record.get("decision_source"), str)
        or not record["decision_source"].strip()
        or not isinstance(record.get("recorded_on"), str)
        or not record["recorded_on"].strip()
    ):
        return "fail", "collection authorization record lacks explicit train_fit human decision"
    return "pass", "separate train_fit authorization record and switch are present"


def _clock_record_matches(path: Path, fields: Mapping[str, Any]) -> str | None:
    record, error = _read_json_object(path)
    if error is not None or record is None:
        return error or f"{path.name} unreadable"
    for key, expected in fields.items():
        if record.get(key) != expected:
            return f"{path.name} field {key} is {record.get(key)!r}"
    return None


def _clock_status() -> tuple[str, str]:
    """Compare the accepted probe with the live simulator clock contract."""
    for path, fields in ((_CLOCK, _CLOCK_FIELDS), (_VELOCITY_CLOCK, _VELOCITY_CLOCK_FIELDS)):
        mismatch = _clock_record_matches(path, fields)
        if mismatch is not None:
            return "fail", mismatch
    decision, error = _decision_record()
    if error is not None or decision is None:
        return "fail", error or "decision record unreadable"
    if decision.get("sim_time_window") != "option_3":
        return "fail", "decision record does not select option 3"
    return (
        "pass",
        "option 3 window delta is 0.016 s after one task step_physics; "
        "VelocityAction.step delta is 0; Env.step observation-window time is recorded in schema 0.2",
    )


def _contains_all(path: Path, needles: tuple[str, ...]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(needle in text for needle in needles)


def _read_inventory_counts() -> dict[str, Any]:
    document = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    return {
        "collection_status": document.get("collection_status"),
        "scene_split_version": document.get("scene_split_version"),
        "counts": document.get("counts") or {},
    }


def precollection_checklist() -> dict[str, Any]:
    """Static checklist. A passing item is a code or file fact, not collection permission."""
    inventory = _read_inventory_counts()
    counts = inventory["counts"]
    collect_source = _COLLECT.read_text(encoding="utf-8")
    serve_source = (_REPO / "experiments/world_model/serve_collect.py").read_text(encoding="utf-8")
    validate_source = _VALIDATE.read_text(encoding="utf-8")
    pair_source = _PAIRS.read_text(encoding="utf-8")
    count_match = all(counts.get(key) == value for key, value in INVENTORY_COUNTS.items())
    admission_present = _contains_all(
        _LATENT,
        (
            "def load_accepted_run",
            "training input was not loaded through load_accepted_run",
            "missing a consistent measured simulator clock",
        ),
    ) and _contains_all(
        _VALIDATE,
        ("sim_time_mismatch", "budget_labeled_complete", "RENAME_NOREPLACE"),
    )
    pair_ready = (
        PAIRS_AUTHORIZE_TRAINING is False
        and '"admits_training": False' in pair_source
        and "final_habitat_action" in pair_source
        and "image_hash_mismatch" in pair_source
        and "image_path" in pair_source
        and "_require_same_collection_identity" in pair_source
        and "feature scene or route does not match the episode" in pair_source
        and "scene_cross" in pair_source
        and "Siglip" not in pair_source
    )
    floors_locked = (MIN_FIT_SCENES, MIN_FIT_ROUTE_GROUPS, MIN_DEV_SCENES, MIN_DEV_ROUTE_GROUPS) == (
        8,
        20,
        3,
        5,
    )
    formal_runs = _formal_run_directories()
    decision, decision_error = _decision_record()
    decision_accepts_protocol = (
        decision_error is None
        and decision is not None
        and decision.get("protocol_accepted") is True
        and decision.get("scale_thresholds_accepted") == "whole_set"
        and decision.get("sim_time_window") == "option_3"
        and decision.get("collection_authorized") is False
        and decision.get("dev_collection_executed") is False
        and "2026-10-27" in str(decision.get("decision_source"))
    )
    progress = scale_quota_progress()
    quota_counted = progress["counts_rows"] or any(
        progress[name] != 0
        for name in (
            "fit_scenes",
            "fit_route_groups",
            "fit_episodes_per_scene_min",
            "fit_non_budget_episodes",
            "dev_scenes",
            "dev_route_groups",
            "dev_non_budget_episodes",
            "paired_latent_samples",
        )
    ) or any(progress["fit_steps_by_action_kind"].values()) or any(
        progress["dev_steps_by_action_kind"].values()
    )
    if decision_error is not None or decision is None or decision.get("collection_authorized") is not False:
        authorization_status = "fail"
        authorization_evidence = decision_error or "historical decision record was changed or is invalid"
    else:
        authorization_status, authorization_evidence = _collection_authorization_status(formal_runs)
    dev_source = (_REPO / "experiments/world_model/dev_collect.py").read_text(encoding="utf-8")
    fit_gate_ready, dev_gate_ready, gate_evidence = _partition_gate_status()
    dev_entry_safe = (
        "def main" in dev_source
        and '"--partition", "train_dev"' in dev_source
        and "collect_main(forwarded)" in dev_source
        and "assert_dev_collectable" in dev_source
        and "habitat.Env" not in dev_source
    )
    try:
        _require_research_scope("train_dev")
        dev_auth_status, dev_auth_evidence = "pass", "dated train_dev authorization record is present"
    except CollectionGateError as exc:
        dev_auth_status, dev_auth_evidence = "fail", str(exc)
    try:
        _require_research_scope("training")
        training_auth_status, training_auth_evidence = "pass", "dated training authorization record is present"
    except CollectionGateError as exc:
        training_auth_status, training_auth_evidence = "fail", str(exc)
    items = [
        _item(
            "schema_unchanged",
            "pass" if SCHEMA_VERSION == "wm-transition/0.2" and APPROVAL_SCHEMA == "wm-approval/0.1" else "fail",
            f"{SCHEMA_VERSION} {APPROVAL_SCHEMA}",
            required=True,
        ),
        _item(
            "collection_authorization_snapshot",
            "pass" if authorization_status == "pass" else authorization_status,
            authorization_evidence,
            required=False,
        ),
        _item(
            "sim_time_writer_records_measured_values",
            "pass"
            if '"dt_sim_s": ack.get("dt_sim_s")' in collect_source
            and '"sim_time_s": info.get("wm_sim_time_s")' in collect_source
            and "get_world_time" in collect_source
            and "sim_time_before_s" in collect_source
            and "sim_time_after_s" in collect_source
            else "fail",
            "observations and executed transitions store measured Habitat-Sim world time",
            required=True,
        ),
        _item(
            "sim_time_validator_checks_consistency",
            "pass"
            if all(
                needle in validate_source
                for needle in ("sim_time_mismatch", "np.isfinite", "sim_time_before_s", "sim_time_after_s")
            )
            else "fail",
            "validate_snapshot checks finite measured clock endpoints and delta consistency; measured zero is valid",
            required=True,
        ),
        _item(
            "budget_not_relabeled_complete",
            "pass" if "budget_labeled_complete" in validate_source else "fail",
            "runner_budget with completion completed is a hard error",
            required=True,
        ),
        _item(
            "train_dev_not_in_fit_collector",
            "pass" if fit_gate_ready else "fail",
            f"assert_collectable accepted fit and rejected dev in {gate_evidence}; no Habitat loaded",
            required=True,
        ),
        _item(
            "inventory_capacity_matches_record",
            "pass" if count_match and inventory["collection_status"] == "not_collected" else "fail",
            json.dumps(counts, sort_keys=True),
            required=True,
        ),
        _item(
            "code_and_offline_acceptance",
            "pass" if admission_present else "fail",
            "source still contains the admission refusals; the earlier offline suite was not repeated",
            required=True,
        ),
        _item("sim_clock_semantics", *_clock_status(), required=True),
        _item(
            "paired_latent_interface",
            "pass" if pair_ready else "fail",
            "alignment binds final_habitat_action, one output_name, and the episode scene and route",
            required=True,
        ),
        _item("formal_feature_cache", *_formal_cache_status(), required=False),
        _item(
            "protocol_human_acceptance",
            "pass" if HUMAN_ACCEPTED and decision_accepts_protocol else "fail",
            "2026-10-27 decision accepts the protocol and scale; separate authorization record controls train_fit collection",
            required=True,
        ),
        _item(
            "scale_proposal",
            "fail" if quota_counted or not decision_accepts_protocol else "pass" if SCALE_ACCEPTED else "fail",
            (
                f"fit scenes>={MIN_FIT_SCENES} routes>={MIN_FIT_ROUTE_GROUPS} "
                f"episodes/scene>={FIT_EPISODES_PER_SCENE} "
                f"each of {','.join(ACTION_KINDS)}>={FIT_PER_ACTION_KIND} "
                f"non_budget>={FIT_NON_BUDGET_EPISODES}; "
                f"dev all {INVENTORY_COUNTS['train_dev_scenes']} scenes "
                f"routes>={MIN_DEV_ROUTE_GROUPS} each kind>={DEV_PER_ACTION_KIND} "
                f"non_budget>={DEV_NON_BUDGET_EPISODES}. "
                "Not inventory capacity, not the loader floor, and not proof of training sufficiency."
            ),
            required=True,
        ),
        _item(
            "dev_collection_path",
            "pass" if dev_gate_ready and dev_entry_safe else "fail",
            f"assert_dev_collectable accepted dev and rejected fit in {gate_evidence}; entry does not construct an environment; executing it is not authorized",
            required=True,
        ),
        _item(
            "formal_collection_authorization",
            authorization_status,
            authorization_evidence,
            required=True,
        ),
        _item("dev_collection_authorization", dev_auth_status, dev_auth_evidence, required=True),
        _item("training_authorization", training_auth_status, training_auth_evidence, required=True),
        _item(
            "eligibility_floor",
            "pass" if floors_locked else "fail",
            (
                f"fit {MIN_FIT_SCENES} scenes/{MIN_FIT_ROUTE_GROUPS} routes; "
                f"dev {MIN_DEV_SCENES} scenes/{MIN_DEV_ROUTE_GROUPS} routes"
            ),
            required=True,
        ),
        _item(
            "collector_consults_checklist",
            "pass"
            if "assert_collection_not_blocked()" in collect_source
            and "assert_collection_not_blocked()" in serve_source
            else "fail",
            "the real collector and the habitat server consult the checklist before creating an environment",
            required=True,
        ),
    ]
    return {
        "collection_authorized": COLLECTION_AUTHORIZED,
        "human_accepted": HUMAN_ACCEPTED,
        "scale_accepted": SCALE_ACCEPTED,
        "code_fix": "complete",
        "offline_admission_tests": "passed_before_this_protocol",
        "protocol_design": "accepted_20261027",
        "schema_transition": SCHEMA_VERSION,
        "schema_approval": APPROVAL_SCHEMA,
        "qualified_formal_transitions": 0,
        "formal_run_directories": formal_runs,
        "scale_quota_progress": progress,
        "paired_latent_samples": 0,
        "inventory": inventory,
        "items": items,
    }


def trial_blockers_clear(report: Mapping[str, Any]) -> bool:
    """Pure rule. An empty item list or a missing required name stays closed.

    This does not read the live flags. Callers that decide whether collection
    may start must use ``technical_trial_ready``.
    """
    if report.get("collection_authorized") is not True:
        return False
    if report.get("human_accepted") is not True or report.get("scale_accepted") is not True:
        return False
    items = report.get("items")
    if not isinstance(items, list):
        return False
    status: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, Mapping) or "name" not in item:
            return False
        status[str(item["name"])] = item.get("status")
    return all(status.get(name) == "pass" for name in REQUIRED_GATES)


def technical_trial_ready() -> bool:
    """Live checklist only. A caller-supplied report cannot grant readiness."""
    return trial_blockers_clear(precollection_checklist())


def checklist_blocks_collection() -> bool:
    """True when starting collection must be refused."""
    return not technical_trial_ready()
