"""The pre-collection protocol keeps collection scope separate from research gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.world_model.precollection_protocol as protocol
from experiments.world_model import SCHEMA_VERSION
from experiments.world_model.latent_data import APPROVAL_SCHEMA
from experiments.world_model.precollection_protocol import (
    ACTION_KINDS,
    COLLECTION_AUTHORIZED,
    DEV_NON_BUDGET_EPISODES,
    DEV_PER_ACTION_KIND,
    FIT_NON_BUDGET_EPISODES,
    FIT_PER_ACTION_KIND,
    HUMAN_ACCEPTED,
    INVENTORY_COUNTS,
    REQUIRED_GATES,
    SCALE_ACCEPTED,
    assert_collection_not_blocked,
    assert_feature_extraction_not_authorized,
    checklist_blocks_collection,
    feature_cache_gate_state,
    precollection_checklist,
    scale_quota_progress,
    technical_trial_ready,
    trial_blockers_clear,
)

_DOC = Path("docs/world_model_precollection_protocol.md")


def test_precollection_checklist_limits_collection_to_authorized_train_fit(monkeypatch):
    # This pre-collection fixture describes an empty inventory. Real experiment
    # folders in the developer's workspace must not change that starting state.
    monkeypatch.setattr(protocol, "_formal_run_directories", lambda: [])
    report = precollection_checklist()
    assert SCHEMA_VERSION == "wm-transition/0.2"
    assert APPROVAL_SCHEMA == "wm-approval/0.1"
    assert COLLECTION_AUTHORIZED is True
    assert HUMAN_ACCEPTED is True
    assert SCALE_ACCEPTED is True
    assert report["scale_accepted"] is True
    assert report["human_accepted"] is True
    assert report["collection_authorized"] is True
    assert report["qualified_formal_transitions"] == 0
    assert report["paired_latent_samples"] == 0
    assert report["protocol_design"] == "accepted_20261027"
    assert checklist_blocks_collection() is False
    assert technical_trial_ready() is True
    with pytest.raises(TypeError):
        technical_trial_ready(report)  # type: ignore[call-arg]
    status = {item["name"]: item["status"] for item in report["items"]}
    for name in (
        "schema_unchanged",
        "sim_time_writer_records_measured_values",
        "sim_time_validator_checks_consistency",
        "budget_not_relabeled_complete",
        "train_dev_not_in_fit_collector",
        "inventory_capacity_matches_record",
        "eligibility_floor",
    ):
        assert status[name] == "pass", name
    assert status["code_and_offline_acceptance"] == "pass"
    assert status["collector_consults_checklist"] == "pass"
    assert_collection_not_blocked()
    assert_feature_extraction_not_authorized() is None

    assert status["paired_latent_interface"] == "pass"
    assert status["formal_feature_cache"] == "pass"
    cache_item = next(item for item in report["items"] if item["name"] == "formal_feature_cache")
    assert cache_item["required"] is False
    assert "formal_feature_cache" not in REQUIRED_GATES
    assert status["protocol_human_acceptance"] == "pass"
    assert status["sim_clock_semantics"] == "pass"
    assert status["scale_proposal"] == "pass"
    assert status["dev_collection_path"] == "pass"
    assert status["formal_collection_authorization"] == "pass"
    assert status["collection_authorization_snapshot"] == "pass"
    scale = next(item for item in report["items"] if item["name"] == "scale_proposal")
    assert "mostly_forward" in scale["evidence"]
    assert "Not inventory capacity" in scale["evidence"]
    granted = {
        "collection_authorized": True,
        "human_accepted": True,
        "scale_accepted": True,
        "items": [{"name": name, "status": "pass"} for name in REQUIRED_GATES],
    }
    assert trial_blockers_clear(granted) is True
    granted["collection_authorized"] = False
    assert trial_blockers_clear(granted) is False
    granted["collection_authorized"] = True
    granted["items"] = []
    assert trial_blockers_clear(granted) is False
    granted["items"] = [{"name": name, "status": "pass"} for name in REQUIRED_GATES if name != "sim_clock_semantics"]
    assert trial_blockers_clear(granted) is False
    assert technical_trial_ready() is True
    decision = Path("docs/world_model_scale_decision.md").read_text(encoding="utf-8")
    assert "docs/world_model_scale_decision.md" in _DOC.read_text(encoding="utf-8")
    assert "按门槛分别决定" in decision
    assert "SCALE_ACCEPTED" in decision and "false" in decision
    assert "已整组接受" in decision
    assert "方案 3" in decision
    counts = report["inventory"]["counts"]
    for key, value in INVENTORY_COUNTS.items():
        assert counts[key] == value
    assert report["formal_run_directories"] == []
    pilot_rows = Path("output/world_model_data/pilot_20261023_train3_r2/transitions.jsonl")
    if pilot_rows.is_file():
        assert all("pilot_" not in path for path in report["formal_run_directories"])
    assert report["qualified_formal_transitions"] == 0
    assert report["scale_quota_progress"]["counts_rows"] is False
    progress = scale_quota_progress(
        [
            {"quality": {"status": "valid"}, "action": {"dt_sim_s": None, "action_kind": "mostly_forward"}},
            {"quality": {"status": "valid"}, "action": {"dt_sim_s": 0.016, "action_kind": "turning"}},
        ]
    )
    assert progress["ignored_rows"] == 2
    assert progress["fit_non_budget_episodes"] == 0
    assert progress["dev_non_budget_episodes"] == 0
    assert set(progress["fit_steps_by_action_kind"]) == set(ACTION_KINDS)
    assert all(value == 0 for value in progress["fit_steps_by_action_kind"].values())
    assert all(value == 0 for value in progress["dev_steps_by_action_kind"].values())
    assert report["inventory"]["collection_status"] == "not_collected"


def test_protocol_document_states_the_four_gates():
    text = _DOC.read_text(encoding="utf-8")
    for heading in (
        "## 1. 样本契约",
        "## 2. 数据质量门槛",
        "## 3. 数据规模与覆盖",
        "## 4. 采集前检查清单",
        "## 5. 历史 schema 变更记录",
    ):
        assert heading in text
    assert "`train_fit` 以外的正式采集未授权" in text
    assert "clock_unchanged" in text
    assert "get_world_time" in text
    assert "dt_command_s" in text
    assert "feature_layer_selected" in text
    assert f"各自至少 {FIT_PER_ACTION_KIND} 条" in text
    assert f"至少 {FIT_NON_BUDGET_EPISODES} 条非预算 episode" in text
    assert f"至少 {DEV_PER_ACTION_KIND} 条" in text
    assert f"至少 {DEV_NON_BUDGET_EPISODES} 条非预算 episode" in text
    assert "53 场景、2977 路线组、8938 episode" in text
    assert "8 场景、626 路线组、1881 episode" in text
    assert set(ACTION_KINDS) == {"stop_threshold", "turning", "mostly_forward"}
    assert "不分配版本号" in text
    assert "wm-transition/0.2" in text
    assert "实测零差值保留 `0.0`" in text
    assert "不是当前实现要求" in text
    assert "`wm-transition/0.1` 时的历史约束" in text


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({"contract_ready": False, "offline_accepted": False, "technical_ready": False, "independently_authorized": False}, ("fail", "contract_missing")),
        ({"contract_ready": True, "offline_accepted": False, "technical_ready": False, "independently_authorized": False}, ("blocked", "offline_acceptance_incomplete")),
        ({"contract_ready": True, "offline_accepted": True, "technical_ready": False, "independently_authorized": False}, ("blocked", "technical_conditions_incomplete")),
        ({"contract_ready": True, "offline_accepted": True, "technical_ready": True, "independently_authorized": False}, ("blocked", "awaiting_independent_authorization")),
        ({"contract_ready": True, "offline_accepted": True, "technical_ready": True, "independently_authorized": True}, ("pass", "authorized_and_checked")),
    ],
)
def test_feature_cache_gate_reasons_are_pure(inputs, expected):
    assert feature_cache_gate_state(**inputs) == expected
    assert protocol._formal_cache_status()[0] == "pass"


def test_partition_checklist_calls_the_fit_and_dev_gates(monkeypatch):
    original_fit = protocol.assert_collectable
    original_dev = protocol.assert_dev_collectable
    report = precollection_checklist()
    status = {item["name"]: item["status"] for item in report["items"]}
    assert status["train_dev_not_in_fit_collector"] == "pass"
    assert status["dev_collection_path"] == "pass"

    monkeypatch.setattr(protocol, "assert_collectable", lambda _inventory, episodes, **_kwargs: episodes)
    status = {item["name"]: item["status"] for item in precollection_checklist()["items"]}
    assert status["train_dev_not_in_fit_collector"] == "fail"
    assert status["dev_collection_path"] == "pass"

    monkeypatch.setattr(protocol, "assert_collectable", original_fit)
    monkeypatch.setattr(protocol, "assert_dev_collectable", lambda _inventory, episodes, **_kwargs: episodes)
    status = {item["name"]: item["status"] for item in precollection_checklist()["items"]}
    assert status["train_dev_not_in_fit_collector"] == "pass"
    assert status["dev_collection_path"] == "fail"
    monkeypatch.setattr(protocol, "assert_dev_collectable", original_dev)


def test_formal_authorization_requires_a_new_record_and_survives_run_creation(monkeypatch, tmp_path):
    record_path = tmp_path / "collection_authorization.json"
    monkeypatch.setattr(protocol, "_AUTHORIZATION", record_path)
    assert protocol._collection_authorization_status([])[0] == "fail"
    assert protocol._collection_authorization_status(["output/world_model_data/formal_run"])[0] == "fail"

    monkeypatch.setattr(protocol, "COLLECTION_AUTHORIZED", True)
    assert protocol._collection_authorization_status([])[0] == "fail"
    record_path.write_text(json.dumps({"collection_authorized": True, "scope": "train_dev"}), encoding="utf-8")
    assert protocol._collection_authorization_status([])[0] == "fail"
    record_path.write_text(
        json.dumps(
            {
                "collection_authorized": True,
                "scope": "train_fit",
                "recorded_on": "future-date",
                "decision_source": "separate human authorization in a future task",
            }
        ),
        encoding="utf-8",
    )
    assert protocol._collection_authorization_status([])[0] == "pass"
    assert protocol._collection_authorization_status(["output/world_model_data/formal_run"])[0] == "pass"
    monkeypatch.setattr(protocol, "_formal_run_directories", lambda: ["output/world_model_data/formal_run"])
    report = precollection_checklist()
    item = next(item for item in report["items"] if item["name"] == "formal_collection_authorization")
    assert item["status"] == "pass"
    assert trial_blockers_clear(report) is True
