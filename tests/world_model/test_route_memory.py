import numpy as np
import pytest

from experiments.world_model.route_memory import ObservedRouteJournal
from tests.world_model.test_tssm_runtime import setup


def motion(y, *, epoch=0, valid=True, yaw=0.):
    return {"local_pose": [0., y, yaw] if valid else None, "pose_epoch": epoch,
            "measurement": {"valid": valid, "source": "actual_rgbd_pnp"},
            "actual_execution_acknowledged": True, "continuity_broken": False}


def test_actual_span_is_retrievable_but_never_an_arrival_or_direct_exit(tmp_path):
    _, bank, session, history, target = setup(tmp_path)
    session.begin_episode("house", "first", history, "initial")
    journal = ObservedRouteJournal(session, tmp_path / "actual_motion.json")
    journal.observe("source", history[0, -1], motion(0))
    journal.observe("target", target, motion(.7))
    assert bank.count("house") == 1 and journal.records[0]["next_event_arrival_claimed"] is False
    session.end_episode()
    assert bank.retrieve(context="house", before_sequence=1, source_feature=history[0, -1],
        predicted_future=target, branch_direction=[0, 1], source_floor=-1.) == []
    matches = bank.query_actual_outcomes(context="house", before_sequence=1, predicted_future=target,
        source_feature=history[0, -1], branch_direction=[0, 1], exclude_episode="second")
    assert len(matches) == 1 and matches[0].outcome_kind == "observed_span"
    assert bank.query_actual_outcomes(context="house", before_sequence=0, predicted_future=target,
        source_feature=history[0, -1], branch_direction=[0, 1], exclude_episode="second") == []
    bank.close()


def test_rotation_or_disconnected_odometry_never_creates_a_traversed_span(tmp_path):
    _, bank, session, history, target = setup(tmp_path)
    session.begin_episode("house", "first", history, "initial")
    journal = ObservedRouteJournal(session, "actual_motion.json")
    journal.observe("a", target, motion(0))
    journal.observe("rotation", target, motion(0, yaw=1.))
    journal.observe("broken", target, motion(0, valid=False))
    journal.observe("new_epoch", target, motion(2., epoch=1))
    assert bank.count("house") == 0
    with pytest.raises(ValueError, match="RGB-D"):
        bank.record_observed(context="house", episode_id="first", event_id="fake", source_observation="s",
            target_observation="t", execution_ref="e", source_feature=np.eye(12)[0], observed_future=target,
            branch_direction=[0, 1], executed=True, arrived=False, outcome="observed_span")
    bank.close()
