from unittest.mock import patch

import numpy as np
import pytest

from tests.world_model.test_tssm_runtime import setup


def fact(bank, source, target, *, event="known", direction=(0, 1), context="house"):
    return bank.record_observed(context=context, episode_id="past", event_id=event,
        source_observation="past-source", target_observation=event, execution_ref="past-execution",
        source_feature=source, observed_future=target, branch_direction=direction, executed=True, arrived=True)


def test_memory_hit_skips_world_prediction_but_real_feedback_still_updates(tmp_path):
    p, bank, session, history, target = setup(tmp_path, learning=True)
    fact(bank, history[0, -1], target)
    session.begin_episode("house", "now", history, "actual-source")
    before = session.updates.parameter_hash()
    with patch.object(p.model, "transition", side_effect=AssertionError("prediction must be skipped")):
        ticket = session.forecast("known", [0, 1])
    assert not ticket.world_model_called and ticket.prior_state is None
    np.testing.assert_array_equal(ticket.future_feature, target)
    assert session.updates.parameter_hash() == before and session.state.tokens.shape[1] == 0
    result = session.observe_event(ticket, target, target_observation="new-actual-target",
        execution_ref="actual-execution", executed=True, arrived=True)
    assert result["deferred_prior_after_actual_execution"] and session.updates.version == 1
    assert bank.count("house") == 2 and session.state.tokens.shape[1] == 1
    bank.close()


def test_conflicting_observed_outcomes_request_observation_without_predicting(tmp_path):
    p, bank, session, history, target = setup(tmp_path)
    fact(bank, history[0, -1], target)
    fact(bank, history[0, -1], np.eye(12, dtype=np.float32)[2], event="conflict")
    session.begin_episode("house", "now", history, "source")
    with patch.object(p.model, "transition", side_effect=AssertionError("ambiguous memory is not a prediction")):
        ticket = session.forecast("ambiguous", [0, 1])
    assert ticket.future_feature is None and ticket.route_reason == "inconsistent_outcomes"
    assert not session.query_actual_memories(ticket)
    with pytest.raises(ValueError, match="requires observation"):
        session.commit(ticket)
    assert bank.count("house") == 2
    bank.close()


def test_live_direction_alignment_is_required_and_transports_old_exit(tmp_path):
    _p, bank, session, history, target = setup(tmp_path)
    fact(bank, history[0, -1], target, direction=(1, 0))
    session.begin_episode("house", "now", history, "source")
    rejected = session.forecast("unresolved", [0, 1], direction_transform=lambda *_: None)
    assert rejected.future_feature is None
    accepted = session.forecast("aligned", [0, 1],
        direction_transform=lambda *_: (np.array([0, 1], dtype=np.float32), {"method": "test_actual_image_alignment"}))
    assert accepted.evidence_kind == "observed_memory_candidate"
    assert session.query_actual_memories(accepted)[0].alignment_info["method"] == "test_actual_image_alignment"
    bank.close()


def test_geometric_revisit_survives_viewpoint_similarity_below_point_98(tmp_path):
    p, bank, session, history, target = setup(tmp_path)
    source = history[0, -1].copy()
    other = np.roll(source, 1)
    other -= source * (source @ other)
    other /= np.linalg.norm(other)
    past = .8 * source + .6 * other
    fact(bank, past, target)
    session.begin_episode("house", "now", history, "source")
    # The same broad appearance candidate cannot bypass geometric verification.
    rejected = session.forecast("unaligned", [0, 1], direction_transform=lambda *_: None)
    assert rejected.world_model_called and rejected.evidence_kind == "model_prior"
    with patch.object(p.model, "transition", side_effect=AssertionError("direct geometric hit skips prior")):
        accepted = session.forecast("aligned", [0, 1], direction_transform=lambda *_:
            (np.array([0, 1], dtype=np.float32), {"method": "current_depth_pnp"}))
    assert accepted.evidence_kind == "observed_memory_candidate"
    assert accepted.memory_matches[0].source_similarity == pytest.approx(.8)
    bank.close()
