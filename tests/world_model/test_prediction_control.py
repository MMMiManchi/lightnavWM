from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.world_model.event_runtime import BranchEvidence
from experiments.world_model.prediction_control import PredictionControl
from experiments.world_model.task_planning import TaskBranchDecision


def context():
    first, second = np.eye(3, dtype=np.float32)[:2]
    copy = SimpleNamespace(sequence=1, observed_future=first, future_similarity=.62, outcome_kind="arrived")
    hit = SimpleNamespace(sequence=2, observed_future=second, future_similarity=.67, outcome_kind="arrived")
    a = BranchEvidence(first, (), (copy,), 2, False, "tssm_model_prior")
    b = BranchEvidence(second, (), (hit,), 2, False, "tssm_model_prior")
    return SimpleNamespace(predicted_branches=(("A", a), ("B", b))), (copy, hit)


def test_prediction_must_change_cited_retrieval_to_gain_control():
    c, matches = context()
    d = TaskBranchDecision("task", "B", "enter visible B", (2,))
    assert PredictionControl().assess(c, d, matches)["accepted"]
    assert not PredictionControl().assess(c, replace(d, evidence_sequences=()), matches)["accepted"]
    assert not PredictionControl().assess(c, replace(d, candidate_id="A", evidence_sequences=(1,)), matches)["accepted"]
    c.predicted_branches = (("B", dict(c.predicted_branches)["B"]),)
    assert not PredictionControl().assess(c, d, matches)["accepted"]
    single = PredictionControl().assess(c, d, matches, current_feature=matches[0].observed_future)
    assert single["accepted"] and single["comparison"] == "current_observation"


def test_identical_predictions_cannot_gain_control_even_if_they_cite_a_memory():
    c, matches = context()
    a, b = [p for _, p in c.predicted_branches]
    c.predicted_branches = (("A", replace(a, future_feature=b.future_feature)), ("B", b))
    assert not PredictionControl().assess(c, TaskBranchDecision("task", "B", "enter B", (2,)), matches)["accepted"]
    with pytest.raises(ValueError):
        PredictionControl(minimum_angle_degrees=0)


def test_low_similarity_retrieval_cannot_gain_control():
    c, matches = context()
    first, selected = [p for _, p in c.predicted_branches]
    weak = SimpleNamespace(sequence=2, observed_future=matches[1].observed_future,
                           future_similarity=.3607, outcome_kind="arrived")
    c.predicted_branches = (("A", first), ("B", replace(selected, retrieved_evidence=(weak,))))
    decision = TaskBranchDecision("task", "B", "enter visible B", (2,))

    result = PredictionControl().assess(c, decision, matches)
    assert result["accepted"] is False
    assert result["reason"] == "retrieval_similarity_below_threshold"
    assert result["retrieval_similarity"] == pytest.approx(.3607)
    assert PredictionControl(minimum_retrieval_similarity=.35).assess(c, decision, matches)["accepted"]




def test_route_span_analogy_cannot_directly_control_an_entrance():
    c, matches = context()
    first, selected = [p for _, p in c.predicted_branches]
    route_span = SimpleNamespace(sequence=2, observed_future=matches[1].observed_future,
                                 future_similarity=.89, outcome_kind="observed_span")
    c.predicted_branches = (("A", first), ("B", replace(selected, retrieved_evidence=(route_span,))))
    decision = TaskBranchDecision("task", "B", "enter visible B", (2,))

    result = PredictionControl().assess(c, decision, matches)
    assert result["accepted"] is False
    assert result["reason"] == "retrieval_is_not_verified_arrival"
    assert result["outcome_kind"] == "observed_span"
@pytest.mark.parametrize("threshold", [-1.01, 1.01])
def test_retrieval_similarity_threshold_must_be_cosine_bounded(threshold):
    with pytest.raises(ValueError):
        PredictionControl(minimum_retrieval_similarity=threshold)
