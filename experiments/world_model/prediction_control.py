"""Let branch-specific predictions affect one bounded entrance option."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionControl:
    minimum_angle_degrees: float = 10.
    minimum_retrieval_margin: float = .02
    minimum_retrieval_similarity: float = .5

    def __post_init__(self):
        if (not 0 < self.minimum_angle_degrees < 180 or not 0 <= self.minimum_retrieval_margin <= 2 or
                not -1 <= self.minimum_retrieval_similarity <= 1):
            raise ValueError("invalid prediction control thresholds")

    def assess(self, context, decision, copy_matches, current_feature=None):
        predictions = dict(context.predicted_branches)
        selected = predictions.get(decision.candidate_id)
        if selected is None or not selected.retrieved_evidence:
            return {"accepted": False, "reason": "no_retrieved_outcome"}
        cited = [m for m in selected.retrieved_evidence if m.sequence in decision.evidence_sequences]
        if not cited:
            return {"accepted": False, "reason": "planner_did_not_use_selected_outcome"}
        chosen = selected.retrieved_evidence[0]
        outcome_kind = getattr(chosen, "outcome_kind", None)
        if outcome_kind != "arrived":
            return {"accepted": False, "reason": "retrieval_is_not_verified_arrival",
                    "outcome_kind": outcome_kind}
        similarity = float(getattr(chosen, "future_similarity", -1.))
        if similarity < self.minimum_retrieval_similarity:
            return {"accepted": False, "reason": "retrieval_similarity_below_threshold",
                    "retrieval_similarity": similarity,
                    "minimum_retrieval_similarity": self.minimum_retrieval_similarity}
        alternatives = [p for key, p in predictions.items() if key != decision.candidate_id]
        angles = [math.degrees(math.acos(float(np.clip(selected.future_feature @ p.future_feature, -1, 1))))
                  for p in alternatives]
        comparison = "other_visible_exits"
        if not alternatives and current_feature is not None:
            # A narrow camera often exposes only one entrance. The meaningful
            # counterfactual then is source-copy retrieval versus imagining that
            # entrance's consequence, rather than inventing an unseen second exit.
            angles = [math.degrees(math.acos(float(np.clip(selected.future_feature @ current_feature, -1, 1))))]
            comparison = "current_observation"
        if not angles or min(angles) < self.minimum_angle_degrees:
            return {"accepted": False, "reason": "predicted_exits_not_separated", "angles": angles}
        if not copy_matches:
            return {"accepted": False, "reason": "no_copy_retrieval_comparison"}
        # Require a change in the actual retrieval ranking. Merely predicting
        # a different vector or citing the source-copy hit cannot earn control.
        copy = copy_matches[0]
        margin = float(selected.future_feature @ (chosen.observed_future - copy.observed_future))
        changed = chosen.sequence != copy.sequence and chosen.sequence in decision.evidence_sequences
        accepted = changed and margin >= self.minimum_retrieval_margin
        return {"accepted": accepted, "reason": "prediction_changed_cited_retrieval" if accepted else "copy_retrieval_not_improved",
                "angles": angles, "retrieval_margin": margin, "predicted_sequence": chosen.sequence,
                "copy_sequence": copy.sequence, "comparison": comparison,
                "retrieval_similarity": similarity,
                "minimum_retrieval_similarity": self.minimum_retrieval_similarity}
