"""Route current exits to observed memory, prediction, or active observation."""

from dataclasses import dataclass, replace

from experiments.world_model.scene_memory import unit_vector


@dataclass(frozen=True)
class MemoryRoute:
    kind: str
    reason: str
    matches: tuple = ()
    inspected: int = 0


@dataclass(frozen=True)
class MemoryRoutingPolicy:
    source_floor: float = 0.98
    candidate_floor: float = 0.5
    branch_floor: float = 0.98
    outcome_agreement: float = 0.9
    maximum_candidates: int = 128
    geometry_source_floor: float = 0.5

    def resolve(self, bank, *, context, cutoff, episode_id, current, direction, transform=None):
        candidates = bank.retrieve(context=context, before_sequence=cutoff, source_feature=current,
            predicted_future=current, branch_direction=direction, exclude_episode=episode_id,
            source_floor=self.candidate_floor, query_kind="visual", top_k=self.maximum_candidates)
        if len(candidates) == self.maximum_candidates:
            return MemoryRoute("observe", "candidate_limit_prevents_consistency_check", inspected=len(candidates))
        aligned, unresolved = [], False
        for item in candidates:
            branch, info = item.branch_direction, None
            if transform is not None:
                result = transform(item.source_observation, branch.copy())
                if result is None:
                    unresolved |= item.source_similarity >= self.source_floor
                    continue
                branch, info = result
                branch = unit_vector(branch, 2)
            similarity = float(direction @ branch)
            if similarity >= self.branch_floor:
                aligned.append(replace(item, branch_similarity=similarity,
                    aligned_branch_direction=branch.copy(), alignment_info=info,
                    score=0.8 * item.source_similarity + 0.2 * similarity))
        # Embedding similarity proposes a revisit. Actual image geometry can
        # establish its camera/exit alignment even after the viewpoint changed.
        # Without image alignment retain the stricter appearance-only rule.
        strong = sorted((m for m in aligned if m.source_similarity >= self.source_floor or
                         (transform is not None and m.alignment_info is not None and
                          m.source_similarity >= self.geometry_source_floor)),
                        key=lambda m: (-m.score, -m.sequence))
        if strong:
            conflict = any(float(a.observed_future @ b.observed_future) < self.outcome_agreement
                           for i, a in enumerate(strong) for b in strong[i + 1:])
            if conflict or unresolved:
                return MemoryRoute("observe", "inconsistent_outcomes" if conflict else "unresolved_source_alignment",
                                   tuple(strong), len(candidates))
            return MemoryRoute("memory", "consistent_observed_source_and_exit", tuple(strong), len(candidates))
        if unresolved:
            return MemoryRoute("observe", "source_looks_familiar_but_alignment_unverified", inspected=len(candidates))
        return MemoryRoute("predict", "no_reliable_observed_exit", tuple(aligned), len(candidates))
