"""Task-specific completion and versioned, image-bound navigation options."""

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GoalContract:
    kind: str
    camera_relation: str
    require_current_target: bool
    instruction: str
    additional_relations: tuple[str, ...] = ()

    @classmethod
    def from_instruction(cls, instruction):
        doorway = re.search(r"\b(?:in|at|on)\s+(?:(?:the|a)\s+)?(?:doorway|threshold|entrance)\b", instruction, re.I)
        entering = re.search(r"\benter(?:ing)?\b|\b(?:go|walk|move|head)\s+into\b|\bstop\s+in\s+(?:the\s+)?(?:room|kitchen|bedroom|bathroom)", instruction, re.I)
        looking = re.search(r"\b(?:look|face|facing)\b", instruction, re.I)
        if doorway:
            return cls("stop_at_entrance", "at_entrance", False, instruction)
        if entering and looking:
            return cls("enter_and_look", "facing_target", True, instruction, ("inside_target_room",))
        if entering:
            return cls("enter_room", "inside_target_room", False, instruction)
        if looking:
            return cls("look_at", "facing_target", True, instruction)
        return cls("approach_target", "at_requested_position", True, instruction)


@dataclass(frozen=True)
class SkillCommand:
    command_id: str
    version: int
    source_observation: str
    candidate_id: str
    bearing_rad: float
    image_region: tuple | None
    issued_step: int
    mode: str
    task_key: str

    @classmethod
    def bind(cls, branch, *, version, step, task_key, mode):
        body = f"{task_key}\0{branch.source_observation}\0{branch.candidate_id}\0{version}"
        return cls(hashlib.sha256(body.encode()).hexdigest()[:20], version, branch.source_observation,
                   branch.candidate_id, branch.bearing_rad, branch.image_region, step, mode, task_key)
