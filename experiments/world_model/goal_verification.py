"""Check the requested camera relation and independently ground visible targets."""

import json
import math
import re
from dataclasses import asdict

import numpy as np

from experiments.world_model.task_progress import QwenProgressVerifier, StageEvidence
from experiments.world_model.visual_harness import parse_object

OBJECTS = {
    "television": r"\b(?:television|tv)\b", "table": r"\btable\b", "chair": r"\bchair\b",
    "sofa": r"\b(?:sofa|couch)\b", "bed": r"\bbed\b", "sink": r"\bsink\b", "toilet": r"\btoilet\b",
    "bathtub": r"\b(?:bathtub|tub)\b", "stairs": r"\b(?:stairs|staircase)\b", "picture": r"\b(?:picture|painting)\b",
    "cabinet": r"\bcabinet\b", "door": r"\bdoor\b", "window": r"\bwindow\b", "mirror": r"\bmirror\b",
}


class QwenStrictProgressVerifier:
    def __init__(self, vision, images):
        self.vision, self.images = vision, images
        self.intermediate = QwenProgressVerifier(vision, images)
        self.last_trace = {}

    def __call__(self, progress, current_reference):
        if progress.active.kind != "goal":
            evidence = self.intermediate(progress, current_reference)
            self.last_trace = self.intermediate.last_trace
            return evidence
        instruction = progress.active.instruction
        doorway = bool(re.search(r"\b(?:in|at|on)\s+(?:(?:the|a)\s+)?(?:doorway|threshold|entrance)\b", instruction, re.I))
        inside = not doorway and bool(re.search(r"\benter(?:ing)?\b|\b(?:go|walk|move|head)\s+into\b", instruction, re.I))
        required = "at_entrance" if doorway else "inside_target_room" if inside else "at_requested_position"
        targets = [name for name, pattern in OBJECTS.items() if re.search(pattern, instruction, re.I)]
        expected = targets[-1] if len(targets) == 1 else None
        context = {"original_instruction": progress.instruction, "final_stage": asdict(progress.active),
            "required_camera_relation": required,
            "recent_actual_commands": list(progress.executions.values())[-12:]}
        prompt = '''Audit FINAL TASK ARRIVAL from actual first-person observations.
Target visibility and physical task completion are SEPARATE questions. Seeing a room's landmark THROUGH a doorway
does not mean the camera has entered that room. A threshold, hallway or exterior view is not inside the target room.
For enter-room instructions require visual evidence that the camera crossed the relevant threshold and is inside.
If the instruction explicitly requests stopping IN a doorway, at_entrance is the requested relation; do not require entering the room.
For stopping-position instructions require the actual requested spatial relation, not merely seeing or approaching furniture.
Images can contain black missing-mesh triangles: these are not televisions, pictures or landmarks. A dark blob without
recognizable object structure is unconfirmed. Policy STOP and issued commands do not prove arrival. Admit uncertainty.
Return JSON with required fields: reason (specific identity AND spatial evidence or uncertainty), target_visible
(boolean), relation_satisfied (boolean), camera_relation (string), target_bbox (array or null), confidence (0..1).
There are no default verdicts. Target visibility and relation are independent: a target can be visible even
when the requested relation is not achieved. State both accurately instead of making all flags agree.
camera_relation is inside_target_room, at_entrance, outside_target_room, at_requested_position, or unknown.
If a target object is identifiable, give its tight target_bbox as [x1,y1,x2,y2] in 0..1000 normalized coordinates.
CONTEXT: ''' + json.dumps(context)
        initial = min(progress.observations, key=progress.observations.get)
        step = progress.observations[current_reference]
        recent = []
        for offset in [6, 1]:
            candidates = [(t, ref) for ref, t in progress.observations.items() if t <= step - offset]
            if candidates:
                recent.append(max(candidates)[1])
        refs = list(dict.fromkeys([initial, progress.anchor, *recent, current_reference]))
        items = [("ACTUAL CURRENT VIEW" if ref == current_reference else "ACTUAL EARLIER TASK VIEW",
                  self.images.get(ref)) for ref in refs]
        raw = self.vision(prompt, items)
        value = parse_object(raw)
        if not all(isinstance(value[k], bool) for k in ["target_visible", "relation_satisfied"]):
            raise ValueError("goal audit requires explicit boolean identity and relation")
        relation = value["camera_relation"]
        if relation not in {"inside_target_room", "at_entrance", "outside_target_room", "at_requested_position", "unknown"}:
            raise ValueError("invalid camera relation")
        supported = value["target_visible"] and value["relation_satisfied"]
        supported = supported and relation == required
        grounding = None
        if supported and expected is not None:
            box = value["target_bbox"]
            if (not isinstance(box, list) or len(box) != 4 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)):
                raise ValueError("identified final target requires an actual current-image box")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError("invalid target box")
            current = self.images.get(current_reference)
            w, h = current.size
            crop = current.crop((int(w*x1/1000), int(h*y1/1000), int(w*x2/1000), int(h*y2/1000)))
            if min(crop.size) < 8 or float(np.mean(np.all(np.asarray(crop) == 0, axis=-1))) >= 0.9:
                supported, grounding = False, {"reason": "crop_missing_or_unusable"}
            else:
                # The requested target and instruction are deliberately absent
                # from this independent crop classification prompt.
                audit_prompt = ("Identify the actual visible object in this crop, without assuming a task. "
                    "Black missing-mesh triangles or unrecognizable surfaces are unknown. Require recognizable structure. "
                    "Return JSON with object_type (one of " + ", ".join([*OBJECTS, "unknown"]) +
                    "), visible_structure (string), confidence (0..1).")
                audit_raw = self.vision(audit_prompt, [("ACTUAL UNLABELLED CURRENT CROP", crop)])
                grounding = parse_object(audit_raw)
                confidence = float(grounding["confidence"])
                supported = (grounding["object_type"] == expected and bool(grounding["visible_structure"]) and
                             math.isfinite(confidence) and 0.8 <= confidence <= 1)
                grounding = {**grounding, "raw_output": audit_raw, "expected_object_not_in_crop_prompt": expected}
        evidence = StageEvidence(progress.active.stage_id, progress.anchor, current_reference, bool(supported),
                                 value["target_visible"], float(value["confidence"]), str(value["reason"]))
        self.last_trace = {"context": context, "raw_output": raw, "grounding_audit": grounding,
                           "evidence": asdict(evidence), "spatial_relation_checked_separately": True}
        return evidence
