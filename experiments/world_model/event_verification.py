"""Visual event evidence; elapsed time and policy STOP never establish arrival."""

import json
from dataclasses import asdict

from experiments.world_model.task_progress import StageEvidence
from experiments.world_model.visual_harness import parse_object


class QwenEventVerifier:
    def __init__(self, vision, images):
        self.vision, self.images = vision, images
        self.last_trace = {}

    def __call__(self, tracker, branch, current_reference):
        payload = {"selected_source_exit": asdict(branch),
            "recent_actual_commands": list(tracker.executions.values())[-12:],
            "measured_execution_feedback": getattr(tracker, "execution_feedback", {}),
            "support_is_visual_not_ground_truth": True}
        prompt = '''Determine whether an ACTUALLY EXECUTED local entrance option has reached its NEXT EVENT.
Compare the actual source with actual later first-person views. The camera is the robot.
The selected visible exit must have been crossed, with a distinct new room entered or a new decision location reached.
For a passive monitor the exit is only a source hypothesis: the original policy may have taken another route.
Never assume this exit was followed because it was proposed. Judge which entrance was actually crossed.
Rotating to face an opening, seeing through it, moving slightly inside the same room, a fixed elapsed duration,
and LightNav STOP do not establish a next event. Commands are issued actions, not measured motion.
Use actual visual evidence and admit uncertainty. This is not final task-goal verification.
Return JSON with required fields: reason (concrete evidence or uncertainty), selected_exit_followed (boolean),
event_visible (boolean), event_kind (string), confidence (number 0..1). There are no default verdicts.
The flags must be booleans. event_kind is next_room, next_decision, or unknown.
CONTEXT: ''' + json.dumps(payload)
        items = [("ACTUAL SELECTED EXIT SOURCE", self.images.get(tracker.anchor))]
        step = tracker.observations[current_reference]
        earlier = [(t, ref) for ref, t in tracker.observations.items() if 0 < t < step and ref != tracker.anchor]
        if earlier:
            _, reference = sorted(earlier)[max(0, len(earlier) - 4)]
            items.append(("ACTUAL INTERVENING VIEW", self.images.get(reference)))
        items.append(("ACTUAL CURRENT VIEW", self.images.get(current_reference)))
        raw = self.vision(prompt, items)
        value = parse_object(raw)
        if not all(isinstance(value[k], bool) for k in ["selected_exit_followed", "event_visible"]):
            raise ValueError("event evidence flags must be booleans")
        if value["event_kind"] not in {"next_room", "next_decision", "unknown"}:
            raise ValueError("unknown event kind")
        supported = value["selected_exit_followed"] and value["event_visible"] and value["event_kind"] != "unknown"
        evidence = StageEvidence("event", tracker.anchor, current_reference, supported,
                                 bool(value["event_visible"]), float(value["confidence"]), str(value["reason"]))
        self.last_trace = {"context": payload, "raw_output": raw, "evidence": asdict(evidence),
                           "event_kind": value["event_kind"]}
        return evidence
