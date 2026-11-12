"""Conservative gates for machine node labels; agreement is still not truth."""

from __future__ import annotations

import math

PROMPT = """Examine only this CURRENT navigation RGB view. Locate architectural exits,
not arbitrary free-space directions. A useful decision point has at least two
separate IMMEDIATE architectural passages (open doorways, corridor mouths, or
stair entrances) leading out of the immediate space. Different ways around a
sofa/table/pillar within one open room are NOT different architectural exits.
Do NOT count chair gaps, furniture edges, windows, closed doors, black missing
scan areas, or a far doorway seen through a nearer doorway. Two sides of one
wide entrance are ONE passage. Admit uncertainty if architecture is unclear.
Return label decision only when a choice between distinct immediate exits is
visible here. An ordinary room interior or a corridor before reaching a split
is ordinary. Blind/blocked views needing another look are observe; otherwise
unknown. Do not assume any candidate junction, route or target was given.
For every supported exit give its bounding box [x0,y0,x1,y1] and an entry point
[u,v] on the floor inside it; all coordinates are 0..1000. No hidden exits.
Return only JSON: {"label":"decision|ordinary|observe|unknown",
"confidence":0.0,"exits":[{"kind":"doorway|corridor|stairs",
"bbox":[0,0,0,0],"floor_entry":[0,0],"clear_architecture":true}],
"reason":"at most 30 words"}.
"""


def supported_exits(prediction: dict) -> list[dict]:
    exits = []
    for exit_ in prediction.get("exits", []):
        if not isinstance(exit_, dict) or exit_.get("clear_architecture") is not True:
            continue
        if exit_.get("kind") not in {"doorway", "corridor", "stairs"}:
            continue
        box, point = exit_.get("bbox"), exit_.get("floor_entry")
        if not isinstance(box, list) or len(box) != 4 or not isinstance(point, list) or len(point) != 2:
            continue
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1000 for x in [*box, *point]):
            continue
        x0, y0, x1, y1 = box
        u, v = point
        if x1 - x0 < 30 or y1 - y0 < 60 or not (x0 <= u <= x1 and y0 <= v <= y1):
            continue
        exits.append(exit_)
    return exits


def two_distinct_exits(exits, minimum_angle_degrees=35.0) -> bool:
    for i, a in enumerate(exits):
        for b in exits[i + 1:]:
            aa, bb = a["bbox"], b["bbox"]
            intersection = max(0, min(aa[2], bb[2]) - max(aa[0], bb[0])) * max(0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
            smaller = min((aa[2] - aa[0]) * (aa[3] - aa[1]), (bb[2] - bb[0]) * (bb[3] - bb[1]))
            if intersection / smaller > 0.25:
                continue
            bearings = [math.atan((0.5 - e["floor_entry"][0] / 1000) * 2 * math.tan(math.radians(60))) for e in (a, b)]
            if abs(math.degrees(bearings[0] - bearings[1])) >= minimum_angle_degrees:
                return True
    return False


def consensus_label(first: dict, second: dict) -> tuple[str, str]:
    if float(first.get("confidence", 0)) < 0.8 or float(second.get("confidence", 0)) < 0.8:
        return "unknown", "low_teacher_confidence"
    label = second.get("label")
    if first.get("label") != label:
        return "unknown", "teacher_disagreement"
    exits = supported_exits(second)
    if label == "decision" and two_distinct_exits(exits):
        return "decision", "two_architectural_exit_proposals_with_separation"
    if label == "ordinary" and len(exits) <= 1:
        return "ordinary", "both_teachers_ordinary"
    if label == "observe":
        return "observe", "both_teachers_observe"
    return "unknown", "insufficient_architectural_evidence"
