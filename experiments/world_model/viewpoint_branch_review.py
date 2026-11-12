"""Current-image checking of proposed branch directions, not future labels."""

PROMPT = """Check the CURRENT first-person image. Two geometric direction proposals
are marked A (cyan) and B (magenta). These proposals can be WRONG. Their lines
are bearings, not proof of a doorway. No future image, map or goal is supplied.
For EACH direction decide what is actually visible. A valid immediate exit is
an open architectural doorway, a corridor mouth/continuation, or a stairway.
Do not count ordinary open floor, two sides of one broad entrance, routes around
furniture or a pillar in one room, chair gaps, a mirror/reflection, a window,
a closed door, or black missing scan geometry. A remote doorway seen through a
nearer entrance is not a separate immediate exit. Ordinary room interiors
before reaching the choice location are not decision nodes.
If the view supports a choice HERE between TWO DISTINCT immediate architectural
exits at A and B, decision_here=true. Otherwise false; admit uncertainty.
Return JSON only: {"A":{"kind":"doorway|corridor|stairs|open_floor|mirror|window|closed_door|furniture|missing_scan|uncertain",
"immediate":false},"B":{"kind":"same enum","immediate":false},
"distinct_architectural_exits":false,"decision_here":false,
"scene_state":"decision|ordinary|observe|uncertain","confidence":0.0,
"reason":"at most 35 words"}. Confidence is not calibrated accuracy.
"""


def proposed_node_passes(value):
    valid = {"doorway", "corridor", "stairs"}
    return (
        value.get("decision_here") is True
        and value.get("distinct_architectural_exits") is True
        and value.get("scene_state") == "decision"
        and float(value.get("confidence", 0)) >= 0.8
        and all(isinstance(value.get(k), dict) and value[k].get("kind") in valid
                and value[k].get("immediate") is True for k in ("A", "B"))
    )
