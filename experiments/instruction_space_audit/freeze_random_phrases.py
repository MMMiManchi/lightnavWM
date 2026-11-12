#!/usr/bin/env python3
"""Freeze clause phrases for sample indices 7-100 before those frames are reviewed.

The split is textual. It does not read images or assign semantic labels.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "random_audit_sample.csv"
OUT = ROOT / "random_audit_phrases.csv"

VERBS = (
    "keep|keeping|proceed|go|going|walk|walking|turn|turning|pivot|pivoting|"
    "travel|traveling|continue|continuing|enter|entering|stop|stopping|wait|waiting|"
    "stand|standing|move|moving|head|heading|pass|passing|follow|following|exit|exiting|"
    "take|taking|step|stepping|remain|remaining|face|facing|hang|hanging|make|making|"
    "veer|leave|leaving|cross|crossing|approach|approaching|climb|climbing"
)
CLAUSE = re.compile(
    rf"(?<=[.!;])\s+"
    rf"|,\s+(?=(?:and\s+|then\s+)?(?:{VERBS})\b)"
    rf"|\s+then\s+(?=(?:{VERBS})\b)"
    rf"|\s+and\s+(?=(?:a\s+)?(?:{VERBS})\b)"
    rf"|\s+(?=(?:stop|wait)\b)",
    re.IGNORECASE,
)


def phrases(text: str) -> list[str]:
    parts = [part.strip(" ,") for part in CLAUSE.split(text.strip())]
    return [part for part in parts if part]


def main() -> None:
    rows = []
    for route in csv.DictReader(SAMPLE.open()):
        if int(route["sample_index"]) < 7:
            continue
        for index in (1, 2, 3):
            episode_id = route[f"episode_id_{index}"]
            for phrase_index, phrase in enumerate(phrases(route[f"instruction_{index}"]), start=1):
                rows.append(
                    {
                        "sample_index": route["sample_index"],
                        "route_group_id": route["route_group_id"],
                        "episode_id": episode_id,
                        "phrase_index": phrase_index,
                        "phrase": phrase,
                        "verdict": "not_viewed",
                        "evidence": "",
                    }
                )
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("phrases", len(rows), "instructions", len({row["episode_id"] for row in rows}))


if __name__ == "__main__":
    main()
