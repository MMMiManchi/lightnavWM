"""Scene-serial scheduling and matched navigation comparisons."""

from collections import defaultdict

import numpy as np


def scene_shards(tasks, workers):
    if workers < 1 or not tasks:
        raise ValueError("nonempty tasks and positive worker count required")
    scenes, seen = defaultdict(list), set()
    for task in tasks:
        key = (task["scene"], str(task["episode_id"]))
        if key in seen:
            raise ValueError("duplicate episode in evaluation schedule")
        seen.add(key)
        scenes[task["scene"]].append(task)
    shards = [[] for _ in range(workers)]
    # Longest-scene-first packing balances load without splitting its history.
    for scene in sorted(scenes, key=lambda s: (-len(scenes[s]), s)):
        index = min(range(workers), key=lambda i: (len(shards[i]), i))
        shards[index].extend(scenes[scene])
    return shards


def matched_navigation(base, treatment):
    def index(rows):
        result = {}
        for row in rows:
            key = (row["scene"], str(row["dataset_episode_id"]))
            if key in result:
                raise ValueError("duplicate result in comparison")
            result[key] = row
        return result
    a, b = index(base), index(treatment)
    if a.keys() != b.keys() or not a:
        raise ValueError("comparisons require identical, nonempty completed episode sets")
    for key in a:
        if a[key]["initial_rgb_sha256"] != b[key]["initial_rgb_sha256"]:
            raise ValueError("paired initial observations differ")
    keys = sorted(a)
    sr = np.array([[r[k]["evaluation_only"]["success"] for k in keys] for r in [a, b]], dtype=float)
    spl = np.array([[r[k]["evaluation_only"]["spl"] for k in keys] for r in [a, b]], dtype=float)
    changed = [k for k in keys if b[k].get("prediction_controlled_rounds", 0) or b[k].get("memory_controlled_rounds", 0)]
    return {"episodes": len(keys), "base_sr": float(sr[0].mean()), "treatment_sr": float(sr[1].mean()),
            "base_spl": float(spl[0].mean()), "treatment_spl": float(spl[1].mean()),
            "sr_delta": float((sr[1] - sr[0]).mean()), "spl_delta": float((spl[1] - spl[0]).mean()),
            "success_gained": int((sr[1] > sr[0]).sum()), "success_lost": int((sr[1] < sr[0]).sum()),
            "controlled_episodes": len(changed),
            "controlled_success_gained": sum(b[k]["evaluation_only"]["success"] > a[k]["evaluation_only"]["success"] for k in changed),
            "controlled_success_lost": sum(b[k]["evaluation_only"]["success"] < a[k]["evaluation_only"]["success"] for k in changed)}
