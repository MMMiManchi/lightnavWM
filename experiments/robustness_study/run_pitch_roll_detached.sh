#!/bin/bash
# Finish the second pitch/roll run outside the Cursor terminal, then compare.
set -u
cd /home/szgx/LightNav-0
mkdir -p output/r2r_camera_jitter
exec > output/r2r_camera_jitter/detached.log 2>&1
echo "===== DETACHED $(date -Is) ====="

python3 - << 'PY'
import json, os
from pathlib import Path
root = Path("/home/szgx/LightNav-0")
allow = json.loads((root / "experiments/robustness_study/pitch_roll_allow.json").read_text())
done = set()
path = root / "output/r2r_camera_jitter/seed22/results.jsonl"
if path.is_file():
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = os.path.basename(os.path.dirname(str(row["scene_id"]).rstrip("/")))
        done.add((scene, str(row["habitat_episode_id"])))
rest = [row for row in allow if (row["scene"], str(row["episode_id"])) not in done]
out = root / "experiments/robustness_study/pitch_roll_remaining.json"
out.write_text(json.dumps(rest, indent=2) + "\n")
print(f"remaining {len(rest)} of {len(allow)}; seed22 already {len(done)}")
PY

remain=$(python3 -c 'import json; print(len(json.load(open("experiments/robustness_study/pitch_roll_remaining.json"))))')
if [[ "$remain" -gt 0 ]]; then
  out="output/r2r_camera_jitter/seed22_rest"
  mkdir -p "$out"
  rm -f "$out/ready" "$out/results.jsonl"
  CUDA_VISIBLE_DEVICES=3 HABITAT_SIM_GPU_ID=0 \
    /home/szgx/miniconda3/envs/habitat/bin/python -u experiments/robustness_study/serve_pitch_roll.py \
    --allow experiments/robustness_study/pitch_roll_remaining.json \
    --config habitat_server/configs/vlnce_r2r.yaml \
    --seed 2202 --port 5681 --gpu-id 0 \
    --ready-file "$out/ready" \
    > "$out/serve.log" 2>&1 &
  spid=$!
  ready=0
  for _ in $(seq 1 180); do
    if [[ -f "$out/ready" ]]; then ready=1; break; fi
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "server died"; tail -30 "$out/serve.log"; exit 1
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "server not ready"; kill "$spid" || true; exit 1
  fi
  CUDA_VISIBLE_DEVICES=2 /home/szgx/LightNav-0/.venv/bin/python -u -m lightnav.cli.eval_habitat \
    --model_path checkpoints/LightNav-0 \
    --backend vllm_local \
    --episodes -1 \
    --max_steps 500 \
    --gpu_memory_utilization 0.45 \
    --server tcp://127.0.0.1:5681 \
    --output_dir "$out" \
    > "$out/client.log" 2>&1
  status=$?
  wait "$spid" || true
  echo "client exit $status"
  if [[ "$status" -ne 0 ]]; then
    echo "client failed"; tail -30 "$out/client.log"; exit "$status"
  fi
fi

python3 experiments/robustness_study/compare_pitch_roll.py
echo "===== DONE $(date -Is) ====="
