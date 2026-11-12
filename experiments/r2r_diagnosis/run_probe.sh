#!/bin/bash
# Paired Go2-W interventions on the fixed 30-episode allowlist.
# Does not touch output/r2r_official, output/r2r_go2w, or the LoRA checkpoint.
set -u
cd /home/szgx/LightNav-0
HAB=/home/szgx/miniconda3/envs/habitat/bin/python
PY=/home/szgx/LightNav-0/.venv/bin/python
OUT=output/r2r_probe
mkdir -p "$OUT/logs"
ALLOW=experiments/r2r_diagnosis/probe_episodes.json

run_one() {
  local gpu="$1" port="$2" mode="$3"
  local ready="$OUT/${mode}.ready"
  rm -f "$ready"
  echo "===== ${mode} GPU ${gpu} $(date -Is) ====="
  CUDA_VISIBLE_DEVICES="$gpu" "$HAB" experiments/r2r_diagnosis/serve_probe.py \
    --allow "$ALLOW" --port "$port" --gpu-id 0 --ready-file "$ready" \
    > "$OUT/logs/serve_${mode}.log" 2>&1 &
  local srv=$!
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" experiments/r2r_diagnosis/probe_client.py \
    --mode "$mode" --server "tcp://127.0.0.1:${port}" --ready "$ready" \
    --backend vllm_local --gpu-mem 0.45 \
    --classes "$ALLOW" --out "$OUT/${mode}.jsonl" \
    > "$OUT/logs/client_${mode}.log" 2>&1
  local rc=$?
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  echo "===== ${mode} exit ${rc} $(date -Is) ====="
  return "$rc"
}

run_one 2 5671 expert
run_one 2 5671 model
run_one 2 5671 oracle_stop
echo "===== PROBE DONE $(date -Is) ====="
