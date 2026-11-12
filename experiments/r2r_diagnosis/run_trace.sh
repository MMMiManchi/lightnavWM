#!/usr/bin/env bash
# One diagnostic replay. Picks idle GPUs at launch. Does not train or load a LoRA.
set -u
cd /home/szgx/LightNav-0

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0 < 1000 {print $1}')
if [ "${#FREE[@]}" -lt 1 ]; then
  echo "no idle GPU"
  exit 1
fi
SIM="${FREE[0]}"
if [ "${#FREE[@]}" -ge 2 ]; then
  INF="${FREE[1]}"
else
  INF="${SIM}"
fi
PORT=5681
OUT=output/r2r_probe/trace_20261020
mkdir -p "$OUT/logs"
if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "port ${PORT} is in use"
  exit 1
fi
printf 'sim_gpu=%s\ninfer_gpu=%s\nport=%s\nstarted=%s\n' "$SIM" "$INF" "$PORT" "$(date -Is)" > "$OUT/run_meta.txt"
echo "habitat physical GPU ${SIM} -> process cuda:0 --gpu-id 0; inference physical GPU ${INF} -> process cuda:0"

CUDA_VISIBLE_DEVICES="$SIM" /home/szgx/miniconda3/envs/habitat/bin/python -u experiments/r2r_diagnosis/serve_probe.py \
  --allow experiments/r2r_diagnosis/probe_episodes.json \
  --config habitat_server/configs/vlnce_r2r_go2w.yaml \
  --port "$PORT" \
  --gpu-id 0 \
  --ready-file "$OUT/ready" \
  > "$OUT/logs/serve.log" 2>&1 &
echo $! > "$OUT/serve.pid"

CUDA_VISIBLE_DEVICES="$INF" /home/szgx/LightNav-0/.venv/bin/python -u experiments/r2r_diagnosis/probe_client.py \
  --mode trace \
  --server "tcp://127.0.0.1:${PORT}" \
  --ready "$OUT/ready" \
  --model checkpoints/LightNav-0 \
  --backend vllm_local \
  --gpu-mem 0.45 \
  --classes experiments/r2r_diagnosis/probe_episodes.json \
  --out "$OUT/episodes.jsonl" \
  --steps "$OUT/steps.jsonl" \
  --frames "$OUT/frames" \
  > "$OUT/logs/client.log" 2>&1
status=$?
echo "client_exit=${status}" >> "$OUT/run_meta.txt"
if [ -f "$OUT/serve.pid" ]; then
  kill "$(cat "$OUT/serve.pid")" 2>/dev/null || true
fi
exit "$status"
