#!/usr/bin/env bash
# Collect the remaining train-fit houses, then redraw the scene graph.
# Detach this script. Closing the editor must not stop it.
set -u
cd /home/szgx/LightNav-0
export PYTHONPATH=/home/szgx/LightNav-0
export PYTHONDONTWRITEBYTECODE=1
LOG=/home/szgx/LightNav-0/output/world_model_data/collect_remaining_20261103.log
echo "pid $$ start $(date -Is)" >> "$LOG"
/home/szgx/LightNav-0/.venv/bin/python -u experiments/world_model/fit_house_remaining.py >> "$LOG" 2>&1
collect_status=$?
echo "collection_exit ${collect_status} $(date -Is)" >> "$LOG"
/home/szgx/miniconda3/envs/habitat/bin/python -u experiments/world_model/scene_graph_entry.py >> "$LOG" 2>&1
graph_status=$?
echo "graph_exit ${graph_status} $(date -Is)" >> "$LOG"
echo "done $(date -Is)" >> "$LOG"
exit "$collect_status"
