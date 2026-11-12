# Model assets and parameters

Checkpoint layout, the action decoder, `eval_config.json` and every server / CLI parameter.

## Checkpoint

A Hugging Face directory (or a directory containing `hf_ckpt/`) with `config.json`,
`model*.safetensors`, `tokenizer*`, `processor_config.json`, and ideally
`eval_config.json`. The architecture must be stock `Qwen3VLForConditionalGeneration`; the
trajectory / action tokens are ordinary rows of the embedding table.

## Action decoder

An **RVQ action-tokenizer bundle**: a directory with `manifest.json`, the per-level
codebook `.npy` files and the `jacobian_weights` file it names. The released checkpoints
ship it as `action_tokenizer/`, referenced from `eval_config.json`, so it is found without
flags; `--action_tokenizer_bundle` overrides it.

## `eval_config.json` (processing parameters)

If present next to or above the checkpoint (the search walks up four parents, so
`hf_ckpt/`, the `global_step_*` directory and the run root are all covered), it supplies the
processing parameters that must match training and cannot be guessed:

```json
{
  "version": 1,
  "common": {"video_size": [224, 320], "pool_enable": true, "pool_spatial": 2,
             "pool_mode": "avg", "pool_stage": "pre_vit"},
  "tasks": {
    "trackvla": {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "slowfast_tiers": null,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "/path/to/action_tokenizer"}},
    "vlnce":    {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer"}}
  }
}
```

| Key | Meaning |
|---|---|
| `common.video_size` | model input frame size `[H, W]`; every frame is resized to it |
| `common.pool_*` | spatial pooling of vision tokens (`pool_stage` `pre_vit` or `post_vit`) |
| `tasks.<task>.num_history_frames` | history window (frames) fed to the model per step |
| `tasks.<task>.predict_horizon` | trajectory horizon `H` (rows per chunk) |
| `tasks.<task>.video_fps` | frame rate the checkpoint was trained at (send frames at this rate or faster; waypoint rows carry no time base of their own, see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `tasks.<task>.slowfast_tiers` | optional multi-rate history layout (SlowFast checkpoints keep the whole episode) |
| `tasks.<task>.action_tokenizer` | decoder snapshot: `{"method": "rvq", "bundle_path"}`; relative paths resolve against the config's directory |

`tasks` is keyed by task: `trackvla` for tracking (`--task tracking`), `vlnce` for
navigation (`--task vln`, and the Habitat evaluation). Values are read automatically; the CLIs'
`--num_history_frames` (and `--pool_spatial`, where offered) override them. Without the file the code
falls back to conservative defaults (`num_history_frames=16`, no pooling,
`video_size=(224, 320)`) which will not match a trained checkpoint — treat a missing
`eval_config.json` as a warning sign. Full schema: `src/lightnav/eval_config.py`.


### Shipping the action decoder with the checkpoint

`tasks.<task>.action_tokenizer.bundle_path` may be a **relative path**; it is resolved
against the directory holding `eval_config.json` (falling back to the checkpoint
directory). The released checkpoints ship their decoder this way:

```
hf_ckpt/
  config.json, model-*.safetensors, tokenizer*, processor_config.json
  eval_config.json          # bundle_path: "action_tokenizer" (both tasks)
  action_tokenizer/         # manifest.json + codebook_l*.npy + jacobian_weights.npy (+ alpha)
```

With such a layout `lightnav-serve`, `lightnav-predict`, `lightnav-eval-habitat` and
`build_tracking_agent()` need no `--action_tokenizer_bundle`: the
decoder is taken from the task entry that matches the server task (`--task tracking` →
`trackvla`, `--task vln` → `vlnce`), then from the other tasks, then from the sibling
directories `action_tokenizer/<task>` or `action_tokenizer/`. The explicit flag always
wins.

## Server / CLI parameters

`lightnav-serve` flags (environment names are what `scripts/start_servers.sh` and
`docker/entrypoint.sh` read):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--model_path` | `MODEL_PATH` | required | checkpoint directory |
| `--action_tokenizer_bundle` | `ACTION_TOKENIZER_BUNDLE` | from the checkpoint | RVQ bundle dir; overrides the decoder the checkpoint ships |
| `--task` | `TASK` | `tracking` | `tracking` (tracking prompt, `tasks.trackvla`) or `vln` (VLN prompt, `tasks.vlnce`) |
| `--backend` | `BACKEND` | `vllm_local` | `vllm_local` or `hf` |
| `--gpu_memory_utilization` | `GPU_MEM_UTIL` | `0.85` | fraction of GPU memory handed to the vLLM engine |
| `--max_batch_size` | `MAX_BATCH_SIZE` | `8` | max sessions per scheduler tick; also sizes vLLM `max_num_seqs`. `1` = strictly serial |
| `--max_wait_ms` | `MAX_WAIT_MS` | `8` | max wait to fill a batch; a lone request is flushed after 2 ms |
| `--quantization` | `VLLM_QUANT` | none (bf16) | `fp8_llm_only`: fp8 LLM, bf16 ViT (~1.5× on Jetson Thor). On SM 11.0 also set `VLLM_DISABLED_KERNELS`; `scripts/serve_thor.sh` does both. See [JETSON_THOR.md](JETSON_THOR.md) |
| `--vit_cache_entries` | `VLN_VIT_CACHE_ENTRIES` | auto (512 w/ SlowFast) | ViT tubelet LRU capacity (`vllm_local`; speed only, never output). Long robot sessions want `1024` |
| `--num_history_frames` | `NUM_HISTORY_FRAMES` | checkpoint config | history-window override (normally leave unset) |
| `--pool_spatial` | `POOL_SPATIAL` | checkpoint config | spatial-pooling override |
| `--aspect_mode` | `ASPECT_MODE` | `stretch` | `stretch`: resize every frame to `video_size` (training behaviour); `keep`: per-session size with the camera's aspect ratio at the same pixel budget (4:3 → 288×384 for a 256×448 checkpoint) |
| `--max_new_tokens` | `MAX_NEW_TOKENS` | `8` | lower bound for the per-step decode cap (see below) |
| `--host` / `--port` | `HOST` / `PORT` | `0.0.0.0` / `8050` | bind address |
| `--ready_file` | – | none | file touched once the port is bound (used by the launchers) |
| `--no_warmup` | – | off | skip the synthetic warm-up inference before binding the port |
| `--record_dir` | `RECORD_DIR` | off | record every connection's episodes (frames + per-step records) for `lightnav-render` (see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `--record_fps`, `--record_timeline`, `--no_record_images` | `RECORD_FPS`, `RECORD_TIMELINE`, `RECORD_IMAGES` | `10`, `realtime`, images on | recording defaults written to the manifest / frame storage |
| `--cam_hfov_deg`, `--cam_height`, `--traj_forward_offset`, `--waypoint_dt_s` | `CAM_HFOV_DEG`, `CAM_HEIGHT`, `TRAJ_FORWARD_OFFSET`, `WAYPOINT_DT_S` | `90`, `0.5`, auto, `0.1` | client-camera geometry and HUD convention used only by the rendered overlay |

### MuJoCo demo parameters

`mujoco_demo` is a separate `uv` project. Run `./run.sh` for its bundled
TurtleBot, or use `uv run --extra microduck vln-mujoco ...` when selecting the
optional MicroDuck backend. MicroDuck needs two external files, the MJCF from
`pollen-robotics/microduck_rl` and an ONNX policy from
`pollen-robotics/microduck-policies`; the download steps are in
[mujoco_demo/README.md](../mujoco_demo/README.md#microduck-optional).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8088` | web-console bind address |
| `--vln-server` | empty | default LightNav WebSocket URL shown in the console |
| `--robot` | `turtlebot` | robot backend: `turtlebot` or `microduck` |
| `--robot-model` | none | external MicroDuck MJCF; required only with `--robot microduck` |
| `--walking-policy` | none | external MicroDuck ONNX policy; required only with `--robot microduck` |

Engine-level environment variables (no flag):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLN_KV_CACHE_GIB` | auto | vLLM KV-cache size in GiB (auto-scales with `max_num_seqs`, floor 2 GiB) |
| `VLN_VLLM_ENFORCE_EAGER` | `0` | `1` disables CUDA-graph capture (faster start, slower steps) |
| `LIGHTNAV_ATTN` | `sdpa` | attention implementation for the `hf` backend (e.g. `flash_attention_2`) |
| `VLN_EVAL_TEMPERATURE` / `TOP_P` / `TOP_K` | greedy | sampling knobs for experiments; benchmark numbers assume they are unset |

**Decode token budget.** A step emits a *grounding prefix* followed by the action token(s):
one `<act_l{d}_*>` per RVQ level. The server
probes the tokenizer for the grounding families it knows and caps generation at
`max(--max_new_tokens, prefix + action_tokens)`, which saves the decode step otherwise spent
reaching `eos` without ever truncating the action.

| checkpoint family | grounding prefix | tokens |
|---|---|---|
| tracking, legacy | `<tpos_k>` | 1 |
| tracking + grid pointing | `<opos_k>` | 1 |
| navigation + dual pointing | `<apos_A><opos_O>` | 2 |

**GPU sizing.** The 4B checkpoint needs ~10 GB for weights in bf16 plus the vLLM KV cache;
one 24 GB GPU comfortably serves one engine with `--gpu_memory_utilization 0.85`. When
several servers share a GPU, `scripts/start_servers.sh` divides the utilisation by
`SERVERS_PER_GPU`.

Note what `--gpu_memory_utilization` does *not* do here. The engine passes an explicit
`kv_cache_memory_bytes` (`VLN_KV_CACHE_GIB`, floor 2 GiB) and therefore skips vLLM's
profile run, so real usage is roughly weights + that KV cache + CUDA graphs (~15 GB for
the released 4B checkpoint) whatever the fraction says. The fraction still acts as the
ceiling vLLM checks against free memory at start-up, so lower it when the GPU is shared
with another job, and raise `VLN_KV_CACHE_GIB` — not the fraction — when you want a
bigger KV cache.

---

## Input aspect ratio (`--aspect_mode`)

The checkpoints are trained on frames resized to a fixed `common.video_size` (256×448 for
the released checkpoint, ≈16:9). By default (`stretch`) every client frame is resized to
that size regardless of its aspect ratio, exactly as in training — a 4:3 camera therefore
arrives horizontally squeezed.

`--aspect_mode keep` (env `ASPECT_MODE=keep`; also `lightnav-predict --aspect_mode`,
`lightnav-eval-habitat --aspect_mode`, `build_tracking_agent(aspect_mode=)`) instead picks,
per session from its first frame, the size with the **source aspect ratio** whose area is
closest to `video_size` and whose sides are multiples of 32 (patch × merge; and of the pre-ViT
pooling factors when the checkpoint pools before the ViT). The vision-token count per frame
stays at the training budget, so history length and sequence length are unchanged:

| camera | `stretch` | `keep` |
|---|---|---|
| 16:9 (480×270, 1920×1080) | 256×448 | 256×448 (identical) |
| 4:3 (640×480) | 256×448 (squeezed) | 288×384 |
| 1:1 | 256×448 | 352×352 |

Frames within one session must share one size (the first frame decides; `reset` starts
over). `keep` avoids geometric distortion but changes the token grid the model sees, which
the released checkpoint was not trained on — validate on your robot before relying on it.
Full native resolution (feeding the camera's own pixel count) is not supported.

## Event latent world-model experiment

This independent research trainer predicts next-event frozen visual features;
it does not change `lightnav-serve` or launch a navigation environment. Its data,
model, GPU allocation and bounded first experiment are documented in
[world_model_event_training_v1.md](world_model_event_training_v1.md).

| Command | Flag | Meaning |
|---|---|---|
| `python scripts/train_event_world_model.py` | `--config PATH` | Required JSON experiment config; starts preparation, eight encoding workers and eight independent model/control jobs. |
| `python -m experiments.world_model.event_train` | `--config PATH` | Required JSON config containing the new run directory, approved source runs and model settings. |
| same | `--phase {prepare,encode,pack,train,summarize}` | Required explicit pipeline stage. CUDA is required for encode/train; there is no CPU fallback. |
| same | `--worker N` | Encoding shard or experiment index, default 0; GPU visibility is assigned by the launcher. |

Cross-episode persistent history and its causal replay are documented in
[world_model_cross_episode_history_20261104.md](world_model_cross_episode_history_20261104.md).

| Command | Flag | Meaning |
|---|---|---|
| `python scripts/eval_cross_episode_memory.py` | `--run-dir PATH` | Required completed event-latent training directory containing fixed features, pairs and checkpoint predictions. |
| same | `--output PATH` | Required new replay directory; refuses overwrite. Replays actual recorded episodes with causal memory writes and no neural parameter updates. |

Residual event-model optimization keeps the first experiment immutable. It adds
current-image patch columns, copy-preserving dynamics and a learned reader of
strictly earlier-episode real experience. Eight fixed ablations use an internal
scene split to select update counts, then refit on all formal-fit scenes. The
previous development set remains exploratory, rather than a new blind test.

| Command | Flag | Meaning |
|---|---|---|
| `python scripts/optimize_event_world_model.py` | `--phase {prepare,encode,train,launch,summarize}` | Required pipeline stage; `launch` runs eight GPU encoding shards and eight model/control jobs. Encode/train require CUDA. |
| same | `--base-run PATH` | Required for prepare; completed, admitted v1 feature/pair directory. |
| same | `--output PATH` | Required for prepare; new directory, never overwritten. |
| same | `--config PATH` | Required for other stages; config written by prepare. |
| same | `--worker N` | Encoding shard/model job, default 0; launcher assigns the visible GPU. |
| same | `--variant {residual,anchored_memory}` | Prepare-time variant, default residual. Anchored memory starts at the direct-history control and learns bounded corrections; no-history queries explicitly retain current state. Three paired branch/no-branch seeds plus two gate-only controls. |

`scripts/smoke_event_lightnav.py` connects an internally selected event model,
prior-episode real memory and the unchanged LightNav checkpoint. The recorded
interface probe uses an explicitly declared teacher direction. Since 2026-11-05,
live mode exposes environmental evidence and retains the current episode's
instruction by default; a matched past direction no longer becomes a local goal.
`TaskGoalRouter` accepts a separate current-task planner's choice among branches
grounded in the current image. The optional frozen Qwen3-VL-4B-Instruct proposer
and planner inspect actual images. Each currently visible candidate gets a
separate world-model future query; its vector searches actual past outcome keys,
and the returned historical images are supplied to the current-task planner as
evidence. Unsupported forecasts abstain. Native LightNav receives its full
chronological RGB prefix; explicit local goals preserve that history, and local
STOP is returned as an unverified tool result. This remains a bounded research
smoke, without verified node discovery, arrival checks or online memory writes.
Planning validates explicit allowed memory ids; image indices are separate.
A rejected JSON/provenance response gets at most one model repair attempt with
the validation error. Invalid citations are never silently removed. Each live
planning image, successful or rejected query trace, raw visual-tool response and
dispatched action is saved incrementally.

| Flag | Meaning |
|---|---|
| `--training-run PATH` | Required optimization directory. Select by internal score; require gain over step zero and non-regression against equal-history retrieval before loading LightNav. |
| `--output PATH` | Required new artifact directory; refuses overwrite. |
| `--model-path PATH` | LightNav checkpoint, default `checkpoints/LightNav-0`. |
| `--backend {vllm_local,hf}` | Default vllm_local; requires CUDA and preserves existing version guards. |
| `--server ADDRESS` | Optional running Habitat ZMQ server for additional bounded live execution. No simulator is launched by this script. |
| `--live-steps N` | Live action budget, 1–60, default 55. Visual matches return environmental evidence only. The match threshold comes from internal fit-validation scenes. Without a current-task planner and currently grounded branches, the original episode instruction remains active; no historical direction is followed automatically. |
| `--planner-model PATH` | Optional pinned `Qwen/Qwen3-VL-4B-Instruct` snapshot plus `asset_manifest.json` copied from `experiments/world_model/contracts/qwen3_vl_4b_instruct_assets.json`; requires CUDA. The general visual planner is separate from the unchanged LightNav checkpoint. Absent by default. |
| `--planner-device DEVICE` | Visual planner device, default `cuda:0`. No CPU fallback. |
| `--planning-interval N` | Positive live-step interval between RGB branch proposals and task planning, default 8; local STOP also returns to planning. Decision views are history anchors, not verified topological nodes. |
| `--query-mode {predicted,direct,shuffled}` | Default predicted. Compare future-vector outcome search with equal-prefix direct-history retrieval, or cyclically shuffle candidate predictions as a diagnostic control. The planner is not shown the mode label. |
| `--maximum-observation-turns N` | Default 2, range 0–4. Budget for active observation requests, each at most 30 degrees; motion uses the policy's existing speed bounds. |

`scripts/probe_predictive_planner.py` runs the real frozen visual proposer,
world-model queries and task planner on recorded current images, without
LightNav or a simulator. Candidates and causal memory prefixes are held fixed
across two task instructions and the three retrieval modes. Selection uses
acquisition order, not model success. The query's future target is never input.
This tests data consumption and decisions, not navigation improvement.

| Flag | Meaning |
|---|---|
| `--training-run PATH` | Required completed anchored-event optimization directory; select the checkpoint by internal validation score. |
| `--planner-model PATH` | Required pinned visual-planner assets and their SHA-256 manifest. |
| `--output PATH` | Required new artifact directory; refuses overwrite. |
| `--device DEVICE` | Default `cuda:0`; requires CUDA. |
| `--samples N` | First 1–5 eligible causal-memory development queries in acquisition order, default 2. |
| `--sample-indices I...` | Optional 1 to 5 explicit eligible development rows; the probe sorts them by acquisition order and rejects rows outside causal-memory `train_dev`. |

`scripts/eval_scene_familiarity.py` compares nested budgets of real prior-episode
experience with frozen weights. Eight workers first replay the 46 recorded scenes,
then each drives one selected scene under four conditions: cold, one older
episode, up to three older episodes, and direct retrieval with the same larger
history. The current task and initial RGB are held fixed across conditions.
Three development scenes and five fit scenes have sufficient prior episodes;
their results are reported separately. Historical experience is admitted recorded
data, not newly verified outcomes from these live attempts. A global LightNav STOP
is dispatched to the simulator so Success/SPL can be measured; local STOP alone
does not establish arrival. No goal distance or true pose is sent to a policy.

| Flag | Meaning |
|---|---|
| `--phase {prepare,launch,worker,summarize}` | Required. Prepare fixes scenario selection, protocol and input/source hashes; launch runs eight isolated CUDA workers and summarizes; worker and summarize also support diagnostics. |
| `--training-run PATH` | Required for prepare; completed anchored-event optimization run, selected by internal validation score. |
| `--output PATH` | Required for prepare; new output directory, refuses overwrite. |
| `--config PATH` | Required for other phases; config produced by prepare. |
| `--worker N` | Worker number 0–7, default 0; launcher assigns its physical CUDA GPU and simulator port. |
| `--live-steps N` | Prepare-time episode limit, default 120, range 40–200; simulator config retains its separate 500-step cap. |
| `--baseline-run PATH` | Optional completed familiarity experiment for prepare. Keeps its eight tasks and step limit, adds only all-prior-history predicted/direct conditions (16 runs), and reuses its immutable offline replay artifacts rather than recomputing them. |

`scripts/diagnose_familiarity_queries.py` re-encodes saved current planning RGB
after a completed familiarity experiment. It separates empty memory, missing
visual source matches and direction-gate rejection, and checks agreement with
runtime support flags. This is post-run diagnosis, with no pose/target input or
policy/gate change.

| Flag | Meaning |
|---|---|
| `--run-dir PATH` | Required completed familiarity directory. |
| `--output PATH` | Required new diagnostic directory; refuses overwrite. |
| `--device DEVICE` | Default `cuda:0`; real frozen SigLIP re-encoding requires CUDA. |

`scripts/eval_online_scene.py` runs eight-GPU, four-arm continual-learning replay:
frozen, memory only, scene-parameter learning only, and both. Each episode is
predicted before its targets are learned; only later episodes can use published
scene parameters. A deterministic reserved portion of already observed support
gates publication, with per-scene checkpoints and optimizer/replay persistence.
The next-event horizon is unknown at prediction time (input sentinel 0), never
the true future arrival duration. This is recorded replay, not fresh navigation.

| Flag | Meaning |
|---|---|
| `--training-run PATH` | Required completed event optimization directory; internally selected frozen checkpoint. |
| `--output PATH` | Required new replay directory; parent refuses overwrite. |
| `--worker N` | Internal worker 0–7; omit to launch all eight CUDA workers and summarize. |
| `--update-steps N` | Parent registration only; SGD budget per completed episode, default 40, range 10–200. |

`scripts/audit_event_readiness.py` is a CPU-only analysis of a completed
four-arm replay. It distinguishes place-label coverage from identical-history
branch controls, measures the actual change from scene adaptation, and scores
retrieval against strictly earlier same-scene outcomes. Missing exact targets
are reported as coverage gaps, without an uncalibrated rejection threshold.
It does not collect new data, train a model or create a blind test.

| Flag | Meaning |
|---|---|
| `--base-run PATH` | Required prepared event dataset with pairs and frozen feature tensors. |
| `--replay-run PATH` | Required completed eight-worker, four-arm online-scene replay; its original data hashes must match. |
| `--output PATH` | Required new audit directory; refuses overwrite and records input hashes and source snapshots. |

`scripts/eval_continual_harness.py` executes fresh episodes with an unchanged
LightNav/visual encoder, observed-span writeback, optional scene SGD, strict
image/depth entrance alignment, ordered task progress and STOP interception.
Eight workers pair four fixed scenes under memory-only and memory+SGD. A
6-step observed control span is a separate target type from old next-node event
labels; it asserts actual observation/execution provenance, not node arrival.
The task verifier supplies temporal visual support, not privileged goal truth.
Scene checkpoints bind the control model, simulator configuration, fixed
prediction horizon and waypoint conversion code as well as the feature space.
Implementation details, measured results and verifier limitations are recorded
in [the continual learning validation report](world_model_online_closed_loop_20261105.md).

| Flag | Meaning |
|---|---|
| `--phase {prepare,launch,worker,summarize}` | Required. Prepare registers tasks and source/global-model hashes; launch uses eight GPUs; other stages support diagnostics. |
| `--training-run PATH` | Required for prepare; event-model training directory. |
| `--output PATH` | Required new prepare directory; refuses overwrite. |
| `--config PATH` | Required for other phases; prepared config. |
| `--worker N` | Worker 0–7, default 0; launcher isolates CUDA and assigns a simulator port. |
| `--episodes N` | Prepare-time episodes per scene/method, default 3, range 1–3. Distinct route groups are preferred for task coverage using offline metadata only. |
| `--live-steps N` | Per-episode budget, default 120, range 30–200. |

`scripts/probe_entrance_alignment.py` tests RGB-only camera alignment on actual
stored diagnostic views. Pose selects near-stationary camera pairs and scores
their angle error; it is never passed to the aligner. Other-scene pairs serve as
negative cases. These images are not used for world-model gradient updates.

| Flag | Meaning |
|---|---|
| `--frames PATH` | Diagnostic frame manifest, default `output/world_model_diagnostics/official_nodes_20261104_04/frames.jsonl`. |
| `--output PATH` | Required new output directory; refuses overwrite. |
| `--limit N` | First qualifying positive pairs, default 40, range 10–100; up to20 other-scene negative pairs. |

`scripts/replay_lightnav_endpoint.py` runs in the installed Habitat Python 3.9
environment. It applies the original smoke's saved controls without a policy
call, verifies the original simulator config hash, and saves replay images,
pose and goal distance. These are replay measurements, not missing original
endpoint measurements, and diagnostic pose/goal information never changes actions.

| Flag | Meaning |
|---|---|
| `--source-smoke PATH` | Required completed smoke directory with result, original source manifest and dispatched live actions. |
| `--output PATH` | Required new endpoint replay directory; refuses overwrite. |
| `--gpu-id N` | Habitat GPU index, default 1. |

`scripts/collect_branch_benchmark.py` collects same-state, multi-exit event
consequences. It writes the protocol, exposure, and input hashes before Habitat
starts. A geometric follower aims at one already-seen exit; the records say
this is not LightNav. Two successive junction views after at least 0.8 m count
as the next event. A fixed distance, a fixed duration, and a local stop do not.
Failures and truncations stay in the run. `--verify` only rereads a finished run.

| Flag | Meaning |
|---|---|
| `--output PATH` | Collection directory. A new path is created and the protocol is written before Habitat starts. The same path later continues unfinished executions and does not rewrite the protocol. |
| `--gpu N` | Use this GPU only. Separate processes can pass 0–7 and claim different unfinished exits. |
| `--prepare-arrivals PATH` | Freeze junctions reached inside an existing run as new same-house origins. Does not start Habitat or change the event rule. |
| `--verify PATH` | Recheck an existing run and rewrite `integrity.json`. Does not start Habitat. |
| `--refresh-supervision PATH` | Write node-discovery and future-consequence labels for an existing run. These labels are not instruction-action imitation targets. Does not start Habitat. |

`scripts/curate_branch_benchmark.py` creates a separate, immutable-input data
revision. It reads fit/development trajectories and MP3D floor polygons to
confirm entry into a different annotated region. This target covers room-like
region transitions; it does not certify decision nodes or same-room junctions.
Original outcomes remain intact, including censored walks that contain a valid
earlier region entry. Sealed trajectories and their house annotations are not
opened. Optional Qwen annotation sees current/past RGB only and produces machine
silver node labels. It does not train LightNav or start a simulator.

| Flag | Meaning |
|---|---|
| `--phase {curate,annotate,summarize}` | Required. Curate registers the rules, source hashes, deduplicated region events and causal node review queue. Annotate runs one frozen visual-teacher worker. Summarize checks full worker coverage and writes weighted silver node examples. |
| `--output PATH` | Required. Curate requires a new directory; subsequent phases use that prepared directory. Worker files refuse overwrite. |
| `--batch PATH` | Raw source batch, default `output/world_model_data/branch_benchmark_20261105_01`. Raw files are never rewritten. |
| `--scenes PATH` | MP3D directory containing `<scene>/<scene>.house`; default `data/scene_datasets/mp3d`. Only visible-role houses are read. Geometry is offline supervision, excluded from student inputs. |
| `--legacy PATH` | Historical event dataset, default `output/world_model_training/event_latent_v1_20261104_01`. Indexes old fit rows separately for weak auxiliary training; old development rows stay excluded. |
| `--model PATH` | Frozen local teacher, default `checkpoints/world_model_planner/Qwen3-VL-4B-Instruct`. Verified asset manifest; no download. |
| `--worker N` | Annotation shard, default 0. Set `CUDA_VISIBLE_DEVICES` to one available GPU; the worker uses logical `cuda:0`. |
| `--workers N` | Total annotation shards, default 8. Summarize requires all shards and exactly one result per prepared example. |

`scripts/refine_branch_node_labels.py` retains the first teacher outputs and
adds an architectural-exit review. Agreement and separated bounding boxes are
still machine silver labels: they do not establish physical traversability or
human-verified node truth. The final reviewed release supersedes both drafts.

| Flag | Meaning |
|---|---|
| `--phase {prepare,worker,summarize}` | Required. Prepare freezes a new review protocol under a curated batch. Worker runs a current-image-only teacher shard. Summarize checks coverage and writes consensus drafts. |
| `--output PATH` | Required child directory of the curated batch, e.g. `.../node_architecture_review`; prepare refuses overwrite. |
| `--model PATH` | Local frozen Qwen checkpoint, same default as the curator. |
| `--worker N` | Shard index, default 0; logical `cuda:0` after CUDA device isolation. |
| `--workers N` | Total shards, default 8. |

`scripts/encode_curated_branch_data.py` caches frozen SigLIP pooler features
and 27 horizontal spatial columns for visible-role RGB only. It packs region
prediction tensors without training any parameters. The target type remains
annotated region entry, and source histories contain only one valid frame.

| Flag | Meaning |
|---|---|
| `--data PATH` | Required curated batch directory. |
| `--output PATH` | Required new feature directory; refuses overwrite. |
| `--normalization-run PATH` | Existing fit-only normalization, default `output/world_model_training/event_latent_v1_20261104_01`. Preserves the feature coordinates used by existing memory/model checkpoints. |
| `--encoder PATH` | Audited local SigLIP snapshot; default cached revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. Weight and preprocessor hashes must match. |
| `--batch-size N` | Encoding batch size, default 4, range 1–32. Uses logical `cuda:0`; no CPU fallback. |

`scripts/finalize_curated_branch_data.py` combines explicit review decisions
and local geometry corroboration into a separate release. Ambiguous examples
remain masked. Reviewed node labels are weak auxiliary supervision; they are
not human gold labels. It also prepares replay recipes for recorded promising
nodes without starting collection or altering the sealed split.

| Flag | Meaning |
|---|---|
| `--data PATH` | Required curated batch directory with features and node review artifacts. |
| `--reviews PATH` | Required JSON with reviewer identity, per-sample verdicts and the allowed missing-scan fraction. |
| `--output PATH` | Required new release directory; refuses overwrite. |

`scripts/recollect_reviewed_branches.py` runs a separate recorded-node trial.
Preparation uses the inference venv; workers use the installed Habitat Python
3.9 environment with one isolated CUDA device. Eight real source observations
are saved before choosing an exit. The controller first reaches a reviewed
entrance checkpoint, then follows the nearest current opening. Arrival requires
a stable, sufficiently deep entry into another annotated region. Same-room
depth gaps never establish completion. Old batches and sealed data are untouched.

| Flag | Meaning |
|---|---|
| `--phase {prepare,worker,summarize}` | Required. Freeze a new trial, run a task shard, or assemble actual outcomes. |
| `--output PATH` | Required trial directory; prepare and execution folders refuse overwrite. |
| `--origins PATH` | Reviewed replay recipes; default `output/world_model_data/branch_curated_20261105_01/release/recollection_origins.jsonl`. |
| `--source-batch PATH` | Immutable source batch containing replay prefixes; default `output/world_model_data/branch_benchmark_20261105_01`. |
| `--environment-step-budget N` | Prepare-time cap including source replay, default 4000. Rejects plans whose worst case exceeds the cap. |
| `--worker N` | Worker index, default 0. |
| `--workers N` | Shard count, default 8. |

`scripts/mine_branch_sources.py` runs CPU geometry in Habitat Python 3.9. It
uses included official viewpoint positions, annotated region floors, and local
navigation rays; it does not use graph degree or adjacency to declare a node.
Two different first-region witnesses and a free rear direction rank a candidate.
Current RGB must still establish distinct immediate architectural exits.

| Flag | Meaning |
|---|---|
| `--output PATH` | Required candidate directory; shard files refuse overwrite. |
| `--legacy PATH` | Existing event dataset defining allowed fit/dev houses; default `output/world_model_training/event_latent_v1_20261104_01`. |
| `--worker N` | CPU shard, default 0. |
| `--workers N` | Total shards, default 8. |
| `--per-scene N` | Maximum spatially separated proposals per house, default 12. These are not accepted node labels. |

`scripts/collect_viewpoint_branches.py` previews and collects new offline source
poses in the existing fit/dev pools. Viewpoint ids, source placement and geometry
are collection metadata only. Each source is placed offline, observed with eight
actual pan actions, and its final image hash must match on every branch repeat.
Workers explicitly rewind only the singleton allowlisted episode and use unique
action ids across resets. No actor imitation targets or goal-route supervision
are written. The archived v3 pilot plan caps fit/dev sources at 50/20, branch
length at 200 steps and total preview/collection steps at 60000. Use the v4
entry below for the currently authorized 1000-step event collection.

| Flag | Meaning |
|---|---|
| `--phase {prepare-previews,preview-worker,prepare-collection,collection-worker}` | Required. Prepare view poses/allowlists, render a scene shard, freeze reviewed source tasks, or execute their branch walks. GPU workers run in Habitat Python 3.9. |
| `--output PATH` | Required batch directory. Published protocols and actual observation folders refuse overwrite. Metadata-only preparation can resume before a protocol or previews exist. |
| `--candidates PATH` | Candidate mining directory, default `output/world_model_data/branch_source_candidates_20261106_01`. |
| `--legacy PATH` | Existing role/episode metadata, same default as the miner. Fit entry also respects the episode route gate. |
| `--reviews PATH` | Required for prepare-collection: explicit current-image reviewer and per-candidate decision verdicts. Teacher draft acceptance alone is insufficient. |
| `--worker N` | Scene shard, default 0. |
| `--workers N` | Shard count, default 8. Isolate one CUDA device per worker. |

`scripts/collect_event_branches.py` reuses reviewed current-image previews in a
new v4 batch. Both native Habitat limits and the wrapper episode limit are
disabled in this mode; the collector enforces **1000 executed branch steps**,
excluding the eight source pan observations. It ends earlier on a stable entry
into a different room, an independently reviewed next decision location, or an
observed physical revisit. Hallways, stairs and foyers are recorded as
intermediate regions. A revisit is not proof of a topological loop. The local
teacher controller can scan and attempt recovery when motion stalls; an
unrecovered stall remains incomplete. In v4.1, real three-dimensional progress
resets the local recovery counter. Earlier scans in another location cannot
exhaust the current location's recovery allowance. A short free local native
probe can advance the offline teacher across a stair landing without requiring
a long body-height gap in the current depth image.

The fallback status `censored_max_steps` is never a successful future. Every
executed command and observation is committed together. Creating a `PAUSE` file
in the batch root, or low disk space, saves `paused_resource`; remove `PAUSE` and
use `--resume` to restore all preceding commands with exact RGB verification.
Direct pose snapshots are recorded but are not trusted as exact restoration.
Restore steps are reported separately and do not create training duplicates.
Automatic resume applies only to resource pauses; censored walks are retained
without silently increasing the authorized 1000-step ceiling.

| Flag | Meaning |
|---|---|
| `--phase {prepare,worker}` | Required. Freeze reviewed source tasks or execute a scene shard in Habitat Python 3.9. |
| `--output PATH` | Required new batch directory. Preparation refuses overwrite. |
| `--previews PATH` | Repeatable preview roots; required for prepare. |
| `--reviews PATH` | Frozen explicit current-image review with accepted candidate ids; required for prepare. |
| `--label-landmarks PATH` | Optional frozen parent landmarks for consistent event labels during a controller correction. Offline labels only; never controller targets or model inputs. |
| `--max-branch-steps N` | Prepare-time fallback, default 1000, allowed range 1–1000. Records incomplete trajectories at the limit. |
| `--worker N`, `--workers N` | Scene shard index/count, default 0/8. Isolate one CUDA device per worker. |
| `--task-id ID` | Repeatable optional execution subset for smoke verification. |
| `--resume` | Continue resource-paused trajectories using verified action replay. Completed and censored results are skipped. |

The collection-only RPC server `experiments/world_model/serve_collect.py`
accepts mutually exclusive `--max-steps N` (bounded legacy mode, 1–500) or
`--event-driven` (disable native step/time limits and wrapper step limit; the
calling collector must enforce its event and resource rules). The archived
controlled trial refuses event-driven mode so its original budget stays fixed.

`scripts/prove_event_driven_collection.py --previews PATH --output PATH` runs a
native regression diagnostic past Habitat's former 500-step boundary, verifies
nontermination and bounded retry-image retention, and reports whether a direct
pose snapshot reproduces RGB exactly. Its observations are not training data.

`scripts/review_viewpoint_sources.py` uses frozen local Qwen on current RGB with
two proposed bearing markers. It receives no future regions or maps. Outputs
remain teacher drafts until explicit visual review; indexed contact sheets are
written by summarize.

| Flag | Meaning |
|---|---|
| `--phase {worker,summarize}` | Required. Run a teacher shard or verify full coverage and create visual-review sheets. |
| `--output PATH` | Required prepared viewpoint batch. |
| `--model PATH` | Local verified teacher checkpoint, default `checkpoints/world_model_planner/Qwen3-VL-4B-Instruct`. |
| `--worker N` | Teacher shard, default 0; uses logical `cuda:0`. |
| `--workers N` | Total teacher shards, default 8. |

`scripts/assemble_branch_dataset.py` creates a separate release from actual
recorded futures. It verifies identical causal histories within intervention
groups, preserves censored walks, deduplicates exact repeats, balances sampling
and excludes targets with more than 20% missing black pixels from primary data.
Geometry witness regions never substitute for observed outcomes.

| Flag | Meaning |
|---|---|
| `--source PATH` | Required completed viewpoint collection batch. |
| `--output PATH` | Required new assembly directory; refuses overwrite. |
| `--pilot PATH` | Optional completed recorded-node recollection trial, kept identifiable in provenance. |
| `--old-curated PATH` | Optional previous curation directory, indexed as separate auxiliary sources rather than merged into primary counts. |

`scripts/assemble_event_branches.py --source PATH --output PATH` assembles v4
committed trajectories into a new release. It checks contiguous executed-step
indices, the 1000-step ceiling, identical paired source histories and actual
future image hashes. Room/decision/outdoor events are primary; physical revisits
have their own files. Censored, stalled, erroneous and quarantined outcomes stay
in the execution index. Exact repeats collapse before sampling weights and pair
counts are computed. Nearby source-pose components are reported separately;
they are not asserted to be distinct semantic junctions.
Optional repeatable `--supplement PATH` replaces every branch and repeat of a
corrected origin with its new executions, regardless of outcome. Partial
origin replacements are refused; all parent raw evidence remains preserved.

`scripts/audit_event_branch_assets.py --source PATH [--source PATH ...]
--release PATH --output PATH` verifies every retained RGB/depth file hash,
contiguous execution counts, frozen inputs and versioned source snapshots.
Superseded controller attempts remain included in the audit.

`scripts/package_branch_node_labels.py` writes current-view silver labels without
promoting them to human gold. Unknown images stay masked; ordinary labels apply
to the visible view and do not assert that unseen directions have no exits.

| Flag | Meaning |
|---|---|
| `--previews PATH` | Repeatable current-view preview roots. |
| `--reviews PATH` | Required explicit accepted and ordinary candidate ids with reviewer identity. |
| `--output PATH` | Required new JSONL file; refuses overwrite. The v4 assembler copies `node_labels_review.jsonl` into its release. |

`scripts/encode_event_branch_data.py` uses the audited frozen SigLIP encoder and
historical fit-only normalization. All eight real source observations are
retained in tensors; target frames and spatial column features remain actual
recorded images. It performs no world-model training or LightNav updates.

| Flag | Meaning |
|---|---|
| `--data PATH` | Required assembled v4 release with primary events and node labels. |
| `--output PATH` | Required new feature directory; refuses overwrite. |
| `--normalization-run PATH` | Existing fit-only coordinates, default `output/world_model_training/event_latent_v1_20261104_01`. |
| `--encoder PATH` | Audited local SigLIP revision and weights, same default as the curated encoder. |
| `--batch-size N` | Default 8, range 1–32. Uses isolated logical CUDA device 0. |

`scripts/finalize_event_branch_data.py --data PATH` adds offline place-positive
labels to a separate `features/training_tensors.npz`, verifies all eight valid
history slots, and runs a CPU forward/backward smoke check on actual fit data.
It performs zero optimizer steps. Room aliases share a scene-scoped region
label; reviewed decision targets use their landmark identity so different
decisions in one hallway are not merged. `data_release.json` binds the final
files and explicitly records that legacy training-launcher configurations need
adaptation to this new dataset schema.

`scripts/train_curated_branches.py` trains a bounded predictor pilot from the
immutable v4.1 release. It checks release hashes, disjoint fit/development
houses, identical paired histories, normalization, label aliases and the fit
mask. Only fit samples contribute gradients. Frozen visual features are reused;
SigLIP and LightNav are not loaded. The existing world-model authorization gate
remains active. No sealed data or navigation evaluation is opened.

Preparation registers eight arms: a width-128/two-layer predictor with and
without exit direction for seeds 11/23/47, plus the existing width-256/four-layer
model with and without direction for seed 11. Each arm uses at most 2000 updates,
batch 32, learning rate 0.0002, and early stopping on development scene-macro
feature cosine. Development metrics therefore include checkpoint selection and
are not a blind test. Paired-exit evaluation uses the same source history and
two different actual future labels; equal predictions receive a 0.5 tie score.
It does not use a development future-image retrieval gallery.

| Flag | Meaning |
|---|---|
| `--phase {prepare,train,summarize}` | Required. Freeze inputs/baselines, train one arm, or aggregate all arms and export a fixed-seed candidate. |
| `--data PATH` | Required for prepare: immutable release with `data_release.json`. |
| `--output PATH` | Required new run root. Preparation and arm folders refuse overwrite. |
| `--worker N` | Arm index 0–7, default 0. Isolate one CUDA device and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before each worker. |
| `--initial-checkpoint PATH` | Optional prepare-time initialization from an existing world-model checkpoint. Registers four matched branch/no-branch arms with seeds 11/23, learning rate 0.00002 and at most 1000 updates. Both controls start from the same weights. |
| `--initial-normalization PATH` | Required with initial-checkpoint: its normalization NPZ. Mean/scale must exactly match the released feature coordinates. |

`candidate_predictor.pt` uses the existing raw-feature `EventFeaturePredictor`
interface with bound fit-only normalization. Exporting a trained candidate does
not enable it for route selection or establish navigation success.
Fine-tuning also retains the step-zero checkpoint; if updates do not improve
the registered development score, it reports that unchanged best step rather
than forcing a degraded new checkpoint.

## Stateful event world model (experimental, opt-in)

`scripts/prepare_tssm_sequences.py` builds contiguous observed sequences from
the immutable v4.1 release. Both actual repeats are retained, with the 90/36
independent-event denominator unchanged. Intermediate route anchors use offline
path distance and actual RGB; they are not new decision-node labels. Coordinates
and future executed controls are not neural prior inputs. Frozen SigLIP and
historical fit-only normalization are reused.

| Flag | Meaning |
|---|---|
| `--phase {prepare,encode}` | Required: freeze sequence references or encode missing anchors and pack tensors. |
| `--data PATH` | Required for prepare: released event dataset. Encode reads the bound source from protocol metadata. |
| `--output PATH` | Required sequence root. Prepare refuses overwrite; encode expects prepared metadata. Use a fresh prepared directory for each encoding release. |
| `--batch-size N` | Default 8. New RGB encoding requires CUDA and audited local SigLIP weights. |

`scripts/train_event_tssm.py` trains explicit deterministic/stochastic state
transitions. Actual-observation posterior state feeds the next transition;
multi-step imagination uses priors only. Global gradients use fit episodes
only, including only past fit support for memory augmentation. Cold development
evaluation has no memory. Development selects checkpoints and is not a blind test.

| Flag | Meaning |
|---|---|
| `--phase {prepare,train,summarize}` | Required: register eight arms, train one, or aggregate and export the fixed `tssm_pair_seed11` candidate. |
| `--data PATH` | Required for prepare: packed sequence root. Train uses its registered path. |
| `--output PATH` | Required run root. Prepare and individual arm folders refuse overwrite. |
| `--worker N` | Arm index 0–7, default 0. Seeds 11/23 each compare full, no paired loss, no state feedback and no direction. Isolate a CUDA device and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. |

Defaults are width 128, stochastic width 32, two dynamics layers, four heads,
16 transition tokens, at most 1000 updates, batch 24 and learning rate 0.0002.
Objectives include direct event prediction, observed route transitions,
three-step overshooting, KL, contrastive and paired-exit losses. Best and last
checkpoints are retained. `load_tssm_predictor` loads `event-tssm/0.1` bundles;
raw `[B,8,1152]` history and `[B,2]` direction produce normalized `[B,1152]`
future features. Supply the valid-history mask when padding observation slots.

`scripts/eval_tssm_replay.py` compares frozen prediction, memory, scene updates
and their combination. Each query is saved before the real outcome is revealed.
Development support is used only after reveal in this separate adaptation
experiment. Replay order is declared, not simulator wall-clock chronology.
Same-view repeats do not validate cross-view relocation or navigation success.

| Flag | Meaning |
|---|---|
| `--data PATH` | Required packed sequences and raw feature cache; replays development episodes. |
| `--model PATH` | Required `event-tssm/0.1` checkpoint; its global weights stay frozen. |
| `--output PATH` | Required new replay root; refuses overwrite. Retains facts, scene adapters, traces, predictions and source snapshots separately for all four modes. |
| `--update-steps N` | Default 20 scene optimizer steps after each actual verified event. |

`StatefulTSSMHarness` explicitly connects `TSSMBranchQueryService` to the
existing task planner. Create a `TSSMSession` with a feature-contract-bound
`PersistentSceneMemory`, checkpoint hash and storage directory. `use_memory`
defaults to true; `online_learning` defaults to false. `begin_episode` resets
latent motion state while retaining real facts and scene parameters. Candidate
`forecast` calls fork the same actual state; `commit` selects the executed exit.
Supply actual route feedback with `report_observed_milestone`, and verified
terminal feedback with `report_verified_event`. Terminal feedback requires
actual observation/execution references, `executed=True` and `arrived=True`.
Report actual direction when the executor deviates from its planned exit.

The rank-16 scene decoder adapter has 59,088 parameters. It updates after real
outcomes, and applies only to previously observed input source/direction
conditions (cosines at least 0.98/0.98); eligibility does not inspect target
values. Posterior state assimilation is distinct from parameter optimization.
Global TSSM, SigLIP and LightNav weights stay frozen online. Real facts inform
the current instruction; old task instructions are not replayed as route
commands. This integration requires explicit opt-in and actual executor
feedback; the existing default navigation entry point is unchanged.

## Stateful Harness navigation loop

`scripts/run_stateful_harness.py` connects the event TSSM to actual LightNav
execution, persistent observation images, geometric exit alignment, task
progress verification and visually supported event writeback. It uses existing
local checkpoints and an existing separate Habitat environment; it does not
download datasets, change global model weights or start robot hardware.

Queries now route before future prediction. Consistent real source/exit
matches return the recorded outcome without calling the TSSM transition or
decoder. Source cosine must be at least 0.98, aligned exit cosine at least
0.98, and pairwise historical outcome cosine at least 0.9. In the live loop,
actual image geometry must support heading transport. Familiar-looking sources
with failed alignment, conflicting outcomes or an exhausted lookup budget
request active observation. A miss may call the world model. These thresholds
are conservative heuristics, not calibrated probabilities or proof of physical
place identity. Prediction traces identify which route was used.

Skipping candidate prediction does not disable actual-state assimilation.
For a memory hit, any prior needed for posterior feedback or scene optimization
is deferred until after actual execution. It receives the original source and
direction, not the target feature. Unexecuted alternatives are never training
facts. Beginning another episode resets motion state while retaining facts,
images, scene weights and optimizer state.

| Flag | Meaning |
|---|---|
| `--tasks PATH` | Required JSON list of `{scene, episode_id}` engineering tasks. Declared order is preserved, including explicit repeats; sealed scene `Vvot9Ly1tCj` is refused. Identities select simulator tasks only and are not neural position/node inputs. |
| `--output PATH` | Required new run root; refuses overwrite. Stores configuration, bound input hashes, source snapshots, actual commands, queries, visual evidence and results. |
| `--experience PATH` | Optional persistent experience directory; default `<output>/experience`. Reuse it to retain facts, hashed observation images and scene adapters across process restarts. Use separate directories for comparison conditions. |
| `--world-model PATH` | Default `output/world_model_training/event_tssm_v2_20261106_01/candidate_tssm.pt`. Must use `event-tssm/0.1`. |
| `--lightnav-model PATH` | Default `checkpoints/LightNav-0`. Uses the existing `vllm_local` backend, CUDA and local checkpoint; no fallback. |
| `--planner-model PATH` | Default audited local `checkpoints/world_model_planner/Qwen3-VL-4B-Instruct`. |
| `--encoder PATH` | Default audited local SigLIP revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`; weights and preprocessing hashes are checked. |
| `--habitat-python PATH` | Default `/home/szgx/miniconda3/envs/habitat/bin/python`; must already exist. |
| `--sim-config PATH` | Default `experiments/world_model/configs/collect_train_go2w_radius010.yaml`. Its `velocity_control` section must explicitly contain `min_abs_lin_speed` and `min_abs_ang_speed`; the runner reads them to intercept implicit zero-velocity STOP commands. |
| `--dataset-split {train,val_seen,val_unseen}` | Simulator episode split, default `train`. R2R val-unseen evaluation passes `val_unseen`. The yaml split is overridden. |
| `--port N` | Simulator ZMQ port, default 5870. |
| `--sim-gpu N` | Physical simulator device, default 0. Inference uses isolated logical CUDA device 0. |
| `--max-steps N` | Navigation evaluation budget, default 500, allowed range 1–500. User policy: evaluate with 500 steps; the separate event data collector retains its 1000-step fallback. Reaching a client/environment limit does not establish arrival or produce an event training label. |
| `--planning-interval N` | Default 12 executed steps. A committed event is not overwritten by periodic replanning. |
| `--verification-interval N` | Default 6 executed steps between visual checks/state milestones. This is a check cadence, not a future-target horizon or arrival criterion. |
| `--observation-budget N` | Default 8 requested views per recovery phase. Actuator-limited yaw can span multiple steps; all count against the evaluation cap. Exhaustion switches recovery strategy rather than automatically failing the whole episode. See task-aware recovery below. |
| `--online-learning` | Enable scene decoder updates after visually supported actual next events; disabled by default. Global TSSM, SigLIP and LightNav stay frozen. |
| `--update-steps N` | Default 20 scene adapter optimizer steps per verified event. |

Local LightNav STOP proposals cannot directly end the global task. Unverified
local completion restores the original instruction and replans the already
observed frame without appending it twice to native history. A positive local
event proposal awaits a second motion-separated actual view. A semantically
confirmed global STOP requires ordered task stages, the requested target and
requested stop relation, supported by repeated actual observations. The default
policy may also dispatch an explicitly unverified original-policy STOP after
bounded recovery, as described below. Memory, model predictions and policy STOP
alone do not establish completion. Goal verification is visual, not an oracle
distance check, and can still make semantic errors.

The earlier controller used `QwenStrictProgressVerifier` for the final stage;
it remains available for regression. The current runner uses the task-aware
verifier described below, which can retain grounded identity across views.
Target visibility and the camera's requested spatial relation are separate
required outputs. Entering a room requires `inside_target_room`; stopping in
a doorway explicitly permits `at_entrance`. Seeing a landmark through a door
does not establish entry. When the final subtask names one supported common
object, a positive proposal also requires a current-image box and a second
crop classification whose prompt omits the task and expected target. Missing
mesh triangles are not admitted as landmarks. This additional audit reduces
one observed hallucination failure; it does not provide ground-truth arrival
or cover every compound target description.
Boxes in either 0–1 or 0–1000 normalized coordinates are accepted and converted
to 0–1000; invalid, too-small and nearly full-image boxes cannot ground a target
or direct observation. If the cropped object's blind classification is unknown,
one additional blind call may inspect the crop alongside surrounding context of
at most 60% of the image. It classifies the original crop, without receiving the
task or expected category. A confident different-object result is not retried.
Both calls use the same vision model and their agreement is not independent
ground truth; the fallback has only selected-development-case validation.

Only an actually executed selected entrance, forward-command prerequisites,
and two motion-separated visually supported next-room/next-decision views can
write an arrived event. The current controller also requires a measured source
aperture crossing before asking the event verifier: two depth-supported boundary
strips must define a plausible opening with deeper interior, and continuous
local RGB-D motion must cross that segment inside its aperture. Unknown source
geometry, a broken motion chain, a closed/window-like surface or a
crossing outside the selected aperture prevents writeback. The plane is a sensor
estimate, not ground truth; these requirements prioritize attribution over recall.
Actual source depth is retained with a hash in each query record for audit.
Active turns, fixed-duration spans, interrupted options
and exhausted budgets do not become arrived facts. New event and final-task
verification are separate. Persisted event evidence is explicitly labelled
`temporal_visual_event_support_not_ground_truth`; it is not automatically
admitted as gold world-model training data. Real goal-distance/success metrics
are retained only in evaluation logs and never passed into the controller.

## Task-aware completion and execution recovery

The stateful runner now uses `QwenTaskAwareProgressVerifier`. Its final contract
distinguishes entering a room, stopping in a doorway, approaching a target and
looking at one. A room target independently grounded earlier in the SAME episode
may establish identity after it leaves view, if actual visual continuity and the
requested crossing are supported. Looking at a target still requires its current
visibility. Compound enter-and-look goals retain both relations. Identity is
never copied from a previous task or an imagined future. Intermediate stages may
record final-target identity without demanding the final destination prematurely.
When the verifier claims the current frame visibly contains a specific target but
the current crop fails the independent object check, the previous identity cannot
rescue that conflicting claim. An honestly out-of-view target may still use the
earlier same-episode identity if the room relation and continuity are supported.
Task clauses must be literal, ordered spans of the original instruction and
cover its constraints; invalid decomposition falls back to the original request.
On a failed decomposition, the base policy still receives that original request,
but task progress is not marked semantically complete from the destination view
alone. An explicit `then` sequence cannot be certified as one undivided stage.
Intermediate completion uses a separate model call with only the active clause,
prior completed clauses and actual execution/views. Its confidence is independent
of final-target visibility and final-goal confidence. Final-target grounding can
continue independently without controlling admission of the intermediate clause.

Original-policy global STOP has three outcomes: visually confirmed, positively
contradicted, or unknown. Unknown receives bounded recovery-view requests. With
the default `defer` policy, a repeated original GLOBAL STOP after those views is
dispatched with semantic status UNVERIFIED. This is not task-progress completion
and does not produce arrived facts or scene training targets. Local STOP is never
promoted this way: the original task is restored and re-inferred first. Positive
contradiction prevents uncertainty fallback. Use `abstain` for a strict control.
If an intermediate task stage already has a first supporting view and there is
observation budget left, the controller requests its second view before taking
the unknown-STOP fallback. When a stage advances, recovery counters reset in the
same controller step, so the next stage is not stopped using stale counters. With
no budget left, the configured fallback applies.
An absent target, a different object, or an obstruction alone is not a proven
goal contradiction. A room-entry contradiction needs a grounded target and an
explicit outside relation; a doorway-before-entry contradiction also needs an
independently grounded door. An unusably small target box cannot direct a view.
Benchmark success and semantic confirmation at dispatch are recorded separately;
an improvement from STOP handling alone is not a world-model navigation gain.
Habitat also treats a velocity command whose actual linear and angular speeds
are both below its configured thresholds as STOP. Before dispatch, the controller
decodes normalized speeds using the simulator ranges and intercepts any such
command that was not explicitly authorized as STOP. It requests a distinct view
within the observation budget; if no non-STOP observation is available it
abstains instead. Repeated look targets cannot generate zero-angle observations.
The runner records both interception and the simulator's termination reason,
so a hidden environment STOP cannot be mislabeled as reaching the 500-step cap.
An experience directory marked `DO_NOT_ADMIT_EXPERIENCE.json` is refused by
`PersistentSceneMemory` before opening its database. This prevents reuse of the
two quarantined task-execution trials whose exit attribution failed review.

RGB-D PnP measures local camera motion independently of issued controls. Missing
geometry stays unknown. Frame observations, command id/version, motion evidence,
front clearance, skill completion/abort and recovery actions are retained.
In the live runner, two positive views for a task stage or event also require
same-epoch measured camera displacement of at least 0.10 m (with at least three
executed steps), or a measured relative yaw of at least 15 degrees. These are
view-separation heuristics, not goal distances or world-model prediction horizons.
On a broken or unavailable motion chain the confirmation is re-anchored and
cannot be certified by issued controls alone.
For walking, entering and other translation-requiring stages, the controller
also requires at least 0.10 m of net measured displacement from an available
pose within that stage and the current odometry epoch. A turn between the two
confirmation views may establish distinct viewpoints, but does not replace this
stage-level displacement. An initial missing pose or a broken epoch can make
this conservative test miss a valid movement. Displacement and visual support do
not prove that the whole route instruction was followed.
Front clearance is not goal distance. Exit control initially orients toward its
grounded bearing; failed or superseded execution cannot be assigned to a new
option. Recovery can replan or return to a recent ACTUALLY observed local anchor
only while odometry and current depth support it. Recovery completion does not
advance a required task clause.

For an approach-to-object goal, a currently grounded candidate box can supply
its actual current RGB-D region to a separate spatial-relation check. A box is
not a pixel-level object mask: its 10th–90th percentile spread must be no more
than the larger of 0.25 m or 25% of its median, otherwise the depth is treated
as mixed and no single target range is claimed. This conservative consistency
test is an uncalibrated heuristic; even a uniform region may be background.
The model receives the region estimate with that limitation, not an asserted
object or reference-goal distance. For an explicitly `next to`, `beside`, `near`,
or `close to` relation, missing, unbound or mixed depth cannot support a positive
completion; it remains unknown. A separate visual relation check is still
required, without a universal hard-coded arrival distance.

Unvalidated world-model predictions are monitored during original LightNav
execution by default, without overriding its route. A confirmed actual entrance
can still write memory/scene updates. Reliable direct memory can support active
options. Optional exploration overrides remain explicitly configurable.
Passive hypotheses whose source geometry or motion chain is unavailable are
refreshed at new planning views instead of occupying the full option lifetime.
Hypotheses with supported geometry continue tracking the actual crossing.

The observation budget is per recovery phase, not a lifetime eight-view episode
limit. It resets after measured translation or a supported task-stage advance,
not mere image changes or repeated STOP. An observation option may span several
actuator-limited yaw steps to reach its requested view; those steps still count
against the 500-step evaluation cap. Measured RGB-D yaw corrects the turn when
available; otherwise its completion uses an issued-control estimate and is not
proof of an actual change in viewpoint. Final returned frames are checked for
diagnostics/writeback, never for retroactive STOP or benchmark success.

| Additional flag | Meaning |
|---|---|
| `--unknown-stop-policy {defer,abstain}` | Default `defer`: preserve original global policy STOP after unresolved bounded recovery, explicitly unverified. `abstain` requires visual support and can end without STOP. |
| `--recovery-cycles N` | Default 3 distinct replan/backtrack cycles per progress phase. Failure is not automatically marked complete or skipped. |
| `--option-max-steps N` | Default 100, range 1–500. Local control/observation option lifetime; an expired option is aborted without an arrived label. Does not change the data collector's 1000-step limit. |
| `--depth-scale X` | Default 10 metres per normalized depth unit for the installed Habitat sensor. Set to 1 for metre-valued depth. This calibrates local sensing, not target coordinates. |
| `--camera-hfov X` | Default 120 degrees, matching the RGB/depth sensor. Used by perception bearings, direction alignment and RGB-D motion estimation. |
| `--exploration-overrides` | Opt in to regular active overrides supported only by exploratory predictions. Disabled by default; unknown exits can be monitored while original LightNav acts. Supported recovery may still invoke an active exploratory option. |

`scripts/calibrate_harness_completion.py` replays a manifest of actual labelled
RGB cases without executing controls or changing model parameters. Expected
labels and evaluation distances are not sent to the verifier. It evaluates
completion proposals, not full temporal STOP acceptance or blind accuracy.

| Flag | Meaning |
|---|---|
| `--cases PATH` | Required JSON cases: original instruction, a `stage` or ordered `stages`, anchor, actual frame refs/hashes up to the current step, recent actual controls and evaluation-only accepted verdicts. Optional `expected_active_completed` checks the independent active-clause result. Optional `current_depth` binds a depth `.npy` to the current image `reference`, its file `path`, `sha256`, and `metres_per_depth_unit`; image dimensions, hash and scale are verified. Future frames are refused. |
| `--output PATH` | Required new output root; refuses overwrite. Saves source snapshots, input bindings, raw calls and verdicts. |
| `--model PATH` | Default audited local Qwen3-VL-4B visual planner. Requires CUDA; no fallback. |
## Branch prediction and persistent-memory decision experiments (2026-11-10)

`scripts/refine_branch_tssm.py` refines an existing `event-tssm/0.1` bundle
using only fit examples, including their actual repeated executions. Paired exits use
primary examples and must have identical input histories
and distinct observed outcomes. The objective combines future cosine, per-exit
two-way retrieval cross entropy, and matching the observed difference between
the two outcomes. Development houses select checkpoints by per-exit retrieval,
then simultaneous correctness of both exits, then feature cosine. Development
selection is not a blind evaluation.

| Flag | Meaning |
|---|---|
| `--data PATH` | Required existing TSSM sequence directory. Its fit/development house split and normalization must agree with the initialization. |
| `--initial-checkpoint PATH` | Required existing event TSSM bundle. |
| `--image-features PATH` | Optional admitted `image_features.npz`, indexed by actual RGB SHA256. Enables a current-exit visual encoder. Only current-source patch columns are read as inputs. Old bundles retain `use_exit_visual=false`. |
| `--output PATH` | Required new output directory; existing results are refused. |
| `--steps N` | Maximum fit updates, default 400; evaluate every 25. |
| `--seed N` | Training seed, default 11. |
| `--pair-weight FLOAT` | Weight on the discriminative objective, default 2.0. |
| `--sequence-weight FLOAT` | Weight on actual route posterior reconstruction and overshooting, default 0.0. Use a positive value to retain the feedback dynamics during exit refinement. |
| `--learning-rate FLOAT` | AdamW learning rate, default 0.0002. |

The optional visual encoder reads actual current SigLIP patch columns with the
camera projection for the proposed bearing. A visual bundle requires these
columns at inference; it cannot silently run without that input. Motion updates
still observe real feedback. Predicted features are never committed as facts.

`scripts/run_stateful_harness.py` additionally supports:

| Flag | Meaning |
|---|---|
| `--prediction-overrides` | Allow one local entrance option only when the planner cites a retrieved `arrived` event, the future differs sufficiently from other visible exit futures (or current state when only one exit is visible), retrieval clears both similarity/margin gates, and its top retrieval changes relative to copying current state. `observed_span` remains analogy-only. Default off. |
| `--minimum-prediction-angle DEGREES` | Default 10.0, strictly between 0 and 180. This is an experimental control threshold, not a calibrated probability. |
| `--minimum-retrieval-margin FLOAT` | Default 0.02. Predicted query score advantage of its selected, cited outcome over the source-copy top outcome; range 0–2. |
| `--minimum-retrieval-similarity FLOAT` | Default 0.5, range -1 to 1. Rejects prediction control when the predicted future has weaker cosine similarity to the retrieved historical future. This initial engineering gate is not calibrated; a matching `observed_span` remains route analogy, not proof of an exit outcome. |
| `--base-policy-only` | Run the unchanged LightNav policy directly. No Harness operations, semantic STOP handling, or memory queries. Cannot enable exploration overrides, prediction overrides, or online learning. |
| `--run-id TEXT` | Optional 1–128 character alphanumeric/underscore/hyphen reference namespace. Evaluation preparation shares one namespace across independent conditions so arbitrary UUID text does not change VLM prompts. Do not resume an already begun episode id into the same bank. |
| `--disable-memory` | Suppress direct memory, prediction conditioning, and retrieved outcome evidence. Real execution may still be journaled for auditing. |
| `--current-episode-memory` / `--no-current-episode-memory` | Default enabled for live runs. Recall facts already written by actual execution in this episode as well as previous episodes, within the query's fixed sequence cutoff. Offline session APIs retain prior-episode-only defaults. Preloaded future transitions are not admitted by this live setting. |
| `--route-memory` / `--no-route-memory` | Default enabled. Journal continuous RGB-D route spans with 0.6–2 m measured net displacement. These are typed `observed_span`; they are retrieval analogies, never next-room arrivals or direct confirmed exits, and do not update model parameters. |
| `--policy-stop-roundtrip` / `--no-policy-stop-roundtrip` | Default enabled with `--unknown-stop-policy defer`. Latch an uncertain original LightNav STOP, inspect in place, return toward its original heading, then dispatch that STOP without requiring another forward navigation proposal. Fresh explicit nonarrival evidence after the scan prevents dispatch. `abstain` retains strict behavior. |

Direct exit lookup first proposes appearance candidates at cosine 0.5. Without
an image transport it retains the 0.98 source and exit agreement checks. With
accepted SIFT/RANSAC yaw or RGB-D/PnP alignment, a source below 0.98 may be
accepted after exit transport and outcome consistency checks. Unverified
alignment never establishes a known exit. An observed route span cannot use this
direct next-event route. The shortlist cap still requests observation when a
complete consistency check cannot be made.

Query logs separate `past_sequences` (conditioning facts), `retrieved_sequences`
(actual outcomes shown), `used_evidence_sequences` (planner citations),
`prediction_changes_top_retrieval`, and `control_source`. Result counters separate
predicted control, direct-memory control, verified arrivals, and measured route
observations. A nonzero model-call count alone establishes none of these effects.

`scripts/prepare_harness_evaluation.py` prepares independent base, monitor,
memory, and predictive conditions. One worker owns every episode of a physical
scene, preserving its input order. Each condition starts with a separate empty
experience directory; online parameter updates remain disabled. Execute the
generated `run_all.sh` to run conditions sequentially, or a condition's `run.sh`.
Worker failures propagate through the launcher.

| Flag | Meaning |
|---|---|
| `--tasks PATH` | Required, repeatable task-list inputs; duplicate scene/episode pairs are refused. |
| `--output PATH` | Required new schedule directory. |
| `--world-model PATH` | Required candidate shared by the Harness conditions. |
| `--workers N` | Default 8, range 1–8. Scenes are packed intact, largest first. |
| `--dataset-split {train,val_seen,val_unseen}` | Default train. |
| `--port N` | First worker port, default 6050. Conditions reuse the range sequentially. |
| `--sim-config PATH` | Default `habitat_server/configs/vlnce_r2r_go2w.yaml`. |
| `--habitat-python PATH` | Default existing Habitat Python 3.9 environment. No Habitat packages are installed in the inference environment. |

`scripts/audit_live_memory_routes.py --source RUN --output NEW_DIR
[--maximum-queries 200]` replays logged source/exit queries. It opens the source
SQLite files read-only and creates a separate replay journal. Only facts from
episodes whose result was written before the query's timestamp are admitted.
It compares fragmented legacy lookup, merged legacy lookup, and merged
geometric lookup. Source depth hashes are verified. Current held-out outcomes
are used only for post-query diagnostics, never for retrieval. This is a routing
replay, not a navigation rerun.

`scripts/summarize_harness_evaluation.py --tasks TASKS --run base=BASE_DIR
--run predictive=PREDICTIVE_DIR --output REPORT` accepts repeatable labeled run
directories. It checks duplicate episodes and compares SR/SPL only when both
conditions complete the exact requested task set with identical initial RGB
hashes. It reports success gained/lost among controlled episodes. Incomplete
condition scores remain explicitly marked partial.
