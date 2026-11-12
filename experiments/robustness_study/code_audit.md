# LightNav-0 代码审查：扰动、解码、历史和停止

审查日期：2026-10-20。只读 `src/lightnav`、`habitat_server` 和已有评测日志。没有改模型结构，没有加载 `checkpoints/LightNav-0-go2w`。

结论先写在这里：仓库里没有相机抖动实现。评测解码是贪心。Pointing 不进入控制。历史是画面缓冲，没有指令进度状态。2026-10-20 的 30 条逐步日志已经补上航点、Pointing、位姿和执行前后航向；10 月 15 日的全量 `results.jsonl` 仍然只有回合级字段。

## 1. 图像进入模型的路径

Habitat 配置只设置相机位置、分辨率和水平视场，没有方向噪声。

官方 R2R：`habitat_server/configs/vlnce_r2r.yaml` 第 49–61 行。`rgb_sensor` 为 480×270、`hfov: 120`、`position: [0, 0.88, 0]`。没有 `orientation`，没有噪声字段。

Go2-W：`habitat_server/configs/vlnce_r2r_go2w.yaml` 第 36–51 行。机身 `height: 0.50`、`radius: 0.215`，相机 `position: [0, 0.50, 0]`，分辨率和视场与官方相同。

仿真把 RGB 和深度都放进观测。`habitat_server/lightnav_habitat/base.py` 第 88–116 行按 `rgb` / `head_rgb` 和 `depth` / `head_depth` 取出图像。策略只读 RGB：`src/lightnav/habitat/policy.py` 的 `TrajVocabVLNCEPolicy.act` 第 158–167 行调用 `self.agent.observe(obs["rgb"])`，深度没有传入引擎。

预处理在 `src/lightnav/inference/frame_preprocessing.py` 的 `rgb_frame_to_model_tensor`（第 57–77 行）：HWC `uint8` 转成浮点，除以 255，双线性缩放到检查点的 `video_size`，再映射到 `[-1, 1]`。检查点 `checkpoints/LightNav-0/eval_config.json` 第 5–7 行把这个尺寸定为 256×448。这里没有旋转、平移或加噪。

全仓库检索 `perturb`、`jitter`、`camera_noise`、`orientation` 噪声，评测路径上没有命中。同事笔记里的「相机 ±1° 抖动」不在本仓库的配置或调用链里。它作用在原始图像、相机外参还是机器人航向，无法从这份源码判断。

## 2. 随机性和采样

`src/lightnav/habitat/runner.py` 第 6 行写明评测解码是贪心，没有采样旋钮。

实际参数在 `src/lightnav/inference/engine.py` 的 `_get_sampling_params`（第 27–40 行）：

| 环境变量 | 默认 |
|---|---|
| `VLN_EVAL_TEMPERATURE` | `0.0` |
| `VLN_EVAL_TOP_P` | `1.0` |
| `VLN_EVAL_TOP_K` | `0` |
| `VLN_EVAL_TRAJ_TOP1` | 关闭 |

HF 路径在第 408–422 行：`temperature > 0` 才 `do_sample=True`。vLLM 路径在 `llm_generate_batch` 第 640–657 行把同一组参数放进 `SamplingParams`。这份代码没有向 `SamplingParams` 传 `seed`。2026-10-20 在空闲的 2 号卡上启动引擎时，vLLM 0.19.1 自己的日志写了 `seed=0`，`enforce_eager=False`。

回合重置把种子显式设成空。`src/lightnav/habitat/remote_env.py` 第 115–116 行发送 `{"seed": None, "options": None}`。数据集顺序因此是确定的，不是每个回合重新抽初始航向。

`VLNCEEnv` 的提前停止默认关闭：`habitat_server/lightnav_habitat/vlnce.py` 第 147–148 行，`early_stop_rotation=0`、`early_stop_steps=0`。

可能让两次运行不一致、但当前默认没有打开的因素：温度大于 0、`VLN_EVAL_TRAJ_TOP1`、改 `max_new_tokens` 到截断动作 token、vLLM 的 CUDA graph（`src/lightnav/inference/vllm_utils.py` 第 394–398 行，`VLN_VLLM_ENFORCE_EAGER` 默认关闭）。这些都不是相机抖动。

## 3. Pointing

词表在 `src/lightnav/vln_utils.py` 第 170–185 行。`apos` 是 48×27 的格子，1–1296 为像素格，1297 左转、1298 右转、1299 停止。`opos` 只有 0–1296，0 表示目标不可见。格子相对画面，不使用相机内参。

解析在 `src/lightnav/serving/protocol.py` 的 `pointing_payload`（第 133–212 行）。输出是客户帧上的像素、`apos_state` / `opos_state`，以及是否夹在边缘。第 156–158 行写明：`apos` 的 stop 不等于响应顶层的 stop。顶层 stop 来自解码后的航点是否接近零。

控制链不读 Pointing。`policy.act` 只把 RVQ 解出的航点交给 `first_waypoint_to_velocity_cmd`。Pointing 出现在可视化 `runner.py` 第 457–466 行，以及诊断日志。

诊断里的「Pointing 与航向是否一致」是事后规则，不是模型的一部分。`experiments/r2r_diagnosis/analyze_trace.py` 第 63–95 行：把 `apos` 像素按 120° 水平视场折成方位角，与指令航向差在 30° 以内算一致；`rot_left` / `rot_right` 只看航向符号。

## 4. RVQ、航点和速度

检查点动作头是 RVQ。`checkpoints/LightNav-0/action_tokenizer/manifest.json`：3 层，每层 256，`horizon: 10`，`representation: se2_diff`。停止码是 `[6, 122, 174]`（`stop.l0 = 6`）。

解码在 `src/lightnav/traj_vocab.py`：

- `RVQBundle.is_stop` 第 189–196 行：只认 level-0 码等于 `stop_l0`，不用「接近零」来认停止码。
- `decode_waypoints` 第 198–212 行：各层码字相加，除以 Jacobian 权重，再按 SE(2) 累加成机体坐标系绝对航点。
- `TrackingAgent.decode_waypoints`（`src/lightnav/tracking.py` 第 124–141 行）在 `is_stop` 或航点整体接近零（`_RVQ_STOP_ATOL = 5e-3`，第 39 行）时改成精确零航点。

航点一行是 `[forward_m, lateral_m, yaw_rad]`，侧向左为正，航向逆时针为正。`src/lightnav/velocity.py` 第 19–56 行的 `first_waypoint_to_velocity_cmd` 丢掉侧向，取 `v = forward / dt`、`w = degrees(yaw) / dt`，再归一化到 Habitat 的 `[-1, 1]`。Go2-W 和官方 yaml 的速度范围都是线速度 `[0, 2.5]` m/s、角速度 `[-300, 300]` deg/s、`time_step: 0.1`，所以一步最多约 0.25 m、30°。负的前进量会被线速度下界截成 0。

`select_action_waypoint`（`policy.py` 第 41–47 行）跳过开头纯零的行，用第一个前进或航向非零的航点。

## 5. 指令和历史

每一步把整段指令原样放进提示，没有子目标下标，没有已完成地标。

提示模板 `src/lightnav/prompts.py` 第 84–90 行 `UNIFIED_TRAJ_PROMPT_TEMPLATE`。RVQ 只改最后一句，第 100–126 行 `to_rvq_prompt`。检查点 `prompt_style` 是 `unified_traj`（`eval_config.json` 第 64 行）。

历史是画面，不是语言状态。`NavigationPolicy.observe`（`src/lightnav/inference/policies.py` 第 106–145 行）把帧放进缓冲。SlowFast 打开时保留整段，不按 64 帧截断。采样在 `src/lightnav/slowfast.py`，推理侧由 `src/lightnav/inference/samples.py` 第 29–68 行的 `_build_slowfast_eval_sample` 调用，保证和训练同一套分段。检查点的层是 current / fast / mid / long / anchor（`eval_config.json` 第 24–63 行）。`num_history_frames` 记的是 64，SlowFast 实际用的是更长的绝对帧号。

`vlnce.py` 第 349 行附近会算测地进度，它留在环境信息里，不进入模型提示。

## 6. 停止和成功

成功是 Habitat 的定义：调用了停止，并且 `DistanceToGoal < success_distance`。本环境半径 3.0 m（`vlnce.py` 第 144 行）。`vlnce.py` 第 450–451 行写明不覆盖 `info["success"]`。

速度控制下，线速度和角速度都低于 `min_abs_lin_speed: 0.01`、`min_abs_ang_speed: 0.5` 时 Habitat 把该步当成停止（yaml 第 36–37 行；`vlnce.py` 第 426–434 行）。模型侧对应零航点，也就是 RVQ 停止码或近零解码。

全量官方和 Go2-W 的 `termination_reason` 都是 `agent_stop`，1839/1839。没有步数截断把成功清零的回合。

## 7. 日志里有什么

全量评测 `run_habitat_eval`（`runner.py` 第 715–734 行）每个回合写一行：成功、SPL、nDTW、步数、最终距离、最近距离、指令、停止原因。不写逐步航点、Pointing、位姿。`output/r2r_official/results.jsonl` 和 `output/r2r_go2w/results.jsonl` 都是这个粒度。只有打开 `--save_video` 或 `record_dir` 时，`runner.py` 第 470–484 行才逐帧记录。

2026-10-20 的诊断回放补上了逐步字段。文件是 `output/r2r_probe/trace_20261020/steps.jsonl`，3259 步、30 个回合。每步有 `raw_text`、`rvq_codes`、`waypoints_robot`、`apos_id`、`apos_state`、`apos_px`、`opos_*`、执行前后 `pose_xyz` / `pose_yaw`、测地距离、专家航向、`yaw_follow_rad`、碰撞、`infer_s`、`history_frames`、`prompt_tokens`，以及 JPEG 路径。客户端在 `experiments/r2r_diagnosis/probe_client.py` 第 160–161 行用 imageio 存 RGB JPEG。

这 30 条只有一次闭环。没有第二次同输入、也没有相机扰动对照，所以不能从这份逐步日志里读出「第一次分歧发生在第几步」。

## 8. 失败分类

分类函数在 `experiments/r2r_diagnosis/analyze_failures.py` 第 45–62 行。距离都是测地距离。无穷距离单独成 `unreachable`。其余用最近距离、最终距离和起点距离：

| 标签 | 条件 |
|---|---|
| `left_after_1m` / `left_after_3m` | 最近距离 ≤ 3 m，最终距离 > 3 m；最近距离 ≤ 1 m 时用前一个名字 |
| `stop_just_outside` | 最终距离比最近距离高出不超过 0.5 m，且最近距离 ≤ 5 m |
| `stop_far_closest` | 同样几乎停在最近点，最近距离 > 5 m |
| `wrong_way` | 从起点到最近点的进展 < 1 m |
| `big_drift` | 离开最近点 ≥ 3 m |
| `mild_drift` | 其余离开 |

成功回合如果直接套这个函数，会因为「停在最近点附近」被标成 `stop_just_outside`。统计失败时要先去掉成功回合。走反、走偏、进圈后离开、停在圈外，都是这条规则，不是模型自己的类别。

## 9. 全量数字，供和同事笔记对照

本地重算 `output/r2r_official/summary.json` 与 jsonl：1839 回合，SR 67.70%，SPL 0.6238，nDTW 0.6334，有限 NE 3.999 m，测地无穷 0。

`output/r2r_go2w/results.jsonl`：同一 1839 个 `(scene, episode)`，SR 61.39%，SPL 0.6069，nDTW 0.6198，测地无穷 90。官方成功而 Go2-W 失败 270 条，反过来 154 条。这是机身差异，不是 ±1° 抖动。

失败且按上面的规则分类（Go2-W，710 条失败）：走反 194，轻度偏离 98，走近后大偏离 83，停在 3–5 m 80，停在更远 74，进 3 m 后离开且最近距离 > 1 m 66，进 1 m 后离开 25，测地无穷 90。特权最短路径和特权 STOP 只在 30 条诊断集上跑过，见 `experiments/r2r_diagnosis/probe_results_audit.md`，不能写成可部署 SR。

同事笔记的 67.54 / 0.621 接近本地官方数，离 Go2-W 的 61.39 更远。那次抖动实验更像标准机身评测，不是这次 Go2-W 评测。两边的逐回合文件没有对齐过。
