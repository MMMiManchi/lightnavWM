# 世界模型数据契约

日期：2026-10-23。契约版本 `wm-transition/0.1`，与采集器写入的 JSON 一致。核对对象是已验收运行 `output/world_model_data/pilot_20261023_train3_r2`（3 条官方 R2R VLN-CE train episode、2 个场景、60 条 `valid` transition、63 帧 RGB、63 帧 depth，`validation_report.json` 的 `hard_error_count` 为 0）。字段以该目录中的记录和 `experiments/world_model/` 的写入代码为准。

早先的提案文本曾写“契约尚未实现”。那句话已过时。下面只保留这次读到的字段。提案里有、文件里没有的项，记为未实现，不补造。

## 1. 一次运行的文件

| 文件 | 已验收运行中的作用 |
|---|---|
| `manifest.json` | 数据源、allowlist、served、reset、seed、配置和代码指纹 |
| `episodes.jsonl` | 一条 episode 一行 |
| `observations.jsonl` | reset 和每次 step 之后的一帧一行 |
| `transitions.jsonl` | 一次已确认的环境 step 一行 |
| `privileged_labels.jsonl` | 离线位姿、目标、测地距离、reference path、评测指标 |
| `assets/*.npy` | RGB `uint8 (270,480,3)` 与 depth `float32 (270,480)` |
| `validation_report.json` | 校验器计数和硬错误 |
| `habitat_server.log`、`habitat.ready` | 服务端日志和就绪标记 |

`output/world_model_data/pilot_20261023_train3` 含 `failure.json`，没有通过校验，不得进入训练集。`output/r2r_probe/` 与 `output/go2w_expert/` 存在，来自 val_unseen 诊断，不得用于训练、归一化统计或模型选择。名为 `instruction_triplet_20261021` 的目录在仓库根下未找到；加载器仍按该目录名拒绝。

## 2. 四类信息

一条 transition 把四类信息分开：

1. **模型输入。** 时刻 t 的 RGB 路径、可选的指令原文、以及已经提交给 Habitat 的物理控制量。v1 预测器不读 depth，不读指令编码器，因为本地还没有核定过的冻结文本编码器。
2. **离线标签。** 由特权位姿算出的机体坐标增量 `(forward_m, left_m, delta_yaw_rad)`，以及下一帧 RGB 路径。下一帧 RGB 只供冻结编码器产生 `z_t1`，不能进入 t 时刻的导航输入。
3. **动作记录。** `prediction` 是 LightNav 的文本、RVQ 和航点。`client_request` 是客户端请求。`execution_ack.final_habitat_action` 才是提交给底层 step 的命令。三者不相等时，训练动作只取最后者。
4. **特权标签。** `goal_position`、`geodesic_distance_m`、`source_geodesic_m`、`reference_path`、`metrics` 里的 success/spl/ndtw/oracle_success，以及世界坐标位姿。只出现在 `privileged_labels.jsonl`。

在线加载器不得打开 `privileged_labels.jsonl`。`experiments/world_model/online_input.py` 的白名单只有指令、当时已经存在的观测 id，以及 `habitat_time_step`、`lin_vel_range`、`ang_vel_range`。

## 3. 图像如何和 transition 关联

`transitions.jsonl` 不内嵌像素。它用 `obs_before_id`、`obs_after_id` 指向 `observations.jsonl`。观测记录里的 `rgb.path`、`depth.path` 是相对运行目录的 `.npy` 路径，并带 sha256、shape、dtype。

时间关系看编号，不看文件名是否相邻：

- 观测的 `step_index`：reset 为 0，第 k 次成功 step 之后的帧为 k。
- transition 的 `t` 等于动作前的 `step_index`，动作后的观测是 `t+1`。
- 已验收运行中，60 条 transition 都有前后 RGB 文件。
- `terminal_observation` 标在预算或终止 step 返回的那一帧上。该运行有 3 帧，`step_index` 都是 20。

## 4. 动作语义

LightNav `generate_from_frames` 返回 `(text, latency)`。文本里的三个 RVQ token 解码成一条 `(10, 3)` 局部轨迹，单位是米、米、弧度，顺序为 `[forward, left, yaw]`。`select_action_waypoint` 只执行第一个可用航点。十个航点不是十个候选动作，也不是十次已经执行的 step。`waypoint_times` 在 60 条记录里都是 null，原因是解码航点没有物理时间轴。不得把 `k * 0.1` 填进去。

`src/lightnav/velocity.py` 把该航点变成归一化速度：`v = forward / dt`（m/s），`w = degrees(yaw) / dt`（deg/s），再线性映到 `[-1, 1]`。横向分量不进入命令，记在 `lateral_m_not_in_command`。Go2-W 的范围是线速度 `[0, 2.5]` m/s、角速度 `[-300, 300]` deg/s、`dt = 0.1` s。归一化线速度 `-1` 是 0 m/s，`0` 不是停止。

Habitat 在 `habitat/tasks/nav/nav.py` 的 `VelocityControlAction.step` 做反向缩放。线速度和角速度的绝对值都低于 `min_abs_lin_speed`（0.01 m/s）和 `min_abs_ang_speed`（0.5 deg/s）时，它设置 `is_stop_called` 并在积分之前返回。否则它以 `dt_command_s` 积分，再经 navmesh 的 `try_step`。`allow_sliding` 在采集配置里为 true。

因此 v1 的执行动作 `u_t` 定义为：

- 来源字段：`action.execution_ack.final_habitat_action`
- 线速度、角速度用记录里的 `control_units` 反归一化
- 时间用 `dt_command_s`，本运行中与 `control_units.control_dt_s` 同为 0.1 s
- `dt_sim_s` 全是 null。Habitat 的 `_elapsed_seconds` 使用 `time.time()`，不是模拟时钟
- 低于停止阈值时，`u_t` 的速度记 0，不再假装发生了积分

`server_override` 在这 60 条里都是 false。字段仍然必须保留。若为 true，训练继续使用最终命令，并保留 `override_reason`，不得改回客户端请求。

## 5. 位姿标签

特权位姿含 `world_position_xyz_m`（米）、`world_rotation_xyzw`，以及 `derived_yaw_rad`。这 63 条位姿的 `yaw_method` 都是 `habitat_sim.quat_rotate_vector`，与 `wm_quat_v1` 的交叉校验最大差为 `1.84e-7` rad。

机体增量由动作前的航向定义，实现在 `experiments/world_model/pose_math.py` 的 `body_frame_delta`：

- yaw 0 面向世界 -Z，正 yaw 和正 left 为左转
- `forward_m`、`left_m` 只使用水平位移
- `delta_yaw_rad` 用 `wrap_pi`
- `up_m` 是世界 Y 的变化。本运行中其绝对值不超过 `1.7e-6` m，不作为 v1 目标

在这 60 条上，用最终命令做一阶积分 `forward = v * dt`、`left = 0`、`yaw = w * dt`，与测得航向的绝对误差最大为 `7.7e-8` rad。前向绝对误差平均 `1.1e-3` m，最大 `4.6e-2` m。横向绝对误差最大 `5.6e-2` m。航向闭合说明角度单位和 `dt_command_s` 与记录一致。前向和横向残差的来源尚未分解到碰撞或 navmesh；碰撞字段全部是 `unknown` / `not_read`，不能把残差标成碰撞。

## 6. 终止语义

`outcome.primary_end_source` 按 `wm-end-priority/0.1` 从这些来源里取第一个：`transport_error`、`server_early_stop`、`runner_budget`、`decode_fallback`、`model_stop`、`habitat_stop`、`habitat_limit`、`manual_abort`。都没有时记 `unknown`。这三种含义必须分开，不能互相填补：

| 含义 | 目前用什么字段表示 | 不能把它当成 |
|---|---|---|
| 已确认没有终止事件 | 还没有单独字段。`terminated=false` 只说明已检查的停止标志为假 | `primary_end_source=unknown` |
| 终止来源未知 | `primary_end_source=unknown` | 停止预测的负样本 |
| 评测或采集预算截断 | `primary_end_source=runner_budget`，episode 的 `completion.state=budget_truncated` | 模型主动 STOP |

本运行的 60 条：

| 计数 | 记录 |
|---|---|
| 57 | `primary_end_source=unknown`，且 `terminated=false`、`truncated=false`。来源未知，不是“已确认没有终止事件” |
| 3 | `primary_end_source=runner_budget`。同时带有 `habitat_stop` 和 `habitat_limit`，因为采集器在第 20 步发送了零速度命令，Habitat 也把它当成停止 |
| 3 | `prediction.inference_called=false`，`model_stop=null`，`raw_text=null`。预算步没有调用模型 |
| 0 | `model_stop=true` |

没有模型 STOP 正样本，所以不训练终止预测头。预算截断不得写成模型停止决策。模型没有做出的决定，不能由数据处理脚本补成决定。

## 7. 缺失值

| 现象 | 本运行中的记录方式 |
|---|---|
| RGB 缺失 | `rgb_missing=true`，不保存全零替代图。本运行 63 帧都没有缺失 |
| depth 缺失 | 同样不保存全零图。本运行没有缺失 |
| 模拟时间 | `sim_time_s=null`，`dt_sim_s=null`，并写原因 |
| 航点时间 | `waypoint_times=null`，并写原因 |
| 碰撞 | `state=unknown`，`source=not_read` |
| step 异常 | 不写 `obs_after_id`，`execution_ack.status=error`，质量为 `quarantined`。本运行没有这种行 |
| seed | manifest 的 `seed.status` 为 `partial`。`applied_in_reset` 在三条 episode 上都是 false |

depth 的数值不是米。解析后的传感器配置是 `normalize_depth=true`、`min_depth=0`、`max_depth=10`、`hfov=120`。Habitat 在 `habitat/sims/habitat_simulator/habitat_simulator.py` 中把它写成 `(depth_m - min) / (max - min)`。63 个保存数组的取值都在 `[0, 1]`，shape 被 `BaseHabitatEnv._extract_depth` 收成 `(270, 480)`。若要米，只能按这个已记录的配置反算。v1 模型不把 depth 当作输入。

## 8. 划分字段

每条 episode 有 `scene`、`episode_id`、`route_id_native`、`route_group_id`、`route_geometry_hash`、`partition`、`instruction_id_native`、`instruction_id_derived`。本运行里 `instruction_id_native` 为 null。两条 `7y3sRwLe3Va` episode 的 `route_group_id` 都是 `7y3sRwLe3Va:4`，必须留在同一侧。`partition` 的记录值全是 `train_fit`，它来自路线组哈希，不是按场景切分。视觉模型使用另一条预注册规则，见设计文档，不回写这些 JSON。

## 9. 禁止泄漏的字段

模型输入字典只允许 `rgb_t`、`rgb_t1`、`instruction_text`、`executed_twist`。下列内容不得进入该字典，也不得用于拟合归一化统计：

- 目标位置、测地距离、reference path、success、spl、ndtw、oracle_success
- 世界坐标位置和旋转
- 解码航点、选中航点、RVQ、客户端请求
- depth
- 任何 `val_seen`、`val_unseen`、`test` 记录
- 中断运行 `pilot_20261023_train3`

`rgb_t1` 只用于生成监督目标 `z_t1`。它不是 t 时刻决策可见的图像。
