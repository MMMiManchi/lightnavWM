# 世界模型正式采集前协议

> 2026-10-28 实现状态覆盖说明：仿真时钟已从拟议/保持 null 更新为 wm-transition/0.2。WorldModelVLNCEEnv.step 在 Habitat Env.step 前后读取同一 Simulator.get_world_time()，采集器把观测时间和 transition 时间差写入记录。0.0 是有效测量；不可读或倒退才是 null，并使该 transition 不具备训练资格。以下原始 2026-10-27 盘点中关于仍为 null、未来 schema 的段落是当时状态，不是当前行为。旧 0.1 日志不会自动回填。

日期：2026-10-27。这是采集前的设计标准，不是采集许可。

当前状态：2026-10-27 的用户指示已经接受本协议正文，并整组接受 fit/dev 报告门槛。`HUMAN_ACCEPTED` 和 `SCALE_ACCEPTED` 为 true，历史决定来源在 `experiments/world_model/records/human_decisions_20261027.json`。另有 `records/collection_authorization.json` 明确授权 `train_fit` 正式数据采集，`COLLECTION_AUTHORIZED` 为 true。`train_fit` 以外的正式采集未授权；该授权也不包含特征提取、训练或正式评测。本协议不会自动启动任何运行。

现行 transition schema 为 `wm-transition/0.2`，批准 schema 仍为 `wm-approval/0.1`。`feature_layer_selected` 保持 null。现有 pilot 不进入训练集。

机器可读的门槛和静态检查在 `precollection_protocol.py`。检查只读仓库文件，不构造模拟器，不签发批准。

## 1. 样本契约

一条可训练的单步样本是同一个 `transition_id` 上的五件事。像素、位姿和可用的 Habitat 仿真时间按这个关系对齐；潜在特征仍未正式提取。

| 符号 | 今天的记录 | 不能换成 |
|---|---|---|
| `o_t` | `obs_before_id` 指向的观测。`step_index == t`。RGB 是该观测的 `.npy` 字节，sha256 与记录一致 | 文件名相邻的另一张图 |
| `a_t` | `action.execution_ack.final_habitat_action`，经 `executed_twist` 反归一化。来源必须是 `final_habitat_action` | `client_request`、解码航点、RVQ 文本 |
| `o_{t+1}` | `obs_after_id`。同一 `episode_uid`，`step_index == t+1` | t 时刻决策可见的输入 |
| 位姿变化 | `body_frame_delta`：动作前机体坐标下的 `(forward_m, left_m, delta_yaw_rad)`。`up_m` 不是 v1 目标 | 命令积分 `v * dt_command_s` |
| 仿真时间 | action 两端点与前后 observation 的 `sim_time_s` 一致，`dt_sim_s` 等于观测窗口差；实测零可写 `0.0` | `dt_command_s`、`dt_wall_s`、`arrival_monotonic_ns`、`k * 0.1` |

`a_t` 的控制量是 `(v_mps, w_rad_s, dt_command_s)`。第三项是已提交速度命令的积分时域，也就是 `VelocityControlAction.step` 传给 `integrate_transform` 的 `time_step`。`ResidualLatentPredictor` 现在读的就是这三项。时钟已经测过，采用哪一种时间仍未决定，所以第三项保持 `dt_command_s`。

位姿标签和命令积分是同一条 transition 上的两个量。位姿来自动作前后的特权位姿。命令积分是基线 B，左向固定为 0，低于停止阈值时增量为 0。二者之差不是碰撞标签。碰撞记录仍是 `unknown` / `not_read`。

`z_t` 和 `z_{t+1}` 不写进 transition。它们将来是 `wm-feature-cache/0.2` 里、同一 `output_name` 下、分别绑定 `obs_before_id` 和 `obs_after_id` 的两行。两行的编码器 revision、权重 sha256 和预处理 sha256 必须相同。特征行的 `image_sha256` 必须等于对应观测记录的 `rgb.sha256`。两行的 `source_run`、`scene`、`route_group_id` 和 `scene_partition` 必须相同，并且 `scene` 与 `route_group_id` 必须等于这条 transition 所属 episode 记录。`scene_partition` 必须等于该场景 id 的登记划分，不能改成路线组上的 `partition`。观测上的 `sim_time_s` 是采集服务读到的该帧时间，不要求为 null。`rgb_t1` 只用来产生 `z_{t+1}`，不能进入 t 时刻的模型输入。

`feature_layer_selected` 继续为 null。`pooler_output` 和 `last_hidden_state` 都只是候选。缓存行用 `output_name` 标明自己是哪一个。选定某一层是以后的实验决定，不是采集字段。诊断 probe 的 `diagnostic_only` 输出不能充当这两行。

目前没有正式特征缓存，所以成对潜在样本数是 0。没有这一对，单步潜在预测没有监督目标。

### 仿真时间的定义

三条时钟不能混用。现行实现从同一个 Habitat simulator 的 `get_world_time()` 读取动作前后观测端点，`dt_sim_s` 是两个观测时间之差；零差值保留为有效的 `0.0`。读数不可用、非有限或倒退时才写 null，此类行不能通过训练准入。

历史测量说明：

1. `habitat.core.env.Env._elapsed_seconds` 是 `time.time() - episode_start`。这是墙钟。`DT_SIM_REASON` 已经写明这一点。它不能写入 `dt_sim_s`。
2. 配置里的 `time_step: 0.1` 是命令积分时域。服务器只在最终动作是 `velocity_control` 且已经解析出这个时域时填写 `dt_command_s`。把它抄进 `dt_sim_s` 仍然是命令时间。
3. `habitat_sim.Simulator.get_world_time` 在 habitat-sim 0.3.1 上存在。`VelocityControlAction.step` 只做 `integrate_transform(time_step)`、navmesh `try_step` 和 `set_agent_state`，不调用 `step_world`。采集器走的是 `habitat.Env.step`：`VLNTask` 没有覆盖 `step`，所以进入 `EmbodiedTask.step`。动作返回之后、读出下一步观测之前，它调用 `_sim.step_physics(1 / physics_target_sps)`。`step_physics` 就是 `step_world`。仓库配置没有改 `physics_target_sps`，默认值是 60。`VelocityControlAction` 返回 None，随后 `HabitatSim.step(None)` 只调用 `get_sensor_observations`，不再次 `step_world`。因此动作前观测和动作后观测之间隔着一次 `step_physics(1/60)`，不是隔着命令时域 0.1 秒。隔离探针没有创建 `habitat.Env`、没有 episode、没有传感器，所以没有像素。它按这个顺序调用了同一个 `step_physics`：`try_step` 加 `set_state` 位移 0.3904685378074646 米时时钟仍是 0；`step_physics(1/60)` 之后时钟是 0.016 秒；再调用一次是 0.032 秒。物理子步是 0.008 秒，请求的 1/60 秒不会原样返回。更早一次 `step_world(0.1)` 得到的是 0.096 秒，那不是这条路径上的调用。2026-10-27 的决定选择方案 3，测量见 `docs/world_model_scale_decision.md`。当前实现记录动作前后观测时钟，不把该次隔离测量的 0.016 秒硬编码为每步时长。

本节描述的 `wm-transition/0.1` 规则是历史规则：当时字段必须为 null，校验器把非空值记为 `dt_sim_forged`。当前 `wm-transition/0.2` 已替换该规则；校验器检查同一 simulator 时钟的端点和差值一致性。读数不可用或倒退时保留 null 并隔离样本。当前 schema 不分配版本号；schema 常量为 `wm-transition/0.2`。

## 2. 数据质量门槛

文件合法和样本够训练是两道门。

文件门已经在加载器里。一条 transition 要标 `valid`，必须同时有：执行确认 `status=executed` 且 `executed` 为 true、确认里的 `action_id` 与动作一致、最终 Habitat 动作、前后观测、步号相差 1、图像哈希相符、前后位姿且 `observation_id` 与对应观测一致。`quality` 不是 `valid` 的行拒绝整个 run，不能跳过坏行后留下其余行。`failure.json`、`diagnostic_only`、以及 manifest 里的 `training_use` 字符串，都不能授予训练资格。多个增量 run 应先各自批准，再由 `load_accepted_run_set` 合并校验；拟合数据只能通过 `admitted_training_rows` 取出，开发验证数据只能通过 `admitted_validation_rows` 取出。`read_recorded_run` 返回路径，不授予准入。

历史 `wm-transition/0.1` 允许时间字段为空；在现行 `wm-transition/0.2` 中，`dt_sim_s` 和观测 `sim_time_s` 必须是可验证的有限值才能进入训练。其他字段的空值仍按各自准入规则处理。预算步的 `model_stop` 必须是 null，`inference_called` 为 false。不得把模型没做的决定补成决定。

轨迹完整和预算截断用现有字段区分，不新增标签：

| 记录 | 含义 | 不能当成 |
|---|---|---|
| `completion.state=budget_truncated`，且至少一步的 `primary_end_source=runner_budget` | 采集器步数预算发出了停止 | 完整轨迹，也不是模型 STOP |
| `completion.state=completed`，且没有任何一步的 `primary_end_source=runner_budget` | 采集循环不是因为 runner 预算而结束 | 导航成功。`success` / `spl` 仍是特权字段 |
| `primary_end_source=unknown` | 终止来源没有单独记录 | “已经确认没有终止事件” |
| `completion.state=infrastructure_error`，或 `quality=quarantined` / `partial` | 失败样本 | 正式训练行 |

`completed` 这个词是采集器的剩余标签。Habitat 自己的步数上限若在非预算步把 episode 截断，现有代码也会写成 `completed`，同时 `end_sources` 里应保留 `habitat_limit`。非预算配额不擦掉这个来源，也不把该 episode 算作到达目标。

`runner_budget` 与 `completion.state=completed` 同时出现是硬错误 `budget_labeled_complete`。

失败样本的规则：

- RGB 或 depth 缺失时不写全零替代图。缺 RGB 的起始帧使该 episode 成为 `infrastructure_error`，不产生可训练 transition。
- step 抛错时 `obs_after_id` 为空，`execution_ack.executed` 为 false，质量为 `quarantined`。整个 run 不能进入 `admitted_training_rows`。
- 动作确认缺失、超时或传输失败同样不是 `executed`。`valid` 要求确认已执行。
- 部分行有效不能让 run 获得训练资格。

来自预算截断 episode 的、本身为 `valid` 的单步，只在第 1 节的仿真时间和成对特征都存在之后，才可以计入按 `action_kind` 分组的单步配额。在那之前，行数再多也不满足训练门。这些单步永远不计入非预算 episode 配额。

## 3. 数据规模与覆盖

三层数字不要叠成一个“够用”：

| 层 | 数字 | 作用 |
|---|---|---|
| 加载资格 | `train_fit` 至少 8 个场景、20 个路线组；`train_dev` 至少 3 个场景、5 个路线组。两侧场景和路线组不相交 | 现有 `require_formal_split`。达不到就拒绝加载。这不是泛化保证 |
| 清单容量 | `train_fit` 53 场景、2977 路线组、8938 episode；`train_dev` 8 场景、626 路线组、1881 episode。`collection_status` 为 `not_collected` | 可采集的上界。不是本次要采完的定额 |
| 训练报告门槛 | 见下面两组待人审数字 | 只为让 E2/E3 能按场景和 `action_kind` 出表。不是功效分析，也没有人接受 |

划分继续用 `wm-scene-split/0.1` 的场景哈希。一个场景的全部路线留在同一侧。不按帧、不按相邻 step 划分。`val_seen`、`val_unseen`、`test` 和 pilot 不进入训练、归一化或模型选择。

`assert_collectable` 只接受清单里的 `train_fit` 路线，遇到 `train_dev` 场景直接拒绝。`dev_collect.py` 目前只有分区计划/拒绝逻辑，不会构造 Habitat 环境；要采集开发集，仍需实现实际入口并获得单独授权。本协议不放宽现有门。

待人审的 `train_fit` 下限，且只有在仿真时间和成对特征都存在之后才开始计数。`scale_quota_progress` 对当前行一律返回 0；`dt_sim_s` 为空或被填成非空，都不会增加这些数字。

- 至少 8 个事先点名的 `train_fit` 场景，至少 20 个路线组。
- 每个被采集的场景至少 2 条 episode。
- 三种 `action_kind`（`stop_threshold`、`turning`、`mostly_forward`）各自至少 20 条 `valid` 单步。
- 至少 20 条非预算 episode：`completion.state` 为 `completed`，且该 episode 没有 `runner_budget`。

待人审的 `train_dev` 下限，同样等待独立采集入口和训练门：

- 使用清单里预注册的全部 8 个 `train_dev` 场景，不在看过 fit 误差之后再挑子集。
- 路线组至少达到加载资格的 5 个。不必为了起步而采完 1881 条 episode。
- 每种 `action_kind` 至少 10 条 `valid` 单步。
- 至少 5 条非预算 episode。

这些数字写在 `precollection_protocol.py`。`HUMAN_ACCEPTED` 和 `SCALE_ACCEPTED` 已分别接受，而且互不代替。`COLLECTION_AUTHORIZED` 当前只授权 `train_fit` 数据采集。合格的正式 transition 数是 0，成对潜在样本数是 0；仿真时间与成对特征仍未满足，故采集覆盖不能计作训练配额。`technical_trial_ready` 只读现场清单，调用方传入的报告不能改变它。通过清单不授予特征提取、训练、dev 采集或评测权限。

## 4. 采集前检查清单

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q tests/world_model/test_precollection_protocol.py
```

通过表示静态事实与本文件一致。`collect.py` 和 `serve_collect.py` 在创建环境之前调用 `assert_collection_not_blocked()`；正式采集还需通过 train_fit 场景门和 campaign 参数校验。`--dry-run` 不创建环境，也不表示已经采集。

| 检查 | 期望 |
|---|---|
| schema | `wm-transition/0.2` 与 `wm-approval/0.1` |
| 采集开关 | `COLLECTION_AUTHORIZED` 为 true，授权范围仅为 `train_fit` |
| 仿真时间写入 | 服务端实测值进入 action 和 observation；`0.0` 是有效值。不可读或倒退才是 null；墙钟仍单独记录 |
| 仿真时钟语义和时间字段 | `pass` 表示 action 前后端点、观测时钟及差值通过同一 simulator 时钟一致性校验；隔离探针不是固定步长 |
| 协议人工接受 | `pass`。`HUMAN_ACCEPTED` 为 true，来源是 2026-10-27 的决定记录 |
| 规模与 fit 覆盖 | `pass`。`SCALE_ACCEPTED` 为 true，整组接受。这不是清单容量，也不是加载资格 |
| 开发集采集路径 | `pass` 表示入口只计划 `train_dev` 且不构造环境。执行采集仍未授权 |
| 正式采集授权 | `pass` 仅表示独立 `train_fit` 授权记录与代码开关一致；不授权 dev、特征、训练或评测 |

正式采集硬预算：每个 episode 最多 200 步；单次 collector run 墙钟默认最多 7200 秒，`--wall-clock-budget-s` 只能设得更短。受控诊断试采仍最多 1200 秒。超过预算会终止当前 run，不自动重试或扩大 allowlist。每个正式 run 必须提供稳定的 `--campaign-id` 与 `--campaign-manifest`；后续追加 run 到同一 manifest，通过 `python -m experiments.world_model.campaign --manifest <campaign_manifest.json>` 重新校验并汇总跨 run 覆盖。campaign 报告只统计采集覆盖，`training_admitted` 恒为 false。
| 仿真时间一致性 | 校验器比较 action 时钟端点、前后 observation 时钟及 `dt_sim_s`；非有限、缺失或不一致的值不能进入训练 |
| 预算冒充完整 | 校验器含 `budget_labeled_complete` |
| 开发场景 | `collection_gate.py` 对 `train_dev` 拒绝采集。行为测试仍是 `tests/world_model/test_collection_gate.py`，本清单不重跑官方 split |
| 清单容量 | 第 3 节的六个计数，且 `collection_status=not_collected` |
| 成对特征接口 | 用 `final_habitat_action` 对齐同一个 `output_name` 的前后观测，场景和路线等于所属 episode，且 `PAIRS_AUTHORIZE_TRAINING` 为 false |
| 正式特征缓存 | `blocked`，且不是采集启动的必要项。采集先于缓存；没有正式缓存时提取入口仍拒绝。协议检查通过不会把它改成 `pass` |
| 加载资格 | 与 `latent_data.py` 的四个下限一致 |
| 总开关 | `checklist_blocks_collection` 为 false；这仅反映当前 train_fit 采集准入，不会自动启动运行 |

目录 `fsync` 的运维语义保持不变：`renameat2(RENAME_NOREPLACE)` 成功之后，目录同步失败不回滚已经发布的完整批准，重试也不会覆盖。这不是半成品批准。绑定挂载仍不能由现有路径检查识别，本协议不增加挂载命名空间要求，也不声称该风险已经消除。同进程 `object.__setattr__` 可以改掉封存字节，这是同一进程内的恶意代码边界，不是公开写入接口的失败。本协议不为此增加进程隔离。

## 5. 历史 schema 变更记录

本节保留 2026-10-27、`wm-transition/0.1` 时的历史约束，不是当前实现要求。当前 schema 不分配版本号，版本常量保持 `wm-transition/0.2`。

当时 `wm-transition/0.1` 已有 `dt_sim_s` 键，但合法值只有 null，非空值会被记为 `dt_sim_forged`。旧提案曾把实测零差标为 `clock_unchanged` 并置 null；这不适用于现行版本。当时提案要求升级 schema 后，服务端才能写入观测窗口时钟。该升级已在 2026-10-28 落地为 `wm-transition/0.2`：同一服务端 step 内记录动作前后 simulator 时钟和差值，并与两帧 observation 对齐。实测零差值保留 `0.0`；时钟不可读或倒退才为 null。

其余约束不变：`dt_command_s`、墙钟或 `k * 0.1` 不能替代实测时钟；位姿标签来自特权位姿机体增量；`z_t` 与 `z_{t+1}` 留在独立的 `wm-feature-cache/0.2`，`feature_layer_selected` 仍为 null。
