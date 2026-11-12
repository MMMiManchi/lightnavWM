# Habitat 仿真时间记录实现

日期：2026-10-28。本文原为仿真时间 schema 提案；实现已落地，现行版本为 `wm-transition/0.2`。本文件保留原测量依据并说明当前字段，不再代表待批准的下一版方案。

## 当前实现

`WorldModelVLNCEEnv.step` 在 Habitat `Env.step` 前后，对同一 Habitat simulator 调用 `get_world_time()`。动作后时钟在 Habitat 原有 physics step 和传感器观测完成后读取，不额外推进模拟器。服务端将端点和 `dt_sim_s = after - before` 写入 action 记录；reset 与后续观测的时钟写入 observation 记录。校验器验证 action 端点与前后观测时钟一致。

- 有限、非倒退的实测差值（包括 `0.0`）保留为数值；零不是缺失。
- 时钟不可读、非有限或倒退时记录 null 和原因。此类 transition 不满足带时钟训练准入。
- `dt_command_s` 是动作积分时域，`dt_wall_s` 是墙钟耗时；两者不能替代 `dt_sim_s`。
- 旧 `wm-transition/0.1` 记录不回填、不自动晋升。当前 schema 字符串仍是 `wm-transition/0.2`，没有再分配版本。

## 测量依据

决定记录选择方案 3：动作前观测和动作后观测各自的 `get_world_time`。

2026-10-27 的隔离测量使用 habitat-sim 0.3.1 和真实的 `habitat.tasks.nav.nav.VelocityAction.step`。没有 `habitat.Env`、episode、传感器，也没有写 transition。动作参数是 `linear_velocity=1.0`、`angular_velocity=0.0`、`time_step=0.1`。

| 读数 | 值 |
|---|---|
| 命令时域 `dt_command_s` | 0.1 秒；控制命令积分参数，不是世界时间 |
| 动作前世界时间 | 0 |
| `VelocityAction.step` 返回后 | 0 |
| 动作本身世界时间差 | 实测 0 秒；0 是读数，不是缺失 |
| 水平位移 | 0.25 米 |
| 动作返回值 | null |
| 随后的 `step_physics` | 1 次，即 `EmbodiedTask.step` 原有调用 |
| `step_physics` 请求时长 | `1/60` 秒（0.016666666666666666 秒） |
| 物理子步 | 0.008 秒 |
| 该物理步后世界时间 | 0.016 秒 |
| 该物理步测得的时钟增量 | 0.016 秒，不等于请求值 `1/60` |
| 方案 3 的观测窗口差 | 0.016 秒 |
| 额外物理步 | 0 |

动作改变了位置，但动作路径本身没有推进世界时间。0.016 秒来自任务在读取动作后观测之前原有的一次 `step_physics`。该隔离测量没有运行 `habitat.Env.step` 或 episode，不能据此断言每条导航 transition 的时长固定为 0.016 秒。

## 字段与时间语义

- `sim_time_before_s`：动作前观测对应的 simulator 世界时间。
- `sim_time_after_s`：动作后观测对应的 simulator 世界时间，读数发生在动作路径及任务原有物理步之后。
- `dt_sim_s`：观测窗口 `sim_time_after_s - sim_time_before_s`。它既不是动作世界时间差，也不是命令积分时域或物理步请求时长。
- observation 的 `sim_time_s`：该帧观测对应的 simulator 世界时间。
- `dt_command_s`：已提交速度命令的积分时域；预测器的第三个输入目前仍使用此字段。
- `dt_wall_s`：服务端 step 的墙钟耗时。

不从位姿、`dt_command_s`、墙钟或 `k * 0.1` 推算仿真时钟。`z_t` 与 `z_{t+1}` 仍由独立的 `wm-feature-cache/0.2` 管理；`feature_layer_selected` 保持 null。

## 历史提案记录

以下是 2026-10-27、`wm-transition/0.1` 尚在使用时的提案文字，不是当前契约。旧提案把“观测窗口确实测得零”也置为 null，并认为非空 `dt_sim_s` 应由 `dt_sim_forged` 拒绝；这两点已被 2026-10-28 的 0.2 实现取代。当前唯一合法的 null 理由是读数不可用、非有限或时钟倒退，而不是实测差值为零。
