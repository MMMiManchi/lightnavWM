# 规模门槛与仿真时钟：待决定事项

> 2026-10-28 更新：此前“dt_sim_s 继续保持 null”的记录是 2026-10-27 的历史状态。当前采集实现使用 Habitat-Sim get_world_time 记录动作前后观测时钟，现行 schema 为 wm-transition/0.2；旧记录不回填。

日期：2026-10-27（2026-10-28 状态补记）。这份材料记录已接受决定和授权范围；文档本身不改变 `HUMAN_ACCEPTED`、`SCALE_ACCEPTED` 或 `COLLECTION_AUTHORIZED`，也不启动采集。三个开关各管一件事：协议正文、规模数字、正式采集。打开其中一个不会打开另外两个。

协议、规模整组和方案 3 已接受。另有独立用户授权只允许 `train_fit` 数据采集；特征提取、训练、`train_dev` 采集和正式评测仍未授权：

| 要决定的事 | 开关或清单项 | 当前决定 |
|---|---|---|
| 采集协议正文是否按现在的文字使用 | `HUMAN_ACCEPTED` | 已接受。来源是 2026-10-27 的用户指示，记录在 `experiments/world_model/records/human_decisions_20261027.json` |
| 下面第 2 节的规模数字是否整组接受。只接受几行或改任何一行，都仍然是未决定 | `SCALE_ACCEPTED` | 已整组接受。来源同上 |
| 第 3 节四个仿真时间读数窗口选哪一个 | `sim_clock_semantics` | 已选方案 3。2026-10-27 的隔离探针测得 `VelocityAction.step` 的世界时间差是 0、随后观测窗口差是 0.016 秒。2026-10-28 已实现 `wm-transition/0.2`，按每次真实 `Env.step` 读取时钟；隔离探针数值不作为固定步长 |
| 是否另做 `train_dev` 采集入口并单独授权 | `dev_collection_path` | 入口已编写，并且拒绝 `train_fit`。执行采集未授权 |
| 是否授权正式采集。此授权只覆盖 train_fit 数据采集，不启动运行，也不授权特征、训练、dev 或评测 | `COLLECTION_AUTHORIZED` | 已授权 train_fit；常量 true，来源见 `records/collection_authorization.json` |

正式特征缓存的状态是 `blocked`。它表示缓存还没建、提取入口会拒绝，不是上表里的一项决定。

## 1. 三层数字不是同一个门槛

| 层 | 现在的值 | 决定什么 | 不决定什么 |
|---|---|---|---|
| 加载资格 | fit 至少 8 个场景、20 个路线组；dev 至少 3 个场景、5 个路线组 | `require_formal_split` 能否返回样本 | 样本够不够训练；采集是否被批准 |
| 清单容量 | fit 53 场景、2977 路线组、8938 episode；dev 8 场景、626 路线组、1881 episode。`collection_status=not_collected` | 预注册划分里还有多少可点名的 train 路线 | 必须把这些 episode 都采完 |
| 报告门槛 | fit 每种动作至少 20 条、至少 20 条非预算 episode，且每个被采场景至少 2 条 episode；dev 用全部 8 个预注册场景，每种动作至少 10 条、至少 5 条非预算 episode | 以后 E2/E3 按 `action_kind` 出表时，单元格低于这个数就不称报告已齐 | 模型一定学得会。这不是功效分析 |

动作类型只有三种：`stop_threshold`、`turning`、`mostly_forward`。依据是 `action_kind` 已有的 0.05 rad 航向步长，以及实验计划要求按这个字段分组报告。20 和 10 是“分组表不至于只有个位数”的审查下限，不是从误差曲线拟合出来的。

非预算 episode 指 `completion.state=completed` 且该 episode 没有 `runner_budget`。它不是导航成功。`success` 和 `spl` 不进入这个计数。

fit 的 8 个场景和 20 个路线组与加载资格相同，因为比加载器更低的配额写了也无法被加载。dev 要求使用清单里全部 8 个场景，是为了避免看过 fit 误差后再挑验证场景。现有 `assert_collectable` 会拒绝 `train_dev`；独立的 `dev_collect.py` 已有分区规划与拒绝入口，但不能构造环境，执行 dev 采集仍未授权。

## 2. 按门槛分别决定

`SCALE_ACCEPTED` 记录 2026-10-27 对下面整组数字的接受，现在是 true。它不能只接受下面某一行。只接受其中几行、或改掉任何一行，都算修改；那样的话这个开关必须回到 false。接受这一组数字不创建 allowlist，也不运行采集器。独立授权记录允许 train_fit 数据采集，但不允许特征提取、训练、dev 采集或正式评测。

加载资格和清单容量不在这个开关里。加载资格已经写在 `latent_data.py`，清单把它锁在 8/20 和 3/5。清单容量是划分文件里的计数，不是要人接受的定额。

| 门槛 | 依据 | 接受 | 改低 | 改高 | 暂不接受 |
|---|---|---|---|---|---|
| fit 至少 8 个场景、20 个路线组 | 与加载资格相同。更低的数写进文档也不会被加载 | 采集计划至少要点名这么多 `train_fit` 场景和路线组，报告才算齐 | 加载器仍然拒绝，文档里的更低数字不生效 | 第一次采集要覆盖更多场景或路线，规模更接近清单上界，启动批准仍然另计 | `scale_proposal` 继续阻塞，E2/E3 没有登记过的场景数 |
| fit 每个被采场景至少 2 条 episode | 一条 episode 看不出同一场景里的第二条路线。不是功效分析 | 只有一条 episode 的场景不能算作该场景已覆盖 | 单条 episode 就能算覆盖一个场景 | 每个场景要更多 episode 才算覆盖 | 没有登记过的场景内重复要求 |
| fit 每种 `action_kind` 至少 20 条 `valid` 单步 | 实验计划按 `stop_threshold`、`turning`、`mostly_forward` 分组。20 只是让单元格不要只有个位数 | 少于 20 条的动作类型不能称为报告已齐。这些单步在仿真时间契约和成对特征都存在之前不计 | 分组表可以用更薄的单元格发表 | 每种动作要更多单步，采集量上升 | E2/E3 没有登记过的单元格大小 |
| fit 至少 20 条非预算 episode | `completion.state=completed` 且没有 `runner_budget`。不是导航成功 | 预算截断的 episode 永远不能填这个名额 | 更少的完整采集循环就能称报告已齐 | 需要更多不是预算截断的 episode | 没有登记过的完整轨迹条数 |
| dev 使用全部 8 个预注册场景 | 避免看过 fit 误差后再挑验证场景。现有入口会拒绝 `train_dev` | 开发集报告必须用清单里的全部 8 个场景，而且要先有单独的采集入口 | 若改成子集，就必须在看 fit 结果之前冻结子集，否则就是事后挑选 | 不能高于 8，除非先改划分；那不是这次的开关 | 开发集路径继续阻塞 |
| dev 至少 5 个路线组 | 与 dev 加载资格相同 | 开发集计划至少要点名 5 个路线组 | 加载器仍然拒绝 | 开发集要覆盖更多路线，仍不必采完 1881 条 episode | 没有登记过的开发集路线数 |
| dev 每种 `action_kind` 至少 10 条 `valid` 单步 | 开发集用于选择，格子比 fit 薄，同样不是功效分析 | 少于 10 条的动作类型不能称为开发集已齐。计数同样等待仿真时间和成对特征 | 选择阶段可以用更少的单步 | 选择阶段要更多单步 | 没有登记过的开发集单元格大小 |
| dev 至少 5 条非预算 episode | 与 fit 的非预算定义相同，条数更少 | 预算截断不能填这个名额 | 更少的完整开发集轨迹就能称选择集已齐 | 需要更多完整开发集轨迹 | 没有登记过的开发集轨迹条数 |

三处数字必须一起改：`precollection_protocol.py`、本文件、协议正文。把 fit 提高到接近 8938 条 episode，或把 dev 提高到 1881 条，第一次采集就变成大规模采集，启动批准仍然另计。当前规模门槛已整组接受，`SCALE_ACCEPTED` 为 true；若将来修改其中任何一行，必须重新决定并将该开关设回 false。数据仍须按 train_fit 采集并通过质量/覆盖报告；规模接受和采集授权均不表示训练就绪。

## 3. 仿真时钟需要的决定

2026-10-27 的隔离探针使用 habitat-sim 0.3.1，场景文件 `17DRP5sb8fy.glb`，`enable_physics` 为 true，渲染器关闭，没有传感器，没有 `habitat.Env`，没有 episode，没有写出 transition。读数写在 `experiments/world_model/records/sim_clock_step_physics_20261027.json`。清单读这份记录以核对历史测量；窗口选择则由决定记录确认，已选方案 3。核对文件不等于本轮重跑探针。

| 操作 | `get_world_time` |
|---|---|
| `pathfinder.try_step` 后再 `set_state`，水平位移 0.3904685378074646 米 | 0 |
| 同一次进程里接着调用 `step_physics(1/60)` | 0.016 秒 |
| 再调用一次 `step_physics(1/60)` | 0.032 秒 |
| 另一次进程里单独调用 `step_world(0.1)` | 0.096 秒 |

物理子步是 0.008 秒。`1/60` 和 `0.1` 都不会原样写进时钟。

采集代码里的顺序是：`VelocityControlAction.step` 移动智能体且不调用 `step_world`；`EmbodiedTask.step` 随后调用 `step_physics(1/60)`，也就是 `step_world`；动作返回 None 后，`HabitatSim.step(None)` 只读传感器，不再推进时钟。动作后观测落在这次 `step_physics` 之后。上面的 0.016 秒是隔离调用测到的，不是在 `habitat.Env.step` 或某条 episode 里测到的。当前服务端在每次 `Env.step` 前后读取世界时间，写入 `wm-transition/0.2`；隔离测量不代表每步固定为 0.016 秒。

四个选项里，2026-10-27 的决定选择方案 3。隔离测量调用了真实的 `VelocityAction.step`，没有 `habitat.Env`，没有 episode。动作参数 `linear_velocity=1.0`、`angular_velocity=0.0`、`time_step=0.1`。动作前后世界时间都是 0，差是 0，水平位移 0.25 米。随后只调用一次 `EmbodiedTask.step` 里原有的 `step_physics(1/60)`，没有第二次。观测窗口的世界时间差是 0.016 秒。该结果只记录历史探针；当前 `dt_sim_s` 来自每次实际 `Env.step` 的观测窗口端点差，实测 0 也保留。记录在 `experiments/world_model/records/sim_clock_velocity_control_20261027.json`。

在 2026-10-27 的旧状态下，选项 3/4 不能改变 `wm-transition/0.1` 的含义，因此当时只保留提案、不分配版本号。该阻塞已于 2026-10-28 通过实现 `wm-transition/0.2` 解决；当前 schema 常量仍固定为 0.2，未创建更新版本。

## 4. 开发集路径

模型选择需要 `train_dev`。fit 采集入口 `assert_collectable` 仍然拒绝 `train_dev`。独立入口 `assert_dev_collectable` 只接受 `train_dev`，并拒绝 `train_fit`。`dev_collect.main` 不能构造环境；执行 dev 采集仍未授权，`COLLECTION_AUTHORIZED` 的现有范围仅为 `train_fit`。
