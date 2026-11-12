# LightNav-0 事件级世界模型数据可行性审计

审计日期：2026-11-03  
工作位置：CEC:/home/szgx/LightNav-0  
审计范围：只读核对现有 train_fit/train_dev transition、特征缓存、准入记录和在线接口；不启动 Habitat，不采集、不提取特征、不训练、不评测。

## 结论

现有数据足以构造连续的逐步观测—动作—结果序列，也有已执行动作确认、世界时间、RGB/depth 和 GT 位姿标签；它们可以作为事件数据的底层素材。**现有数据尚不能直接监督或评价用户目标中的事件级分支世界模型**：schema 没有岔路/门口/房间/地标事件 ID，没有动作发生前的候选分支集合，也没有同一已确认节点上多种分支的结果关联。严格可确认的事件标签和显式候选分支集合为 0。

最终复核共享工作区时发现另有 event_spans/event_place 诊断代码和输出：使用 GT 位姿切出 165 个转弯/episode-end proxy span，其中 train_fit 85、train_dev 80；几何近邻算法报出 11 个有 2–3 个离开朝向的候选地点。这是值得人工核验的新线索，不是 11 个已确认岔路：地点仅按水平位置 0.75 m 邻近与朝向阈值聚类，可能合并不同位置，且输出没有给出候选地点到成员 span/观察 ID 的完整映射。事件类型标签仍为 0。

现有 LightNav 控制链每步只产生一条 RVQ 轨迹并把一个航点转换成单个 velocity_control 命令；它不是 Harness 的分支候选接口。在线策略实际只把 RGB 历史和指令送入推理，不消费 depth。普通 RemoteEnv 路径也没有采集器里的逐动作 execution_ack。因此，TSSM 和 Harness 不能只靠把模型接在现有 policy.act 前面就获得因果闭合的数据闭环。

结论不是“没有可用数据”：它是“逐步数据很完整，但事件、反事实分支和导航相关记忆标签缺失”。暂不训练事件级 TSSM；下一步先做小规模事件/分支标注盘点，判定现有多路线重访能否提供真实替代结果，不能则再做定向分支采集设计。

## 数据来源和质量

所有列出的采集 run 均记录 source_split=train；train_dev 是按场景隔离的开发划分，不并入 train_fit。schema 均为 wm-transition/0.2，不使用早期 0.1 文档推断当前字段。

| Run | Partition | Episode | Scene / route group | Transition | Episode steps min / median / max | Budget 截尾 |
|---|---|---:|---:|---:|---:|---:|
| formal_fit_20261028_04 | train_fit | 24 | 8 / 24 | 1,816 | 29 / 68 / 180 | 0 |
| formal_dev_20261028_01 | train_dev | 24 | 8 / 10 | 3,249 | 65 / 84 / 500 | 1 |
| formal_fit_radius010_20261029_01 | train_fit | 120 | 29 / 40 | 9,420 | 24 / 62.5 / 500 | 1 |
| formal_fit_radius010_20261029_02 | train_fit | 120 | 27 / 41 | 8,733 | 26 / 61.5 / 276 | 0 |
| formal_fit_radius010_20261029_03 | train_fit | 120 | 26 / 40 | 9,284 | 28 / 65 / 353 | 0 |
| formal_fit_radius010_20261029_04 | train_fit | 120 | 31 / 40 | 9,681 | 29 / 67 / 441 | 0 |
| formal_fit_radius010_20261029_05 | train_fit | 140 | 28 / 48 | 10,061 | 30 / 63.5 / 214 | 0 |
| formal_fit_radius010_20261029_06 | train_fit | 20 | 8 / 8 | 1,550 | 45 / 79 / 111 | 0 |

六个 radius010 run 合计 640 episodes、48,729 transitions、47 个去重场景、214 个去重 route_group_id。加上 fit04 后 train_fit 共 664 episodes、50,545 transitions、47 个去重场景和 237 个去重 route_group_id；没有重复 episode_uid。fit04 与 dev01 的 8 个场景互不重叠。dev01 是 24 episodes、3,249 transitions。

本轮直接读取 8 份 validation_report.json：每份 hard_error_count=0、每个 transition 均 valid 且有 executed=true 的 execution_ack，dt_sim_s 非空；每份的 RGB/depth missing 数和 observation sim_time_s 缺失数均为 0。transition 数与 episode 记录相符；各 run 的步数范围见上表。fit04 和 radius010-02 至 06 的 episode completion 均为 completed；radius010-01 有 1 个 budget_truncated；dev01 有 1 个 budget_truncated。completed 只是采集 episode 完成状态，不代表导航成功。

这些是现存 validation report 的证据，不是本轮重新运行 validate_snapshot。validate.py 的 _load_array_bytes 会对每个 RGB/depth 资源核验 SHA256、numpy shape 和 dtype（experiments/world_model/validate.py:464-475），validate_snapshot 对 observations 的资源逐项调用该检查（:555-566）。因此报告为 0 硬错误意味着采集时校验过整包图像资源；本轮没有重新读取所有图像文件。

半径和传感器以实际 run 配置为准：collect_train_go2w_radius010.yaml 中 agent radius=0.10 m、相机高 0.50 m、RGB/depth 480x270、HFOV=120 度；manifest 记录 depth_sensor 为 HabitatSimDepthSensor 且 normalize_depth=true。不要将该 depth 数值不经反归一化直接解释为米制距离。LightNav policy 当前并不把 depth 送入模型。

manifest 的 training_use 字段仍为 not_approved；另有独立 approval sidecar。8 份 approval 均通过 verify_run_approval 的 schema/run_name/MAC 检查（传入 approval 自带的已签名 files map），并且当前 manifest.json 与 validation_report.json 的哈希与各自 approval map 一致。为避免加载和复制全部 RGB/depth 快照，本轮没有重新计算所有 JSONL 和 asset 的 approval digest，也没有运行 load_accepted_run_set。因此，这里确认的是批准记录真实性及两份关键元数据与批准快照一致，**不宣称本轮重新完成了训练准入加载**。已有首版训练报告记录的 fit/dev 来源仍是 fit04/dev01，不是 48,729 条 radius010 campaign。

## 事件、分支与记忆标签

- episode 记录有 scene、route_group_id、route_geometry_hash、instruction；observation 有 step_index、RGB/depth、sim_time_s、available_at_decision 和 pose_label_ref；transition 有前后 observation 引用、一个执行动作、预测轨迹和 ack；privileged_labels 有 GT pose、goal、geodesic distance、reference_path、collision 和导航 metrics。
- 当前记录中没有显式 event_id、event_type、junction/door/room/landmark/node ID 或 branch_candidate 集合。严格事件节点/转移数为 0；当前决策时可用的备选分支集合为 0。
- 每条 transition 记的是唯一实际执行动作与实际后果。不同 episode 可能重访相同位置，但没有经标注的节点同一性和候选分支集合；本审计不从 pose 邻近或动作变化推断成“同一岔路”。显式标注的同节点多分支 outcome 数仍为 0。
- GT pose、goal、reference_path、goal distance、collision 和 metrics 只可用于离线构造或评估 oracle/proxy，不能进入线上查询。共享工作区的 event_spans 代理规则为累计航向变化至少 45 度后到稳定帧，直行 yaw 阈值 0.03 rad、0.5 m 摆动重置；event_place 另用同场景 0.75 m 水平距离及至少 0.5 rad 离开朝向差聚类。这些规则得到 165 spans、11 个候选地点；没有标门/岔路/房间/地标，本审计没有运行生成器。
- 11 个候选地点包含 47 个成员 span；9 个地点跨 2 个及以上 episode，2 个仅来自同一 episode 的非连续回访。由于地点聚类使用 union-find 连通分量，链式相邻点可能使最终 cluster 直径超过 0.75 m。results.json 仅给聚合数，不给稳定的 cluster_id/member span IDs，尚不能直接人工核验每个分支。
- 指令文本和未来帧可支持离线人工标注，但现有数据没有独立的“这条历史经验对当前导航有帮助”标签。视觉相似只表示表观相似，不等于导航相关。不能据此报告有意义的 navigation Recall@k。

最低事件数据契约应显式区分观察、预测和未知：

| Record | 最低字段 | 约束 |
|---|---|---|
| EventNode | event_id、episode/run/scene/route、observation_id、step_index、event_type、label_source、confidence、available_at_decision | 真实标注与 proxy 使用不同 event_type/provenance；未知值保留 unknown |
| BranchCandidate | candidate_id、event_id、candidate action/local subgoal、generator_id/version、created_at_decision、policy_id | 必须在选定动作和 outcome 之前生成；不能带未来观测/未来路线结果 |
| ObservedOutcome | candidate_id、action_id、execution_ack、final_habitat_action、obs_before/after、dt_sim_s、terminal/truncated/end_source | 只有确实执行并观测到的后果标为 observed；未走分支为 unknown，不伪造 outcome |
| MemoryRecord | memory_id、source observation/event、feature vector、encoder/output/version、scene/route/episode、available_at、observed_or_predicted、label provenance | predicted feature 不能伪装成真实观察；训练/查询必须做时间和路线泄漏过滤 |
| LearningTransition | 当前 event history、BranchCandidate、ObservedOutcome 引用、instruction/landmark 标签来源、split、policy_id、截尾状态 | train_fit 与 train_dev 分开；dev 只用于选型/验证，不回流 fit |

## 特征缓存和可评价性

现有缓存统一为 wm-feature-cache/0.2、google/siglip-so400m-patch14-384、output_name=pooler_output、shape=[1152]、float32、feature_layer_selected=null。

- pooler_output/features.jsonl 有 5,113 行，恰好与 fit04 的 1,840 条和 dev01 的 3,273 条 RGB observation 一一对应；按 observation 图像文件名集合核对，missing=0、extra=0。pooler_output_failed_hardlinks_20261028 的 manifest 与 features.jsonl 哈希都与 pooler_output 完全相同，是重复记录，不重复计数。
- motion_supplement_20261030_01 有 582 行，来自 radius010 run01-run05；run06 为 0。对 48,729 个 radius010 transition，仅 485 条的 obs_before 与 obs_after 两端都能在该缓存的 observation ID 集合中找到。这是基于 ID/文件名的候选相邻对数量，尚未逐条调用 align_transition_latents，不等于 485 个已通过 pair contract 的样本。
- 582/49,369 个 radius010 observations 约 1.18%；候选相邻对 485/48,729 transitions 约 1.00%。因此新采的 48k 数据没有对应的完整 SigLIP 缓存，不能把它们直接当成已缓存的世界模型特征数据。
- 共享工作区的 event_place 输出另有 213 个端点 SigLIP 向量（diagnostic endpoint_pooler.npz），不是正式缓存。针对 79 条路径至少 1 m 的转弯 proxy，真实起点—到达向量余弦中位数 0.873，起点—“other”余弦中位数 0.843，中位差 +0.020；59/79 的起点—到达余弦更高。它比较的是已发生路径的真实到达画面，不是世界模型预测，也不测试分支选择。所谓 other 是同一 split 排序后的下一条 span 到达帧，代码没有强制不同 episode/route，不能当作无泄漏记忆检索评测或 navigation Recall@k。

若查询使用真实未来 SigLIP feature，可以在严格排除目标帧、同一未来片段、同路线泄漏后做离线视觉相似度上界；但没有独立导航相关性标签，所得结果不能说明预测检索会提升路线选择。本轮没有运行检索指标。

## LightNav 实际接口

| 部分 | 当前代码事实 | 对目标链路的影响 |
|---|---|---|
| Runner 生命周期 | runner reset 后每步调用 policy.act，随后 env.step（src/lightnav/habitat/runner.py:614-694） | 有当前观测→动作→下一观测的顺序；没有显式 after_step/episode-memory callback |
| 输入策略 | policy.reset 提取 instruction；act 只把 obs.rgb 放入 TrackingAgent，再调用 generate_from_frames（src/lightnav/habitat/policy.py:142-168） | depth 不进当前模型；模型看到 RGB 历史和指令 |
| 控制输出 | 单个 RVQ/trajectory 输出解码为 waypoint 序列，跳过前导原点后选一个航点并变成单个 velocity_control（src/lightnav/habitat/policy.py:41-47, 170-198） | 当前不是可供 Harness 排序的多分支候选列表 |
| 普通环境 RPC | RemoteEnvClient.step 只返回 server info；RemoteEnvServer 转发环境的 obs/reward/done/info（src/lightnav/habitat/remote_env.py:115-127；habitat_server/lightnav_habitat/remote_server.py:91-107） | 普通评测链没有 collector 的 wm_action_id/execution_ack 契约 |
| 采集器动作确认 | WorldModelVLNCEEnv 要求 action id，记录 final_habitat_action 和跨 Env.step 的 dt_sim_s（experiments/world_model/server_adapter.py:433-503） | 可借鉴其 ack 语义，但它是独立采集 adapter，不代表普通在线接口已支持 |
| 在线信息隔离 | online_observation 只返回 RGB/instruction，online_info 只放行速度配置；调用位置在 collect.py（experiments/world_model/online_input.py:11-52；collect.py:447-463） | ordinary runner 没调用此 projection；当前 policy 自身只选 RGB/指令，但未来 Harness 必须显式使用安全投影，不能接收 raw info |
| GT 泄漏面 | Habitat obs/info 可含 goal_distance、progress、reference_path、goal_position、distance_to_goal 和 metrics（habitat_server/lightnav_habitat/vlnce.py:377-382, 438-462, 510-517） | 这些只能用于评测/离线 oracle；不能作为 TSSM/Harness 线上状态 |
| 视频特征 | LightNav ViTResult.video_embeds 是 base vision tower 的 post-pool packed embeddings，且另带 video_grid_thw（src/lightnav/inference/engine.py:18-23, 458-509） | 它不是缓存中的独立 SigLIP pooler_output；要用于 TSSM 必须新增明确的时序/空间索引契约 |
| 记忆生命周期 | NavigationPolicy.reset 会清空 history/cache；SlowFast 按 episode 内 absolute frame index 构造显式 video_segments（src/lightnav/inference/policies.py:82-95, 106-152；src/lightnav/inference/samples.py:29-69） | 跨 episode 记忆需要独立 MemoryStore，不能把不同 episode 的图像拼成连续视频 |
| 主动观察/回溯 | 当前 policy/runner/server 未发现 observe-only、候选分支生成、物理回溯或恢复节点接口 | 这些需新增 Harness/environment 能力，不是现有 controller 参数 |

## 证据快照

本审计保留服务器现有工作区状态：branch local/research-snapshot、HEAD 4933c66f32898175cd5b8ce870c40d8ac26c8765、dirty=true。审计自身只新增本报告和 summary.json，没有修改源码/原数据/授权开关/权重/现有结果，没有运行 pytest、Habitat、特征提取、训练或导航评测。最终复核时共享工作区已存在未跟踪的 event_spans/event_place 代码、测试及诊断输出；来源/执行者/具体运行时间不能从当前 Git 状态归属到本审计，故本审计没有重跑它们。event_place 的 entry 源码明确会加载 SigLIP 并编码 213 个端点，输出声明未写正式缓存；该输出与本审计自身未提取特征是两件事。输入和这些代理诊断产物的 SHA256 见同目录 summary.json。

## 唯一下一步

T1：先补齐 event_spans 诊断的稳定 cluster_id/member span/observation 映射，并给 11 个几何候选地点生成人工核验清单。按 RGB/route 上下文核验它们是否真是同一决策地点、是否为门/岔路/单纯转弯，以及每个离开方向是否都有真实后续观测；同时检查 cluster 的最大空间直径、split/route 去重和 9 个跨 episode、2 个单 episode 回访的差异。把无法确认的标为 proxy/unknown，不借模型生成补标签。此任务只验证现存轨迹能否形成分支监督/记忆评价集；若不够，再单独设计定向分支采集。不要先训练 TSSM，也不要把 train_dev 用于训练。
