# 最新世界模型数据：质量、规模与训练可用性审查

日期：2026-11-04，北京时间。工作目录 `/home/szgx/LightNav-0`，分支 `local/research-snapshot`，HEAD `4933c66`。

架构参考按用户明确的分工：LightNav-0 是基座；HarnessVLN 参考编排；Memoir 参考预测检索思路；SC²-WM 参考世界模型设计。本次检查已保存的数据与当前读取/采集代码，没有启动仿真、补采、提取新特征或训练，没有修改原始数据或实验实现。

## 结论

**存量足够启动小规模预训练和可行性试训；最新探针与场景图还不能整包直接当成已验收的事件分支训练集。** 原始图像文件整体完整，主要缺口是下一事件标签、分支执行证据、图像与图节点的对应关系，以及独立的分支验证数据。

这个判断不要求先补齐所有房间/物体语义标签：真实未来 RGB 本身就能监督视觉状态预测。需要先明确目标是“规则切分后的真实未来观察”还是“已确认的下一决策节点”，并保持实际分支、终止原因、数据来源和监督目标可追溯。

不建议因为当前问题就把所有数据重采。已生成 89 条分支审查索引、2,191 条训练侧运动片段代理索引，以及场景图 RGB 对应清单，可以直接支持下一步样本整理。

## 实际数据量

| 数据 | 数量 | 独立性与用途 |
|---|---:|---|
| 旧正式 train_fit | 50,545 transitions，664 episodes，47 scenes，237 route groups | 完整的逐步执行/观察素材；episode 的同路线指令变体不能当独立路线 |
| 旧正式 train_dev | 3,249 transitions，24 episodes，8 scenes，10 route groups | 与 fit 场景、路线和几何哈希均不重叠；继续保留为开发来源 |
| 新探针全部保存观察 | 6,819 条，6,290 个不同 RGB 文件哈希 | 含回放、旋转和重复尝试，不代表 6,819 个事件 |
| 新分支尝试 | 原始 90 次，按同一目标保留最新重试后 89 条 | 来自 38 个场景、44 个源 episode/route、55 个决策上下文 |
| 程序标记到达 | 44 条，来自 33 个决策上下文 | 尚未确认的下一路口代理标签；到达净位移中位数 1.63 m，步数中位数 43.5 |
| 两批房屋探针 | 45 个场景；31 处找到候选节点并各尝试两个出口；14 个未找到路口 | 两个分支均标到达的节点只有 11 处 |
| 最新场景图 | 112 places，208 occurrences，111 edges，45 scenes | 混合 fit/dev 和新探针的诊断图，不能按 111 条训练样本计数 |

去重后的 89 条分支终止状态：44 条 `next_decision`、30 条 `truncated_blocked`、9 条 `truncated_distance`、6 条 `truncated_budget`。受阻和预算终止保留为实际观察或截尾素材，不当作下一事件的正例。

官方 train 共 61 个场景，现行按 scene 的划分为 53 个 train_fit、8 个 train_dev。旧 fit 的 47 个场景与新探查的 45 个场景重叠 39 个，合计已经触及全部 53 个 fit 场景；新增场景是 `HxpKQynjfin`、`Pm6F8kyY3z2`、`XcA2TqTSSAj`、`YmJkqBEsHnH`、`dhjEzFoUFzH`、`gZ6f7yhEvPG`。这是场景触达，不代表每个房间、岔路或分支均已覆盖。

划分核对使用 `latent_data.scene_partition`。`group_manifest.assign_groups` 仍返回旧的 route 派生 partition，不能用它代替 scene 划分审计。

## 文件与序列完整性

- 新探针全量重新读取了 **13,638 个 RGB/depth NPY 文件，约 6.19 GB**；记录哈希、尺寸、dtype、可解码性和有限数值均通过，缺失/损坏/哈希不符为 0。RGB 为 270×480×3 uint8，depth 为 270×480 float32，深度值均在 [0,1]。
- 旧八个 run 全量重新核对元数据、文件存在性、transition 顺序、前后观察链、pose 引用、action/ack ID、执行状态及世界时钟一致性，未发现这些结构错误。另每个 run 按观察 ID 等距抽取 16 条观察，共 256 个 RGB/depth 文件重新哈希并解码，全部通过。
- 旧 run 原有 validation report 均为 0 hard errors。本轮没有重新全读旧数据的全部图像或重跑完整的冻结准入 loader；旧图像检查的本轮范围就是上述抽样。
- 新分支的 89 条记录均能找到保存的末帧，记录的步数与保存分支帧数相符，报告位移与末帧位姿相符。58 条记录提供恢复位置/朝向误差且均为 0；其余 31 条为首次从搜索节点直接出发，未单独保存恢复误差，不算恢复失败。

文件完整性通过不等于画面有足够环境信息。新数据中 34 帧超过一半像素接近纯黑，其中 23 帧超过 80%；深度也有对应的大面积零值。人工查看最高零值的 4 帧，可见画面主要是黑色空洞，仅少量表面残留。两个分支末帧超过一半近黑，均在 `YmJkqBEsHnH/1415` 的受阻探针中。不要将这些作为有效房间后果或几何可通行性正例。

另外，`vyrNrziPKCB/10076/exit_1` 保存了 452 帧，但位置全程不动，最终预算截尾。原地观察有其用途，不能按这些帧数增加独立行走事件计数。

## 当前不能直接投入正式事件训练的原因

### 1. 到达判定仍可能停在原节点附近

`branch_stop` / `arrival_stop` 以“距离起点至少 0.8 m，当前开口至少两个”为到达条件，没有验证离开原节点并进入另一节点。44 条到达记录中，5 条从始至终都观察到多个开口，另有 1 条最初 0.4 m 位移方向与候选方向相差超过 25°。

这六条是审查标记，不是已经证明的六个错误；25° 只是方向筛查阈值，绕障也可能造成偏差。44 条中的 38 条没有触发本次这几项筛查，也仍不能自动升级为真实事件标签。

优先复核的五条全程多开口记录：

- `old:5LpN3gDmAk7:2996:90:0`。
- `old:cV4RVeZvu5T:2129:129:1`。
- `house:5q7pvUzZiYa:10100:0` 和 `:1`。
- `house:p5wJjkQkbXX:10046:0`。

### 2. 分支步的实际命令、ack 和世界时间没有保存

新探针的 **6,715 个分支步，已保存的 command、execution_ack、sim_time_s 覆盖均为 0**。当前 `_save_frame` 保留 RGB/depth、pose、开口数以及可选 mode/moved_m。`search_commands.json` 只记录搜索/回放前缀，不记录分支控制序列。

这不抹掉真实 RGB 的自监督价值，也不意味着按预先选择的分支描述预测末帧完全不可训练。但它使执行一致性、控制策略版本、物理时间以及部分截尾原因无法严格复验，不能把新探针假装成旧 `wm-transition/0.2` 完整 transition。后续 recorder 应在实际执行时保存这些字段；旧样本不能用重新推算的动作冒充已记录动作。

### 3. 最新图中有大量位姿节点没有图像

最新图的 111 条连接中，**45 条来自 train_dev，训练侧只有 66 条**。训练侧进一步追溯后：

| 条件 | 数量 |
|---|---:|
| 两端 RGB 均能映射回真实文件 | 27 / 66 |
| 其中还匹配到原候选出口 | 7 |
| 缺少至少一端 RGB | 39 / 66 |
| 图中缺 RGB 的到达 occurrence | 49 |

原因是新 `_search` 的中间步骤只写了 `path.jsonl` 位姿，RGB 只保留 start/node；`_fit_house_episodes` 随后给所有搜索位姿生成 observation_id，并在这些位置上产生图节点。生成 ID 不会补出实际图像。例如 `GdvgFV5R1Z5/4115/search#54` 有位置和生成的 ID，但没有对应观察资产。

因此 `places.jsonl/edges.jsonl` 适合几何诊断，当前不能直接作为完整视觉事件 dataset。新分支自身的 start/end 配对可以绕过图继续利用；缺失的搜索帧需排除或按明确需要补齐。

111 条 walked 出口还有 80 条是候选匹配失败后追加的 `exit_id="walked"`；训练侧为 44 / 66。它们证明实际走过一段路径，尚未证明执行前候选与该结果正确关联。图的扇区角度、近邻节点合并问题仍见前次审查，不能把目前 14 个环当成完成核验的拓扑真值。

### 4. 两个出口不必产生两种不同后果

已人工查看 **全部 11 个“两条分支都标到达”的房屋节点**，每处共享起点和两张末帧，共 33 个展示位次。图见本次产物的 `visual_review/paired_arrivals_01.png` 与 `_02.png`。

其中 `1LXtFkjw3qL/10181` 等有明显不同的真实后续视角，说明已采数据有价值。`5q7pvUzZiYa/10100` 的两个终点相距约 0.20 m、`PX4nDJXEHrG/10232` 相距约 0.13 m，图像也显示相同宽泛空间。它们可能汇合，也可能候选分割或执行有偏差；需要按路线过程核验，不能强制把不同 exit_id 当作对比学习负例。

31 对房屋分支中，另有 `YmJkqBEsHnH/1415` 的两次受阻终点几乎相同，图像差的均值约 0.0014 / 255。它不提供两条不同的有效到达结果。

### 5. 验证集和特征契约尚未配齐

新分支全部在 train_fit，与现有 dev 的 scene/route/route_geometry_hash 无重叠，这是通过项。现有 dev 仍是旧 on-policy 路线，没有对应新探针控制器的成对分支结果。不能只在 31 个节点上随机拆帧验证；需要按 scene、route、node 分组并固定独立验证单元。

现有 SigLIP pooler cache 为 1152 维特征。对新图像按真实图像哈希核对，只有 24 个不同 RGB 已在缓存中，**新分支两端特征都齐备的样本数为 0**。这属于提特征/对齐工作，不是必须重采的理由。预测头、冻结目标 encoder 与查询索引需要约定同一特征空间。

SC²-WM 的已核对实现包含候选特征重建网络及反馈 gate/delta，可作为首个小型预测模块的结构参考；接到 LightNav 时仍需重新定义本项目的输入与监督契约。[固定版本模型源码](https://github.com/sunrise-yuze/SC2_WM/blob/f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b/vlnce_baselines/models/etp/vilmodel_cmt.py)

## 现有 50k 数据还能提供什么

本轮对全部八个正式 run 运行了已有的 `segment_episode` 规则，生成独立 proxy sidecar，没有改原数据、事件标签或 schema：

| 划分 | 运动/结束片段 proxy | 路径至少 1 m | 净位移至少 1 m | 截尾末段 |
|---|---:|---:|---:|---:|
| train_fit | 2,191 | 1,496 | 1,444 | 1 |
| train_dev | 80 | 60 | 59 | 1 |

规则依据 GT 航向变化与回合结束，只用于离线确定目标观察；GT 不作为在线输入。它没有门、岔路或房间真值，没有执行前多分支清单。片段仅标 `true_event_type=unknown`、`proposal_only=true`、`decision_time_branch_candidates=null`，不能重命名成 2,191 条真实分支事件。

这些片段有完整的原始 RGB、观察 ID 和逐步执行证据，可以用于：冻结视觉表示后的未来状态预训练、历史建模、近邻/当前状态基线及数据管线试训。对于“相同节点选择不同出口会怎样”的监督，仍应使用并核实已有多分支探针。

## 能否训练与下一步

| 训练/评价目标 | 本次判断 |
|---|---|
| 固定 encoder + 小型未来特征预测头，检验流程、目标和 loss | **可以进入准备与首轮试训**：优先整理 2,191 个代理片段及新分支真实图像对，明确各自目标 |
| 小规模分支条件末帧预测 | **可做探索性试训**：先完成实际候选—起点—末帧配对与质量筛查；结果按弱监督端点预测报告 |
| 正式的下一事件世界模型及多分支对照 | **当前不能整包直接训练**：44 条弱标注到达、11 个双到达节点，且候选关联/节点标签/执行证据未核验 |
| 预测检索是否帮助导航、持续学习是否改善陌生分支 | **当前无法据此验收**：缺独立的任务相关记忆标签、分支验证样本和在线闭环实验 |

“数量够不够”必须按独立事件/路线/场景判断。现有存量足够启动第一轮小模型可行性实验；仅凭这些静态计数，不能承诺完整模型的泛化效果。正式分支监督目前偏少，且同一位置的多帧或重复尝试无法增加相同数量的独立样本。用按 scene 分组的学习曲线和分支有无条件对照决定后续补采量，不先规定一个没有依据的总帧数门槛。

**下一张任务卡：整理现有事件预测 dataset 并完成小规模试训准备。** 输出明确的两类样本：完整轨迹派生的 motion proxy；有真实起点、预先候选和已观测后果的 branch proxy。保留截尾与 unknown，排除缺图的图连接，优先复核上述 6 个到达筛查标记和近似相同的分支后果；补上不可恢复字段的 unknown 标记与后续 recorder 修复。随后固定 encoder/目标空间、scene 分组与基线，再提特征并进行有限试训。只按实际缺口决定补采，无需把所有场景推倒重来。

## 本次产物与验证

正式产物目录：`output/world_model_diagnostics/data_training_audit_20261104_02/`。

- `summary.json`：数据、图像、划分、图和特征覆盖计数。
- `extension_summary.json`：黑色像素覆盖、旧轨迹 proxy 与到达筛查统计。
- `input_manifest.json`、`input_stability.json`：287 个输入元数据/源码/配置文件哈希；完成时无变化。每个已检查图像的哈希和检查结果另见 `asset_checks.jsonl`。
- `branch_review.csv`、`branch_review.jsonl`、`branch_quality_flags.json`：89 条去重分支的源文件、图像对、方向、终止与待核验项。
- `formal_motion_span_proxies.jsonl`：2,271 条 fit/dev 代理片段，划分和真实资产引用显式保存。
- `graph_edges_rgb_resolution.jsonl`、`missing_graph_rgb_occurrences.json`：图连接到真实 RGB 的对应与缺失证据。
- `house_pairs.json`、`visual_review/`：31 组成对分支统计；11 组双到达联系图及最高零深度样本。
- `audit.py`、`extend_audit.py`：实际离线分析脚本，未 import Habitat/torch 或加载权重。初轮 `_01` 是探索草稿，使用修正 scene 划分和 search 初帧映射后的 `_02`。

实际命令：

```bash
.venv/bin/python /tmp/lightnav_data_training_audit_20261104.py
.venv/bin/python output/world_model_diagnostics/data_training_audit_20261104_02/extend_audit.py
.venv/bin/python -m pytest tests/world_model/test_branch_exit.py tests/world_model/test_scene_graph.py tests/world_model/test_key_nodes.py tests/world_model/test_decision_nodes.py -q
.venv/bin/ruff check experiments/world_model/fit_house_entry.py experiments/world_model/fit_house_remaining.py experiments/world_model/branch_exit.py experiments/world_model/branch_exit_entry.py experiments/world_model/scene_graph.py experiments/world_model/scene_graph_entry.py tests/world_model/test_branch_exit.py tests/world_model/test_scene_graph.py --output-format concise
```

本次目标代码检查：26 tests passed，ruff passed。pytest 另提示当前环境未加载 `asyncio_mode` 对应插件；上述测试均为同步离线测试。这些通过项不证明事件标签和图节点语义正确。本次未改 `src/`，未重复运行已知受环境依赖影响的全量 CPU 套件。

审计日志：`/tmp/lightnav_data_training_audit_20261104_02.log`、`/tmp/lightnav_data_training_audit_extension_20261104.log`。重跑审计应使用新输出目录，避免覆盖本轮记录。

## 后续路线更新：优先复用官方 viewpoint

本轮后续核对了 Memoir 的节点来源和本地 connectivity。53 个 fit 场景已有 6,578 个 included viewpoint、13,476 条无向边。下一步优先建立官方 viewpoint 的 Habitat/观察适配清单，降低继续猜测和聚类自采节点的优先级；现有完整轨迹与分支探针用于连续执行及视角适配。上述数据质量计数保留。方案对比与边界见 [官方 viewpoint 与在线节点生成](world_model_viewpoint_vs_online_nodes_20261104.md)。
