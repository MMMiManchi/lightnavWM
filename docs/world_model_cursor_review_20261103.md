# Cursor 工作区审查与进展核对

审查日期：2026-11-03（北京时间）。工作目录：`/home/szgx/LightNav-0`。
分支：`local/research-snapshot`，HEAD：`4933c66`。

本次只审查、离线读取和运行 CPU 测试，未修改实验实现，未启动 Habitat、采集、提取特征或训练。本文是本次新增的审查记录。共享工作区没有逐项提交记录，不能仅凭 Git 状态将全部改动归属给 Cursor。

## 当前进展

当前处在事件与分支数据、离线场景图原型阶段，尚未完成事件世界模型训练或与 LightNav 的在线闭环集成。旧审计文档只覆盖早期状态；工作区已经出现后续探针采集结果，不能继续概括成“只做了只读审计”。

| 阶段/产物 | 本次直接核对的事实 | 可支持的结论 |
|---|---|---|
| 工作区 | 审查开始时 21 个已跟踪文件修改，73 个未跟踪文件，未跟踪 Python 共 13,821 行；无 `src/` 修改、无暂存改动 | 包含多天的累积实验，不能按单次小补丁看待 |
| `decision_nodes_20261103_01` | 5,113 帧，298 个满足深度开口规则的帧，聚成 77 个节点；54 个匹配实际离开方向；58 个看见但未走的开口 | 深度几何候选，未形成门/岔路语义真值 |
| `branch_exit_20261103_01` / `_02` | 同一出口的两次探针；前者预算截尾，后者距离截尾，均未标到达 | 保留了尝试和失败证据 |
| `branch_exit_20261103_03` | 26 个出口探针：程序标记到达 8、距离截尾 8、预算截尾 4、受阻截尾 6 | 已有真实后续 RGB/depth，但“到达下一节点”尚需复核 |
| `fit_houses_20261103_01` | 8 个场景：6 个场景各尝试 2 个出口；12 个分支中程序标记到达 9、受阻 3；另外 2 个场景没有找到路口 | 有补充分支观察；这 8 个场景均曾在 radius010 数据中出现 |
| `scene_graph_20261103_04` | 16 场景，62 地点，114 次到达，66 条连接，108 条 unknown 出口，程序统计 6 个环 | 可作为离线审查图，地点/出口和环的正确性尚未验收 |

最新场景图只使用 `formal_fit_20261028_04` 和 `formal_dev_20261028_01`，没有并入六批 radius010 或上述分支补采。fit/dev 各 31 个地点，分别有 21/45 条连接。62 个地点中 26 个跨 episode，28 个有多次到达。

66 个 walked 出口中只有 13 个使用原候选 `e1/e2`；其余 53 个是未匹配候选后追加的 `exit_id="walked"`。66 条真实行走连接不能直接等同于 66 条已成功对齐的“执行前候选→真实结果”样本。图依赖 GT 位姿、目标和全场景 navmesh，属于离线 oracle/proxy，不能直接作为 RGB-only 在线可用状态。

探针 protocol 标记 `training_use=not_approved`、`lightnav_used=false`、`navigation_eval=false`；最新图标记 `training_started=false`。本次没有发现这些新产物已接入训练或线上策略的证据。

## 需要修正的问题

### R1 [P1] 仍在原路口观察两个开口，会被标成“到达下一路口”

位置：`experiments/world_model/branch_exit.py:298-303,371-376`。

`branch_stop` / `arrival_stop` 仅要求离起点至少 0.8 m、当前开口数至少 2，没有记录离开原节点、重新进入另一节点或节点同一性。最小复现：未更换节点、始终观察两个开口，调用 `branch_stop(18, 0.80, 2)` 和 `arrival_stop(18, 0.80, 2)`，两者均返回 `next_decision`。

17 个程序标记到达的分支中，4 个在整个已保存探针过程中开口数始终至少 2：

- `branch_exit_20261103_03/exits/5LpN3gDmAk7_2996_step90_exit0`。
- `branch_exit_20261103_03/exits/cV4RVeZvu5T_2129_step129_exit1`。
- `fit_houses_20261103_01/houses/5q7pvUzZiYa_10100/exit_0`。
- 同一 house 的 `exit_1`。

这不能证明上述 4 个一定没到新节点，但现有判定没有提供区分原节点与新节点的证据。把它们写成已确认的事件间转移，会污染监督标签。应增加节点离开/到达状态与同一性核验；现有结果保持原样，通过新的审查 sidecar 标为待核验 proxy。

### R2 [P1] 新探针没有保存分支动作和执行确认，无法还原动作条件转移

位置：`experiments/world_model/branch_exit_entry.py:153-181,341-357`，`experiments/world_model/fit_house_entry.py:219-234`。

`_save_frame` 保存 RGB/depth、position/yaw、opening_count；可选字段只有 mode/moved_m。当前分支步没有保存实际 command、`wm_action_id`、server 返回的 `wm_execution`、世界时间和逐步终止标记。实际检查 `_03` 的 observation 行也没有这些字段，`fit_houses` 多了 mode/moved_m，但仍缺 command/ack。`search_commands.json` 只覆盖寻找/回放前缀，不能替代分支执行记录。

因此这批数据目前可做观察审查，不能直接满足 action-conditioned LearningTransition 契约。后续采集应复用现有 collector 的动作/ack/时间记录；已存在探针若无法从其他日志恢复，明确标记 execution/time unknown，不以当前源码重建的动作冒充已记录动作。

### R3 [P2] 跨 ±π 的出口扇区用算术平均，方向会偏到另一侧

位置：`experiments/world_model/scene_graph_entry.py:520-525`。

代码合并首尾角度组后，用 `sum(group) / len(group)` 计算方向。只开放正后方三条射线 `[-180°, -157.5°, +157.5°]` 的最小复现，返回 -60°；正确方向应在 ±180° 附近，相差 120°。这会导致候选出口方向、行走匹配和灰色未知出口示意错误。应使用圆周均值，并对整圈开放/方向不确定的扇区显式处理。

本复现只提取函数并使用合成几何回调，没有 import Habitat 或加载 navmesh。

### R4 [P2] 近距离 A→B→A 会直接合并两个节点，删除真实环路

位置：`experiments/world_model/scene_graph_entry.py:282-295`；相近的风险也存在于 `:313-335`。

`_merge_revisits` 遇到 A→B→A 且代表位置可近距离走通时，直接把 B 合进 A，没有验证两者是否是不同的决策区域。合成两个相距 1.8 m、均标 `junction` 的节点，给出 A→B→A，函数只留下一个节点。`_collapse_splits` 也能仅凭单一后继和距离将相邻决策点合并。

这会删除实际短环和近距离的两个岔路。应在合并前验证决策区域同一性，保留明确不同的出口/节点，并测试真实短环、相邻岔路和链式合并。当前“6 个环”是该算法的输出，不是经过复核的完整环路数量。

### R5 [P2] “新房子”排除集只写死旧 8 场景，漏掉已有 radius010 数据

位置：`experiments/world_model/fit_house_entry.py:58-67,450-457`。

`WALKED_SCENES` 只有 fit04 的 8 个场景，没有汇总六批 radius010。逐条核对旧 `episodes.jsonl` 后，所选 8 个场景全部已经采过；具体 episode 不重复。

| 本批 scene | 已有来源示例 |
|---|---|
| `17DRP5sb8fy` | radius010 run04 |
| `1LXtFkjw3qL` | run01、run05 |
| `29hnd4uzFmX` | run02、run03、run04、run05 |
| `5q7pvUzZiYa` | run02、run04、run05、run06 |
| `759xd9YjKW5` | run01、run04、run05 |
| `8WUmhLawc2A` | run01、run02、run04、run05 |
| `B6ByNegPMKs` | run02、run03、run04、run05 |
| `D7G3Y4RVNrH` | run03、run04 |

这些探针仍有分支观察价值，但不能据此声称扩展了未见场景覆盖。若“新”仅相对于 fit04，应在协议中限定；若要求从未采过，则从明确列出的全部有效 run/campaign 动态计算排除集。

## 其他验收边界

- `key_nodes.goal_stop` 取整条路线中水平距离目标最近且不超过 3 m 的帧，没有确认实际 STOP、楼层或测地可达性。合成目标在另一层时也会返回起始帧；实际数据中有 5 条路线的此帧比 episode 结束早超过 10 步。应标成离线目标接近 proxy，不能将该名字直接当事件真值。
- 本次复查的是当前源码和已保存结果，缺少各轮 dirty 源码哈希与输入快照清单，不能保证当前 entry 精确复现每个历史输出。例如 `_03` 保存行没有当前 `_save_frame` 调用传入的 mode/moved_m。后续需固定代码、输入和参数快照。
- `docs/world_model_event_data_audit_20261103.md` 的“下一步”仍停留在 11 个旧候选的人工清单阶段，需由上述已发生的进展和审查问题更新，避免重复劳动。

## 实际验证

| 命令 | 结果 |
|---|---|
| `.venv/bin/python -m pytest tests/world_model -q -m 'not gpu'` | 188 passed，1 failed |
| `.venv/bin/ruff check experiments/world_model tests/world_model --output-format concise` | 9 项错误：import 排序或未使用 import |
| `.venv/bin/ruff check .` | 22 项错误，其中上述目录 9 项；其余在本次未修改路径 |
| `make test PYTHON=.venv/bin/python` | 收集阶段因 `lightnav_habitat` 不在路径而失败 |
| `PYTHONPATH=src:habitat_server .venv/bin/python -m pytest -q -m 'not gpu' --tb=short` | 放开沙箱本地 socket 限制后 756 passed、62 failed、1 skipped；其中 61 项缺异步测试插件，另 1 项是下述目录状态断言 |

world_model 唯一失败位于 `tests/world_model/test_precollection_protocol.py:109`：断言 `formal_run_directories == []`，实际已有 15 个数据目录。测试依赖工作区“尚未采集”的过期状态，需改为隔离的临时目录测试。不能据此声称核心逻辑整体失败，也不能报告测试全通过。

完整 CPU 复跑日志：`/tmp/lightnav_cursor_review_cpu_20261103.log`。本次未安装依赖；`pytest-asyncio` 已列在项目 test extra 中，当前环境未加载该插件。本次退出沙箱只重跑 CPU 测试，没有启动真实 Habitat 仿真。

另运行了不写文件的合成复现：原路口未离开仍返回 next_decision、±π 扇区方向、A→B→A 合并和跨楼层 goal_stop；以及对现有 JSONL 的到达/截尾计数、出口 ID、场景重复和 observation 字段检查。

## 下一张任务卡

唯一下一步：修正并验收现存节点/分支证据，产出一份可追溯的候选—执行—结果清单，再决定是否需要补采。

1. 修复 R1/R3/R4，补充真正区分旧节点、角度边界和短环的 CPU 用例；修复过期目录状态断言与本改动的 lint。
2. 固定输入清单、源码哈希、参数和 split；将 53 个未匹配出口单列，保留 unknown 和 oracle/proxy 来源。
3. 复核 17 个程序标记到达的分支，先审查上述 4 个全程多开口样本；新建 sidecar 记录确认、proxy 或 unknown，不覆盖旧产物。
4. 明确 R2 所缺执行证据能否恢复，重算有效“事件节点 + 预先可见候选 + 实际执行 + 下一节点”数量；汇总已有 48k 数据的场景排除集，修正“新房子”的覆盖口径。

完成后以有效样本数、候选匹配率、节点同一性/环路用例和测试结果验收。目前不以“62 个地点”“17 个 arrival=true”作为已具备训练与线上闭环的依据。

## 本次核对的关键 SHA256

```text
81dadd8690d40957c1a02bd639dbd1bdb3fe9980c344ace86aa9741d5490fa12  experiments/world_model/scene_graph.py
1714632ca1adc03bfbb5366288e9c8e3dd49f551622aee0c29747daf9aeaeea1  experiments/world_model/scene_graph_entry.py
2c1d64f94fce9441ce42a871a13989e83f2c9c055ac0f9e975700bfa2a3b1017  experiments/world_model/branch_exit.py
8bc7d24a19304e0e199f9dbf524bc6593c053853199313298c2fc86d0cfed60a  experiments/world_model/branch_exit_entry.py
81a1158c0eb48e94919393372b826137f29392e890a2ba593c8ff41aafaaab79  experiments/world_model/fit_house_entry.py
3f968b99a3c4c8aa6a29203a445a554b2ed34208da03d3dfb1aa4416662c16ec  output/world_model_diagnostics/scene_graph_20261103_04/results.json
c379a5d7723775b6d77fee391f811fad1a85b361b6ccab22ad55d3b81e43da6f  output/world_model_diagnostics/scene_graph_20261103_04/branches.jsonl
adf966eca237e8a217a5bdcfc986f745d7bad97e5febd804adaa1f205de4a409  output/world_model_data/branch_exit_20261103_03/results.jsonl
ab85db7266dd2c52b8b8f8c2a9b76b441f741109a12b6572e1ed2aa0046ed637  output/world_model_data/fit_houses_20261103_01/results.jsonl
```
