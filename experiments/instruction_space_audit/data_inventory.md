# 数据清单

只读检查，没有改现有日志，也没有重跑评测。分组字段是从文件内容确认的，不是事先假定的。

## 全量对应

`data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz` 有 1839 条 episode。每条的字段是 `episode_id`、`trajectory_id`、`scene_id`、`start_position`、`start_rotation`、`goals`、`instruction`、`reference_path`、`info`。指令文本在 `instruction.instruction_text`。没有 `path_id`，也没有官方的详细或简略标记。

同一条路线的三句指令用 `(scene_id 文件名去掉 .glb, trajectory_id)` 分组。613 组，每组正好 3 条。

`output/r2r_go2w/results.jsonl` 有 1839 行，字段包括 `habitat_episode_id`、`scene_id`、`instruction`、`success`、`final_distance`、`min_distance`、`steps`、`termination_reason`、`forced_stop`。没有逐步位姿、Pointing、RVQ 或相机画面。用 `(场景名, habitat_episode_id)` 和数据集连接，1839 条全部对上，没有缺行，也没有重复。

成功按 `success >= 0.5`。成功 1129，失败 710。`termination_reason` 全部是 `agent_stop`。其中 `forced_stop=true` 的有 13 条，12 条失败、1 条成功。

## 重算后的路径统计

| 项目 | 数量 |
|---|---|
| 三句都成功 | 210 条路线 |
| 只成功一句 | 133 条路线 |
| 成功两句 | 183 条路线 |
| 一句或两句成功，合计 | 316 条路线 |
| 三句都失败 | 87 条路线 |
| 失败 episode | 710 |
| 失败且同路线有成功指令 | 449 |
| 失败且同路线没有成功指令 | 261 |

此前汇总的 710、449、87 和这次相同。相同是因为用的就是这个分组和成功门槛，不是把旧数字抄进来。

87 不能全部理解成“三句指令都写错或模型都失败”。已有的 `infinite_geodesic_episodes.csv` 里有 90 条 episode，在重烘焙的 Go2 导航网格上没有路径。这 90 条正好构成 30 条三句都失败的路线。剩下 57 条三句都失败的路线，结果距离是有限的。261 条没有成功兄弟的失败 = 90 条无路径 + 171 条有限距离。449 条有成功兄弟的失败里没有这 90 条。

`route_triplets.csv` 的 `go2_path` 列把这 30 条标成 `no_path`，其余为 `finite`。无路径来自已有诊断文件，这次没有重跑导航网格。

## 逐步信息能拿到什么

| 内容 | 全量 1839 | 30 条筛查轨迹 | 另 8 条兄弟回合 |
|---|---|---|---|
| 指令原文、成功、最终距离、最近距离 | 有 | 有 | 有 |
| 逐步相机画面 | 无 | `trace_20261020/frames`，3259 步 | `instruction_triplet_20261021`，445 步 |
| 机器人位姿 | 无 | 3259 步都有 | 有 |
| 测地距离 | 无 | 3259 步都有 | 有 |
| RVQ 和航点 | 无 | 3259 步都有 | 有 |
| Pointing（apos/opos） | 无 | 3253 步有 apos 状态，缺 6 步 | 有 |
| 模型停止 | 只有回合级 `agent_stop` | 逐步 `model_stop` | 逐步 `model_stop` |

30 条轨迹在 `output/r2r_probe/trace_20261020/steps.jsonl`。8 条兄弟是 115、117、215、216、143、144、172、173，在 `output/r2r_probe/instruction_triplet_20261021/steps.jsonl`。其余失败没有逐步 Pointing、位姿或画面。

`val_unseen_gt.json.gz` 有 1839 个键，内容是 `locations`、`actions`、`forward_steps`。这是另一套专家演示，不是这次 Go2 测地路径，审计没有拿它当模型输入，也没有拿它给指令打分。

## 已有分析文件

这些文件都在，本次没有覆盖。

- `experiments/r2r_diagnosis/paired_episode_comparison.csv`：30 行。专家、模型、提前停止三种对照，不是 1839 的路径分组。
- `experiments/r2r_diagnosis/failure_type_comparison.csv`：6 行失败类型计数。
- `experiments/r2r_diagnosis/failure_index.jsonl`：1839 行，含 Go2 和另一套官方结果的对照。路径统计以 `results.jsonl` 为准，没有用这里的官方列。
- `output/r2r_probe/visual_audit_20261020/visual_audit.csv`：14 行画面记录。
- `output/r2r_probe/visual_audit_temporal_20261020/temporal_visual_audit.csv`：247 行。
- `output/r2r_probe/error_screening_full_episode_20261021/`：30 条的逐步筛查、回放和 90 条无路径清单。

画面审计的旧结论没有直接抄进 `manual_review.csv`。本轮标签只来自重新打开的原图，或来自可复核的航向和距离日志。
