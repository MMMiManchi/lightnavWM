# 跨 episode 历史：持久场景记忆首版

2026-11-04。用户要求得到跨 episode 的历史，作为同场景越走越熟的支撑。现已实现独立的持久记忆、预测接口和因果重放验证。它是冻结世界模型的记忆增强版，没有把“继续训练更多 epoch”冒充历史积累。

## 保留什么

`PersistentSceneMemory` 使用 SQLite 保存以下真实经历：

- 工作区上下文、episode 身份、原始指令和经历写入顺序。
- 每次事件的实际起始观察、实际执行方向、执行记录引用和真实后果观察。
- 起始与后果的冻结视觉特征，维度和归一化契约与世界模型保持一致。
- 到达、受阻、截尾的区别。受阻/截尾可以记为真实经历，但不会当作已到达下一事件的目标来检索。

同一工作区开始新 episode 不会清空这些记录。原 episode 的指令、观察引用和执行引用可通过 `episode_history`、`experience_history` 查阅；数据库关闭、重开后仍可读取。相同 event 不重复写入，也不能用另一后果改写。

工作区是调用者维护的物理场景/session 命名空间，负责隔离不同房屋。实验用数据集场景标签分组，不作为模型输入或节点查询键；真实系统需要有正确的工作区边界管理。此版本没有解决跨建筑自动识别和重定位。

## 怎样参与未来预测

```text
当前视觉历史 + 选定方向
        ↓ 冻结事件世界模型
      未来特征假设
        ↓ 用当前视觉、方向、未来假设查询
此前 episode 的真实经历库
        ↓ 返回实际“起始状态—执行方向—真实后果”
    历史上下文 + 记忆增强的未来特征
```

`EpisodeMemorySession` 默认 `scope="cross_episode"`：只读当前 episode 开始前已存在的记录，排除当前 episode。`scope="all_past"` 还允许当前回合中已经真实完成的更早转移。切换 episode 只清理未完成的预测请求，不删除场景记忆。

`predict_with_scene_memory` 接已有 `EventFeaturePredictor`；`predict_cached` 用于严格控制的离线重放。返回值分别保留基础模型预测、记忆增强预测、历史来源和融合权重。`history_context()` 提供可供后续模型读取的类型明确的三元组：

| 字段 | 形状/用途 |
|---|---|
| source_features | K×1152，过去实际开始状态 |
| executed_directions | K×2，过去实际所走方向 |
| observed_future_features | K×1152，过去实际后果 |
| record_sequences | 可见性顺序，必须不晚于当前快照 |
| episode_ids_for_audit | 来源核验，不作为神经模型的封闭类别输入 |

不同 episode 的帧没有被伪装成一段连续视频。第一版用固定检索/融合规则验证历史价值，尚未专门训练跨 episode 注意力读取器、场景 adapter 或 LoRA。现有基础模型的神经参数保持冻结。

检索不用官方 point ID、邻居列表或真值坐标。规则固定为 0.5×起始视觉相似度＋0.3×预测/真实后果相似度＋0.2×方向相似度；起始相似度至少 0.5、方向差最多35°，取3条。融合权重上限0.75，随视觉/方向相似度减小。这些是首轮未调优规则，命中相似经历不等于已经确认拓扑同一地点。

## 写入的因果顺序

必须先预测，真实执行后评分，再调用 `observe_real` 写入真实后果。预测向量不写入真实观察库。接口要求实际执行方向：若控制器走了别的出口，记录归到实际方向，不把后果绑定到原计划。

写入序号表示“结果已经确认并可见”的顺序，不冒充跨 reset 的仿真时间。异步结果不能用旧预测请求写进新 episode。特征契约变化时拒绝混用原库，避免视觉编码器/标准化漂移。

```python
bank = PersistentSceneMemory(memory_path, contract_id=fixed_feature_contract_id)
session = EpisodeMemorySession(bank, scope="cross_episode")
session.begin_episode("workspace-A", unique_episode_id, original_instruction)

prediction = predict_with_scene_memory(
    predictor, session, raw_visual_history, valid_history, selected_direction,
    event_id=event_id, source_observation=current_observation_ref,
)
cross_episode_history = prediction.history_context()
future_feature = prediction.future_feature

# 执行并实际观察后，将真实图像用同一 encoder 和 predictor.normalize 转成特征。
session.observe_real(
    prediction.prediction_id, actual_normalized_future_feature,
    target_observation=actual_observation_ref, execution_ref=actual_execution_ref,
    executed_branch_direction=actual_taken_direction, executed=True, arrived=True,
)
session.end_episode()
bank.close()
# 下一 episode 可重新打开同一 SQLite 文件，继续读取已发生的历史。
```

## 已有验证结果

最终产物：`output/world_model_training/cross_episode_memory_v1_20261104_04/`。用已有开发数据按原始 episode 记录顺序重放：18 个 episode、6 个场景、44 次转移，评估3个原始世界模型 checkpoint，没有新训练或实际导航。

| 历史可见范围 | 离线未来特征 R@1（三 checkpoint 均值） |
|---|---:|
| 无额外历史 | 22.73% |
| 仅当前 episode 已发生的历史 | 19.70% |
| 仅此前 episode 的真实历史 | **40.91%** |
| 当前及此前 episode 的真实历史 | 40.15% |

严格跨 episode 模式每个 checkpoint 有26次查询读到旧回合记录。首个 episode 的13次查询没有历史，输出保持基础模型预测。随后31次查询的未来特征平均余弦由0.4495升到0.5850。

收益主要来自已走过的区域/分支：16次先前已见起点/出口组合的 R@1 从20.83%升到70.83%。27次后续查询的当前图像哈希此前未出现，这组从16.05%升到38.27%。这不是大幅换朝向重定位的完整证明。

独立新路线仅5次，R@1从33.33%到20.00%；陌生路线未显示收益。同路线不同指令的 episode 较多，不能把本实验说成稳定的新路线泛化或导航成功率改善。复制当前特征的参考 R@1 为27.27%、宏余弦0.5456，跨 episode 方案也没有在所有指标上超过该参考。

所有未来图库只用于离线评分，不属于在线可读记忆。每次先预测，再观察、评分和写入；逐回合关闭重开数据库，逐条检查序号与来源隔离。报告保留冷启动和记忆增强的不同评价条件。

## 产物与检查

- `episode_history.json`：已完成重放后的全历史摘要，18个episode、44条真实转移。用于查看；运行时不能把完整重放后的库当作第一回合可见历史。
- `history_context_example.npz/json`：一次预测当时确实可见的跨 episode 历史特征及来源，不含该查询的未来。
- `*.sqlite`：各对照/检查点的真实经历库。
- `*_audit.jsonl`：每次预测所读历史、可见序号、结果到达后写入的序号。
- `results.json`、`subgroup_summary.json`、`report.md`：整体和分组结果。
- `input_manifest.json`、`source_snapshot/`、`verification.json`：来源和复核证据。

44项相关 CPU 测试通过，覆盖持久化、场景隔离、序号与episode可见性、禁止想象结果、实际/计划分支不一致、预测器确实使用旧回合历史以及参数未更新；新增源码 Ruff 通过。未修改原始采集记录、首轮模型权重或 LightNav 服务代码。

重放命令（输出目录必须不存在）：

```bash
.venv/bin/python scripts/eval_cross_episode_memory.py \
  --run-dir output/world_model_training/event_latent_v1_20261104_01 \
  --output output/world_model_training/cross_episode_memory_new_run
```

后续在同一日期完成了可学习的历史读取/校准与真实基座接口冒烟，见
[world_model_optimization_v3_20261104.md](world_model_optimization_v3_20261104.md)。
本文40.91%仍表示原版固定规则重放结果，不替换成后续成绩。新接口的记忆契约还绑定实际标准化均值/尺度，旧数据库不会被静默混用或覆盖。
