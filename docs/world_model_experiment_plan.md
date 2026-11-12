# 世界模型实验计划

日期：2026-10-23。本文件定义对照、判据和结果格式。正式训练不运行。已验收试点 `output/world_model_data/pilot_20261023_train3_r2` 只用于接口和单帧诊断，不用于模型选择。

编码器契约是 `wm-feature-cache/0.2`，见 `docs/world_model_feature_contract.md`。`feature_layer_selected` 保持 null。

代码入口：

- `experiments/world_model/latent_data.py`：读取已验收运行、场景划分、监督样本
- `experiments/world_model/baselines.py`：基线 A 和 B
- `experiments/world_model/predictor.py`：基线 C 的残差网络和分开的损失
- `experiments/world_model/feature_cache.py`：缓存契约检查，不写正式缓存
- `tests/world_model/test_latent_v1.py`：划分和预测器的 CPU 测试

## 1. 实验

### E0 契约与划分门

对照：没有模型。检查字段、泄漏、预算截断和场景划分。

成功判据：`tests/world_model/test_latent_v1.py` 通过；试点上的 `require_formal_split` 拒绝把它当正式训练集；预算步的 `model_stop` 为 null。

数据量：使用已有的 60 条完整 transition。不新增采集。

```bash
.venv/bin/python -m pytest tests/world_model/test_latent_v1.py tests/world_model/test_collection_contract.py -q
```

### E1 正式数据是否够用

场景清单 `experiments/world_model/splits/wm_scene_split_0.1.json` 已经达到下限：`train_fit` 53 个场景、2977 个路线组，`train_dev` 8 个场景、626 个路线组。这是元数据。这些场景的 transition 还没有采集。

E2 和 E3 在正式采集通过独立验收之前不运行。试点、val_seen、val_unseen 和 test 不能补进训练集。路线组哈希不是视觉划分。

### E2 基线

在通过 E1 的正式 `train_dev` 上计算，不更新参数。两个特征臂都要报告。

- A：`copy_latent(z_t)`，与 `z_t1` 比 MSE 和余弦距离
- B：`kinematic_delta`，与机体位姿增量比平移（米）和 wrap 后的航向（弧度、度）

A 只和潜在状态比，B 只和位姿比。B 使用命令时间步，是近似积分。报告按场景、路线组和 `action_kind` 给出，并写明 `wm-feature-cache/0.2` 的编码器哈希。

### E3 两个候选特征的单步模型

对照是 A 和 B，以及两个臂彼此。模型是 `ResidualLatentPredictor`，输入已提交物理控制 `(v_mps, w_rad_s, dt_command_s)`。损失保持 `latent_mse`、`translation_huber`、`yaw_cosine` 三项，分开记录，不在模块里加总。

| 臂 | 特征 | 结构 | 参数量 |
|---|---|---|---:|
| 池化 | `pooler_output`，1152 维 | `ResidualLatentPredictor(latent_dim=1152, hidden=64)` | 153,219 |
| token | `last_hidden_state`，729×1152 | 同一个残差块按 patch 共享权重，损失对 729 个 token 取平均 | 153,219 |

不把 token 展成 839,808 维再送进这个 MLP。那样约有 `129 × 839808 + 4611 = 108,339,843` 个参数，比的是容量。两个臂的优化器、学习率、按 transition 计数的 batch 和步数在开跑前写成同一份配置。这些数现在还没有选定。

单帧缓存体积是 4,608 字节和 3,359,232 字节，不含元数据。正式存储、压缩和索引还会增加开销。

成功判据分三项写：潜在状态是否低于 A，位姿是否低于 B，以及第 4 节的局部检索探针。`latent_mse` 更低不把 `feature_layer_selected` 写成该臂，也不表示导航更好。权重只在 `train_fit` 上更新。失败时保留分组误差。不用特权目标距离筛选样本。

### E4 预测引导记忆

前提：E3 的分组结果已经写明两个臂分别相对 A 和 B 的差。对照是用 `z_t` 检索、用 `z_hat_t1` 检索、最近帧和随机历史。oracle 查询 `z_t1` 只用于离线评分。

检索限制在同一 episode。不读目标点。Recall@k、MRR，以及检索结果是否更接近未来观测，分开报告。这一阶段不改导航输出。

局部空间探针只用 `train_dev`：用池化向量或 patch 特征检索同一 episode 的历史观测，记录最近邻是否落在预先写死的空间邻近范围内。阈值在开跑前固定。它回答表示是否还分得开局部位置，不回答导航是否成功。

### E5 导航闭环

前提：E4 的检索结果经过可关闭适配层进入 LightNav 现有视频历史。对照是适配层关闭时的原策略。

成功判据才包括 SR、SPL、NE、nDTW，并且测试场景不在训练和模型选择里出现。v1 不实现这一实验。

## 2. 结果记录

每个实验一个新目录，已存在则拒绝覆盖。建议 `output/world_model_runs/<experiment>_<utc>/result.json`，至少包含：

```json
{
  "experiment": "E3",
  "eligible_formal_split": false,
  "scene_split_version": "wm-scene-split/0.1",
  "feature_cache_version": "wm-feature-cache/0.2",
  "feature_layer_selected": null,
  "train_fit_scenes": 0,
  "train_fit_route_groups": 0,
  "train_dev_scenes": 0,
  "train_dev_route_groups": 0,
  "rows": 0,
  "seed_status": "partial",
  "git_commit": "3015508b70fb6e30bbd66701e94c9b916dcb8df8",
  "encoder": {
    "id": "google/siglip-so400m-patch14-384",
    "revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    "frozen": true
  },
  "checkpoint": null,
  "metrics_by_scene": [],
  "metrics_by_route_group": [],
  "baseline_delta": {"latent_vs_A": null, "pose_vs_B": null},
  "counts": {
    "missing_observation": 0,
    "terminal_observation": 0,
    "runner_budget": 0,
    "model_stop_true": 0,
    "server_override": 0,
    "quarantined": 0
  },
  "navigation_metrics": null
}
```

`navigation_metrics` 在 E5 之前保持 null。`feature_layer_selected` 在比较实验完成之前保持 null。没有正式 transition 时，不得填写声称模型有效的 `baseline_delta`。

## 3. 现在不执行的事

编码器身份和预处理已经固定。输出层没有选定。当前可以运行的是 CPU 契约测试，不是 E2 或 E3。

正式采集开始之前的三项验收分别写在 `docs/world_model_collection_gate.md`、`docs/world_model_cache_integrity.md` 和 `docs/world_model_reproducibility.md`。这三份记录的是 mock 测试和诊断前向。通过它们只表示可以进入正式采集启动评审。本计划仍然不启动采集、不写正式特征缓存、不训练预测器。
