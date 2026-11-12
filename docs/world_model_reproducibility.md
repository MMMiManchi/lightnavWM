# 世界模型跨进程特征稳定性

日期：2026-10-23。`feature_layer_selected` 保持 null。这次比较只使用已有 RGB，输出只在诊断目录，并且标记 `diagnostic_only`。这些帧没有加入正式训练数据。

结论：在预先规定的数值标准下，同一张 GPU、同一软件环境中的两个进程达到了稳定。跨 GPU 没有测。这只允许进入正式采集启动评审。

## 1. 预先规定的容差

容差写在 `experiments/world_model/siglip_repro_probe.py`，并被 `tests/world_model/test_siglip_repro.py` 在 GPU 测量之前锁住：

| 量 | 通过标准 |
|---|---:|
| 最大绝对误差 | 0 |
| 均方误差 | 0 |
| 余弦相似度 | 1 |
| 形状、dtype、有限值 | 两次输出完全一致，dtype 为 float32 |

依据是同一软件栈上同进程重复前向曾经逐位相同，而这次没有另外的噪声模型。因此同一张 GPU 上的两个进程也按逐位相同验收。`1e-5` 只作为以后若出现非零误差时的评审候选，不参与本次是否通过的判断。测试用 `1e-8` 的扰动确认：只要最大绝对误差非零，函数就返回失败。

两个数组逐元素相等时，余弦相似度记为 1。最大绝对误差和均方误差来自差值本身。

## 2. 这次测量

命令：

```bash
.venv/bin/python -m experiments.world_model.siglip_repro_probe --gpu 2
```

退出码 0。耗时约 37 秒。父进程不占用 GPU。每个子进程由脚本把 `CUDA_VISIBLE_DEVICES` 设为 `2`，因此 PyTorch 只看得到这一张卡，设备序号是 0。四个子进程的 pid 分别是 1158835、1158987、1159131、1159279。

诊断目录：`output/world_model_diagnostics/siglip_repro_20261023T093701Z/`。`repro_report.json` 中 `diagnostic_only` 为 true，`formal_feature_cache`、`formal_collection`、`training` 和 `used_as_formal_sample` 都是 false，`feature_layer_selected` 是 null。缓存检查器会拒绝来自 `world_model_diagnostics` 或带 `diagnostic_only` 的行。

两张图都来自已有的 pilot 观测，只作诊断：

| 场景 | 文件 sha256 |
|---|---|
| `5LpN3gDmAk7` | `37a66987b9b151393c4e1efe74750e920942cc16a490e49d67013f76417f54d1` |
| `7y3sRwLe3Va` | `c7ccdabea55552d853085be6fdbb753e2d758b55c1dbdf2f669ebcb8f500a3ff` |

每个场景由两个独立进程加载同一 snapshot、同一权重和 `experiments/world_model/contracts/siglip_so400m_patch14_384_preprocessor.json`。子进程重新计算的权重 sha256 是 `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`，预处理 sha256 是 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`。模型类是 `SiglipVisionModel`，`local_files_only=True`，并设置了 `HF_HUB_OFFLINE=1`。

| 场景 | 输出 | 形状 | dtype | 有限 | 最大绝对误差 | 均方误差 | 余弦相似度 |
|---|---|---|---|---|---:|---:|---:|
| `5LpN3gDmAk7` | `pooler_output` | `(1152,)` | float32 | 是 | 0 | 0 | 1 |
| `5LpN3gDmAk7` | `last_hidden_state` | `(729, 1152)` | float32 | 是 | 0 | 0 | 1 |
| `7y3sRwLe3Va` | `pooler_output` | `(1152,)` | float32 | 是 | 0 | 0 | 1 |
| `7y3sRwLe3Va` | `last_hidden_state` | `(729, 1152)` | float32 | 是 | 0 | 0 | 1 |

四组都落在预先规定的标准内。

运行环境，四个进程相同：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB，物理序号 2 |
| PyTorch | `2.10.0+cu128` |
| CUDA | 12.8 |
| Transformers | 5.8.0 |
| `cudnn.deterministic` | false |
| `cudnn.benchmark` | false |
| `matmul.allow_tf32` | false |
| `cudnn.allow_tf32` | true |
| `deterministic_algorithms` | false |

这些确定性开关保持进程默认值，没有为了让结果变成 0 而改写。跨 GPU 一致性没有测。

## 3. 三项结论

1. 采集器已经被证明只消费合法 train 清单。证据在 `docs/world_model_collection_gate.md`。
2. 缓存写入器已经被证明能在写入前阻止不合规记录。证据在 `docs/world_model_cache_integrity.md`。
3. 跨进程特征达到了预先规定的数值稳定性标准。范围是同一张物理 GPU 和同一软件环境。

三项都通过，因此允许进入正式采集启动评审。评审通过之前不启动正式采集，不写正式特征缓存，不训练预测器。即便进入评审，下一步仍应先做一次受限的小批量采集，确认实际执行记录、图像哈希、特征契约和 split 元数据可以闭环，再决定是否扩大数据规模。

## 4. 命令、退出码和测试数量

在 `/home/szgx/LightNav-0` 执行。

| 命令 | 退出码 | 结果 |
|---|---:|---|
| `.venv/bin/ruff check` 本轮改动的采集、缓存和复现文件 | 0 | All checks passed |
| `.venv/bin/ruff check .` | 1 | 13 个错误，全部位于既有的 `experiments/robustness_study/`、`mujoco_demo/tests/`、`scripts/diagnostics/` 和 `tests/test_fp8_quant.py`。本轮改动的文件不在其中，没有顺手修改这些文件 |
| `.venv/bin/pytest -q tests/world_model` | 0 | 41 passed |
| `.venv/bin/python -m experiments.world_model.siglip_repro_probe --gpu 2` | 0 | `status=passed` |

41 个测试包含本轮新增的清单门禁、写入前失败、真实权重哈希和容差锁定，也包含原先的世界模型测试。其中官方清单测试和真实权重测试各有 1 个，已经包含在这 41 个里。

没有执行：真实 Habitat 采集、正式特征缓存、预测器训练、权重下载、跨 GPU 复现。
