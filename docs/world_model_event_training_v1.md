# 事件级未来潜在特征模型：首轮八卡实验

> 2026-11-06 架构复核更正：本页历史实验名中的 `tssm_branch` / “TSSM”
> 对应 `event_latent.py` 的历史 Transformer + 条件潜变量单目标预测器。
> 它没有持续的潜在状态转移及真实观测 posterior 到下一次转移的反馈，
> 因而不能视为完整的状态化 TSSM。保留历史名称、路径和数值以便追溯。
> 新的状态化实现为 `event_tssm.py`，验证见
> [2026-11-06 报告](/home/szgx/LightNav-0/docs/world_model_event_tssm_20261106.md)。

2026-11-04，用户明确授权开始世界模型训练，8 卡均可用。此轮是冻结视觉表示上的事件级预测实验，不是节点分类，也没有接入导航或启动 Habitat。

## 固定的第一版契约

- 数据来源：通过原有 HMAC/不可变像素准入的 8 个正式 fit/dev run。第四版官方节点 sidecar 仅作离线标签；未经正式逐步执行准入的房屋探针不混入首轮训练。
- 单位：同一真实 episode 内，从当前事件到下一不同事件首次进入帧。禁止跨 reset；同地点抖动、少于 3 步、没有可靠方向匹配或缺图的片段排除。started_inside 可以作为已知当前事件状态，不冒充刚进入。
- 分支标签：只从当前帧已有的角度候选且 depth=not_blocked 的出口中，依据实际离开证据关联所走分支，误差最多 35°。这是离线弱监督；不提供未来命令序列、GT 位姿、节点 ID 或邻接给模型。在线从图像提出出口的能力尚未实现。
- 输入：4 个较早事件的真实观察＋4 个当前/近期 RGB 特征，共 8 token；不足历史使用 mask。再输入选定方向的 sin/cos 两维。历史索引不得超过当前时刻。
- 目标：下一事件真实 RGB 的冻结 `google/siglip-so400m-patch14-384` pooler，1152 维；revision 和权重哈希沿用审核契约。本机没有已审核 DINOv2 权重，因此首轮使用现有 SigLIP 空间，不新增下载或宣称两种 encoder 等价。
- 表征规范：仅用 fit 所引用图像估计通道均值和标准差，标准差下限 0.01，然后 L2 归一化。真实记忆键也必须使用同一规范。目标没有可训练投影层。
- 质量筛查：源像素由现有准入 loader 封存后读取；当前/目标超过半幅近黑或文件字节完全相同的样本排除。无效的早期历史用 mask。该筛查不等于人工语义节点验收。

## 网络与损失

`EventLatentWorldModel`：宽度 256、4 层、8 头、FFN1024 的历史 Transformer；分支 MLP；64 维高斯随机状态；条件 prior、使用真实后果的训练 posterior；1152 维特征预测头。以实际 `model_config.json` 记录参数量。该配置是首轮可复现版本，不声称最优。

部署/评价只调用不接收 target 的 `predict`，使用 prior 均值；`imagine` 可采样多个假设，其不确定性尚未校准。训练同时约束 prior 和 posterior 的未来特征余弦误差，加 0.1 倍多正例对比损失及 0.001 倍 KL（每维 free-bits 0.1）。同一真实目标地点的不同观察是多正例，不强行当负例。

8 个实验并行：3 seed 的带分支 TSSM、相同 3 seed 的无分支 TSSM，以及 1 对确定性带/不带分支基线。冻结编码器先用 8 卡分片提特征，再在每张卡运行一个模型，采用多实验并行而非小数据集上的 DDP。

固定上限 1600 次更新、batch64、AdamW lr=2e-4、weight_decay=0.01、100 步 warmup、梯度裁剪 1。每 100 步按开发场景宏平均 prior 余弦选 checkpoint，连续 6 次评价不改善则停止。按同路线/起终点组合逆频率采样。开发集用于选 checkpoint，不能称为盲测。

## 对照与解读

比较复制当前特征、最近训练经历的后果、无分支模型与打乱分支输入。报告 prior 特征相似度、按场景宏平均、离线真实未来图库的 Recall@1/5、MRR、分支响应和训练项；不以 posterior 重建证明未来预测。

离线图库包含真实后果，属于表征评价工具，不是部署时已经可访问的记忆，不证明实际记忆检索或导航收益。此轮若不能超过无分支/复制/近邻基线，应先分析数据、表示和分支条件，不因 8 卡可用而直接扩大训练。

## 启动与产物

```bash
.venv/bin/python scripts/train_event_world_model.py \
  --config experiments/world_model/configs/event_latent_v1_20261104.json
```

主机须能访问全部 8 张 CUDA GPU。受限环境无法访问设备时使用获准的主机执行，不回退 CPU 训练。

配置固定在 `experiments/world_model/configs/event_latent_v1_20261104.json`。输出 `output/world_model_training/event_latent_v1_20261104_01/` 包含授权来源、HMAC 准入结果、源文件哈希、事件对、已准入像素快照、特征规范、日志、最佳 checkpoint、各基线及结果。原始采集记录和旧实验不改写。

`status.json` 是总状态，`logs/encode_gpuN.log`、`logs/train_gpuN.log` 是每卡日志。失败时记录原因，只终止本启动器创建的子进程。模型成功结束后保存 `results.json`；主进程退出后已完成的结果仍可审查。

底层 `python -m experiments.world_model.event_train` 接受 `--config PATH`、`--phase {prepare,encode,pack,train,summarize}`、`--worker N`；worker 是分片/实验编号，设备由 CUDA_VISIBLE_DEVICES 指定。prepare 创建新目录；已有 incomplete prepare 或已有单模型目录不会被覆盖。

## 首轮已完成

实际准入并筛查后为 614 组训练转移、44 组开发转移，覆盖 40 / 6 场景及 166 / 8 路线组。2,798 张冻结特征已编码。8 组模型全部完成并按预设耐心早停，TSSM 最佳步数为 200–400，确定性模型为 700；没有在看到结果后追加更新。

TSSM 带分支三 seed 的开发场景宏平均余弦为 0.3486，无分支为 0.3487；复制当前特征为 0.5456。离线真实未来图库的 R@1 分别为 22.7%、20.5%、27.3%。没有稳定分支收益，目前不能宣称已经学到有效的分支后果或导航增益。完整结果在实验目录 `report.md`、`results.json`、`review_summary.json`。

可直接加载的推理包保存在 `exports/`，含模型、fit-only 均值和尺度。8 组都已通过重新加载验证：CPU 与 GPU 预测最大绝对差约 2.23e-5、向量余弦大于 0.999999、top1 地点检索一致。复核没有把未来标签输入预测器。

```python
from pathlib import Path
import torch
from experiments.world_model.event_inference import load_event_predictor

predictor = load_event_predictor(
    Path("output/world_model_training/event_latent_v1_20261104_01/exports/tssm_branch_seed20261104.pt"),
    device="cuda",
)
with torch.inference_mode():
    # Raw frozen SigLIP pooler features [B,8,1152], valid mask [B,8], direction [B,2].
    future_feature = predictor(raw_visual_history, valid_history, selected_direction)
    # Real memory observations must use the identical frozen encoder and transform.
    memory_keys = predictor.normalize(raw_memory_features)
```

`exports/*_dev_future_features.npy` 给出每组的 `[44,1152]` 预测样例；这是开发集输出，不是实际机器人记忆。编码器本体按已审核的本地 snapshot 加载，不包含在小型推理包里。训练和推理接口没有官方 ID、世界坐标或未来轨迹输入。

本轮结果的边界：它只测冷启动跨场景预测。固定历史窗口不等于跨回合场景记忆，训练中没有真实到达后的场景专属参数更新，因此不能用本轮结果否定“越走越熟”。后续持续适应应另按 [world_model_continual_learning_protocol_20261104.md](world_model_continual_learning_protocol_20261104.md) 设计 support/query 和四组机制对照，不能直接增加旧数据 epoch 或把开发集并入全局训练。
