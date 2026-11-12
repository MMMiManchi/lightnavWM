# 下一步只做一件事：把同一次回放的每一步记下来

## 四个假设

假设 A，停止时机是主要瓶颈：现有实验不支持把它放在前面。特权 STOP 的净新增成功是 6，而且 6 条都已经走进过 3 m。另外 18 条失败的最近测地距离大于 3 m，改变停止时刻不能让它们成功。诊断集上 +0.20 的 SR 来自抽样比例，全量 Go2-W 里对应的上界大约是 91/1839。不优先做终止网络。

假设 B，方向或路线选择是主要瓶颈：整段换成最短路可以恢复 24/24 条失败，说明这 30 条存在一条能走通的路。这不能证明模型曾经生成过这条路、只是排序选错了。模型每步只解出一个 RVQ 航点，日志里没有候选集合。第一步里，Pointing 哨兵和航向符号一致，所以至少这 20 条不是“点对了、RVQ 解反了”。24 条失败里有 11 条第一步航向差小于 20°，偏离发生在后面，现有日志停在第一步。

假设 C，错误发生在运动执行：没有数据。速度指令由航点的前进和航向直接换算，横向分量被丢掉，这是已知的独轮映射。专家组用同一套映射走通了 30 条，所以映射本身能执行正确航点。模型指令和车体实际转角是否一致，没有逐步位姿，不能下结论。

假设 D，这 30 条因为机身几何不可达：不成立。专家组用的就是 Go2-W 配置，半径 0.215 m、高度 0.50 m，30/30 由 Habitat 记成功。这不改变另外 90 条全程测地距离无穷的结论，那些 episode 不在这次允许列表里。

## 选定实验

研究假设：18 条从未进入 3 m 的失败里，模型航点与最短路航向的明显分歧，要么在某一步已经出现且车体跟上了该指令，要么车体转角和指令不一致。前者说明下一步该改决策，后者说明下一步该查执行。

不训练新网络，不加载 LoRA，不重跑已经完成的三组汇总。新增一个只记录的客户端模式，仍用 `checkpoints/LightNav-0` 和 Go2-W 配置，允许列表可以先用这 30 条。

要改的文件：

- `experiments/r2r_diagnosis/probe_client.py`：增加 `trace` 模式。每步在执行前写下测地距离、模型航点、RVQ、原始文本、专家航点；执行后写下位姿增量。专家航点只作对照，不执行。
- `experiments/r2r_diagnosis/serve_probe.py`：在 `info` 里增加执行前后的平面位置和航向。现在的服务没有把位姿发给客户端。

数据：`experiments/r2r_diagnosis/probe_episodes.json`，模型 `checkpoints/LightNav-0`。输出写到新文件 `output/r2r_probe/trace.jsonl`，不覆盖 `model.jsonl`。

运行命令，确认显卡 2 空闲后再开：

```bash
cd /home/szgx/LightNav-0
CUDA_VISIBLE_DEVICES=2 /home/szgx/miniconda3/envs/habitat/bin/python \
  experiments/r2r_diagnosis/serve_probe.py \
  --allow experiments/r2r_diagnosis/probe_episodes.json \
  --port 5671 --gpu-id 0 \
  --ready-file output/r2r_probe/trace.ready
```

另一个终端：

```bash
cd /home/szgx/LightNav-0
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/r2r_diagnosis/probe_client.py \
  --mode trace --server tcp://127.0.0.1:5671 \
  --ready output/r2r_probe/trace.ready \
  --backend vllm_local --gpu-mem 0.45 \
  --classes experiments/r2r_diagnosis/probe_episodes.json \
  --out output/r2r_probe/trace.jsonl
```

每步字段：`scene`、`episode_id`、`step`、`geodesic_m`、`model_forward_m`、`model_yaw_rad`、`rvq_codes`、`apos_id`、`opos_id`、`expert_yaw_rad`、`pose_xy_before`、`pose_yaw_before`、`pose_xy_after`、`pose_yaw_after`。回合结束仍记录 Habitat 的 `success`、`spl`、`ndtw`、`distance_to_goal`。

继续做决策修正的条件：在 18 条从未进圈的失败里，至少一半存在某一步，模型航向和最短路航向差超过 45°，并且该步实际航向变化与指令航向同号、差值在 15° 以内。

否决决策修正、改查执行的条件：这些失败步上，指令航向和实际航向变化经常反号或差 30° 以上。

若两种情况都很少，分歧既不明显、执行也跟上了，就不要接着加停止头或路线评分器，先把轨迹和指令的对应关系再查一层。

## 和已有工作的关系

这次记录实验本身不是新方法，没有创新点要主张。它是为了避免在日志不够时加模块。

如果记录之后仍想做“发现走偏再改一条路”，下面这些工作已经覆盖了相近机制。公开页面核对于 2026-10-20。

| 工作 | 链接 | 状态 | 和可能的后续方案哪里重叠 |
|---|---|---|---|
| LightNav-0 | https://arxiv.org/abs/2608.30935 | arXiv:2608.30935，技术报告，未见到会议接收页 | 已经输出双通道 Pointing 和 RVQ 航点。再做一个 Pointing 头会和现有接口重合。 |
| BudVLN | https://arxiv.org/abs/2602.06356 | arXiv 预印本，2026-02-06，未核对到正式发表页 | 用测地距离这个特权信号做反事实回锚，专门处理走偏。和“用最短路纠正当前状态”重叠很大。 |
| IDEAL-VLN | https://proceedings.mlr.press/v306/xin26a.html | ICML 2026，PMLR 306 | 按测地距离调节专家介入概率，已经是正式发表的干预策略。 |
| SeekVLN | https://arxiv.org/abs/2609.37353 | arXiv 预印本，页面未写会议接收 | 同一状态上比较“先寻找证据”和“直接走”的后续收益。反事实比较这个形式已经被用过。 |
| GroundingVLN | https://arxiv.org/abs/2609.18581 | arXiv 预印本，代码页写明评审后放出 | 把推理锚到图像点，再交给几何规划。若下一步是改 Pointing 而不是改执行，重叠在这里。 |

因此现在不能声称“基于执行反馈的最小纠错”是一个已经站得住的新机制。BudVLN 和 IDEAL-VLN 已经用测地距离做纠偏，SeekVLN 已经做同状态反事实比较。在逐步日志把错误定位到决策或执行之前，不开始新网络。
