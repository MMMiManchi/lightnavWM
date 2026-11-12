# 下一阶段最小实验

日期：2026-10-20。不训练，不加载 `checkpoints/LightNav-0-go2w`，不覆盖 `output/r2r_official` 和 `output/r2r_go2w`。特权最短路径和特权 STOP 不进入结果表。

## 已经完成的测量

开环图像旋转已经跑完，命令是：

```bash
cd /home/szgx/LightNav-0
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/robustness_study/openloop_image_rotation.py
```

当时 0 号和 1 号卡各有一个 `gavln-venv` 进程。2–7 号空闲，实验用了 2 号。结果在 `openloop_results.jsonl` 和 `openloop_summary.json`。它回答了「同一 JPEG 是否稳定」和「图像平面 ±0.5°/±1° 会不会改第一步航点」。它没有回答闭环 SR。

不要把这次图像旋转写成相机外参实验。`frame_preprocessing.py` 和两份 yaml 都没有方向噪声。同事的 ±1° 实现仍然未知。

## 还缺的材料

向同事要三次运行的逐回合成功表和扰动脚本。没有逐条表，就不能报告成功变失败、失败变成功的条数，也不能把 19.8% 的分母写成 1839。本地官方 SR 是 67.70%，笔记是 67.54%，两者接近，但是两次运行。

## 下一件只做闭环记录

目的：看图像平面 ±1° 在闭环里第一次改变可执行航点的步号，以及这 30 条的成功标志是否翻转。执行的动作来自原图解码。扰动解码只写日志。这样测的是 B 会不会在后续步出现，以及如果当时改用扰动动作，至少第一步会不会和成功轨迹不同。不在这一轮改动作。

数据：`experiments/r2r_diagnosis/probe_episodes.json` 的 30 条。模型：`checkpoints/LightNav-0`。配置：`habitat_server/configs/vlnce_r2r_go2w.yaml`。输出目录新建 `output/r2r_probe/perturb_trace/`，不要写进 `trace_20261020`。

实现限制：新逻辑放在 `experiments/robustness_study/` 的诊断客户端里，调用现有 `TrajVocabVLNCEPolicy`。不改 `src/lightnav/habitat/runner.py` 的正式评测。每步对当前 RGB 做与开环脚本相同的中心旋转（±1°，线性插值，边缘复制），`policy.reset` 不能跨视角串历史。原图那次的历史照常累积并用于执行。扰动视角用当时的历史帧加上旋转后的当前帧，单独推一次后丢弃这次 `observe` 对主缓冲的影响。实现时要先把主策略的缓冲拷出来，避免一次额外 `act` 把旋转帧写进正式历史。

每步至少写：`scene`、`episode_id`、`step`、原图 `raw_text` 和航点、两个旋转的 `raw_text` 和航点、`|\Delta\psi|`、`|\Delta f|`、执行前后位姿、测地距离（只作标签）、`infer_s`、两次推理的耗时和。回合结束写 Habitat 的 `success`、`spl`、`ndtw`、`distance_to_goal`。

建议命令，先看 `nvidia-smi`，选一张空闲卡，把 `CUDA_VISIBLE_DEVICES` 设成那张卡。下面用 2 号作为占位，卡上有作业时换掉：

```bash
cd /home/szgx/LightNav-0
CUDA_VISIBLE_DEVICES=2 /home/szgx/miniconda3/envs/habitat/bin/python \
  experiments/r2r_diagnosis/serve_probe.py \
  --allow experiments/r2r_diagnosis/probe_episodes.json \
  --port 5672 --gpu-id 0 \
  --ready-file output/r2r_probe/perturb_trace/ready
```

客户端在诊断脚本落地后使用同一张卡的另一份进程设置，模型仍然是 `checkpoints/LightNav-0`，`gpu_memory_utilization` 不超过 0.45。30 条闭环原先大约每步 0.14 s，加上两次额外前向后，粗算是 1 小时量级，而不是全量 1839 的一天。先跑这 30 条。

对照组：

| 组 | 做什么 | 不做什么 |
|---|---|---|
| 已有 `trace_20261020` | 原图闭环 | 不重跑 |
| 新的 perturb trace | 每步多推 ±1°，执行原图 | 不改动作，不训练 |
| 开环 `openloop_results.jsonl` | 第 0 步 JPEG | 不和闭环 SR 混成一张表 |

指标同时给两次视角下的逐步差，以及原图执行的 SR、SPL、NE、nDTW。原图执行的 SR 应接近已有 6/30。若差很多，先查随机种子和历史缓冲被污染，不解释成扰动效应。扰动动作没有被执行，所以不要给扰动条件单独报一个 SR。

另外报告：第一步动作一致率、第一次 \(|\Delta\psi|>0.05\) rad 或 \(|\Delta f|>0.05\) m 的步号直方图、这类步在成功和失败回合中的比例、每步额外耗时、显存。

## 停止条件

满足任一条就停止，不再加模块：

1. 30 条里，可执行航点被 ±1° 改掉的步，在成功回合和失败回合中的比例接近。检测器分不开成败。
2. 第一次分歧的中位步号大于 20，而且第 0 步几乎不分歧。那说明同事的 19.8% 更像长期分叉或他们的扰动不是图像旋转。在看到他们的逐条表之前，不做投票器。
3. 若把「扰动视角一致则替换原动作」套到已有开环的 `2azQ1b91cZZ/10` 上，成功第一步会被改成右转。这条规则不再进入后续消融。

继续做无训练纠错的条件：失败回合里，早期（前 10 步）世界系动作分歧的比例明显高于成功回合，并且保留原动作、只缩短步长的离线重放能在至少 3 条失败上减小最终测地距离，同时不把 6 条成功改成失败。离线重放不能用测地距离当输入，只能用它当评分。

## 明确不做

- 全量 1839 的第三次抖动评测。30 条若已经分不开成败，全量只是放大同一个空结果。
- 把 76.6% 写成方法的 SR。
- 训练一致性分类器。标签要等闭环分歧和成败的关系出来。
- 接 StereoNav 的立体编码器和目标点渲染。目标坐标会改变 R2R 的任务定义。
- 接 GC-VLN 的约束求解和回溯。
- 在仿真里做候选 3 的执行纠错。跟随误差已经约是 0。
