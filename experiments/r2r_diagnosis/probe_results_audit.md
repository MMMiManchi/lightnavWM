# 30 条诊断集三组结果核对

核对时间：2026-10-20。只读已有日志，没有重跑导航、没有训练、没有加载 `checkpoints/LightNav-0-go2w`。

## 文件和启动方式

| 组 | 结果 | 客户端日志 | 仿真日志 |
|---|---|---|---|
| 原模型回放 | `output/r2r_probe/model.jsonl` | `output/r2r_probe/logs/client_model.log` | `output/r2r_probe/logs/serve_model.log` |
| 特权最短路径 | `output/r2r_probe/expert.jsonl` | `output/r2r_probe/logs/client_expert.log` | `output/r2r_probe/logs/serve_expert.log` |
| 特权 STOP | `output/r2r_probe/oracle_stop.jsonl` | `output/r2r_probe/logs/client_oracle_stop.log` | `output/r2r_probe/logs/serve_oracle_stop.log` |

启动脚本是 `experiments/r2r_diagnosis/run_probe.sh`。仿真端 `experiments/r2r_diagnosis/serve_probe.py` 使用 `habitat_server/configs/vlnce_r2r_go2w.yaml`（机身高 0.50 m、半径 0.215 m，相机高 0.50 m、480×270、水平视场 120°）。客户端 `experiments/r2r_diagnosis/probe_client.py`。模型是 `checkpoints/LightNav-0`，后端 `vllm_local`，`gpu_memory_utilization=0.45`。物理显卡是 2（`CUDA_VISIBLE_DEVICES=2`，进程内 `gpu-id 0`），端口 5671。允许列表是 `experiments/r2r_diagnosis/probe_episodes.json`，30 条。

三组都正常结束。服务日志分别在 07:12:25、07:24:11、07:34:24（UTC）写了 `Habitat environment closed`。六个日志里没有 `Traceback`。每组 jsonl 30 行，重复键 0。三组的 `(scene, episode_id)` 集合相同。

## 评测定义

成功来自 Habitat 的 `Success` 度量，源码在 habitat-lab `habitat/tasks/nav/nav.py`。条件是两件同时成立：任务里 `is_stop_called` 为真，并且 `DistanceToGoal` 小于 `success_distance`。本环境把这个半径设为 3.0 m（`VLNCEEnv` 默认参数）。比较是严格小于，不是小于等于。客户端把 `info["success"] >= 0.5` 记成成功，没有用距离自己再判一次。`vlnce.py` 里写明不覆盖 `info["success"]`。

`DistanceToGoal` 的默认 `distance_to` 是 `POINT`，用 `sim.geodesic_distance` 到目标点。结果里的 `final_distance_m` 和 `min_distance_m` 都是这条测地距离，不是欧氏距离。

SPL 也是 Habitat 的 `spl`。参考长度是回合开始时的测地距离。实际长度是智能体每步位移的欧氏距离之和。这两项都没有单独写进 jsonl，所以平均路径长度记为未记录。专家组的 SPL 0.997 是这条公式在真实执行轨迹上的输出。控制器每步执行的是导航网格最短路上的航点，所以执行轨迹接近那条测地路径，SPL 才接近 1。它不是把最短路径长度直接抄成 SPL。

nDTW 是环境注入的 Habitat `NDTW` 度量，结果字段 `ndtw`。

特权 STOP 在动作前读当前测地距离，小于 3 m 就发零速度停止。停止之后的成功、SPL、nDTW、最终距离仍由 Habitat 计算。30 条里没有“成功标志为真但最终测地距离 ≥ 3 m”的记录。曾经进入 3 m 没有被当成成功；成功要求停止被调用，并且停下时的测地距离小于 3 m。

`stop_trigger_step` 没有单独落盘。`oracle_fired` 为真时，停止动作是该回合日志里的最后一条，0 起点的序号等于 `steps - 1`。这个值写在 CSV 的 `stop_trigger_step`，来源列是 `derived_last_logged_action`。没有触发时该列为空。平均逐步推理延迟没有记录。模型组和 STOP 组只在第一步记了 `first_step.infer_s`，专家组没有推理。

## 和 10 月 15 日 Go2-W 基线的对照

基线文件是 `output/r2r_go2w/results.jsonl`。30 条都能对上。成功标志 30/30 一致。最终测地距离的绝对差中位数是 0 m，最大 0.058 m。`2azQ1b91cZZ/116` 回放最终距离 13.10981559753418 m，步数 121，与基线相同。

## 总体指标

诊断集故意多放了失败样本，下面的 SR 不能当成 R2R val_unseen 的成功率。

| 指标 | 原始 LightNav-0 | 特权最短路径 | 特权 STOP |
|---|---:|---:|---:|
| Episode 总数 | 30 | 30 | 30 |
| 成功数 | 6 | 30 | 12 |
| 失败数 | 24 | 0 | 18 |
| SR | 0.200 | 1.000 | 0.400 |
| SPL 均值 | 0.1777 | 0.9971 | 0.3947 |
| NE 均值（测地距离，m） | 8.899 | 0.891 | 8.523 |
| nDTW | 0.363 | 0.924 | 0.440 |
| 平均路径长度 | 未记录 | 未记录 | 未记录 |
| 平均决策次数 | 108.6 | 35.6 | 92.4 |
| 平均推理延迟 | 未记录 | 未记录 | 未记录 |
| 第一步推理耗时均值 | 0.100 s（30 条） | 未记录 | 0.102 s（30 条） |

| 干预组 | 原模型失败且干预后成功 | 原模型成功但干预后失败 | 净新增成功数 | 30 条诊断集 SR 变化 |
|---|---:|---:|---:|---:|
| 特权最短路径 | 24 | 0 | 24 | +0.800 |
| 特权 STOP | 6 | 0 | 6 | +0.200 |

逐条表在 `experiments/r2r_diagnosis/paired_episode_comparison.csv`。分类汇总在 `experiments/r2r_diagnosis/failure_type_comparison.csv`。重算脚本是 `experiments/r2r_diagnosis/analyze_probe.py`。
