# HALO 参考与导航未确认到达的复核

日期：2026-11-09。HALO 源码复核版本为 `35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f`。
本文是代码与日志分析；未运行 HALO、未移植其模型或机器人部署代码。

## 先明确步数和成功定义

用户最终约定：**数据采集最多 1000 步，导航评测最多 500 步**。当前 `run_stateful_harness.py` 的默认值和允许上限均已改为 500，采集脚本保留 1000。
此前接线试跑显式使用 120/300，覆盖了当时默认值；旧配置和结果保留，不能将它们当成 500 或 1000 步评测。

本地 VLN-CE 评估的成功条件是调用 STOP 且参考目标距离小于成功半径，默认半径 3 米。
当前 Harness 另外要求看见目标并确认本次指令的位置关系，因此“进入评测成功半径”和“通过 Harness 到达验证”是两个不同判断。
评测真实距离被隔离，不是控制器输入；这项隔离不应取消，但需要用离线距离和语义标注校准 STOP 验证器。

| 运行 | 最近距离 / 最终距离 | 实际动作 / 步数预算 | 实际结束原因 |
|---|---|---:|---|
| 旧验证，两次 | 4.66 / 5.91 米；4.53 / 6.09 米 | 各 85 / 120 | 错误视觉确认，发出 STOP |
| 修正后第 1 次 | 2.09 / 6.16 米 | 120 / 120 | 曾进入 3 米范围又走出，预算结束 |
| 修正后第 2 次 | 0.90 / 0.90 米 | 120 / 120 | 已接近参考终点，未发 STOP，预算结束 |
| 延长试跑 | 1.60 / 2.61 米 | 161 / 300 | 8 次主动观察耗尽，未确认后续行动，弃权结束 |

延长试跑第 76 个动作后已进入 3 米范围，累计 86 个动作的评估距离在 3 米以内。
按 STOP 提议之前的上一帧评估状态统计，有 **14 次范围内 STOP 提议被拦截，其中 5 次是局部任务 STOP，9 次是原始全局任务 STOP**。局部 STOP 本来就不能直接终止整项任务；对这 9 次全局 STOP，在当前评测规则下放行可满足成功条件。这不是已经实际取得成功，也不证明每个位置都满足“进入电视房间”的语义要求。
0.90 米那次没有证据表明范围内 STOP 被拦截，不能把它和 2.61 米那次混为一谈。

因此，最后一轮不是 300 步耗尽，更不是 1000 步不足。它受独立的主动观察上限和到达判据影响。
“没有确认”不能直接当作“路线从未接近目标”，更不能仅据这些试跑否定世界模型的全部架构。

## 当前链路中可以确定的问题

1. **局部目标没有被充分落实为可跟踪的出口。** `HarnessLightNavExecutor.commit_goal` 保存方向和来源，但实际给 LightNav 的操作主要是替换文字指令。日志中出现把厨房、楼梯、柜旁缝隙当成通往电视房间方向的判断；选择一个候选不保证实际经过该候选。这是结合代码和日志作出的诊断，不是正式误差归因实验。
2. **到达验证缺少在线几何支持。** 深度目前主要用于历史方向对齐；最终验证仍依赖 RGB、少量实际历史和已下发控制。已下发前进量不等于实际位移，画面变化也可能只是转头。
3. **unknown 没有充分的恢复过程。** 最后一轮反复 STOP 后主要补充原地观察，最终用完全局 8 次观察预算。需要区分目标不明、方向对不齐、物理受阻、执行走偏和已到但当前未看见目标，再选择不同恢复动作。
4. **验证规则与评测规则没有分别校准。** 正确拒绝“黑色缺失网格像电视”的误停，不代表范围内所有拒绝都正确。应独立报告到达范围覆盖、STOP 放行/误拦截、最终语义关系，以及正式成功率。

依据：[执行器](/home/szgx/LightNav-0/experiments/world_model/event_runtime.py:143)、[深度使用位置](/home/szgx/LightNav-0/experiments/world_model/stateful_controller.py:66)、[观察预算](/home/szgx/LightNav-0/experiments/world_model/stateful_controller.py:176)、[评测成功定义](/home/szgx/LightNav-0/habitat_server/lightnav_habitat/vlnce.py:451)。

## HALO 最值得参考的部分

### 技能与完成条件集中定义

`skills.yaml` 为技能统一登记模型提示、适用前提、目标种类和完成外观。`pass_through` 明确区分穿过障碍与走到物体前停下，穿过去之后重复整段技能会过冲。
这些完成外观在 HALO 中主要是模型提示，不是通用几何证明。
我们可以把 `cross_exit`、`approach_landmark`、`look`、`backtrack` 等写成技能合同，分别定义成功、失败、可重试和继续当前技能的条件；局部穿门完成不等于全局任务到达。
[HALO 技能登记](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/orchestrator/halo_orchestrator/skills.yaml)。

### 明确的计划状态与恢复动作

HALO 将 `complete_step`、`retry_step`、`insert_step`、`ask_operator` 和 `finish_mission` 分开，计划有版本、修订号和完成/失败/取消等状态；插入恢复步骤后可恢复原任务。
新任务有 epoch，旧异步规划结果会被丢弃。值得借鉴的是任务状态、执行反馈与恢复的结构。
自主 VLN 评测中，可把请求操作员帮助的分支改成重新观察、重定位、回溯或明确弃权。
不能直接照搬其重试上限后失败并推进下一步的策略，否则会跳过本任务要求的路线约束。
[计划状态](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/orchestrator/halo_orchestrator/plan.py)、[动作决策](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/orchestrator/halo_orchestrator/orchestrator.py)、[运行节点](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/system/halo_system/nodes/orchestrator_zmq.py)。

### 执行器反馈和动作衔接

HALO 的 generator 接收当前 RGB、机器人状态与语言技能，生成约 5 秒动作；tracker 跟踪它。下一段生成读取 tracker 尚未执行的参考动作，并补偿推理延迟。
编排层还读姿态、关节速度、计划活动量和位置变化，区分停稳、仍在执行和无进展。这些不是目标真实距离；位置变化也不能被当成必然向目标接近。
我们应给 LightNav 局部选项补充开始/运行/停下/受阻/偏离/完成的执行报告，并用在线可得的里程计或 RGB-D 运动估计核验实际进展。
G1 的关节计划与站立高度阈值不适用于本项目的导航控制，不能直接移植。
[动作生成与 tracker 衔接](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/system/halo_system/nodes/motiongen_zmq.py)。

### 数据生成、失败反馈与质量版本

HALO 将场景生成、动作候选生成、质量检查、放置与渲染分开。其 motion checker 输出逐项值、阻断项和再生成反馈，并用 checker 版本与配置哈希标识判据，要求用好/坏动作及具体时间窗口校准。
适合借鉴到我们的出口数据完整性、实际执行分支、有效穿门、图像缺损和多出口覆盖检查。
[数据流水线](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/datagen/docs/data_pipeline.md)、[质量检查](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/datagen/motion_gen/motion_check/README.md)。

注意其公开流水线主要服务坐椅技能；失败候选的最佳链也可能保留，由 `final_gen_accepted` 区分，不能把文件数等同于通过数量。检查器还有明确的场景/椅子碰撞限制，不能把人体动作检查直接当作导航可通行性验收。
我们要增加的是同路口不同出口、不同来向、不同终点视角和失败恢复经历，不要求都沿任务最短正确路线采集。STOP 验证需要另建接近但未到、穿过门后目标出视野、错误房间、网格缺损等带独立标注的样本。

## 建议的下一次实现顺序

1. 将评测预算固定为 500，保留采集 1000。局部停滞和观察预算独立记录，并在证据不增加时切换恢复策略。
2. 给出口选项增加可跟踪的源画面/出口框/方向锚点和执行状态。不要只更换一条泛化的文字指令。
3. 将 STOP 判定分为已到、明确未到、不确定；为不确定提供有目的的补证据、重定位和回溯，分别校准评测通过与语义正确。
4. 用在线深度和实际运动估计支持相对距离、门口跨越与受阻判断；目标真实坐标/距离继续只用于离线评估。
5. 小规模标注真实完成/失败后再校准规则，不以不断追加默认否定提示替代独立验证。

HALO 的 `FrameRing` 当前只保留最新图像，任务历史也是有限文本记录，不能替代本项目的跨 episode 环境事实库或 TSSM。应借鉴其任务执行层，同时保留我们的环境记忆与真实观测后的场景更新。
[图像上下文](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/orchestrator/halo_orchestrator/context.py)、[任务文本历史](https://github.com/apple-aiml-research/ml-halo/blob/35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f/orchestrator/halo_orchestrator/history.py)。
