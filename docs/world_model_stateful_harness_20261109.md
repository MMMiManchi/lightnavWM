# 新 Harness 查询与执行链路

日期：2026-11-09。范围为第一阶段：查询分流、LightNav 执行、STOP 验证、真实事件反馈和跨 episode 保存。本轮不重训全局世界模型，不开展多场景正式对照评测。

## 已实现的链路

新增入口 [run_stateful_harness.py](/home/szgx/LightNav-0/scripts/run_stateful_harness.py)，用本地 `vllm_local` LightNav、冻结 SigLIP 和视觉工具连接现有 Habitat 环境。没有修改 `src/`、原服务端默认入口或机器人部署代码。

查询先使用当前实际视觉特征和出口方向查真实经历。源画面相似、方向经过实际图像几何对齐、多个历史后果一致时，直接返回历史实际后果；这条查询不执行 TSSM prior/decoder 或场景适配器预测。
未命中时才预测未来特征；熟悉画面无法对齐、历史后果冲突等情况请求主动观察。实际规划仍依据本次指令和当前可见出口，历史任务不被当成路线命令。

当前直查阈值为 source 余弦 0.98、方向余弦 0.98、历史后果两两余弦至少 0.9。直播入口使用 SIFT/RANSAC、可用时的当前深度支持方向对齐。这些是启发式条件，不保证识别出物理上的同一个地点。
`world_model_called` 指未来预测是否执行；初始化和执行后的潜在状态维护仍然保留。

多个候选从同一实际状态查询，提交执行后才接收反馈。中间真实观测更新 posterior；只有选定出口确实执行、下一房间或下一决策位置得到两次有实际动作间隔的视觉支持，才写入到达后果并更新场景适配器。
实际前进指令只是入场条件，不证明位移或到达。每 6 步检查一次只是默认调用频率，不是“0.6 秒后的画面就是下一地点”。

新事件完成与整项任务完成分别判断。局部 STOP 未验证时恢复原指令并在同一已观察帧上重新规划，保留 LightNav 原生历史；需要补充证据时主动观察。全局 STOP 必须经过本次任务的阶段和最终位置关系检查。观察或步数预算耗尽，不记为任务成功，也不补造事件后果。

跨 episode 保留真实事实、可重启加载的图片索引、场景参数和优化器状态；运动潜在状态与未执行候选被重置。当前查询上下文沿用此前的 prior-episode 历史隔离约束。

## 实际试跑暴露的 STOP 问题

最初直接复用旧的视觉进度验证器。在已有开发任务 `aayBHfsNo7d / 76` 的两次试跑中，都在第 85 个执行动作放行 STOP，评估端距离分别为 5.910 米和 6.087 米，成功均为 0。
原指令是跨过当前房间并进入有电视的房间。验证器把远处黑色缺失网格和隔门视野解释为电视，并把“看见目标”误当成“已经进入目标房间”。失败结果保留在 `stateful_harness_live_20261109_01`。

本轮新增 [goal_verification.py](/home/szgx/LightNav-0/experiments/world_model/goal_verification.py)：最终检查分别要求目标身份与相机空间关系；进房间必须支持相机已在房间内部，不能只看见门后物体。指令明确要求停在门口时，门口关系本身是合法终点。
对于最终子任务中唯一可识别的常见目标类别，正面提议还须提供当前画面框，再对裁剪区域做一次不含任务和预期目标提示的类别识别。

对固定实际画面的独立复核中，先前两张误停图均被拒绝；另一个已有近终点的“停在门口”画面被接受。结果保留在 `strict_goal_regression_20261109_01`。这只是两个负例和一个正例，不是总体误停率或到达验证准确率。

修正版先用相同开发任务执行两次最多 120 步的接线试跑，没有放行 STOP，分别拦截 4 / 1 次提议。结束时评估距离为 6.161 / 0.898 米，均未确认完成。距离较近不能单独证明满足“进入电视房间”，需要结合实际语义位置核验。

随后移除了验证提示中的默认否定示例，分别要求目标身份和位置关系判断，并补充进入区域过程中的实际中间画面。最终版本以最多 300 步预算再执行一次：实际执行 161 个动作，拦截 15 次 STOP，观察预算耗尽后以 `unverified_STOP_no_supported_continuation` 弃权结束；未发送 STOP，评估距离 2.609 米、成功为 0。最后画面实际面对卫生间门口，验证器没有把它当成电视房间。

本轮 5 次同一开发任务试跑共执行 571 个环境动作，所有进程已经结束。前两次旧验证器误放行，后三次未确认完成；不能宣称导航验收通过，也不能把同一开发任务的调试当成盲测效果。
120 / 300 步都是当时的接线预算，不是数据采集的事件截止策略。用户随后明确：导航评测采用 500 步，数据采集仍采用 1000 步。当前评测入口已将默认值和允许上限改为 500；旧运行的实际配置和源码快照保留。

实时试跑均没有得到视觉支持的下一事件，因此新增到达事实和在线参数更新均为 0。这暴露了目前出口提议、局部执行与事件确认仍不能稳定配合。日志中有把厨房、楼梯或柜旁缝隙当成目标方向的探索判断；这些没有被写成已验证分支后果。不能用后台接口测试代替实时积累经历的验收。

## 记忆与在线更新的实际数据核对

另用已有 v4.1 发布包的一条已完成真实事件做回放，未新增导航执行：

- 第一次查询之前记忆为空，执行预测；只有揭示实际后果后才写入并更新场景参数。
- 关闭并重开事实库、图片索引和场景参数，再查询相同实际源画面与出口。
- 实际图像方向对齐通过，直接查回实际后果，余弦为 1.0；测试把 TSSM transition 替换为会抛错的调用，证明查询没有执行未来预测。
- 场景参数实际发生变化，全局模型参数保持不变。

记录在 [真实事件回放结果](/home/szgx/LightNav-0/output/world_model_training/stateful_memory_real_replay_20261109_01/results.json)。这是同视角、已验证旧事件的接口检查，不代表未知出口预测或新的实时导航成功。
最终提示版本的两张误停负例和一个门口正例复核均符合预期，保存在 `strict_goal_regression_20261109_02`；这些样本也只用于开发诊断。

新导航试跑若没有得到视觉支持的下一事件，就不会产生到达事实或在线更新。回放中的更新与实时导航中的更新数量分开统计。

## 验证与复用

104 项相关 CPU 测试通过，包含旧模型/记忆/任务进度回归，以及新增的零预测直查、冲突观察、方向对齐、图片重启、局部 STOP 恢复、全局 STOP 拦截、真实事件写回、参数更新和目标关系检查。
修改涉及的模型、脚本和测试通过定向 Ruff 检查。pytest 仍有既有 `asyncio_mode` 配置警告。

新增核心代码：

- [记忆查询路由](/home/szgx/LightNav-0/experiments/world_model/memory_routing.py)
- [查询与状态反馈](/home/szgx/LightNav-0/experiments/world_model/tssm_runtime.py)
- [Harness 接口](/home/szgx/LightNav-0/experiments/world_model/tssm_queries.py)
- [导航控制循环](/home/szgx/LightNav-0/experiments/world_model/stateful_controller.py)
- [下一事件视觉验证](/home/szgx/LightNav-0/experiments/world_model/event_verification.py)
- [图片持久化](/home/szgx/LightNav-0/experiments/world_model/live_images.py)

命令参数见 [CONFIGURATION](/home/szgx/LightNav-0/docs/CONFIGURATION.md)。示例：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python scripts/run_stateful_harness.py \
  --tasks output/world_model_training/stateful_harness_smoke_20261109_tasks.json \
  --output output/world_model_training/stateful_harness_my_new_run \
  --online-learning --max-steps 500 --sim-gpu 0
```

复用已有经历需显式传 `--experience`。新实验输出目录禁止覆盖；需要公平对照时，各条件应使用独立经历目录。封存场景被入口拒绝。真实目标距离、成功率、轨迹和坐标不进入控制器，评估指标仅在动作之后单独记录。

当前完成的是可运行链路和接线验证，完整实时导航验收尚未通过。世界模型对未知出口的预测仍未达标；视觉地点关联、最终到达判断和下一事件判断仍需扩大独立样本验证。
视觉支持的事件明确标为 `temporal_visual_event_support_not_ground_truth`，不会自动并入原有 gold 训练发布包。旧数据、模型和失败运行均保留。

汇总与完整性核对见 [integration/results.json](/home/szgx/LightNav-0/output/world_model_training/stateful_harness_integration_20261109_01/results.json)。
不同运行分别保留了实际使用的源码快照；最终版增加的提示与门口规则没有追记到旧运行结果中。
