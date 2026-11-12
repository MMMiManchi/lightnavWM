# 交给 Cursor：T0 事件预测数据与真实接口审计

日期：2026-11-03。配套方案：`docs/world_model_event_memory_design_20261103.md`。本任务由Cursor执行；Codex负责review与下一阶段设计。本轮交接文档本身不代表本任务已执行。

## 1. 正确工作位置和启动检查

```bash
ssh CEC
cd /home/szgx/LightNav-0
```

CEC连接信息在用户本机SSH config中已配置；不要再让用户重复提供。2026-11-03核实：分支 `local/research-snapshot`、HEAD `4933c66`、大量未提交修改。先阅读适用AGENTS和当前git status，保留全部已有变更。不reset、restore、clean或批量add；本任务不提交与自己无关的文件。

服务器没有 `rg`；若仍无则使用 `find`/`grep`。Habitat为独立Python3.9环境，推理为仓库 `.venv` Python3.11，不安装混用。

## 2. 要完成的结果

为下一阶段回答一个问题：

> 现有真实train轨迹能否构成“当前事件节点 + 当时可生成的分支 -> 下一事件真实后果 -> 可检索历史经验”的训练和评价单元？如果不能，最少缺哪些数据、标签或接口？

交付必须包含真实文件/字段/计数证据，而不是只定义几个dataclass或搭一个空Harness。先审计数据和语义，再给出最小派生事件索引与后续任务建议；此任务不训练预测器。

## 3. 范围

允许：读取源码、现有manifest/日志/图像元数据和少量必要RGB；运行离线统计；新增只读审计入口、测试及独立诊断产物；仅在不改原始数据/schema的前提下提出或生成明确标注为proposal的事件sidecar。

不做：启动Habitat；采集；重提特征；下载大模型或安装外部完整栈；训练或导航评测；修改默认policy/runner/服务行为；修改权重、原数据、schema、训练配置、授权开关或已有结果。既有广泛授权并不要求在本审计任务中提前执行后续阶段。

建议文件范围（如已存在须先核实，不能覆盖）：

- `experiments/world_model/event_audit.py`：纯离线读取、统计与报告逻辑，不import Habitat或加载大模型。
- `experiments/world_model/event_audit_entry.py`：可选独立CLI，输出新目录，不覆盖；所有输入清单显式记录。
- `tests/world_model/test_event_audit.py`：仅对新增非平凡行为做合成测试。
- `docs/world_model_event_data_audit_20261103.md`：实际审计结论。
- `output/world_model_diagnostics/event_data_audit_20261103_01/`：建议独立目录；已存在就选择新后缀并记录，不覆盖。

这些名字是新任务建议，不是声称当前已经有这些文件。

## 4. 按顺序审计

### A. 现有数据与时间语义

从 `formal_fit_20261028_04`、`formal_dev_20261028_01` 的当前有效manifest/批准记录及只读loader开始；其他fit来源必须列出具体路径、radius/传感器版本、split/route分组与准入状态，不能因为有50k就自动并入。

逐来源统计episode/scene/route/transition、RGB与末帧可读性、完整段长度、漏步/异常/截尾、执行ack与观察一致性。核实原schema目前是 `wm-transition/0.2`；不照抄旧0.1文档，不把命令时间当世界时间，不修写旧记录。只读取必要清单/元数据，不遍历全部官方scene资源或执行仿真。

区分两类信息：离线可读的GT标签和在线实际可获得的信息。保留现有 `online_input.py` 边界，不能把GT pose、goal distance、scene connectivity或未来RGB送进决策。

### B. 事件样本能否成立

检查原数据有没有真实事件/门口/岔路/房间/地标标签、同节点回访和多分支后果；没有就明确写无。

“每固定N步抽帧”“累计GT位移到一定长度”只能作为离线时间/距离片段proxy，不能改名为事件节点真值。可提出人工抽查或后续teacher标注清单，但本任务不新跑VLM或特征模型。若读取少量RGB用于人工核对，写清样本选择规则与数量，不用少量例图外推整体覆盖。

先列出两种样本：可严格确认的事件转移；仅有长片段、待标注的proxy。对每类给出实际数量和可用字段，不能以模型生成文字当真实语义标签。

检查候选分支在决策时是否存在。已执行命令序列是后果审计字段，不能作为一开始未知的未来输入。原RVQ10点是一条轨迹，不能作为多分支清单。

### C. 记忆查询能否被评价

定义决策时可见的候选记忆集合：同episode过去记录、允许的先前同场景记录、允许的跨场景train经验分别列出。排除当前未来帧、同路线泄漏、query目标自身和近重复片段。

列清“相关记忆”的标签来源：视觉相似是代理指标，不能冒充导航帮助。若没有独立的语义/路线相关标签，先提出需要人工标注的最小集合，不伪造Recall@k。真实未来query仅作离线上界，不能进入线上；若能用已有兼容缓存计算上界，须记录特征空间/版本和严格因果过滤；没有兼容特征则记录阻塞，不提新特征。

### D. 实际集成接口

重新核对并给出文件行号：

- runner工厂与reset/step生命周期；是否有完成观测通知路径。
- policy如何observe并生成；避免一帧重复入历史。
- 最终执行ack在哪个server入口可获得；普通RemoteEnv缺什么。
- SlowFast绝对帧和显式video_segments；跨episode记忆图像如何不冒充连续视频。
- 当前可执行候选生成、节点定位、局部子目标、主动观测和物理回溯有没有真实接口；没有就标“需新增”，不能只用函数名字表示能力。
- RGB-only、RGB-D、GT pose oracle边界分别是什么。scene_id选场景库若依赖环境元数据，明确披露。

## 5. 交付物与验收标准

必须交付中文报告、机器可读summary、输入文件清单/哈希或快照ID、代码diff和执行命令。

验收看事实，不要求一定可训练：

1. 对每个来源给出真实数量、split与分组、图像/执行/时间字段覆盖；所有未知项有具体证据和最小核实方法。
2. 将真实事件与proxy分开，明确严格事件样本和多分支后果样本实际有多少，零也如实写零。
3. 给出最小契约草案：EventNode、BranchCandidate、ObservedOutcome、MemoryRecord、LearningTransition；包含available_at、来源/版本、执行策略ID、截尾、观测/预测/未知区分。
4. 证明新审计逻辑不修改原数据、不import仿真、不加载权重；输出无覆盖。
5. 明确下一步唯一任务：若事件/相关性不足，先补标注/定向分支数据；若足够，进入只记忆与oracle-query上界T1。不要一次排满所有训练。

若实现离线审计入口，应做针对性测试；建议命令（仅在实际新增对应文件后执行）：

```bash
.venv/bin/python -m pytest tests/world_model/test_event_audit.py -q
.venv/bin/python -m ruff check experiments/world_model/event_audit.py experiments/world_model/event_audit_entry.py tests/world_model/test_event_audit.py
```

缺ruff时记录依赖现状，不为文档审计安装大栈。没有实现新行为、只写报告时不追加形式化测试。若改src则遵循AGENTS的完整CPU测试要求；本任务应避免改src。失败保留独立产物并报告原因；回滚只撤自己的新代码，不触碰原始数据/已有dirty修改，避免破坏性命令。

完成时返回：做了什么、真实计数、通过/未通过的条件、文件路径、实际命令和结果文件、当前阻塞、下一阶段唯一任务。完成后停止，交Codex review，不自动开始采集或训练。

## 6. Codex复核重点

Codex检查是否偷偷把0.1秒、固定间隔或GT路点重新命名成长期事件；是否把执行后的命令序列当执行前输入；是否把预测写成观察记忆；是否将视觉相似等同导航相关；是否已有数据足够却继续空转扩展准备清单。审计结论必须推动下一张具体任务卡。
