# 后续 Agent 实施计划（本轮不实施）

日期：2026-10-23。当前任务仅新增本目录四份文档，未修改核心实现、权重、评测行为或历史结果。项目根 `/home/szgx/LightNav-0`，当前HEAD `3015508b70fb6e30bbd66701e94c9b916dcb8df8` 且dirty。必须先阅读适用AGENTS，保留现有修改，不reset/clean。后续“Agent”是顺序工作包名称，不表示本轮启动了子代理。

依赖顺序：`02可信采集 → 03表征契约 → 04离线世界模型 → 05预测引导记忆 → 06有界闭环 → 07冻结评测`。任一质量门失败即暂停下游，不一次性实现全部模块。唯一当前最高优先级是02。

## Agent 02：train-only独立transition采集及校验

目标：证明官方train可在当前Go2-W环境实际reset/step，并生成符合 [数据契约](world_model_data_contract.md) 的可靠样本；不训练世界模型。

**建议新增文件范围，均未实现**：

- `/home/szgx/LightNav-0/experiments/world_model/collect.py`：独立launcher，使用runner工厂注入，显式train/allowlist/new run目录。
- `/home/szgx/LightNav-0/experiments/world_model/server_adapter.py`：独立VLNCEEnv subclass/受控adapter，不改默认serve入口；在server STOP替换后的 `_convert_action` 到底层step边界捕获最终命令及全精度ack，服务端读取pose、sensor缺失、源split/IDs。
- `/home/szgx/LightNav-0/experiments/world_model/recorder.py`、`validate.py`、`group_manifest.py`：原子无覆盖写入、严格provenance/group/时间/action校验、恢复策略。
- `/home/szgx/LightNav-0/experiments/world_model/configs/collect_train_go2w.yaml`：独立配置，显式split=train；保留原baseline YAML和500步评测默认。pilot预算最多20步/episode，独立标识。
- `/home/szgx/LightNav-0/tests/world_model/test_collection_contract.py` 等CPU fake测试。

允许读已有runner/policy/base/vlnce/probe/expert逻辑，禁止直接复用expert脚本的目录删除与val结果allowlist。若不改核心无法实现factory外部的hook，先提交具体接口差异供下一任务评审，不在原baseline静默埋采集副作用。服务器adapter捕获提交动作后，还需确认底层step确实返回，ack应区分submitted/executed/error；不能把_convert_action调用本身等同执行成功。

seed：构建独立resolved配置时设置Habitat seed/iterator，在合适初始化阶段调用底层Env.seed（`/home/szgx/miniconda3/envs/habitat/lib/python3.9/site-packages/habitat/core/env.py:331`），设置客户端相关RNG并记录执行证据。现有remote reset的seed=None行为不可据此声称已解决，独立协议/launcher需明确传递路径。不要为了实验改变默认reset行为。

验收：

1. CPU fake覆盖正常移动、旋转、STOP、解码回退、runner预算STOP（无推理、无stale输出）、server STOP替换、丢包/超时/重试、末帧、缺sensor、未知allowlist/重复ID、seed未应用、val污染、写入冲突；错误必须拒收/隔离。
2. train manifest含源hash、组划分、requested/resolved配置、dirty代码版本、模型/RVQ版本和seed状态；指定集合与实际reset集合完全相等。
3. **经后续执行授权**完成2–3个显式train episode、至少2个scene、每个≤20步的有界试采，不做完整评测；包含终止记录（自然STOP或明确budget），保留最后step返回的RGB。模拟server override等罕见路径可fake测，不能为凑验收修改真实baseline规则。
4. 每条valid transition有真实before/after RGB、前后pose、当步预测/未推理状态、请求与最终命令ack、dt来源、终止/异常来源。客户端到服务端step action_id一一对应；服务端替换可被校验，不允许复算代替。
5. 校验报告零硬错误；相同manifest/seed的重复小样本reset集合/顺序、resolved配置一致，首步动作差异被记录分析；不要求GPU逐bit一致。深度单位/量程和真实传感器缺失状态已确认。
6. 关闭collector回到原factory，fake输入下原动作/推理调用次数不变；现有核心文件diff和旧output保持不变。真实pilot成功仅证明小范围可运行，不扩大为全部train无异常。

测试命令（**未来入口提案，目前不能直接运行；本轮未执行**）：

```bash
cd /home/szgx/LightNav-0
.venv/bin/python -m pytest tests/world_model/test_collection_contract.py -q
.venv/bin/python -m experiments.world_model.group_manifest --split train --dry-run
# collector通过独立配置指定Habitat解释器和端口，manifest选择2–3 episode/2 scene
.venv/bin/python -m experiments.world_model.collect --config experiments/world_model/configs/collect_train_go2w.yaml --allowlist <new_train_allowlist.json> --max-steps 20 --run-dir <new_empty_run_dir>
.venv/bin/python -m experiments.world_model.validate --run-dir <new_run_dir> --strict
```

参数/CLI需随实现更新，本计划不把命令存在当验收证据。服务端使用Habitat Python3.9，客户端用.venv3.11，不能在错误解释器下测试后宣称兼容。仅在新端口启动本任务拥有的进程，不停止其他用户服务。

失败回滚：关闭独立入口/只停止本任务进程，隔离失败run并写原因，保留全部样本。新增代码可在审查后按文件撤回自身补丁，不删除旧output、不git reset，不将失败采集扩大到全量。未通过则不进入03/04训练。

## Agent 03：视觉/语言表征的时间契约

依赖02数据门；可以先静态审计，不提前跑大模型。范围：新增 `/home/szgx/LightNav-0/experiments/world_model/features.py`、`feature_contract.md`、`tests/world_model/test_features.py`。若复用LightNav需要暴露接口，提出最小、默认关闭的独立engine adapter；不得擅改 `engine.generate` 的基线返回语义。

审计位置：`/home/szgx/LightNav-0/src/lightnav/inference/engine.py:458` `_vit_forward_from_inputs`、`:511` `vit_forward`、`:552` cached路径、`:669` tubelet cache；`model.py:315` deepstack拼接；`/home/szgx/LightNav-0/src/lightnav/slowfast.py:1` 成对/补帧语义、`inference/samples.py:29` 构样本。明确frame/time→token映射、层/shape/池化、奇数帧padding、HF与vLLM、cache命中等价与未来可见性；未确认前不能输出“每帧latent”。语言表征同样定义冻结encoder和原始指令映射，不能只用字符串占位称语言条件已实现。

验收：用可控短帧序列的CPU fake特征映射测试；必要时另行批准最多少量实际特征提取，不完整导航评测。比较独立冻结encoder方案的资产/许可/计算；仅本地可用资产，不自动下载。确保target encoder冻结、改变未来帧不改变当前online特征、metadata可追溯。形成二选一决定和失败理由。

未来测试命令：`.venv/bin/python -m pytest tests/world_model/test_features.py -q`；未来特征检查入口 `.venv/bin/python -m experiments.world_model.features --manifest <validated_train_manifest> --audit-only --limit 8`。未实现时不能声称通过。

回滚：不发布不明语义的cache版本；禁用adapter，保留原RGB数据；不因特征审计失败覆盖原engine。

## Agent 04：离线轻量世界模型首阶段

依赖02/03。范围：独立 `/home/szgx/LightNav-0/experiments/world_model/{dataset.py,predictor.py,train.py,evaluate_prediction.py}`、独立config与CPU测试；新增权重仅写新实验目录，冻结基座。训练本身需下一任务明确授权，本轮不启动。

输入为RGB latent、语言、因果历史、实际执行命令和dt；目标未来latent+局部pose增量。先copy-current/动作均值/最近邻与小MLP/GRU基线，若标签可靠再增加受阻/进展；位姿回归不是最终研究。数据统计只在train_fit拟合；dev按scene/route分组，禁止val_seen/unseen进入训练loader。

验收：dataset拒收未知ack/缺末帧/val来源；变更privileged标签不改变online输入；动作/pose坐标转换、yaw wrap、STOP独立类型测试通过。报告未见开发组latent/pose误差、预测检索rank、多步误差、方差/坍缩与分层结果、语言/动作/历史消融及计算成本。不能以训练loss下降或dev过拟合当导航提升证据。

未来命令：`.venv/bin/python -m pytest tests/world_model/test_dataset.py tests/world_model/test_predictor.py -q`；`.venv/bin/python -m experiments.world_model.train --config <frozen_train_config> --run-dir <new_run_dir>`；`.venv/bin/python -m experiments.world_model.evaluate_prediction --checkpoint <new_wm_checkpoint> --partition train_dev`。预算、训练步数先在对应任务定义，不自动全量。

回滚：禁用新增模块，隔离失败checkpoint和报告，保留数据/基座；预测不优于简单基线时先检查标签/表征/数据覆盖，不直接上闭环。

## Agent 05：预测引导记忆与实际输入适配

依赖04的独立预测证据。新增 `/home/szgx/LightNav-0/experiments/world_model/{memory.py,retrieval_eval.py,history_adapter.py}` 和对应测试；默认不开启，严禁将计算cache当语义memory。仅episode内历史，reset清空，future mask和available_at约束；动作结果只能在完成后可见。

先离线对比predicted/current/recency/random/oracle query；oracle未来帧只在评价。再将过去检索RGB按真实时间顺序与最新帧去重合入独立history sample adapter，不改原RVQ/控制器；保持token与图像预算匹配。若下游基座输入完全没变，不能宣称检索已接入；如果要改prompt，另设受控实验而非顺带修改。

验收：Recall@k/MRR和检索错误案例；没有跨episode/未来/特权泄漏；相同预算的current-query/recency对照；fake engine能观察预期历史索引变化，关闭adapter输入与输出严格还原。离线相似性好不自动通过闭环收益门。

未来命令：`.venv/bin/python -m pytest tests/world_model/test_memory.py tests/world_model/test_history_adapter.py -q`；`.venv/bin/python -m experiments.world_model.retrieval_eval --partition train_dev --causal-only --compare predicted,current,recency,random,oracle`。

回滚：关闭history adapter，恢复原SlowFast输入；不清空原日志，不把oracle query带入线上。

## Agent 06：小规模闭环与单一恢复机制

依赖05；新增 `/home/szgx/LightNav-0/experiments/world_model/{run_closed_loop.py,recovery.py}` 和独立配置/测试。先比较A基线、B预测shadow、C预测memory；shadow在相同输入时应不改变动作序列。之后仅引入一个WM误差/重复观测触发的有界恢复机制形成D。候选重排另开任务，必须先有train候选后果覆盖；不能把10航点当候选。

验收：预注册train_dev固定清单和预算；无在线GT目标/测地距离；所有额外观察/恢复步计入路径与预算；关闭新增机制精确回基线。失败分感知/语言/规划/控制/STOP/基础设施，报告配对差值与成本，不挑成功案例。若要在val检查，先冻结阈值，记录开发接触；不可用val轨迹反向训练。

未来命令：`.venv/bin/python -m pytest tests/world_model/test_recovery.py tests/world_model/test_closed_loop.py -q`；`.venv/bin/python -m experiments.world_model.run_closed_loop --config <fixed_train_dev_config> --variants baseline,shadow,memory,full --allowlist <frozen_manifest>`。

回滚：关闭恢复/检索，保留全失败轨迹和版本；无证据提升则报告负结果，不修改指标阈值/分母制造收益。

## Agent 07：锁定方案的隔离评测与报告

依赖06，小规模证据充分且另有评测授权才执行。范围：独立实验config、结果聚合/置信区间脚本及报告，不改指标源码。预注册val_seen/unseen用途/次数，最终test留到方案确定并满足官方提交协议后。当前test GT不可得时不伪造NE/nDTW。

验收：固定全部allowlist、机器人/传感器、server终止规则、500步与生成预算；A/B/C/D及current-query同预算memory消融；SR/SPL/NE/nDTW、route/scene组bootstrap区间、成本、失败类型与基础设施异常全报告。基线标注为当前dirty版本，不假定官方原版。所有训练/统计/阈值来源可追溯，无val/test训练样本。

未来命令：`.venv/bin/python -m experiments.world_model.report --manifest <locked_experiment_manifest> --paired --bootstrap-group route,scene`；正式运行命令在冻结配置后由本任务补充，不在这里给出可误触发全量的命令。

回滚：停止本任务拥有的运行，保留已完成和异常结果；分析脚本修复只生成新报告版本，不覆盖旧结果。不得用修改成功半径、删异常episode或挑seed代替研究收益。

## 通用代码检查与退出条件

上述命令均是未来接口设计，不是当前存在的CLI。后续若修改src，须按适用AGENTS执行仓库既定测试（先核实Makefile/解释器要求），再加对应模块测试；文档-only本轮不运行模型/pytest。各任务记录文件范围、diff、测试实际命令/退出码和未验证项，避免用“计划运行”冒充“通过”。任务内只撤回自己可识别的新补丁；遇到重叠用户修改停止并报告，不强制覆盖。

下一步唯一优先交付：Agent 02 的可信train transition采集与严格校验，通过CPU fake测试和2–3 episode有界试采，出具零硬错误报告。**在此之前不开发/训练预测器，更不接闭环。**
