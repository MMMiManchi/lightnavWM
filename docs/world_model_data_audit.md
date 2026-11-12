# LightNav-0 世界模型数据可行性审计

审计日期：2026-10-23。对象：CEC `/home/szgx/LightNav-0`，HEAD `3015508b70fb6e30bbd66701e94c9b916dcb8df8` **加现有工作区修改**。本轮只读调查并新增四份文档；未启动模型、仿真、训练或评测，未下载数据/权重。本文“已确认”指静态代码、文件或官方页面证据；“建议”不是现有功能。

## 1. 官方发布情况与资源边界

|资源类别|已确认的发布/获取方式|依赖、许可与限制|本机状态与结论|
|---|---|---|---|
|LightNav-0 checkpoint|[官方模型](https://huggingface.co/LightOriginsHQ/LightNav-0)，README 的 `hf download` 示例|2026-10-23 HF API 返回 `private=false,gated=false`，模型卡 Apache-2.0；基座 Qwen3-VL-4B-Instruct|`/home/szgx/LightNav-0/checkpoints/LightNav-0/` 存在，权重 `model-00001-of-00001.safetensors` 为 9,695,744,034 bytes，含 tokenizer、eval_config、RVQ bundle；文件存在不等于运行验证|
|完整原始预训练/中训/SFT/RL语料及处理流水线|[论文 v2](https://arxiv.org/html/2608.30935v2)描述 2K+ 场景、4K+ 小时、多来源混合；[官方 README](https://github.com/lightorigins/LightNav-0/blob/main/README.md)提供模型和推理/评测入口|各来源许可不可由代码/权重许可替代；论文列出数据源不等于完整混合语料和处理结果可下载|**未确认完整公开，也未确认本机具备**。已检查 README、模型卡、官方根目录及 scripts 列表，未得到完整语料清单/下载与训练再现证据。下一步最小核实：官方 dataset manifest/release、训练数据准备文档、处理样本及训练入口；必要时向维护者询问版本和授权，不下载大资源|
|INSIGHT 基准|README:160–161 链接 [Light-INSIGHT-Bench](https://github.com/lightorigins/Light-INSIGHT-Bench)|是单独基准入口；本轮未审计其文件及许可|不能把该链接当作全部 LightNav 原始训练语料已发布的证据；完整本地副本未确认|
|R2R 与 VLN-CE 任务数据|原始 R2R 是离散导航任务；当前使用 Habitat 转换的 VLN-CE v1-3 preprocessed。官方 [VLN-CE README](https://github.com/jacobkrantz/VLN-CE/blob/master/README.md):66–78 给出 minimal 与 preprocessed 下载及目录，[格式说明](https://jacobkrantz.github.io/vlnce/data)|任务 JSON/GT 不包含全部可渲染场景。上游 README:267：代码 MIT；数据/模型受 Matterport Terms 和 CC BY-NC-SA 3.0 US 约束，派生图像分享前仍需核对条款|四个 split 均存在，详情见下表；不是完整 LightNav 原始训练集|
|Matterport3D 场景|[MP3D 访问入口](https://niessner.github.io/Matterport/)，VLN-CE README:53–62：通过获准的下载脚本 `--task habitat` 获取 Habitat 场景|需申请/接受 Matterport 使用条款；不应重新分发受限资产。无需为本任务安装离散 Matterport3DSimulator/下载其 panorama skybox|`data/scene_datasets/mp3d/` 中有 90 个 navmesh；train 涉及的全部 `.glb`（非空）和 `.navmesh` 均存在。未实际加载验证，亦未重新审核账号授权|
|Habitat 可执行任务环境|本地 Go2-W YAML + VLNCEEnv + 数据/场景 + 独立 Habitat Python|渲染驱动、CUDA/EGL、navmesh与代理尺寸、Python依赖均需兼容；不是只凭 JSON 即可运行|环境目录、包元数据、场景与代码存在；**本轮未验证 reset/step 成功**|
|本地实验及派生数据|`output/r2r_probe/trace_20261020/steps.jsonl`、`output/go2w_expert/*.jsonl`|采集 split 来源和派生许可继承；不能以生成日志改变其验证集属性|三份主要文件全部匹配 `val_unseen`；仅用于诊断，禁止纳入本研究训练集|

官方核实方式：GitHub contents API 成功读取 README（blob `f541d70508b62b87bb49820089e94ab3276d137d`）和 scripts；官方 scripts 列表为 `eval_evt_bench.sh,eval_habitat.sh,serve_thor.sh,smoke_gpu.sh,start_servers.sh`。本地未跟踪的 `scripts/train_go2w.py`、`collect_go2w_expert.py` 不能作为官方原训练流程公开的证据。HF [API](https://huggingface.co/api/models/LightOriginsHQ/LightNav-0)与模型卡由本机读取成功；服务端访问 HF 失败，属于网络差异。模型卡仍链接 `lightrobo/LightNav-0`，本地 origin 为 `lightorigins/LightNav-0`，引用时记录实际 URL/版本，不能默认链接内容一致。

## 2. 本地数据、依赖和来源

数据根：`/home/szgx/LightNav-0/data/datasets/R2R_VLNCE_v1-3_preprocessed/`。以下为本轮 gzip JSON 静态扫描结果，未生成新轨迹。

|split|episode数|场景数|本地文件|用途|
|---|---:|---:|---|---|
|train|10,819|61|`train/train.json.gz`、`train/train_gt.json.gz`|唯一新增训练采集来源；GT 10,819 项，episode key 缺失数 0|
|val_seen|778|53|`val_seen/val_seen.json.gz`|冻结后的已见场景验证，不参与世界模型训练|
|val_unseen|1,839|11|`val_unseen/val_unseen.json.gz`|冻结后的未见场景验证；现有诊断已造成开发接触，应披露|
|test|3,408|18|`test/test.json.gz`|方案锁定后最终测试；不能据此假设可获得 test GT 或本地完整指标|

train gzip SHA256：`340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea`。
val_seen：`e52a19e08a2f6f936f61b021cf4a0e17ddb8685f8807c60ba6382a61e2463df0`。
val_unseen：`1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`。
test：`4af77f36c1ed1ad3cf19f724970aba7f84b9218a9575d19719941cb6d2230c33`。

train 实际 episode 字段：`episode_id,goals,info,instruction,reference_path,scene_id,start_position,start_rotation,trajectory_id`。instruction 只有 `instruction_text,instruction_tokens`。全部有 trajectory_id，全部 **没有 instruction_id**。按 `(scene_id,trajectory_id)` 得 3,603 组，各组起点、旋转、reference_path、goals 的 JSON 精确表示无冲突。此结果支持路线分组，但未证明不同 trajectory_id 间无重复/增强路线；划分时还应按几何路线指纹合并关联组。不得把 episode_id 当作跨数据版本的全局 ID。

Habitat 解释器目录 `/home/szgx/miniconda3/envs/habitat`；元数据已见 habitat_lab `0.3.20231024`、numpy `1.23.5`、fastdtw `0.3.4`、omegaconf `2.3.1`、lightnav_habitat_server `0.1.0`。推理解释器 `/home/szgx/LightNav-0/.venv/bin/python3.11` 存在。未在本轮 import 大模型/渲染包，habitat-sim 实际版本、推理依赖完整性、GPU/EGL可用性仍待小规模运行确认。推荐安装版本只是 `/home/szgx/LightNav-0/docs/HABITAT_SERVER.md:27` 的文档要求，不等于实际安装验证。

## 3. train 采集调用链与真实接口

下表路径全部为当前 dirty 工作区版本。`R=/home/szgx/LightNav-0`，每个路径均从 R 展开，不代表官方未改版。

|环节|真实文件、函数、行号|事实/采集影响|
|---|---|---|
|split、数据路径|`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/serve.py:68` 参数；`vlnce.py:230` `VLNCEEnv._apply_dataset_overrides`|支持 `--split train`、data/scenes 根覆盖；构造器默认 val_unseen，必须显式 train，并记录解析后的配置|
|Go2-W配置|`/home/szgx/LightNav-0/habitat_server/configs/vlnce_r2r_go2w.yaml:21`、`:28`、`:37`、`:40`|默认 val_unseen；身体高0.50m、半径0.215m；RGB/depth 480×270、HFOV120°、传感器高0.50m；控制dt0.1s，线速0–2.5m/s、角速±300°/s|
|episode解析|`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/vlnce_extensions/vlnce_dataset.py:81` `VLNCEDatasetV1.__init__`，`:121` `from_json`（`:135`起构造episode）|gzip 由 split 格式化；episode_id 转字符串，解析 instruction 和场景；类型支持可选 instruction_id/trajectory_id，不代表数据实际有该字段|
|allowlist|`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/serve.py:102`；`vlnce.py:18` `episode_allow_key`、`:28` `filter_episodes_by_allowlist`、`:240` `_apply_episode_split`|`--episode-ids` JSON 列表 `[{"scene":"7y3sRwLe3Va","episode_id":"1"}]`，scene key 为 glb 父目录；不分片时过滤后直接返回，分片时按场景排序连续切分。部分未匹配可能静默丢失，需新增严格集合相等校验|
|顺序与数量|`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/vlnce.py:53` `_disable_episode_shuffle`；`runner.py:614` 起 episode 循环|`--episodes N` 只是运行数量，既不是随机采样也不是固定 allowlist；采样应离线生成版本化清单，核对实际 reset 的 key|
|可注入入口|`/home/szgx/LightNav-0/src/lightnav/habitat/runner.py:561` `run_habitat_eval`，`:614` reset，`:693` step|接受 env_factory/policy_factory/engine_factory，可在独立 launcher 中注入采集包装器，不改默认 runner 行为|
|策略输入输出|`/home/szgx/LightNav-0/src/lightnav/habitat/policy.py:142` reset、`:158` act、`:200` get_info|当前用 RGB历史+instruction；get_info 提供 raw_text、RVQ、轨迹、选中索引/行。没有稳定 latent、候选评分接口|
|动作分层|`/home/szgx/LightNav-0/src/lightnav/habitat/policy.py:41` `select_action_waypoint`，`:192` act；`/home/szgx/LightNav-0/src/lightnav/velocity.py:19` `first_waypoint_to_velocity_cmd`|选首个前向或yaw非零的航点；不是十候选。映射forward/dt和yaw/dt、裁剪归一化，横向不执行|
|RVQ语义|`/home/szgx/LightNav-0/src/lightnav/traj_vocab.py:64` `compose_to_abs`，`:189` `is_stop`，`:198` `decode_waypoints`；checkpoint `action_tokenizer/manifest.json`|三层256码，解码10×3；SE2差分积分为当前机器人局部绝对航点，forward米、left米、yaw弧度。没有已确认的各航点物理时标；不能把第k航点当k次dt命令。STOP level0=6，完整stop tuple[6,122,174]|
|命令被替换|`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/vlnce.py:519` `step`，`:387` `_convert_action`；`base.py:160` `step`|服务端可先将请求替换成 STOP，再转换后交给底层 env.step。必须在转换后/底层step边界记账；客户端请求或格式化 history 不能证明实际提交命令|
|预算STOP/解码回退|`/home/szgx/LightNav-0/src/lightnav/habitat/runner.py:653`；`policy.py:170`|预算末步可跳过推理直接STOP；解码失败生成零速度（不自动等于离散STOP）。记录当步来源，不能沿用上一推理结果|
|观测与位姿|`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/base.py:88` `_extract_rgb`、`:100` `_extract_depth`、`:109` `_build_observation`；`vlnce.py:438` `_compute_info`|RGB/depth可取但无传感器时有全零回退；info含目标/指标而非完整pose/route/instructionID。GT pose需服务端sim.get_agent_state采集，不能假设客户端已收到|
|已有pose诊断|`/home/szgx/LightNav-0/experiments/r2r_diagnosis/serve_probe.py:58` `PoseEnv`；`scripts/collect_go2w_expert.py:85` `_agent_frame`|有独立server subclass获取前后pose的先例；Habitat世界坐标和姿态转换须固定，yaw=0朝−Z，正yaw向左|
|终止帧|`/home/szgx/LightNav-0/src/lightnav/habitat/runner.py:693`、`:702`；`:488` `final_frame`|step返回后的终止观测可采，但视频final_frame不是原始transition数据接口。包装器必须在任何reset之前保存末帧，异常时不能捏造nextRGB|
|seed|`/home/szgx/LightNav-0/src/lightnav/habitat/remote_env.py:115` reset；`base.py:148` reset|remote reset发送None；base接受seed但不调用底层seed。底层 `/home/szgx/miniconda3/envs/habitat/lib/python3.9/site-packages/habitat/core/env.py:331` `Env.seed` 可设置random/NumPy/sim/task，`:139` iterator另用config.seed；须独立launcher同时处理配置、iterator、env及推理seed并记录实际生效状态|

表中同文件的短写（如 vlnce.py/runner.py）与其上一完整路径同根。准确函数签名以当前文件为准。

### 3.1 可复用 expert 采集器的边界

`/home/szgx/LightNav-0/scripts/collect_go2w_expert.py:135` `ExpertEnv._label` 利用特权最短路生成标签，`:182` `expert_action` 的控制来自几何航点，RVQ量化标签不等于实际控制；`:367` 起从已有评测结果取 allowlist，默认 Go2-W配置仍是 val_unseen；`:386` meta未保存split。`:259` 和 `:325` 存在删除 episode 目录/不完整帧的逻辑，**不能直接复用到现有输出**。可借鉴pose读取和编码，另写无覆盖collector。`scripts/train_go2w.py:32` 的本地专家/纠错读取器没有官方split隔离保证，且不是原始训练流程或世界模型入口。

### 3.2 可行性分层结论

1. 数据静态就绪：train、GT及全部61场景glb/navmesh存在，route可分组。
2. 代码入口具备：split override、allowlist、factory和step返回值支持独立采集。
3. 可靠训练样本采集 **尚未就绪**：缺终止原始帧、全精度最终动作ack、稳定种子/来源契约和严格allowlist校验；不能只复用旧日志。
4. 端到端实际运行 **尚未确认**：需要下一任务运行经授权的2–3个train episode、最多20步/episode的有界试采；本轮没有运行。navmesh存在不保证Go2-W可达性/碰撞几何完全准确，Habitat圆柱体不是完整Go2-W动力学模型。

## 4. 现有日志能否训练状态转移

本轮逐行按规范化 `(scene,episode_id)` 与四个split匹配：

|文件（均在 `/home/szgx/LightNav-0/`）|行数/episode|split|可用性|
|---|---|---|---|
|`output/r2r_probe/trace_20261020/steps.jsonl`|3,259 / 30|全部val_unseen|当前JPEG、指令、输出/RVQ、选中航点、前后GT pose、距离、结束标志；非无泄漏训练数据|
|`output/go2w_expert/samples.jsonl`|66,960 / 1,749|全部val_unseen|专家帧目录、指令、动作码、距离等；缺完整前后transition与实际命令证据|
|`output/go2w_expert/corrections.jsonl`|34,572 / 315|全部val_unseen|纠错标签/model_codes等；同样不能作为train来源|

`experiments/r2r_diagnosis/probe_client.py:159` 在动作前存JPEG，`:181` step后记录pose；`:211` 从action顶层而非action_args读速度。`analyze_trace.py:98` 之后重新计算命令，并写 `action_source=recomputed_same_mapper`；`:373` 会重写 steps.jsonl，**本轮未运行该脚本**。因此 trace 中看似完整的速度也不能当服务端执行凭证。相邻非末步可用于诊断观测对，但末帧、depth、seed、route/instruction标识和全精度实际命令链不完整。不能据此恢复所有可靠 `(o_t,a_exec,o_t1)` 训练样本。

字段归类：RGB/指令/模型输出/pose/距离部分已在trace；depth和最终帧需新增保存；route/native episode与split可从源JSON计算并校验；最终动作、替换来源、执行时长和seed生效需新增服务端/launcher逻辑；当前稳定latent语义未确认。GT pose、目标、测地距离、reference_path属于仿真特权，仅离线标签。

## 5. 评测与隔离

`/home/szgx/LightNav-0/scripts/eval_habitat.sh:19` 起支持SPLIT/HABITAT_CONFIG/OUTPUT_ROOT等；`:65` 起VLN默认val_unseen，MAX_STEPS默认500。`/home/szgx/LightNav-0/src/lightnav/habitat/runner.py:715` 起写episode结果/聚合。不得为采集修改其默认值或覆盖已有output。

底层指标源码 `/home/szgx/miniconda3/envs/habitat/lib/python3.9/site-packages/habitat/tasks/nav/nav.py`：`Success:505`（update_metric中STOP且distance严格小于阈值）、`SPL:549`（成功×最短起终距离/max(最短距离,实际路径长)）、`DistanceToGoal:941`（测地目标距离，作为NE）。R2R阈值由 `/home/szgx/LightNav-0/habitat_server/lightnav_habitat/vlnce.py:81` 注入，默认3m；`:438` 的info不覆盖Habitat成功指标。

`/home/szgx/LightNav-0/habitat_server/lightnav_habitat/vlnce_extensions/vlnce_measures.py:127` `NDTW`，`:169` update_metric，以GT参考轨迹和实际位置计算 `exp(-DTW/(len(GT)*success_distance))`；fastdtw优先。`vlnce.py:98` `_inject_ndtw_measurement` 从同split `_gt.json.gz`加载。不能用进展reward或oracle success替代标准SR。

train内部先按scene留出开发集；不足时按保守route关联组划分并声明非场景泛化。同路线多指令、增强和相邻帧不可跨组。val_seen/unseen不训练、不拟合归一化/编码器/阈值；调参仅在train开发集，验证集稀疏检查并报告开发接触。test待方案锁定后使用，缺GT时遵循官方提交协议，不自造指标。

## 6. 阻塞项与最小核实

|阻塞/风险|最小核实方法|是否阻塞第一版|
|---|---|---|
|实际执行命令与请求不同、缺末帧|独立server adapter在转换后step边界出具action_id关联ack；含server STOP替换及末帧的fake测试+train试采|是，训练前必须解决|
|seed只是API参数、allowlist部分匹配|记录resolved config与seed调用证据，缺任一ID即失败；重复相同manifest的reset/order和首步控制核对|是；不能许诺GPU逐bit确定性|
|视觉特征时间/空间语义不明|engine.py:458/552、model.py:315 的tubelet、deepstack、pooling/frame映射审计；小输入契约测试，不做全评测|阻塞复用内置latent；可选独立冻结编码器绕开|
|依赖/EGL/场景可运行性与depth单位|有界train试采并保存传感器resolved config、depth归一化/量程、missing_sensor标志和有效像素检查|是，样本发布前|
|完整官方训练数据、MP3D权限|查官方manifest/release与本地许可记录|完整原语料不是采集train的必要条件；合法访问是使用前提|

推荐唯一下一步：独立、train-only、无覆盖的transition采集器及校验器，先通过小规模验收。第一版世界模型应联合预测未来视觉latent和局部位姿增量，随后把预测用于因果episode记忆检索；不是把位姿回归器当最终成果。详情见同目录设计、契约和实施计划。
