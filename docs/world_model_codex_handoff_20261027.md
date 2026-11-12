# LightNav-0 世界模型数据链路：盘点与 Codex 交接

日期：2026-10-27。这是给 Codex 的执行交接，取代同日较早的短版。旧的 Cursor Goal 跟踪已按用户要求结束。结束跟踪不表示技术就绪，不表示正式采集、特征缓存、训练或正式评测已获授权。

本轮没有重跑 pytest，没有跑仿真时钟探针，没有改授权开关，没有启动采集或特征提取。本轮对清单的当场读取只调用了 `precollection_checklist()`，不是测试套件。

状态词只使用这五种：

- **已实现**：当前源码里有这个函数或分支。
- **已有历史测试证据**：以前的会话记录过 pytest 通过，仓库里没有对应的结果文件。不能当成刚刚全量通过。
- **本轮刚刚验证**：这次盘点里读过源码，或当场调用过清单函数。
- **尚未验证**：没有在这次盘点里执行，也没有仓库内结果文件。
- **需要授权**：没有用户的新指令就不能做。

## 1. 工作区

分支：`local/research-snapshot`。最近提交是 `4933c66`（Snapshot world-model admission seals, collection gates, and feature-contract checks）。其下还有未提交修改。不要 commit，不要 `git reset`、`git checkout`、`git restore`。

已修改、未提交：

- `experiments/world_model/collect.py`
- `experiments/world_model/collection_gate.py`
- `experiments/world_model/latent_data.py`
- `experiments/world_model/serve_collect.py`
- `experiments/world_model/siglip_forward_probe.py`
- `experiments/world_model/siglip_repro_probe.py`
- `experiments/world_model/validate.py`
- `tests/world_model/test_collection_failure.py`
- `tests/world_model/test_latent_v1.py`
- `tests/world_model/test_training_admission.py`

未跟踪：

- `docs/world_model_precollection_protocol.md`
- `docs/world_model_scale_decision.md`
- `docs/world_model_transition_schema_proposal.md`
- `docs/world_model_codex_handoff_20261027.md`（本文件）
- `experiments/world_model/dev_collect.py`
- `experiments/world_model/feature_pairs.py`
- `experiments/world_model/precollection_protocol.py`
- `experiments/world_model/records/`（三份 JSON）
- `tests/world_model/test_dev_collect.py`
- `tests/world_model/test_feature_pairs.py`
- `tests/world_model/test_precollection_protocol.py`

## 2. 当场读到的开关

`precollection_checklist()` 本轮返回：

| 项 | 值 | 状态 |
|---|---|---|
| `SCHEMA_VERSION` | `wm-transition/0.1` | 本轮刚刚验证 |
| `APPROVAL_SCHEMA` | `wm-approval/0.1` | 本轮刚刚验证（源码常量） |
| `HUMAN_ACCEPTED` | `True` | 本轮刚刚验证 |
| `SCALE_ACCEPTED` | `True` | 本轮刚刚验证 |
| `COLLECTION_AUTHORIZED` | `False` | 本轮刚刚验证。这是当前未授权，不是永久禁令 |
| `technical_trial_ready()` | `False` | 本轮刚刚验证 |
| 正式运行目录 | `[]` | 本轮刚刚验证 |
| 合格正式 transition / 成对样本 / 配额计数 | 0 / 0 / 不计行 | 本轮刚刚验证 |

`COLLECTION_AUTHORIZED` 是 `experiments/world_model/precollection_protocol.py` 里的常量。用户明确授权正式采集之前，不要改成 `True`。用户授权之后，改这个常量才是离开准备阶段的出口，不要再靠追加准备工作等待它自己变成 true。

`experiments/world_model/records/human_decisions_20261027.json` 里的 `collection_authorized: false` 是 2026-10-27 “当天尚未授权”的历史记录。当前清单把它当成现场条件：改成 true 会使 `protocol_human_acceptance`、`scale_proposal`、`formal_collection_authorization` 变成 `fail`，不会授予采集。这是缺陷，见第 7 节。将来授权要改代码常量，并另写新记录，不要改这份旧 JSON。

## 3. 模块实现

### `experiments/world_model/__init__.py`

**已实现。** `SCHEMA_VERSION = "wm-transition/0.1"`。`DT_SIM_REASON` 写明没有读模拟器时钟，`Env._elapsed_seconds` 使用 `time.time()`，所以不是 `dt_sim_s`。包文档写明本包不训练世界模型。

### `experiments/world_model/latent_data.py`

**已实现** 训练准入，不是采集器。

| 函数 | 作用 |
|---|---|
| `scene_partition` | 场景 id 的 sha256 末字节 `% 10 == 0` 为 `train_dev`，否则 `train_fit` |
| `require_formal_split` / `formal_report` | fit ≥8 场景且 ≥20 路线组，dev ≥3 场景且 ≥5 路线组，否则 `FormalSplitError` |
| `unapproved_collection_component` | `pilot_`、`siglip_repro_`、`bounded_acceptance`、`acceptance_`、`world_model_diagnostics`、`r2r_probe`、`go2w_expert`、`instruction_triplet_20261021` 不能进正式训练 |
| `read_recorded_run` | 诊断读取。不授予训练 |
| `issue_run_approval` / `verify_run_approval` | 批准文件在 run 外，schema `wm-approval/0.1`，绑定目录名和快照摘要 |
| `load_accepted_run` | 训练门。拒绝未批准路径、`failure.json`、`diagnostic_only`、非本 schema、非 train、把 seed 标成 `verified`、复验硬错误、批准不符、场景下限不足。像素从同一次快照封存 |
| `supervised_rows` | 非 `valid`、未执行、缺 RGB、缺位姿、非空 `dt_sim_s`、预算步标成 `completed`，都拒绝整包 |
| `admitted_training_rows` | 没有 `admitted_frames` 就拒绝，防止绕过 `load_accepted_run` |

`load_accepted_run` 不读取 `COLLECTION_AUTHORIZED`。它拒绝的是不合格 run，不是“采集开关”。非空 `dt_sim_s` 在 `supervised_rows` 被拒绝，文案是 command time is not simulation time。null 在这里合法，配额不计是另一道门。

对应测试：`tests/world_model/test_training_admission.py`（31 个测试函数）、`test_latent_v1.py`（8）、`test_encoder_and_split_audit.py`（5）。**已有历史测试证据，本轮未重跑。** 这些文件在 git status 里是已修改或依赖已修改的 `latent_data.py`，所以历史通过不能覆盖当前工作区差异。

下一步：没有发现必须马上改的准入漏洞。不要把 null `dt_sim_s` 改成在 `supervised_rows` 里抛错，那会打断现行命令时域路径。

### `experiments/world_model/validate.py`

**已实现** 快照校验和 no-replace 发布。

- `rename_noreplace`：Linux `renameat2(RENAME_NOREPLACE)`。
- `validate_snapshot`：非空 `dt_sim_s` 记 `dt_sim_forged`（约第 698 行）；`runner_budget` 且 `completion.state==completed` 记 `budget_labeled_complete`（约第 732 行）。
- `validate_run` 走快照。`main` 是校验入口，不采集。

对应测试主要在 `test_training_admission.py`、`test_collection_contract.py`（13 个测试函数）、`test_collection_failure.py`（4）。**已有历史测试证据，本轮未重跑。** `validate.py` 有未提交修改，历史结果不覆盖当前 diff。

下一步：无。不要升级 schema。

### `experiments/world_model/precollection_protocol.py`

**已实现** 采集前清单。不采集、不训练。

调用关系：`technical_trial_ready` → `trial_blockers_clear(precollection_checklist())`。函数不接受外部报告。`checklist_blocks_collection` 是它的否定。`assert_collection_not_blocked` 在清单阻断时抛 `CollectionGateError("precollection", ...)`。`assert_feature_extraction_not_authorized` 在缓存项不是 `pass` 时抛 `FeatureContractError`。

`trial_blockers_clear` 要求报告里的 `collection_authorized is True`，且 `human_accepted`、`scale_accepted` 为真，且 `REQUIRED_GATES` 每一项状态都是 `pass`。`formal_feature_cache` 不在 `REQUIRED_GATES`。

`scale_quota_progress` **已实现**为恒定返回 0，`counts_rows=False`。传入的行被忽略，包括 `dt_sim_s` 为 null 或 0.016 的行。

下一步：第 7 节的字符串判断和第 6 节的缓存状态机。不要为了让 `technical_trial_ready()` 变成 true 而放宽条件。

### `experiments/world_model/feature_pairs.py`

见第 5 节。**已实现。** 本轮读过源码，**尚未验证**指没有本轮重跑 `test_feature_pairs.py`。

### `experiments/world_model/collect.py`

**已实现** fit 采集客户端。

- `assert_collectable` 在 `_prepare_plan` 里，早于环境。
- `main`：先看配置 split 是否为 train。`--dry-run` 走 `collection_dry_run`，不调用 `assert_collection_not_blocked`，文档写明不创建环境。非 dry-run：`_prepare_plan` 之后、`RunRecorder` 和 `subprocess.Popen` 之前调用 `assert_collection_not_blocked`。
- `run_session`：仅当 `_is_habitat_env` 为真时，在 `reset` 之前调用 `assert_collection_not_blocked`。
- 写出的 `dt_sim_s` 和观测 `sim_time_s` 是 None。观测原因字符串把“动作差为 0”和“保持 null”用 so 连起来，见第 4 节。

`collect.main` 使用 `CollectClient`（ZMQ），自己不构造 `habitat.Env`。Habitat 在 `serve_collect` 进程里。

### `experiments/world_model/serve_collect.py`

**已实现。** `prepare_server_plan` 在导入 Habitat 之前调用 `assert_collectable`。`main` 要求 Python 3.9，然后 `assert_collection_not_blocked()`，之后才 `import` Habitat 并构造 `WorldModelVLNCEEnv`。未授权时 `SystemExit`，不会执行到构造。

### `experiments/world_model/dev_collect.py`

**已实现** 的是分区规划和拒绝，不是采集。

- `prepare_dev_plan` → `assert_dev_collectable`。不查采集清单，不构造环境。
- `main` 丢弃 argv，不调用 `prepare_dev_plan`。先 `assert_collection_not_blocked`，失败则 `SystemExit`。若清单将来放行，下一句仍是 `SystemExit("train_dev collection does not construct an environment")`。
- 全文件没有 `habitat.Env`、`habitat_sim`、`Simulator`、子进程。

**需要授权** 才能执行 dev 采集或把 Habitat 构造接进来。当前不要接。

### `experiments/world_model/collection_gate.py`

**已实现。**

- `assert_collectable`：`train_dev` 场景或分区 → `CollectionGateError("train_dev", ...)`。只接受清单里的 `train_fit` 路线。
- `assert_dev_collectable`：`train_fit` → `CollectionGateError("train_fit", "... is not collected by the dev entry")`。只接受 `train_dev` 路线。
- 两边都先 `validate_scene_inventory`，对照 `scene_partition` 和 `require_formal_split`。

### `experiments/world_model/server_adapter.py`

**已实现** `WorldModelVLNCEEnv`。`__init__` 调用 `super().__init__`，父类构造 Habitat。种子写入发生在 `habitat.Env()` 之前，证据字符串在 `_apply_dataset_overrides`。`reset` 约第 402 行，`step` 约第 420 行。这个类不自己查 `COLLECTION_AUTHORIZED`；调用方 `serve_collect.main` 在构造之前查。

### 特征提取

- `siglip_forward_probe.main`：解析参数后调用 `assert_feature_extraction_not_authorized()`，然后才可能碰 GPU。**已实现。**
- `siglip_repro_probe.main`：同样在 worker 或写诊断目录之前调用该函数。**已实现。**
- `cache_writer.commit_feature_row`：`_refuse_destination` 拒绝路径中含 `formal_feature_cache`、`world_model_cache`、`world_model_diagnostics` 或 `pilot_`。它不调用提取授权函数，也不调用采集清单。不要在它开头加 `assert_collection_not_blocked`。
- `feature_cache.py`：`FEATURE_CACHE_VERSION = "wm-feature-cache/0.2"`。`check_feature_cache` 不编码。`feature_layer_selected` 必须保持 null。

### 测试文件与证据

`tests/world_model/` 里本轮数到的 `def test_` 个数如下。这不是通过数。

| 文件 | 测试函数个数 | 覆盖 | 证据 |
|---|---|---|---|
| `test_training_admission.py` | 31 | 批准、封存、`load_accepted_run`、非空 `dt_sim_s` | 已有历史测试证据；`latent_data.py` 与 `validate.py` 之后有未提交修改，本轮未重跑 |
| `test_collection_contract.py` | 13 | transition 契约，含 `dt_sim_s is None` | 同上 |
| `test_latent_v1.py` | 8 | 场景划分下限 | 文件本身有未提交修改，本轮未重跑 |
| `test_collection_gate.py` | 6 | fit/dev 隔离，含官方清单上的 dev allowlist 使 `_prepare_plan` 抛 `train_dev`、`prepare_server_plan` `SystemExit` | 已有历史测试证据；`collection_gate.py` 有未提交修改，本轮未重跑 |
| `test_encoder_and_split_audit.py` | 5 | 划分审计 | 已有历史测试证据，本轮未重跑 |
| `test_cache_writer.py` | 5 | 正式目录名拒绝 | 已有历史测试证据，本轮未重跑 |
| `test_collection_dry_run.py` | 4 | dry-run 不创建环境，dev allowlist 被拒绝 | 已有历史测试证据，本轮未重跑 |
| `test_collection_failure.py` | 4 | 失败记录；有未提交修改 | 本轮未重跑 |
| `test_siglip_forward_probe.py` | 3 | 探针契约 | 已有历史测试证据。探针 `main` 已加授权拒绝，本轮未重跑 |
| `test_precollection_protocol.py` | 2 | 清单、开关、`technical_trial_ready()` 无参数、字符串针脚 | 已有历史测试证据：与下一行一起曾 3 passed。仓库无结果文件。本轮未重跑 |
| `test_feature_pairs.py` | 2 | 对齐成功与一批拒绝 | 已有历史测试证据，本轮未重跑 |
| `test_dev_collect.py` | 1 | 分区隔离；`main([])` 匹配 `does not authorize` | 与协议测试一起的那次 3 passed。未执行到第二条 `SystemExit` |
| `test_siglip_repro.py` | 1 | 诊断复现契约 | 已有历史测试证据，本轮未重跑 |

会话里说过更早一批约 80 个 world-model 测试通过。本轮数到 85 个 `def test_`。80 不是当前文件集合的通过证明，也没有结果文件。不要写成“当前全部测试已通过”。

只查源码字符串、不调用被测函数的清单项见第 7 节。行为测试在上表里，但本轮没有重跑。

## 4. Schema 与仿真时间

**已实现** 的现行契约：

- transition schema：`wm-transition/0.1`。批准 schema：`wm-approval/0.1`。
- `dt_command_s`：命令积分时域。配置 `time_step: 0.1`。只在最终动作为 `velocity_control` 且服务器解析出该时域时填写。
- `dt_sim_s`：现行合法值只有 null。非空值被 `validate_snapshot` 记为 `dt_sim_forged`，被 `supervised_rows` 拒绝。
- `sim_time_s`：观测上的世界时间。采集器写成 null。
- 墙钟：`DT_SIM_REASON` 与 `dt_wall_s`。不能写入 `dt_sim_s`。
- 预测器第三项仍是 `dt_command_s`。位姿标签是机体增量，不是时间。

**已有历史测量证据，本轮未重跑探针。** 记录在仓库里，清单用字段相等来核对，这是本轮读到的代码行为，不是本轮新测量。

`experiments/world_model/records/sim_clock_velocity_control_20261027.json`：

| 量 | 值 |
|---|---|
| 动作前世界时间 | 0 |
| `VelocityAction.step` 返回后 | 0 |
| 动作本身的世界时间差 | 0 |
| 水平位移 | 0.25 米 |
| `step_physics` 请求 | `1/60`，记录为 0.016666666666666666 秒 |
| 物理子步 | 0.008 秒 |
| 这次物理步之后 | 0.016 秒 |
| 方案 3 观测窗口差 | 0.016 秒 |
| 额外物理步 | 0 |
| `habitat_env` / `episode` | false / false |

`VelocityAction.step` 不推进世界时间。0.016 秒是动作返回之后、读动作后观测之前那一次已有的 `step_physics`。探针按这个顺序调用，没有跑 `habitat.Env.step`，没有 episode。请求的 `1/60` 秒和测得的 0.016 秒不是同一个数。

`experiments/world_model/records/sim_clock_step_physics_20261027.json` 是另一次探针：位移 0.3904685378074646 米时时钟仍为 0；一次 `step_physics(1/60)` 后 0.016；再一次 0.032；另一次进程 `step_world(0.1)` 为 0.096。不要和 velocity 记录合成同一次实验。

采集器保持 null，是因为 `wm-transition/0.1` 不能保存这次窗口。不是因为动作差为 0。`collect.py` 约第 363 行的 `sim_time_reason` 用 so 把这两件事连在一起。清单要求该字符串同时包含 `option 3`、`VelocityAction.step measured delta 0`、`stays null`。改文案时保留这三处，否则 `dt_sim_writer_is_null` 变 fail。

提案 `docs/world_model_transition_schema_proposal.md` **已实现**为文档，没有版本号，没有 `wm-transition/0.2`。它写了动作差 0 和窗口差 0.016，但字段只准备了观测窗口两端和 `dt_sim_s = after - before`。还没分开：

- 动作已执行、物理步尚未发生时的世界时间，以及动作执行时间差。测得值是 0，0 表示测到了，不是没读。
- 物理步请求时长与测得世界时间增量。
- 0.25 米位移是状态变化，不是时间。

**需要授权** 才能分配版本号、改 `SCHEMA_VERSION`、写入非空仿真时间。修订提案文字不需要授权，也不要为此重跑探针。

## 5. 成对特征

`align_transition_latents` **已实现**。`PAIRS_AUTHORIZE_TRAINING` 是 `False`；若被改成 true，函数会抛 `pair_authorization`。返回值含 `"admits_training": False`。不编码、不开图像、不写缓存。

对齐规则，本轮读源码核对：

- 键是 `transition_id`、`obs_before_id`、`obs_after_id`。两端相同则 `reference_mismatch`。
- `obs_before` 对应 transition 的 `t`，`obs_after` 对应 `t+1`。
- 动作来自 `execution_ack.final_habitat_action`，必须与 `action.final_habitat_action` 相同，`action_id` 必须一致，`quality.status==valid`，`ack.status==executed` 且 `executed is True`。否则 `not_executed` 或 `action_id_mismatch`。
- 两行同一 `output_name`。`_select` 找不到则 `missing_feature`，多于一行则 `duplicate_feature`。
- 同一 `source_run`、`scene`、`route_group_id`、`scene_partition`。分区必须等于 `scene_partition(scene)`。
- episode 的场景和路线必须与特征行一致，`source_split==train`。
- `image_sha256` 等于观测 `rgb.sha256`，`image_path` 等于 `rgb.path`。
- `feature_layer_selected` 必须是 null。`feature_dtype` 为 float32，形状等于 `OUTPUT_SHAPES`。
- 观测 `sim_time_s` 必须存在且为 null，否则 `dt_sim_forged`。
- 非空 `action.dt_sim_s` 同样 `dt_sim_forged`。
- `_reject_row_source` 拒绝 diagnostic、pilot、非 train。

`tests/world_model/test_feature_pairs.py` 的 `test_alignment_rejects_crossed_observations_and_bad_rows` 覆盖了缺输出名、步号、编码器不一致、未执行、伪造 `dt_sim_s`、diagnostic、action_id、缺最终动作、重复特征、两端同 id、形状、pilot、已选层、场景不符、分区不符、episode、image_path、image_hash、观测时钟非空。**已有历史测试证据，本轮未重跑。** 本轮没有发现必须改接口的契约缺口，所以没有改这个文件。

清单项 `paired_latent_interface` 不调用 `align_transition_latents`。它检查常量和源码字符串，并且禁止文档里出现 `check_feature_cache` 或 `Siglip`。这是字符串检查，不是本轮的行为测试。

## 6. 准入路径

```text
fit 采集 CLI
  collect.main
    配置 split 必须是 train
    --dry-run → collection_dry_run → _prepare_plan → assert_collectable
               不调用 assert_collection_not_blocked，不创建环境
    否则 _prepare_plan → assert_collectable
         assert_collection_not_blocked
         然后才 RunRecorder / Popen(serve_collect)

Habitat 侧
  serve_collect.main
    prepare_server_plan → assert_collectable     # 尚未 import Habitat
    assert_collection_not_blocked
    然后 import Habitat，构造 WorldModelVLNCEEnv

进程内 session
  run_session
    若 _is_habitat_env：assert_collection_not_blocked，然后才 reset
    否则不查这道清单

dev
  dev_collect.main → assert_collection_not_blocked → SystemExit
  不调用 prepare_dev_plan，不构造环境

训练
  load_accepted_run → validate_snapshot → verify_run_approval → require_formal_split
  admitted_training_rows → supervised_rows
  不读 COLLECTION_AUTHORIZED

特征
  siglip_*_probe.main → assert_feature_extraction_not_authorized → 才可能加载模型
  commit_feature_row → 按目录名拒绝正式缓存路径
```

未授权时：`assert_collection_not_blocked` 抛错，CLI 变成 `SystemExit`。`serve_collect` 在构造环境之前退出。`dev_collect.main` 同样退出。探针在进 GPU 之前退出。

`_is_habitat_env` **已实现**的识别范围：模块名以 `habitat.`、`habitat_sim`、`lightnav_habitat` 开头；或 `experiments.world_model.server_adapter` 且没有 `_habitat_env` 或其为 None。内层是测试替身时返回 false，离线 session 测试依赖这一点。`test_precollection_protocol.py` 用一个 `__module__` 被改成 `server_adapter`、没有 `_habitat_env` 的类，断言 `run_session` 在 `reset` 前拒绝。**已有历史测试证据，本轮未重跑。**

绕过，**本轮刚刚验证**的是源码分支，**尚未验证**指没有拿真实 Habitat 对象执行：

- `run_session` 对模块名不在上述前缀里的对象不查清单，然后调用 `reset`。同进程包装器若类不在 `server_adapter`，就会走过这道检查。这不经过 `collect.main` 或 `serve_collect.main`。当前 CLI 用的是 `CollectClient`，不是把 Habitat 对象直接交给 `run_session`。
- `prepare_dev_plan` 不查清单。今天没有任何生产入口在它后面构造环境。
- `commit_feature_row` 可以写向不含正式目录名的路径，只要别的契约通过。正式目录名仍被拒绝。探针 main 会被授权函数拦住。
- `--dry-run` 不查采集总开关。它仍会 `assert_collectable`，所以 dev 路线进不了 fit dry-run。

本轮没有改这些路径。包装器绕过是已知边界，不是本轮要加的进程隔离。

fit 与 dev 隔离：**已实现**于两个 assert。`test_dev_collect.py` 和 `test_collection_gate.py` **已有历史测试证据**。清单项 `train_dev_not_in_fit_collector` 只搜索 `"is not collected"`，dev 的错误句里也有这几个词，所以该项不是隔离证明。

## 7. 阻塞清单

本轮 `precollection_checklist()` 的状态如下。判断方式是读源码得出的，不是新的 pytest。

| 名称 | required | 状态 | 依据 | 问题 |
|---|---|---|---|---|
| `schema_unchanged` | 是 | pass | 比较两个常量 | 无 |
| `collection_not_authorized` | 否 | pass | 常量目前为 false。true 时此项变 fail | 这是快照，不是就绪条件 |
| `dt_sim_writer_is_null` | 是 | pass | `collect.py` 源码字符串 | 不执行采集器 |
| `non_null_dt_sim_is_rejected` | 是 | pass | `validate.py` 含 `dt_sim_forged` | 字符串，不调用 `validate_snapshot` |
| `budget_not_relabeled_complete` | 是 | pass | 源码含 `budget_labeled_complete` | 字符串 |
| `train_dev_not_in_fit_collector` | 是 | pass | 源码含 `is not collected` | 误报风险：dev 错误句也能满足。行为在 `assert_collectable` 和 `test_collection_gate.py` |
| `inventory_capacity_matches_record` | 是 | pass | 读划分 JSON 计数，且 `collection_status=not_collected` | 文件事实 |
| `code_and_offline_acceptance` | 是 | pass | `latent_data.py` / `validate.py` 含指定句子 | 不重跑测试。证据字段自己写了 earlier suite was not repeated |
| `sim_clock_semantics` | 是 | pass | 两份 JSON 字段相等，且决定记录是 `option_3` | 不重跑探针。字段不符会 fail |
| `paired_latent_interface` | 是 | pass | 常量 False 加源码字符串 | 不调用 `align_transition_latents` |
| `formal_feature_cache` | 否 | blocked | 见下文 | 没有 pass 分支，五个原因没有分开 |
| `protocol_human_acceptance` | 是 | pass | `HUMAN_ACCEPTED` 且决定 JSON 满足一组条件，其中包括 `collection_authorized is False` | 旧记录被当成永久条件 |
| `scale_proposal` | 是 | pass | 配额函数仍全 0，决定 JSON 有效，且 `SCALE_ACCEPTED` | 不表示已经采够 |
| `dev_collection_path` | 是 | pass | 源码有函数名，且没有子串 `habitat.Env` | 不调用 `main`。pass 不等于可以执行 dev 采集 |
| `formal_collection_authorization` | 是 | `BLOCKED_HUMAN_DECISION` | 旧 JSON 的采集字段必须是 false，且没有非正式目录；常量仍为 false | 这是当前唯一的必要阻塞 |
| `eligibility_floor` | 是 | pass | 四个下限常量等于 8/20/3/5 | 常量比较 |
| `collector_consults_checklist` | 是 | pass | 两个文件含 `assert_collection_not_blocked()` | 不执行这两个 main |

正式特征缓存 **已实现** 的判断只有：

- writer 源码不再含拒绝词 → `fail`
- `formal_feature_cache` 或 `world_model_cache` 目录存在 → `fail`
- 两个探针源码没有调用授权函数 → `fail`
- 否则 `blocked`，证据是提取还没开始

没有这些状态：契约未满足、离线验收未完成、尚未授权、技术条件已满足仍等待独立授权、独立授权且检查通过。`wm-feature-cache/0.2` 已经是检查器契约，不是“接口未写”。接口未写这个判断今天不成立。独立授权常量不存在，所以“技术已满足、只差授权”也无法表示。不要把这个 gate 放进 `REQUIRED_GATES`。采集可以在缓存之前发生；缓存授权也不能代替采集授权。

修复缓存状态机时要验证：现场在没有独立授权时不能变成 `pass`；五个原因能用纯函数测到；不创建目录；现有 `test_precollection_protocol.py` 仍看到 `blocked` 且 `required is False`。**需要授权** 才能把现场状态设为已授权。写纯函数和离线测试不需要。

## 8. 已接受的决定和未授权事项

来源：`experiments/world_model/records/human_decisions_20261027.json`。不要再请用户确认这些。

**已接受，不要改门槛：**

- 协议正文。
- 报告门槛整组。只改一行就必须把 `SCALE_ACCEPTED` 设回 false。
- 方案 3：动作前、动作后观测各自的世界时间。
- 可以写 `train_dev` 入口，不执行采集。
- 正式采集尚未授权。
- 可以分批采集。清单容量不是一次采完的定额。
- 报告门槛不能看完结果再改。

三层数字：

| 层 | 数字 |
|---|---|
| 加载资格 | fit ≥8 场景、≥20 路线组；dev ≥3 场景、≥5 路线组 |
| 清单容量 | fit 53 / 2977 / 8938；dev 8 / 626 / 1881。`collection_status=not_collected` |
| 报告门槛 | fit：每种 `action_kind` ≥20，每个被采场景 ≥2 条 episode，非预算 episode ≥20。dev：全部 8 个预注册场景，路线组 ≥5，每种动作 ≥10，非预算 ≥5 |

动作类型：`stop_threshold`、`turning`、`mostly_forward`。非预算 = `completion.state=completed` 且没有 `runner_budget`，不是导航成功。计数要等仿真时间契约和成对特征都存在。现在配额函数不计行。

**未授权：**

- 正式采集，以及启动受控试采。
- 正式特征缓存和特征提取。
- 下一版 transition 的版本号和非空 `dt_sim_s`。
- 执行 dev 采集，以及在 dev 入口构造 Habitat。
- 训练和正式导航评测。
- 把 `feature_layer_selected` 选成某个层。

`docs/world_model_scale_decision.md` 与上述决定冲突的句子，本轮只记录、不改，以免动到测试要求的固定文本：

- 第 5 行写“当前决定都是未决定”，同页表里协议、规模和方案 3 已是接受。
- 第 29 行写“开发集还没有采集入口”。`dev_collect.py` 已存在。同文件第 71 行写了入口。
- 第 48 行写“暂不接受任何一行，技术就绪检查都会继续失败”。`SCALE_ACCEPTED` 已是 true，`scale_proposal` 本轮是 pass。
- 第 52 行写记录对上不表示已经选定时间窗口。第 65 行写已选方案 3。

改这份文档时必须留下测试用的字面量：`按门槛分别决定`、`SCALE_ACCEPTED`、`false`、`已整组接受`、`方案 3`。协议正文的固定句见 `tests/world_model/test_precollection_protocol.py` 的 `test_protocol_document_states_the_four_gates`。不要整文件格式化。

## 9. 受控试采方案（只供审查）

**未授权，本轮不启动。** 下面没有替用户选数量或时长。

目的：在单独授权后，用有限的仿真步确认采集器、记录和失败退出能跑通。它不是正式训练集，也不是导航评测。

与正式采集的区别：

- 正式采集要 `COLLECTION_AUTHORIZED=True`，输出目录不能被 `unapproved_collection_component` 命中，还要通过 `validate_snapshot` 和 `load_accepted_run`。
- 试采目录必须落在未批准组件里，例如以 `pilot_` 或 `acceptance_` 开头，或放在 `output/world_model_diagnostics`。这样 `load_accepted_run` 会拒绝它。
- 现有 `output/world_model_data/pilot_20261023_train3` 与 `pilot_20261023_train3_r2` 已是这种隔离。它们不是新试采，也不能当训练数据。

尚未决定，不能由接手人填数：

- 场景数、episode 数、最大步数、墙钟上限。
- 试采目录的具体名字。
- 试采记录以后能否晋升为正式候选。晋升至少需要：目录不再属于未批准组件、schema 仍是当时接受的版本、`validate_snapshot` 硬错误为 0、批准文件、场景下限，以及用户另一次书面同意。仿真时间仍为 null 的行可以有文件，但计入不了已接受的报告门槛。

可以写死的边界：

- 不使用真实机器人，不改 `robot_deploy`，不启动机器人 launch。
- 不跑正式导航评测，不把试采当 LightNav 成功率。
- 使用仿真时才涉及 Habitat 和 VLN episode。没授权就不能起 `habitat.Env`。
- 异常时走现有 `abort_collection` / `failure.json`。有 `failure.json` 的 run 不能进 `load_accepted_run`。
- 步数上限现有代码是 1 到 20。这是入口限制，不是用户选定的试采预算。

离线可做的只有：确认未批准路径会被拒绝。这在 `load_accepted_run` 和 `test_training_admission.py` 里已有历史测试。本轮不新增试采测试，也不创建目录。

## 10. 交给 Codex 的任务

可以现在做、不需要新授权的，按这个顺序。不要等采集授权才做第 1 项。

### P1. 修订仿真时间提案

- 状态：提案已有，字段未分开。
- 改：`docs/world_model_transition_schema_proposal.md`。可选地改 `collect.py` 的 `sim_time_reason`，保留三个清单针脚。
- 原因：第 4 节的测量。0.016 不是动作耗时，也不是 `1/60`。
- 验收：提案同时写出命令时间 0.1 秒、动作世界时间差 0、请求 `1/60`、测得增量 0.016、观测窗口 0.016、位移 0.25 米。`SCHEMA_VERSION` 仍是 `wm-transition/0.1`。采集器仍写 null。文件中没有下一个版本字符串。
- 授权：不需要。不要重跑探针。
- 可离线完成：是。

### P1. 修正规模文档里与已接受决定矛盾的句子

- 状态：第 8 节列出的四句。
- 改：`docs/world_model_scale_decision.md` 仅这些句子。
- 验收：矛盾句消失；第 8 节的固定字面量还在；`SCALE_ACCEPTED` 和 `COLLECTION_AUTHORIZED` 的值不变。
- 授权：不需要。不要把这扩成新架构。
- 可离线完成：是。改完只跑 `tests/world_model/test_precollection_protocol.py`。

### P2. 特征缓存状态机的纯函数和离线测试

- 状态：未实现五种原因。现场只能是 `fail` 或 `blocked`。
- 改：`precollection_protocol.py` 里的纯函数，加上 `tests/world_model/` 中的新测试或现有协议测试。现场在没有独立授权常量时必须仍是 `blocked`。不要新增 `FEATURE_CACHE_AUTHORIZED=True`。
- 原因：第 7 节。接口和 `wm-feature-cache/0.2` 已经存在，不能叫“未实现”。
- 验收：纯函数能返回五种原因；现场调用不是 `pass`；不创建缓存目录；`formal_feature_cache` 仍 `required=False`。
- 授权：把现场标成已授权需要用户授权。写失败和等待状态不需要。
- 可离线完成：是。

### P2. 清单里有证据的误判

- 状态：`train_dev_not_in_fit_collector` 和 `dev_collection_path` 是字符串。
- 改：让这两项调用 `assert_collectable` / `assert_dev_collectable` 的合成清单，或改为读取测试已覆盖并在证据里写明“字符串不是证明”。不要重写整个清单。
- 验收：删掉 fit 函数里的 train_dev 拒绝后，该项变 fail；只在 dev 错误句里留 `"is not collected"` 不能再让它 pass。用合成数据，不读官方大 JSON，不启动 Habitat。
- 授权：不需要。
- 可离线完成：是。只跑协议测试和 `test_dev_collect.py`。

### P3. dev 入口第二条退出的测试

- 状态：`main` 在清单失败时到不了“不构造环境”那一句。
- 改：`tests/world_model/test_dev_collect.py`。不要给 `dev_collect.py` 加上 Habitat 构造。
- 验收：在测试里证明未授权退出发生在任何环境工厂之前；文件仍不能构造 Habitat。
- 授权：执行 dev 采集需要授权。这个测试不需要。
- 可离线完成：是。

### 不要做

- 不要改 `COLLECTION_AUTHORIZED`、`HUMAN_ACCEPTED`、`SCALE_ACCEPTED`。
- 不要把 2026-10-27 的 JSON 采集字段改成 true。
- 不要升级 schema，不要写非空 `dt_sim_s`。
- 不要重跑时钟探针、官方 split 遍历，或整套 `tests/world_model`。
- 不要创建 `formal_feature_cache` 或 `world_model_cache`。
- 不要启动采集、试采、训练、导航评测、真实机器人。
- 成对接口没有发现缺口，不要重写。
- 不要把试采目录晋升为训练数据。
- 不要调用 `CreateGoal`。旧 Goal 跟踪已经结束。

## 11. Codex 打开仓库后的第一批

1. 读本文件第 1、2、8 节，确认没有新的未提交文件被误删。
2. 做 P1 的提案修订。
3. 做 P1 的规模文档矛盾句，只跑 `test_precollection_protocol.py`。
4. 把结果写回本文件末尾的执行记录，标明改了哪些文件、跑了哪条命令、哪些仍是历史证据。

P2 可以接着做，但不要和采集授权绑在一起。

## 12. Codex 执行记录（2026-10-27）

本节记录交接后的离线工作，不追溯改写上文的“本轮”盘点。打开仓库时分支为 `local/research-snapshot`、HEAD `4933c66`，上文所列原有未提交修改及未跟踪文件仍在；没有 commit、reset、restore 或清理。

- **已实现**：修订 `docs/world_model_transition_schema_proposal.md`，分开命令时域 0.1 秒、测得动作世界时间差 0、物理步请求 `1/60` 秒、测得增量 0.016 秒、观测窗口 0.016 秒和水平位移 0.25 米；没有分配下一版 schema，现行采集器仍写 null。
- **已实现**：仅修订 `docs/world_model_scale_decision.md` 中与已接受决定冲突的叙述，保留协议测试所需字面量与原数字；授权开关值未变。
- **已实现**：`precollection_protocol.py` 新增无副作用的 `feature_cache_gate_state` 五种原因判断。现场 `_formal_cache_status()` 仍是 `blocked`，`formal_feature_cache` 仍非必要门，未创建缓存目录或独立授权常量。
- **已实现**：fit/dev 清单改为调用合成、足量的 train split 上的 `assert_collectable` 与 `assert_dev_collectable`，不读官方大 JSON、不构造 Habitat。合成门放行交叉分区时，对应项变 `fail`。`test_dev_collect.py` 增补两次退出路径测试；入口本身仍不构造环境。
- **本轮刚刚验证**：`.venv/bin/python -m pytest tests/world_model/test_precollection_protocol.py -q` 为 `2 passed`；之后 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/world_model/test_precollection_protocol.py tests/world_model/test_dev_collect.py -q` 为 `10 passed`。两次均有 `Unknown config option: asyncio_mode` 警告。对三个改动的 Python 文件执行 `.venv/bin/ruff check` 为 `All checks passed!`。
- **尚未验证**：未重跑整套 `tests/world_model`、时钟探针、真实 Habitat 对象、正式采集、特征提取、训练或导航评测。此前其他测试通过记录仍只是**已有历史测试证据**，不是当前 dirty 工作区的全量结果。
- **需要授权**：正式/受控试采、dev 采集、正式特征缓存、schema 升级、非空 `dt_sim_s`、选择特征层、训练和正式评测。`COLLECTION_AUTHORIZED=False`、`HUMAN_ACCEPTED=True`、`SCALE_ACCEPTED=True` 保持不变；旧决定 JSON 未改。

## 13. 后续离线准备与剩余门槛（2026-10-27）

此节是新的执行记录，不把第 1–11 节所述“本轮未重跑”改写为当时已运行。

- **本轮刚刚验证**：对采集分区、dry-run、transition契约、失败记录、训练准入和特征配对运行指定 CPU 测试，`60 passed`；对协议/dev入口测试运行 `11 passed`。均有 pytest 的 `Unknown config option: asyncio_mode` 警告；本节没有运行整套 world-model、仿真时钟探针或真实 Habitat。Ruff 对本节改动的协议文件和测试通过。
- **已实现**：修复将来授权状态转换的死结。`precollection_protocol.py` 保持旧 `human_decisions_20261027.json` 的 `collection_authorized:false` 为历史事实；只有 `COLLECTION_AUTHORIZED=True` 且另有新的 `records/collection_authorization.json` 显式标注 `scope=train_fit`、人类授权来源和记录日期时，正式授权项才可能通过。尚未授权时正式目录仍是 `fail`；授权后正式目录出现不再自动反咬后续 reset。新记录**未创建**，开关仍为 false。离线测试验证了这两侧和完整清单模拟状态；模拟授权不改变现场。
- **本轮刚刚验证**：现场 `precollection_checklist()` 的必要项中仅 `formal_collection_authorization=BLOCKED_HUMAN_DECISION`，`technical_trial_ready()=False`，正式运行目录 `[]`、配额 `counts_rows=False`。清单其余 `pass` 混合了合成行为测试、源码字符串和历史记录字段核对，不等于真实环境全部验证。
- **本轮刚刚验证**：已有 `output/world_model_data/pilot_20261023_train3_r2/validation_report.json` 记 3 个 episode、60 条 valid transition、0 个硬错误，manifest 指明 `source.split=train`、最多20步、部分种子状态。它是先前试采的**已有历史测试证据**，不是本节新采集；目录以 `pilot_` 开头，`dt_sim_s` 为null，没有正式成对特征，不得晋升为训练数据或配额。
- **尚未验证**：正式授权后的真实启动、服务端重置、完整train覆盖、方案3的 `habitat.Env.step` 时钟边界、正式特征提取/配对及资格门槛。现场没有合格正式 transition，不能宣布“数据开采要求已全部满足”。
- **需要授权**：正式采集应由用户明确指定范围及启动；之后才可新建独立授权记录并改变采集常量。非空仿真时间要求另行批准新的 schema 并在真实 episode 路径验证；正式特征缓存、dev采集、训练、评测仍各自受原有边界约束。不得把本次“继续探索”当成这些许可。
