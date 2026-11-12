# 世界模型正式采集启动评审

日期：2026-10-26。这是受限小批量采集的启动评审和 dry-run，不是采集结果。正式采集、正式特征缓存和预测器训练都没有启动。`feature_layer_selected` 仍是 null。

建议结论：**附条件通过**。dry-run 已经证明下面这份计划可以被入口接受，并且没有创建 run 目录。条件写在第 6 节。条件未接受之前，不要启动采集。

## 1. 实际入口和调用链

采集入口是 `python -m experiments.world_model.collect`。本轮给它增加了 `--dry-run`。`experiments/world_model/group_manifest.py` 里原有的 `--dry-run` 只打印官方 train JSON 经路线组哈希之后的汇总；它不读取 `wm_scene_split_0.1.json`，也不检查输出目录。它不能代替采集入口的 dry-run。

不带 `--dry-run` 时，`collect.main` 的顺序是：

1. 解析参数。`--config`、`--allowlist`、`--scene-inventory`、`--run-dir` 都是必填。`--max-steps` 只接受 1 到 20。
2. `_require_train_config` 用 `yaml.safe_load` 读取 `habitat.dataset.split`。正式启动和 `--dry-run` 都先做这一步。值不是字符串 `train` 时变成 `SystemExit`，此时 `RunRecorder` 还没有创建。
3. `_prepare_plan` 读取 allowlist 和官方 train `json.gz`，调用 `check_allowlist_against_train`，再调用 `assert_collectable`。后者调用 `validate_scene_inventory`，其中用清单里的路线组调用 `require_formal_split`。
4. 上述任何失败都变成 `SystemExit`。此时 `RunRecorder` 还没有创建，Habitat 进程也还没有 `Popen`。
5. 检查通过之后才 `RunRecorder(run_dir)`。已有目录会在 `mkdir(exist_ok=False)` 处失败，不会清空，也不会补写 `failure.json` 盖住原目录。
6. `--launch-server` 才启动 Habitat 3.9 的 `serve_collect.py`。服务器在导入 `lightnav_habitat` 之前执行 `prepare_server_plan`，重复清单检查。`serve_collect` 构造环境时传入 `split="train"`。
7. `run_session` 对每个计划行调用 `reset`。服务端公布的 served key 顺序必须等于整个计划；本次 `wm_reset.key` 必须等于当前行的 `(scene, episode_id)`。不一致时在 `save_observation` 和 step 循环之前抛出 `RuntimeError`。`main` 捕获后写 `failure.json` 并返回 1。
8. 服务器侧 `WorldModelVLNCEEnv.reset` 在返回的 episode 不在 `served_keys` 中时抛出，不调用 `step`。`_apply_episode_split` 把迭代器固定为 allowlist 顺序，并在顺序不一致时抛出。`_apply_dataset_overrides` 在父类改写配置之后检查 `self.split` 和 `config.habitat.dataset.split` 都是 `train`，否则抛出。dry-run 没有构造 Habitat 环境，所以这一段只由 mock 测试证明，没有在真实模拟器上执行。

`--dry-run` 先做同一项配置检查，再做第 3 步的计划检查，然后打印计划并返回 0。它不创建 `RunRecorder`，不调用 `Popen`，也不写 transition。传入 `--launch-server` 和 `--habitat-python /bin/false` 时仍然返回 0，run 目录不出现。

清单校验同时做两件事。`validate_scene_inventory` 要求清单内部的 `dataset_sha256` 等于 train `json.gz` 的字节摘要。dry-run 再对清单文件本身调用 `sha256_file`，把文件摘要写进输出。本次官方文件摘要是 `95b22fa97c231c3feea68e6622c8c7e4741f93ecbaa4a474ae6a7998a242d907`。把内部 `dataset_sha256` 改成别的值之后，dry-run 退出码是 1，消息是 `dataset_hash`，提议的 run 目录没有出现。官方清单文件的摘要在这次篡改测试之后仍然是上面的值。

`assert_collectable` 拒绝 `source_split` 不是 train 的行，拒绝清单中标成 `train_dev` 的场景，也拒绝不在该场景 `train_fit` 路线组里的 episode。场景标签必须等于 `scene_partition(scene)`，所以不能把 `train_dev` 场景改标后放进来。allowlist 里的 `partition` 若是 `val_seen` 或 `val_unseen`，在进入场景清单检查之前就被 `check_allowlist_against_train` 拒绝。

配置检查读的是 YAML 结构里的 `habitat.dataset.split`，不是文件里有没有 `split: train` 这几个字。注释里写 `split: train`、字段实际是 `val_unseen` 时，dry-run 和正式启动都拒绝，并且不创建 run 目录。`habitat_server/configs/vlnce_r2r.yaml` 的字段是 `val_unseen`，因此被拒绝。`collect_train_go2w.yaml` 的字段是 `train`。运行时 `WorldModelVLNCEEnv._apply_dataset_overrides` 再要求改写后的配置对象仍是 `train`；构造参数如果是 `val_unseen`，覆盖函数抛出，不会进入 `habitat.Env`。`allow_order` 不由这段覆盖改写。空的 `allow_order` 仍由 `_apply_episode_split` 拒绝。

## 2. 代码已经证明的记录关系

这些结论来自 `collect.py`、`server_adapter.py` 和 `validate.py` 的当前实现，不是来自字段名称。

一条 transition 在写入时带有 `episode_uid`、`t`、`obs_before_id`、`obs_after_id`、`action.action_id` 和 `action.execution_ack`。观测记录带有 `step_index`、`rgb.path` 和 `rgb.sha256`。`validate_run` 在 run 关闭之后重新读取 `.npy`，核对 sha256、形状和 dtype，并检查后一帧的 `step_index` 比前一帧大 1。图像哈希不符会进入 `hard_error_count`。这个检查发生在整次 run 写完之后，不是在下一帧采集之前。`commit_validation` 在 `hard_error_count` 非 0 时写 `failure.json`，状态是 `validation_failed`，并返回 1。目录和已有文件保留。`failure.json` 已存在时不覆盖。写失败记录本身失败时，原始错误和写入错误都打到 stderr。reset 失败、图像保存的 `FileExistsError` 和 Habitat 子进程提前退出走 `abort_collection`，同样写失败记录并返回 1。step 里已经捕获的传感器异常仍写成一条 `quarantined` transition，不把整次目录删掉。只要目录里还有这种记录，训练入口就整包拒绝，不会跳过坏行后留下其余样本。

`action.dt_sim_s` 在 `_write_transition` 里固定写成 `None`。服务器只在最终动作是 `velocity_control` 且已经解析出 `habitat_time_step` 时填写 `dt_command_s`，并单独写 `dt_sim_reason`。YAML 里的 `time_step: 0.1` 是命令时间步的配置来源，不是已经测量的模拟器时间。

结束来源分列保存。客户端在最后一步放入 `runner_budget`，并使用 `model_stop` 为 null 的空预测。`model_stop is True` 才加入 `model_stop`。`habitat_stop_called` 才加入 `habitat_stop`。`truncated` 才加入 `habitat_limit`。来源列表为空时才放入 `unknown`。`primary_end_source` 按固定优先级从这些名字里取一个，其余名字仍留在 `end_sources`。碰撞写在特权标签里，状态固定为 `unknown`，来源是 `not_read`；它不进入 `end_sources`。校验器另外拒绝把 `decode_failed` 和 `model_stop` 同时当成有效 STOP。

采集循环不调用 SigLIP，也不调用 `commit_feature_row`。特征维度、有限值和权重哈希不是这次采集入口能证明的事情。

单个 episode 在首帧缺 RGB 或 step 抛错时，会写下该 episode 的隔离记录并继续后面的 episode。reset 不匹配、served 顺序不匹配和指令文本不匹配则会中断整个 `run_session`。

## 3. 已通过的前置门禁

| 门禁 | 证据 | 本轮是否重新执行 |
|---|---|---|
| 场景清单与 `require_formal_split` | `tests/world_model/test_collection_gate.py` | 是，包含在下面的 45 项里 |
| reset 不在 allowlist 时不 step | `test_collection_dry_run.py::test_reset_outside_allowlist_fails_before_step` | 是，mock 环境 |
| 写入前哈希与违规行拒绝 | `tests/world_model/test_cache_writer.py` | 是，包含在 45 项里 |
| 同 GPU 跨进程特征复现 | `docs/world_model_reproducibility.md` | 否。那次结果不外推到其他 GPU |
| `feature_layer_selected` 必须为 null | 特征契约测试与本次 dry-run 输出 | dry-run 输出里该字段是 null |

隔离修复之后，`.venv/bin/pytest -q tests/world_model` 是 51 passed，退出码 0。新增的是训练入口拒绝测试和采集失败记录测试。旧 pilot 目录没有被读入 allowlist。

## 4. dry-run 命令和副作用

命令：

```bash
.venv/bin/python -m experiments.world_model.collect \
  --dry-run \
  --launch-server \
  --habitat-python /bin/false \
  --config experiments/world_model/configs/collect_train_go2w.yaml \
  --allowlist output/world_model_diagnostics/collection_start_review_20261026/allowlist.json \
  --scene-inventory experiments/world_model/splits/wm_scene_split_0.1.json \
  --dataset-root data/datasets/R2R_VLNCE_v1-3_preprocessed \
  --run-dir output/world_model_data/pilot_bounded_acceptance_20261026T015701Z \
  --max-steps 4
```

退出码 0。第一次标准输出保存在 `output/world_model_diagnostics/collection_start_review_20261026/dry_run_stdout.json`，没有被覆盖。隔离修复后用同一命令重跑，退出码仍是 0，新输出写在同目录的 `dry_run_stdout_isolation.json`。两次都没有创建 run 目录。重跑输出里 `config_split` 是 `train`，`training_use` 是 `not_approved`。内容是：

- `episodes` 2，`scenes` 2
- `keys`：`["VzqfbhrpDEA", "10"]`、`["cV4RVeZvu5T", "13"]`
- `route_group_ids`：`VzqfbhrpDEA:7`、`cV4RVeZvu5T:8`
- `max_steps` 4，`step_upper_bound` 8
- `formal_collection_started` false，`formal_feature_cache` false，`training` false
- `feature_layer_selected` null，`accepted_for_formal_training` false
- `output_exists_after_check` false
- 数据集 sha256 `340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea`
- 清单文件 sha256 `95b22fa97c231c3feea68e6622c8c7e4741f93ecbaa4a474ae6a7998a242d907`
- allowlist sha256 `4ff9e37b8b0c052091905a2adec4c20cbe202f082069749bbbee75958339d2a0`
- 配置 sha256 `3ce54c5c963cf9634bb1bd4f273c6ac871896cea8617a8f6b259de5e28fee987`

副作用：提议的 run 目录在命令前后都不存在。dry-run 没有创建它。评审输入 `allowlist.json` 和这份标准输出副本写在 `output/world_model_diagnostics/collection_start_review_20261026/`，该目录被 `.gitignore` 忽略，不是 transition，也不是特征缓存。

另外三次拒绝：

| 检查 | 退出码 | 结果 |
|---|---:|---|
| `--config habitat_server/configs/vlnce_r2r.yaml`，dry-run 与不带 `--dry-run` 的正式入口各一次 | 1 | `config_split: habitat.dataset.split is val_unseen`，run 目录仍不存在 |
| `--run-dir /tmp/wm_dry_occupied`，其中已有 `keep.txt` | 1 | `run_dir_exists`，`keep.txt` 仍是 `keep` |
| 清单副本的 `dataset_sha256` 被改掉 | 1 | `dataset_hash`，run 目录仍不存在 |

pytest 里的合成清单覆盖了合法 train、`train_dev`、`val_unseen` 分区、场景标签交叉、清单哈希不符、输出目录已占用，以及 dry-run 不产生 `.jsonl`。reset 不在 served allowlist 时 `step_calls` 为 0。

## 5. 受限采集方案

这一批只作为上线验收，不进入正式训练集。规模是两个合法 `train_fit` 场景各一个 episode，每个 episode 最多 4 步，因此 transition 上限是 8。两个场景是 `VzqfbhrpDEA` 和 `cV4RVeZvu5T`。选择时跳过了旧 pilot 使用的 `7y3sRwLe3Va` 和 `5LpN3gDmAk7`。八个 `train_dev` 场景不在 allowlist 中。

建议策略是 `--policy scripted`。脚本动作仍经过 Habitat 的执行回执和图像保存，用来核对数据链路。它不加载 LightNav checkpoint。on-policy 采集是另一次审批。

建议 run 目录：

`output/world_model_data/pilot_bounded_acceptance_20261026T015701Z`

`RunRecorder` 对已存在目录直接失败，所以这个名字只能用一次。父目录 `output/world_model_data` 目前存在。

若以后人工批准并去掉 `--dry-run`、加上 `--launch-server`，入口会创建：

- `episodes.jsonl`、`observations.jsonl`、`transitions.jsonl`、`privileged_labels.jsonl`
- `assets/*.npy`
- `manifest.json`、`validation_report.json`
- `habitat_server.log`、`habitat.ready`
- 只有循环抛错时才有 `failure.json`

同一次命令不会创建正式特征缓存、预测器 checkpoint，也不会改 `wm_scene_split_0.1.json`。

采集后的核对，仍然要另一次人工动作，不要和启动绑在一起：

- `validation_report.json` 的 `hard_error_count` 必须是 0。
- manifest 的 `allowlist_keys`、`served_keys`、`reset_keys` 都等于上面两个 key，顺序一致。
- episode 场景集合只有 `VzqfbhrpDEA` 和 `cV4RVeZvu5T`。
- transition 数不超过 8，观测能用 `obs_before_id` 和 `obs_after_id` 串起来。
- 每张 RGB 的 sha256 与记录一致。
- `dt_sim_s` 仍是 null。`runner_budget`、`model_stop`、`habitat_stop` 和 `unknown` 按第 2 节的字段分别存在，不互相改写。
- 不把这个目录交给 `load_accepted_run` 或训练脚本。
- 不对这批图像写正式特征缓存。特征维度和有限值要等单独的缓存评审。

`wm-transition/0.1` 没有改。采集器仍把 manifest 里的 `training_use` 写成 `not_approved`，这个字符串本身不能批准训练。`load_accepted_run` 还要求一份放在 run 目录外面的 `wm-approval/0.1` 记录：它绑定目录名，并用仓库外的密钥对 manifest、report、四份 JSONL 和 `assets/` 做 HMAC。缺记录、损坏、目录名不符、摘要不符或批准后改了文件，都整包拒绝。加载时会重新跑 `validate_run`，并核对 report 计数；任何不是 `valid` 的 transition，包括带原因的 quarantine，都拒绝整个 run。`diagnostic_only: true` 也拒绝。路径分量另有一条独立规则：以 `pilot_` 或 `siglip_repro_` 开头、含 `bounded_acceptance`、以 `acceptance_` 开头，或名字是 `world_model_diagnostics`、`r2r_probe`、`go2w_expert`、`instruction_triplet_20261021`。因此旧 pilot、计划目录 `pilot_bounded_acceptance_20261026T015701Z`，以及在该目录里伪造 `training_use: formal`，都进不了训练样本。这条规则在读取 manifest 之前执行。特征缓存写入器的路径拒绝不能代替它。`read_recorded_run` 只做诊断读取，不是训练入口。现存 pilot 没有被自动批准。

## 6. 尚未验证的风险和停止条件

以下任一情况出现，都停止，不扩大规模，不写正式缓存，不训练：

- 启动前重跑 dry-run，清单文件 sha256 不是 `95b22fa97c231c3feea68e6622c8c7e4741f93ecbaa4a474ae6a7998a242d907`，或退出码不是 0。
- run 目录已经存在。
- Habitat 进程在 ready 文件出现前退出。
- reset 的 scene、episode 或顺序与 allowlist 不一致。
- `validation_report.json` 的 `hard_error_count` 不是 0，或目录里出现 `failure.json`。这样的目录不能当成验收通过，也不能进入 `load_accepted_run`。
- 出现 `train_dev`、val、test 或旧 pilot 路径里的图像。
- 有人把 `pooler_output` 或 `last_hidden_state` 写入 `feature_layer_selected`。
- 把本次 run 交给正式训练或正式特征缓存。

还没有证据的事项：真实 Habitat reset 的现场记录、新图像的 SigLIP 特征、跨 GPU 数值一致性。`load_accepted_run` 对新 `pilot_` 和 bounded-acceptance 目录的拒绝已由直接调用训练入口的测试覆盖，不是只靠目录命名约定。碰撞字段今天固定是 `unknown` / `not_read`，不能当成接触传感器已经测量。

本轮改动文件的 `.venv/bin/ruff check` 退出码是 0。此前全仓检查的 13 个既有错误没有在本轮修改，也没有重跑全仓来把它们算进这次结果。

## 7. 建议结论

附条件通过，试采仍未批准。训练入口拒绝 pilot、diagnostic 和 bounded-acceptance 路径，并且不再把 manifest 里的 `formal` 当作批准；校验失败和逃出采集循环的异常会留下 `failure.json` 与非零退出。这些代码修复不构成启动许可。两个 episode、每集最多四步只够核对数据链路，不能证明策略性能、特征预测或跨场景泛化。是否进入受限试采，仍要人工决定。在那之前不启动 Habitat 采集，不写 transition，不写正式特征缓存，不训练。
