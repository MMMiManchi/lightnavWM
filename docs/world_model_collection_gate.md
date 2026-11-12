# 世界模型采集器与场景清单闭环

日期：2026-10-23。清单版本 `wm-scene-split/0.1`。正式场景清单文件没有被修改。

采集器在启动前直接读取这份清单，并用 `require_formal_split` 重算门槛。`train_dev` 场景会让启动失败，不会被悄悄丢掉。本轮没有启动 Habitat，没有写正式 transition。

结论：采集器已经被证明只消费合法的 train 清单。这只允许进入正式采集启动评审。

## 1. 实际调用链

`experiments/world_model/collect.py` 的 `main` 现在必须收到 `--scene-inventory`。它在创建 run 目录、拉起 Habitat 进程之前调用 `_prepare_plan`：

1. 读取 `wm-allowlist/0.1`。
2. 读取官方 train `json.gz`，核对 allowlist 的 schema、train split、数据集 sha256、重复 id 和 `val_seen` / `val_unseen` 分区标签。
3. 读取场景清单，调用 `experiments/world_model/collection_gate.py` 的 `validate_scene_inventory`。
4. `assert_collectable` 要求每一个将要 reset 的 `(scene, route_group_id)` 都出现在清单的 `train_fit` 路线组里。

`validate_scene_inventory` 不相信清单里的 `eligible: true` 单独成立。它检查：

- `scene_split_version` 是 `wm-scene-split/0.1`
- `source_split` 是 `train`
- `dataset_sha256` 等于调用方从 train `json.gz` 字节算出的摘要
- 每个场景的 `scene_partition` 等于 `scene_partition(scene)`，不能把 `train_dev` 场景改标成 `train_fit`
- 路线组不跨场景，场景 id 不重复
- `n_route_groups` 与 `route_group_ids` 长度一致
- 用清单里的路线组重建 episode，再调用 `require_formal_split`

门槛仍是训练侧至少 8 个场景、20 个路线组，验证侧至少 3 个场景、5 个路线组。场景侧由场景 id 的 sha256 决定。采集器不再用路线组哈希自己筛一遍官方 JSON 来决定哪些场景可以采集。

Habitat 侧入口 `serve_collect.prepare_server_plan` 在导入 `lightnav_habitat` 之前执行同一组检查。检查失败时进程以 `SystemExit` 结束，模拟器对象不会被构造。Python 3.11 上的 `serve_collect.main` 仍会因为解释器版本不对而拒绝启动，这一条在本轮继续成立，所以没有用 Habitat 3.9 解释器启动服务器。

`WorldModelVLNCEEnv.reset` 在 reset 返回的 episode 不在已固定的 served allowlist 中时立刻抛出，不会再 step。`run_session` 在写观测和 transition 之前比较两件事：服务端公布的 served key 顺序必须等于计划，本次 reset 的 key 必须等于当前计划行。任一不一致都停止，并留下空的 transition 流。

## 2. 官方清单上的核对

文件：`experiments/world_model/splits/wm_scene_split_0.1.json`。

train `json.gz` 的 sha256 是 `340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea`，与清单中的 `dataset_sha256` 一致。`validate_scene_inventory` 对这份文件返回 eligible。`17DRP5sb8fy` 在 `train_fit` 中，`1pXnuDYAj8r` 在 `train_dev` 中。

`_prepare_plan` 用官方 train JSON 和这份清单接受了 `17DRP5sb8fy` 的一个 episode，返回的 `(scene, episode_id)` 与 allowlist 逐项一致。同一调用对 `1pXnuDYAj8r` 抛出 `CollectionGateError`，代码是 `train_dev`。`prepare_server_plan` 对这个 dev episode 抛出 `SystemExit`。清单文件本身没有被写入。

## 3. 测试

| 检查 | 结果 |
|---|---|
| 合法合成 train 清单通过，并且只接受其中的 `train_fit` 路线 | 通过 |
| `train_dev` 场景进入采集计划 | 拒绝 |
| `source_split` 为 `val_seen`、`val_unseen` 或 `test` | 拒绝 |
| 场景标签与场景哈希不一致，或同一路线组出现在两个场景 | 拒绝 |
| schema 版本不符、数据集哈希不符、场景数低于门槛、`eligible` 不为 true | 拒绝 |
| allowlist 分区被改成 `val_unseen` | `_prepare_plan` 抛出 `RuntimeError`，服务器准备函数抛出 `SystemExit` |
| schema 被改掉之后调用 `collect.main` | `SystemExit`，run 目录没有创建 |
| reset 返回 allowlist 之外的 episode | 立即失败，`step` 调用次数为 0 |
| reset 先返回计划中的后一个 episode | `RuntimeError`，`transitions.jsonl` 为空 |
| 官方清单加 `17DRP5sb8fy` | 通过，返回的 episode 与 allowlist 一致 |
| 官方清单加 `1pXnuDYAj8r` | 拒绝 |

对应测试是 `tests/world_model/test_collection_gate.py`。命令和退出码见 `docs/world_model_reproducibility.md` 第 4 节，三份门禁共用同一次 pytest。

## 4. 没有执行的事

没有启动 Habitat 采集，没有调用 Habitat 3.9 解释器，没有把 `train_dev` 场景放进任何 run 目录。正式采集仍要先做一次单独的启动评审，并且先跑受限的小批量，确认实际 reset 记录、图像哈希、特征契约和 split 元数据能够闭环。
