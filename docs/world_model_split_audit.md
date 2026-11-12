# 世界模型 v1 正式 train/dev 场景划分审计

日期：2026-10-23。划分版本 `wm-scene-split/0.1`。清单文件：

`experiments/world_model/splits/wm_scene_split_0.1.json`

生成命令是 `python -m experiments.world_model.scene_inventory`，退出码 0。它只读官方 train JSON，复用 `load_split_rows`、`assign_groups`、`scene_partition` 和 `formal_report`。`--split` 不是 `train` 时进程直接退出。目标文件已存在时拒绝覆盖。

这是元数据清单。`collection_status` 为 `not_collected`。没有按这份清单采集 transition，也没有把它用于训练。

## 1. 数据来源

| 项目 | 值 |
|---|---|
| 文件 | `data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz` |
| sha256 | `340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea` |
| 场景规则 | 场景 id 的 sha256 最后一字节模 10；余数 0 为 `train_dev`，其余为 `train_fit`。同一场景的全部路线组和指令继承该侧 |
| 准入下限 | `train_fit` 至少 8 个场景、20 个路线组；`train_dev` 至少 3 个场景、5 个路线组；两侧场景和路线组都不得相交；`source_split` 必须是 `train` |

8 与 3 是当前规则的最低准入，不是跨场景泛化的统计保证。以后报告指标时按场景分开，不把全部帧合成一个平均数当作泛化结论。

episode 数不等于路线组数。下面两列都保留。

## 2. 计数

`formal_report` 判定 `eligible=true`，原因文本是 “formal split floors are met”。

| 集合 | 场景 | 路线组 | episode |
|---|---:|---:|---:|
| train_fit | 53 | 2977 | 8938 |
| train_dev | 8 | 626 | 1881 |
| 合计 | 61 | 3603 | 10819 |

两侧 `shared_scenes` 与 `shared_routes` 都是空列表。61 个场景 id 在清单中各出现一次。每个场景内的 `route_group_ids` 无重复，且长度等于 `n_route_groups`。

`train_dev` 场景：

| 场景 | 路线组 | episode |
|---|---:|---:|
| `1pXnuDYAj8r` | 85 | 256 |
| `2n8kARJN3HM` | 89 | 268 |
| `Vvot9Ly1tCj` | 92 | 276 |
| `ZMojNkEp431` | 75 | 225 |
| `aayBHfsNo7d` | 48 | 145 |
| `ac26ZMwG7aT` | 91 | 273 |
| `uNb9QFRL6hY` | 53 | 159 |
| `ur6pFq6Qu1A` | 93 | 279 |

## 3. 互斥与不得混入的 split

官方 JSON 与 train 的场景交集：

| split | episode | 场景 | 与 train 的场景交集 | `(scene, episode_id)` 碰撞 |
|---|---:|---:|---:|---:|
| val_seen | 778 | 53 | 53 | 10 |
| val_unseen | 1839 | 11 | 0 | 0 |
| test | 3408 | 18 | 0 | 0 |

val_seen 与 train 共用场景，不能当作未见场景。那 10 个碰撞键的 `trajectory_id` 和指令文本都不同，所以 episode id 跨 split 不表示同一条路线。清单没有收入 val_seen、val_unseen 或 test 的任何 episode，三个 split 的 `included_in_train_inventory` 都是 false。

采集器里的 `wm-group/0.1` 按路线组哈希划分。同一规则下，61 个 train 场景里有 58 个同时含有路线组哈希的两侧。该划分会把一个场景拆进两个集合，不能用作世界模型的视觉划分。世界模型清单使用场景哈希，不改写已经采集的 `partition` 字段。

## 4. 试点排除

`output/world_model_data/pilot_20261023_train3_r2` 写在 `pilot_run_excluded`。它有 3 条 episode、2 个场景（`7y3sRwLe3Va`、`5LpN3gDmAk7`）、60 条 transition。两个场景在场景哈希下都属于 `train_fit`，但该运行只验证采集协议，不进入正式训练或正式验证。`pilot_20261023_train3` 有 `failure.json`，同样排除。

## 5. 测试

`tests/world_model/test_encoder_and_split_audit.py` 覆盖：

- 同一场景被分到两侧；
- 同一路线组被分到两侧；
- 非 train split 不能生成清单；
- 缺失权重、缺失图像、特征维度不匹配、非有限特征、缓存配置哈希不一致。

对官方 train JSON 的测试确认清单达到场景和路线组下限、两侧不相交、val_unseen 与 train 无共同场景，以及 val_seen 的 episode id 碰撞不是同一条轨迹。
