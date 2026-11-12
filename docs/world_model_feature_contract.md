# 世界模型 v1 冻结特征契约

日期：2026-10-23。契约版本 `wm-feature-cache/0.2`，检查代码在 `experiments/world_model/feature_cache.py`。

这份契约已经定稿。它固定编码器和两种候选输出的记录方式。`feature_layer_selected` 保持 null，正式特征缓存没有创建，正式采集和预测器训练保持关闭。

同一进程里同一张图两次前向的最大绝对差为 0。这只说明那一次进程内的重复。它不说明换进程、换 GPU 或换一批图像之后仍然得到同一组数。

## 1. 编码器

| 字段 | 定稿值 |
|---|---|
| `encoder_id` | `google/siglip-so400m-patch14-384` |
| `revision` | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` |
| `model_class` | `SiglipVisionModel` |
| `weight_sha256` | `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a` |
| `transformers_version` | `5.8.0` |
| `torch_version` | `2.10.0+cu128` |
| `cuda_version` | `12.8` |
| `feature_dtype` | `float32` |
| `feature_layer_selected` | null |

加载只使用本地 snapshot，并设置 `local_files_only=True`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。只构造 `SiglipVisionModel`。文本塔不进入模块。前向使用 `eval()` 和 `torch.inference_mode()`，不计算梯度。

权重路径必须指向一个现存文件。manifest 里的 `weight_sha256` 必须等于上表。打开正式缓存之前，调用 `check_encoder_manifest(..., verify_weight_hash=True)` 再对文件字节核对一次。日常逐行检查不重复读取 3.5 GB。

预处理器文件缺失、哈希不符，或 `allow_processor_defaults` 不是 false 时，检查器直接失败。不得改用 `SiglipImageProcessor` 的默认边长 224。Transformers 5.8.0 中名为 `IMAGENET_STANDARD_MEAN` 的常量是 `[0.5, 0.5, 0.5]`，经典 ImageNet 均值是 `IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]`。常量数值碰巧相同，也不能代替下面这份 JSON。

## 2. 预处理

仓库内副本：

`experiments/world_model/contracts/siglip_so400m_patch14_384_preprocessor.json`

sha256 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`，368 字节。来源是 Hub revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` 的 `preprocessor_config.json`，与诊断目录中的副本逐字节相同。Hugging Face snapshot 目录里仍然没有这个文件。

对一张 HWC `uint8` RGB 的处理顺序是：

1. 精确缩放到 384×384，bicubic（`resample=3`），不保持长宽比。
2. 不做 center crop。`do_center_crop` 为空。
3. 乘以 `1/255`。
4. 按均值和标准差 `(0.5, 0.5, 0.5)` 归一化。
5. 得到 `float32`、通道优先的 `(1, 3, 384, 384)`。像素 0 对应 `-1`，像素 255 对应 `1`。

采集帧的原始形状是 `uint8 (270, 480, 3)`。进入编码器之前才缩放到 384。LightNav 的 256×448 与 `[-1, 1]` stretch 只属于导航会话。

`rgb_t` 与 `rgb_t1` 仍由 `obs_before_id` 和 `obs_after_id` 配对，不靠文件名相邻。

## 3. 两种候选输出

| 候选 | 实测形状 | dtype | 单帧特征字节 | 角色 |
|---|---|---|---:|---|
| `pooler_output` | `(1, 1152)` | float32 | 4,608（4.5 KiB） | 低成本预测的第一个实验候选 |
| `last_hidden_state` | `(1, 729, 1152)` | float32 | 3,359,232（3.203125 MiB） | 保留 patch 空间信息的候选 |

字节数只计 float32 特征本身，不含 episode、哈希和索引。token 特征的体积是池化特征的 729 倍。

池化向量适合先看潜在状态预测是否可学。它把每个 patch 收成一个全局向量，定位门、走廊或转角这类局部线索的能力还没有证据。token 序列留下 27×27 的 patch 网格，后续若要做空间记忆检索，需要单独的压缩或聚合，不能把 729×1152 直接当成已经可用的检索键。

单帧前向不能决定哪一个更适合导航。缓存行用 `output_name` 标明自己存的是哪一个候选。manifest 的 `feature_layer_selected` 必须是 null；写成任一候选都会被拒绝。

第 7 节审计里的数值范围、276.3 ms、59.5 ms 和 1,763,706,368 字节峰值，只描述那一张图和那张 GPU。

## 4. 缓存记录

schema `wm-feature-cache/0.2`。一行是一个观测的一个候选输出，不是一条训练样本的全部标签。

每一行绑定：

- `episode_uid`、`step_index`、`observation_id`
- `image_path` 与 `image_sha256`（`.npy` 文件字节的 sha256，与观测记录里的 `rgb.sha256` 同一算法）
- `source_split=train`、`source_run`、`scene`、`route_group_id`
- `scene_partition`，必须等于 `wm-scene-split/0.1` 对场景 id 的哈希
- revision、权重 sha256、预处理 JSON sha256
- `output_name`、`feature_shape`、`feature_dtype`
- `feature_sha256`：按 C 顺序把向量看成 float32 之后的 sha256

调用 `check_feature_cache` 时必须传入已经能通过 `require_formal_split` 的 episode 列表。行里的 episode、场景和路线组必须属于这份列表。正式缓存因此不能绕过场景级划分。划分下限仍是训练侧至少 8 个场景、20 个路线组，验证侧至少 3 个场景、5 个路线组。episode 数不等于路线组数。

缓存中不得出现目标点、测地距离、reference path、世界坐标、success/spl/ndtw 或其他特权字段。这些名字沿用 `latent_data.FORBIDDEN_INPUT_KEYS`。

检查器拒绝：

| 情况 | 错误码 |
|---|---|
| 路径分量以 `pilot_` 开头，包括 `pilot_20261023_train3_r2` | `pilot_source` |
| `source_split` 不是 `train` | `non_train_split` |
| 场景侧与场景哈希不符，或同一路线组跨场景 | `scene_cross` |
| episode 不在通过 `require_formal_split` 的列表中 | `formal_split` |
| 图像文件不存在 | `missing_image` |
| 图像字节哈希与记录不符 | `image_hash_mismatch` |
| 预处理器文件不存在 | `missing_preprocessor` |
| 权重文件不存在 | `missing_weights` |
| revision、权重哈希、预处理哈希、特征校验和或版本不符 | `cache_config_mismatch` |
| 形状或长度与该 `output_name` 的定稿形状不符 | `feature_dim_mismatch` |
| 非有限值 | `non_finite_feature` |
| 特权字段 | `privileged_field` |
| `feature_layer_selected` 不是 null | `feature_layer_selected` |
| 同一观测的同一输出层出现两次 | `duplicate_feature` |

试点 RGB 只用于第 7 节诊断。正式缓存不接收它。val_seen、val_unseen 和 test 不接收。

本仓库现在没有按这个 schema 写出的特征文件。
