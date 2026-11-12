# 世界模型 v1 冻结视觉编码器审计

日期：2026-10-23。兼容性复核读了本地配置、权重头、已安装 Transformers 5.8.0、Transformers 4.37.2 的公开源码，以及 Hub revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` 上的公开配置。没有下载模型权重，没有把 `preprocessor_config.json` 写进本地 snapshot，没有训练，没有采集。视觉塔前向是之后单独做的，结果在第 7 节。

## 结论

`google/siglip-so400m-patch14-384` 的本地 snapshot 与这份 revision 的结构是兼容的。384、patch size 14 和 729 个位置嵌入在当前实现里是同一件事：`384 // 14 = 27`，`27² = 729`。729 不是把输入改成 378 的证据。

正式特征缓存和正式采集仍然关闭。本地 snapshot 里还没有预处理器文件，`last_hidden_state` 和 pooling head 都还没有被选定为 `z_t`。第 7 节是一次受控视觉塔前向的实测，不是特征缓存。

LightNav checkpoint 仍然不能当作逐帧 `z_t`。

## 1. 384、14 与 729

本地 `vision_config` 写着 `image_size=384`、`patch_size=14`。权重 `vision_model.embeddings.position_embedding.weight` 的形状是 `[729, 1152]`，`patch_embedding.weight` 是 `[1152, 3, 14, 14]`，并带有 `[1152]` 的 bias。888 个键里没有 class 或 cls 嵌入。

已安装 `SiglipVisionEmbeddings`（`transformers/models/siglip/modeling_siglip.py` 第 124–134 行）用 `padding="valid"` 的卷积，并计算：

```text
num_patches = (image_size // patch_size) ** 2
num_positions = num_patches
```

对这份配置，结果是 27 和 729。位置表的长度等于 patch 数，没有额外 token。默认前向把这 729 个位置直接加到 patch token 上；`interpolate_pos_encoding` 默认是 False。因此插值和额外 token 不影响这条默认路径。

同一个卷积构造在 CPU 上的形状探针：

| 输入边长 | 输出网格 |
|---:|---|
| 384 | `27×27`，729 个 token |
| 378 | `27×27`，729 个 token |
| 224 | `16×16`，256 个 token |

384 输入的最后 6 行和最后 6 列没有进入任何一个 patch：用全 1 卷积核、零 bias 时，这部分像素的输出和为 0。左上角像素只进入第 `(0, 0)` 个 patch。这是 valid 卷积从左上角取整，不是 center crop，也不是要求先把图缩到 378。把缩放目标改成 378 同样能得到 729 个 token，但得到的是另一张图。

Transformers 4.37.2 的 `SiglipVisionEmbeddings` 使用同一公式，前向也是直接加上固定位置编码，没有插值分支。本地 `config.json` 的 `transformers_version` 是 `4.37.0.dev0`。5.8.0 在默认关闭插值时，token 数与 4.37.2 的这条路径一致。激活函数和 LayerNorm epsilon 没有写进本地 JSON；5.8.0 的 `SiglipVisionConfig` 用默认值 `gelu_pytorch_tanh` 和 `1e-6` 补上。这两个默认值是否与当初保存配置时的库默认值逐位相同，要等真实权重前向才能核对，目前记为待验证。

当前 Transformers 5.8.0 在 `auto_mappings.py` 里内置了 `siglip`，配置中没有 `auto_map`。加载不需要额外模型代码，也不需要 `trust_remote_code`。

## 2. 本地 snapshot 与 Hub revision

`refs/main` 的内容是 `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`。Hub 在这个 revision 上的文件表，与本地两个文件的标识一致：

| 文件 | Hub 标识 | 本地 |
|---|---|---|
| `config.json` | git oid `a513b447f6b05f640f9980d25f4d8a5cff6cd25d`，576 字节 | blob 名相同；sha256 `adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b`，与该 revision 的原始文件逐字节相同 |
| `model.safetensors` | LFS oid `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`，3,511,950,624 字节 | sha256 和字节数都相同 |

同一 revision 的树里还有 `preprocessor_config.json`（git oid `051261b5a784a2e7e5f93de1f86d170d69a9efd9`，368 字节）和 `README.md`（4321 字节）。本地 snapshot 只有 `config.json` 和 `model.safetensors` 两个符号链接，这两份文件都不在本地。本次把公开文件读到临时目录核对，没有写回 snapshot。

该 revision 的 `preprocessor_config.json` 与当前 `main` 上的同名文件逐字节相同，sha256 为 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`。它的字段是：

```json
{
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.5, 0.5, 0.5],
  "image_processor_type": "SiglipImageProcessor",
  "image_std": [0.5, 0.5, 0.5],
  "processor_class": "SiglipProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {"height": 384, "width": 384}
}
```

文件里没有 `do_center_crop`。已安装处理器读入后，`do_center_crop` 是 `None`，预处理分支不会做 center crop。`resample=3` 是 Pillow 的 bicubic。`size` 同时给出 height 和 width 时，Torchvision 与 PIL 后端都把图像精确缩放到 384×384，不保持长宽比。`rescale_factor` 是 `1/255`，再按均值和标准差 0.5 归一化，因此 0 映射到 `-1`、255 映射到 `1`。输出是 `float32`、通道优先的 `(1, 3, 384, 384)`。

`do_convert_rgb=True` 来自处理器类的默认值，不来自这份 JSON。已有采集帧是 RGB。

用临时目录中的这份 JSON，对一张 `uint8 (270, 480, 3)` 合成图分别走 torchvision 和 PIL 后端，输出形状都是 `(1, 3, 384, 384)`，最大绝对差是 `4.47e-8`。这只说明这一张图上的两个后端很接近，不能外推到所有图像。

已安装 `SiglipImageProcessor` 在不读 JSON 时，`size` 仍是 224。它的 `image_mean` 指向常量 `IMAGENET_STANDARD_MEAN`。在 Transformers 5.8.0 的 `utils/constants.py` 里，这个常量是 `[0.5, 0.5, 0.5]`；经典 ImageNet 均值是另一个常量 `IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]`。所以类默认的均值碰巧与官方文件相同，类默认的边长则不同。前向必须使用上面这份 JSON，不能使用类默认边长。

## 3. 权重与库的形状对应

只读 safetensors 头，没有把权重载入模块：

| 张量 | 形状 | 与 5.8.0 的关系 |
|---|---|---|
| `vision_model.embeddings.patch_embedding.weight` | `[1152, 3, 14, 14]` F32 | 3 通道、kernel 14，对应配置里省略后的默认 `num_channels=3` |
| `vision_model.embeddings.position_embedding.weight` | `[729, 1152]` F32 | 等于 `(384 // 14)²` |
| `vision_model.head.probe` | `[1, 1, 1152]` F32 | attention pooling 的 probe，不是第 730 个 patch |
| `text_model.embeddings.token_embedding.weight` | `[32000, 1152]` F32 | 等于文本配置省略后的默认 `vocab_size` |
| `text_model.embeddings.position_embedding.weight` | `[64, 1152]` F32 | 等于默认 `max_position_embeddings` |

`vision_model.*` 的张量合计 1,712,902,400 字节。`text_model.*` 合计 1,798,939,584 字节。其余是 `logit_scale` 和 `logit_bias` 两个标量。全文件 3,511,950,624 字节。一次只加载视觉塔时，权重大约占 1.60 GiB。激活峰值没有测量。

`vision_use_head` 不在配置里。5.8.0 因此把 `use_head` 设为 True。`pooler_output` 是 `SiglipMultiheadAttentionPoolingHead` 的 `hidden_state[:, 0]`，按源码形状是 `(batch, 1152)`。`last_hidden_state` 在 post-LayerNorm 之后，按源码形状是 `(batch, 729, 1152)`。二者都还没有用这份权重实测。

## 4. 许可

本地 snapshot 没有 LICENSE，也没有 README。与本地 config、权重 oid 对应的 Hub revision 上，README 为 4321 字节，sha256 `96a172e58539296eadb939acf544287c14e20a0a426fbbff116518c2c8024a8b`，开头声明 `license: apache-2.0`。这个声明属于该 revision 的模型卡，不是本地文件。对应关系来自 config 逐字节一致和权重 LFS oid 一致。

## 5. LightNav 仍不作为逐帧编码器

| 项目 | 读到的值 |
|---|---|
| 权重 | `checkpoints/LightNav-0/model-00001-of-00001.safetensors`，9,695,744,034 字节，sha256 `ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18` |
| 配置 | `config.json`，sha256 `0e047b5e9ae7981cf0066e74d8b655a2f49d7d7965c25fca386f9eea80b63e58` |
| 视觉结构 | `temporal_patch_size=2`，`patch_size=16`，`spatial_merge_size=2`，`out_hidden_size=2560` |
| 评测预处理 | `eval_config.json` 的 `common.video_size` 是 `[256, 448]` |
| 接口 | `generate_from_frames` 返回 `tuple[str, float]` |

采集 RGB 是 `uint8 (270, 480, 3)`。导航预处理会在默认 stretch 下把它变成 256×448，再映射到 `[-1, 1]`。这不是 SigLIP 的 384×384、bicubic、均值 0.5。`video_embeds` 的时间维和逐帧对齐没有核实，继续排除。

## 6. 前向之前写下的条件

前向之前允许的范围是：只加载 `SiglipVisionModel`，处理器使用 sha256 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8` 的 JSON，不把该文件写入 snapshot，不用类默认边长 224，不把试点帧写成正式缓存。这些条件的实测写在第 7 节。

## 7. 2026-10-23 视觉塔前向实测

命令在物理 GPU 2 上运行，进程退出码 0。`CUDA_VISIBLE_DEVICES=2`，只看得到这一张卡。加载前空闲 84,649,312,256 字节，卡为 NVIDIA A800-SXM4-80GB。PyTorch `2.10.0+cu128`，CUDA 12.8，Transformers 5.8.0。环境变量 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。没有下载权重，没有访问网络补文件，没有启动 Habitat，没有训练。

权重 sha256 实测为 `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`，3,511,950,624 字节，与第 2 节的审计记录一致。任务说明里另写过 `ffc4a925378a881afa761865048eb8d07c55cacf5eaf66548b6641c39f67af18`，那是 LightNav checkpoint 的哈希，不是这个 SigLIP 文件。本次按 SigLIP 审计哈希接受文件。

处理器使用审计时已经核对过的 368 字节 JSON，sha256 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`。它被复制到诊断目录，没有写入 snapshot。加载后的契约是 384×384、bicubic（`resample=3`）、`rescale_factor=1/255`、均值和标准差 0.5、`do_center_crop=null`。类默认边长 224 没有被使用。

图像是试点运行里已有的一帧 train RGB：`obs-R2R_VLNCE_v1-3_preprocessed|train|7y3sRwLe3Va|1-0_rgb.npy`，sha256 `c7ccdabea55552d853085be6fdbb753e2d758b55c1dbdf2f669ebcb8f500a3ff`。这帧只用于诊断，不进入正式样本。预处理输出是 `float32`、`(1, 3, 384, 384)`，442,368 个值全部有限，范围 `[-1, 1]`。同一张图预处理两次的最大绝对差是 0。

`SiglipVisionModel.from_pretrained(..., local_files_only=True)` 加载了 448 个视觉张量。Transformers 把文本塔和 `logit_scale`、`logit_bias` 报为 UNEXPECTED 并跳过；模块类型是 `SiglipVisionModel`，没有 `text_model`。权重 dtype 是 float32，`eval()` 之后 `training` 为 false。输出 `requires_grad` 为 false。

| 项目 | 实测 |
|---|---|
| `last_hidden_state` | `(1, 729, 1152)`，float32，839,808 个值全部有限，最小 -60.837，最大 94.427，均值 0.0236，标准差 1.891 |
| `pooler_output` | `(1, 1152)`，float32，1,152 个值全部有限，最小 -2.394，最大 9.344，均值 -0.0126，标准差 0.599 |
| 加载后占用 | allocated 1,715,535,360 字节，reserved 峰值 1,969,225,728 字节 |
| 第一次前向 | CUDA event 276.3 ms，wall 0.276 s；allocated 峰值 1,763,706,368 字节，reserved 峰值 1,996,488,704 字节 |
| 重复性探针 | 同一像素张量第二次前向，CUDA event 59.5 ms；两个输出的最大绝对差都是 0 |
| OOM | 没有发生 |

第一次前向含有 CUDA 启动开销。59.5 ms 是第二次前向，只说明这一张图在这张卡上的第二次耗时。两个输出都保留为诊断结果，`feature_layer_selected` 是 null。

诊断目录：`output/world_model_diagnostics/siglip_forward_20261023T090439Z/`。里面有 `forward_report.json`、`probe.log` 和处理器副本。没有 latent cache，也没有训练样本。

## 8. 契约已经定稿，输出层仍未选定

第 7 节的前向之后，冻结编码器契约写在 `docs/world_model_feature_contract.md`，schema 为 `wm-feature-cache/0.2`。检查器是 `experiments/world_model/feature_cache.py`。`feature_layer_selected` 保持 null：`pooler_output` 和 `last_hidden_state` 都是候选。

已核验的预处理器 JSON 按原字节放在 `experiments/world_model/contracts/siglip_so400m_patch14_384_preprocessor.json`，sha256 仍是 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`。它没有写入 Hugging Face snapshot。

同一进程、同一张图的重复前向最大绝对差为 0。同一张 GPU 上两个进程的比较写在 `docs/world_model_reproducibility.md`。那次比较没有换成另一张 GPU。正式采集和正式特征缓存仍然关闭。对比实验只写在 `docs/world_model_experiment_plan.md`，没有执行。
