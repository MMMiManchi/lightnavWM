# 世界模型特征缓存写入前校验

日期：2026-10-23。契约版本 `wm-feature-cache/0.2`。`feature_layer_selected` 保持 null。

写入器是 `experiments/world_model/cache_writer.py` 的 `commit_feature_row`。它先做完与检查器相同的校验，再对真实文件算 sha256，最后才用 `O_CREAT|O_EXCL` 创建目标文件。校验失败时目标路径不存在。已有文件不会被覆盖。本轮没有创建正式缓存目录，也没有生成正式 latent cache。

结论：缓存写入器已经被证明能在任何持久化之前阻止不合规记录。这只允许进入正式采集启动评审。

## 1. 写入前的顺序

1. 目标已存在、路径落在 `formal_feature_cache`、`world_model_cache`、`world_model_diagnostics`，或路径含有 `pilot_` 目录时，直接拒绝。
2. 行上 `diagnostic_only` 为 true 时拒绝。检查器 `check_feature_cache` 使用同一条规则，因此跨进程诊断目录不能被正式缓存脚本接受。
3. manifest 的 `feature_layer_selected` 必须存在并且是 null。写成 `pooler_output` 或 `last_hidden_state` 都会失败。
4. `check_feature_cache` 检查图像文件存在且字节 sha256 与行一致，revision、权重 sha256、预处理 sha256、形状、dtype、有限值、`feature_sha256`、正式 train split、特权字段和 pilot 来源。
5. `verify_bound_files` 重新读取两个真实文件：
   - 预处理器必须解析到 `experiments/world_model/contracts/siglip_so400m_patch14_384_preprocessor.json`。另一个路径即使字节相同也会被拒绝。通过路径检查之后，再对这个文件的字节计算 sha256，必须等于 `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`。
   - 权重路径指向本次要加载的 `model.safetensors`。文件字节的 sha256 必须等于 `ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`。
6. 以上都通过之后才创建文件。写入中途失败会删除这个新文件。第二次写入同一个路径会失败，并保留第一次的字节。

日常失败测试在第 4 步就返回，不会读取 3.5 GB 权重。只有行检查已经通过、即将创建文件时才哈希真实权重。

预处理器不在 Hugging Face snapshot 里。写入器把 manifest 中的预处理路径改写为仓库里的契约文件，再哈希该文件。只在元数据里填写期望哈希不够。

## 2. 测试

`tests/world_model/test_cache_writer.py` 覆盖了这些失败，并且每次失败后目标文件都不存在：

| 篡改 | 结果 |
|---|---|
| 记录生成后替换图像字节 | `image_hash_mismatch`，无文件 |
| 行内权重 sha256、revision 或预处理 sha256 与契约不符 | 拒绝，无文件 |
| 特征长度不是 1152 | 拒绝，无文件 |
| 特征含 NaN 或 Inf | 拒绝，无文件 |
| `source_run` 含 `pilot_`，或 `source_split` 不是 train | 拒绝，无文件 |
| 行内出现 `goal_position` | 拒绝，无文件 |
| `diagnostic_only` 或诊断目录来源 | 拒绝，无文件 |
| `feature_layer_selected` 写成某个候选层 | 拒绝，无文件 |
| 预处理 JSON 被复制到另一个路径 | `verify_bound_files` 拒绝，即使字节相同 |
| 权重文件字节不是审计过的 checkpoint | 拒绝，无文件 |
| 目标文件已经存在 | 原字节保持不变 |
| 目标位于 `world_model_diagnostics` | 拒绝，无文件 |
| `os.write` 中途失败 | 部分文件被删除 |

成功路径使用本地 snapshot `9fdffc58afc957d1a03a25b10dba0329ab15c2a3/model.safetensors`。测试在 `verify_bound_files` 返回之前确认目标文件还不存在，返回的权重摘要等于 `ea2abad2…d94a`，预处理摘要等于 `f59da2f8…57e8`。写入内容里 `diagnostic_only` 为 false，`feature_layer_selected` 为 null。紧接着的第二次写入被拒绝，文件字节不变。

这条成功路径写在 pytest 的临时目录里，不是正式缓存。测试结束后临时目录被删除。

## 3. 没有执行的事

没有创建 `formal_feature_cache` 或正式 latent cache。没有把诊断编码写入可被训练读取的缓存。没有下载权重，没有改编码器契约，没有把 `feature_layer_selected` 改成已选定。
