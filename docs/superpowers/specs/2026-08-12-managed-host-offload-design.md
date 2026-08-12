# FastLLM 受预算的进程内模型卸载设计

**日期：** 2026-08-12  
**状态：** 已确认，按 IQ6 256K 带图稳定性范围实施

## 1. 问题

当前 API server 的 memory-tier suspend 会执行 `model.reset()`、清空 paged cache manager 并释放 CUDA idle pool。resume 重新调用 `CreateLLMModelFromFile`，因此即使进程没有退出，也要重新经历 GGUF 读取、tensor 转换/合并、GPU 上传和 warmup。实测 Q6 恢复约 176–280 秒。

把完整 GGUF 和 prefix-cache 目录复制到 `/dev/shm` 不是可接受方案：它没有统一 RAM 预算，会与系统和服务进程竞争内存；权重约 23GB，再加 prefix checkpoint 可能耗尽当前 62GB RAM；tmpfs 仍引入文件系统复制和目录生命周期；内存紧张时没有精确逐块淘汰或可靠回退。

本设计让 FastLLM 自己管理 GPU-ready tensor 的 host 热缓存。suspend 时优先把重建成本高的 warmup/merge 结果从 GPU 原样 D2H，随后释放对应 CUDA allocation；resume 时直接 H2D，成功一个就释放一个 host block。缓存不足的 tensor 按持久化 materialization recipe 从原 GGUF 重建。缓存失败只降低到现有磁盘恢复路径，不能改变模型输出或阻止显存释放。

## 2. 已验证的代码事实

- `Data::ToDevice(DataDevice::CPU)` 已支持：为 `expansionBytes` 分配 host storage、D2H 原样复制、对 model weight 调用 `CudaFreeForData`。
- 反向 `CPU -> CUDA` 成功后，现有实现会删除非 mmap 的 `cpuData`。它已经具备“迁移后释放源副本”的基本语义。
- GGUF loader 先建立 `ReadGGUFTask`，调用 `WeightImportGGUFTensor`，再按 `weightMergeRules` 生成最终权重，最后执行 `OnModelWeightsLoaded()`。
- Qwen3.5/3.6 会生成普通 GGUF 中没有的最终 tensor，包括 `mergeqkv`、`gateup`、linear-attention 合并权重、MTP 合并权重以及 device-specific MTP draft lm-head。
- `OnAutoWarmupFinished()` 还会创建 MTP serving state、linear slot pools、CUDA graph/workspace。GPU-ready `Data` 可以迁移；`cudaGraphExec`、stream、cuBLAS handle 等驱动对象不能序列化，必须销毁后重建。
- persistent prefix cache 已有带校验的 SSD generation 和 fail-open restore。它天然是低成本回归路径，不应优先占用紧缺 RAM。

## 3. 目标与非目标

### 目标

本实现属于现有 `VRAM → RAM → disk` 三级缓存体系：READY 模型在 VRAM；memory-tier suspend 将可复用 tensor 和剩余 prefix KV 放入受预算的 RAM 层；disk tier 继续保存 GGUF/source recipe 与 persistent prefix generation。不得并列建立第二套卸载缓存或第二份独立预算。
- memory-tier suspend 后 FastLLM 进程和 model 拓扑保持存活，GPU 显存回到冷基线。
- 在统一、硬上限的 host RAM 预算内缓存任意比例的 GPU-ready tensor；部分缓存也必须减少恢复工作。
- warmup/merge 派生状态优先于普通权重，普通权重优先于 prefix KV 的 RAM 副本。
- READY 稳态不保留 manager-owned host 权重副本；SUSPENDED 稳态不保留对应 GPU 权重副本。
- 任意缓存分配、校验或 materialization 失败都回落到现有 `CreateLLMModelFromFile + disk prefix restore`。
- 与 IQ6（当前 abliterated i1_Q6_K）256K、Turbo3 KV、MTP2 和 persistent prefix cache 兼容；不得改变 context/KV geometry。

### 非目标

- 不序列化 CUDA graph、stream、event、cuBLAS/cuDNN handle 或 kernel module。
- 不保存活动 request；suspend 仍要求 active/queued request 为零。
- 不复制 GGUF 到 tmpfs，不创建 RAM 文件系统快照。
- 首版只保证单 CUDA device（当前 V100）热恢复。TP/multi-CUDA 检测到后明确回落磁盘，不做不完整迁移。
- 不承诺 Linux 内核 page cache 的物理唯一性；只能保证 FastLLM 自己管理的 tensor 副本唯一。源文件读取后可 best-effort `posix_fadvise(..., DONTNEED)`。

## 4. 核心对象

### 4.1 RAM tier：`HostOffloadManager`

`HostOffloadManager` 是三级缓存的 RAM 层实现，不是独立于 tiering 的旁路。进程内、每个 model 实例一个 manager，负责：

- 读取动态预算；
- 枚举、排序和迁移 candidate；
- 统计 resident bytes；
- 保存 tensor block 与 reload recipe；
- 在 resume 成功后逐块释放 host storage；
- 在错误时整体 invalidate 并触发磁盘回退。

统一预算由模型 host 热缓存和 prefix KV RAM payload 共同消费。默认预算：

```text
budget = min(
  FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES,          # 默认 12 GiB
  MemAvailable - FASTLLM_HOST_SUSPEND_MIN_FREE_BYTES  # 默认保留 12 GiB
)
```

当 `MemAvailable <= reserve` 时 budget 为 0，直接使用 disk tier。manager 按 tensor/chunk 增量分配，每次分配前重新读取 `MemAvailable`；不先申请一个 12GB 连续 arena。新页必须被实际触页后才计入 resident bytes，分配失败只停止增加缓存，不继续逼近 OOM。prefix KV 通过同一预算控制器计费，不能再单独获得额外的 CPU tier 配额。

缓存使用普通 pageable anonymous memory。H2D/D2H 通过一个固定上限的 pinned staging buffer 流水传输，避免 page-lock 数十 GB。

### 4.2 `HostOffloadBlock`

每个 block 记录：

- stable tensor name 和 model instance generation；
- dtype、GGML type、dims、TP layout、bytes；
- 原 CUDA device；
- host pointer/owner；
- 分类、优先级和 measured restore cost；
- checksum（传输后验证）；
- 对应 `WeightMaterializationPlan` id。

block 生命周期：

```text
GPU_RESIDENT -> COPYING_TO_HOST -> HOST_RESIDENT
HOST_RESIDENT -> COPYING_TO_GPU -> GPU_RESIDENT
```

只有 `COPYING_*` 的有界窗口允许双份；目标复制和校验成功后立即释放源。失败时源仍保持有效，或者进入全量磁盘回退，不能留下半初始化 tensor。

### 4.3 `WeightMaterializationPlan`

GGUF 首次加载时持久保存在 model 对象，覆盖最终 tensor，而不是只保留已经被 erase 的临时输入。plan 类型：

1. `GGUF_DIRECT`：file、offset、payload bytes、GGML type。
2. `GGUF_TRANSFORM`：上述字段加 replace/untile/dequant 参数。
3. `MERGE`：输入 plan id、有序 merge 规则、最终 metadata。
4. `MODEL_GENERATED`：模型 hook id；不能独立重建的对象标记 `must_cache`。

每个源文件保存 canonical path、size、mtime 和 GGUF tensor-table fingerprint。resume 前任一 identity 不一致，整份 suspend image 失效并走全量磁盘恢复，不能把旧缓存和新文件混用。

partial resume 对缺失的最终 tensor 只执行它的 plan 子图，使用 bounded staging storage，并填充仍存活 model 中的目标 `Data`。不能验证 plan 的 tensor 不得无复制释放 CUDA storage；若 `must_cache` 总量超过 budget，则本次 suspend 直接采用 disk tier。

## 5. 缓存价值排序

排序不是 LRU，也不是按 tensor 大小；所有模型权重恢复前都需要。排序指标为“每缓存一字节节省多少恢复时间”：

```text
score = estimated_rebuild_ms / bytes
```

首轮使用静态类别，后续使用实测 resume 计时更新 EWMA：

1. `must_cache` 的不可独立重建派生 tensor。
2. GGUF merge/untile/dequant/repack 后的最终权重，包括 Qwen3.5 `mergeqkv`、`gateup`、linear-attention 合并权重和 MTP/device-specific 权重。
3. 普通 GPU-ready model weight。
4. tokenizer/model metadata（本来常驻，体积很小，不计大块预算）。
5. prefix KV 的 RAM payload；仅使用前四类分配后的剩余预算，内存压力时第一个淘汰。SSD generation 始终保留。

CUDA graph exec、linear slot scratch、paged-cache 空页和通用 temp buffer 不进入 host cache。它们没有可移植 host 表示，或重建成本/字节低；resume 后按正常 hook 重建。

## 6. Suspend 流程

1. API server 进入 `SUSPENDING`，拒绝新请求，并再次验证无 active/queued request。
2. 按现有逻辑将 dirty prefix cache checkpoint 到 SSD。失败只记录，不阻止显存卸载；上一个有效 generation 仍可恢复。
3. 模型 hook 销毁不可迁移的 CUDA graph/runtime 对象，停止 TP worker；保留 model/tokenizer/tensor topology。
4. manager 枚举 GPU-resident final tensor，验证每个未缓存 candidate 都有 materialization plan。
5. 计算动态 budget；先处理高 score candidate：D2H、checksum、切换为 `HOST_RESIDENT`、立即释放 CUDA allocation。
6. 对未命中的可重建 tensor，仅在 plan 已验证后释放 CUDA allocation并标记 `SOURCE_EVICTED`。
7. 清理 linear slot scratch、paged manager、idle temp pool 和其余非语义 CUDA storage。
8. prefix KV 只有在权重缓存后仍有 budget 时才保留 RAM payload；否则只保留 SSD generation descriptor。
9. 状态切换到 `SUSPENDED_HOST`；若 budget 为 0 或任一步无法安全形成 partial image，则销毁 model 并进入现有 `SUSPENDED_DISK`。

Suspend 中途失败：

- 还未释放的 CUDA tensor 保持原样。
- 已迁移 tensor 可从 host H2D 回滚；若回滚也失败，销毁整个 model 并进入 disk tier。
- endpoint 返回实际 tier 和 downgrade reason；显存卸载目标优先于热缓存命中率。

## 7. Resume 流程

1. `SUSPENDED_HOST -> RESUMING`，锁住 lifecycle，拒绝并发 resume/request。
2. 校验 model generation、源文件 identity、tensor metadata 和 block checksum。
3. 按依赖顺序恢复 host block：H2D 成功并校验后立即释放该 host block，因此 host resident bytes 单调下降。
4. 使用释放出的 host 空间作为 bounded staging，为 `SOURCE_EVICTED` tensor 执行 materialization plan；上传后立即清空 staging。
5. 调用 model 的 `OnHostResume()`：只重建不可迁移 CUDA runtime（MTP serving handles、linear slot pools、需要的 graph/workspace），不得重复已缓存的 weight merge/repack。
6. prefix 有 RAM payload才恢复 RAM payload；否则沿现有 persistent generation 从 SSD lazy materialize。
7. 运行内部最小一致性检查，状态切换到 `READY`。此时 host weight cache bytes 必须为 0。
8. 任一步失败：记录原因，销毁半恢复 model，清空 host cache，调用现有完整磁盘恢复。磁盘恢复成功才切换 READY。

## 8. 状态机与并发

```text
READY
  -> SUSPENDING
      -> SUSPENDED_HOST
      -> SUSPENDED_DISK
SUSPENDED_HOST|SUSPENDED_DISK
  -> RESUMING
      -> READY
      -> DISK_FALLBACK_LOADING -> READY|ERROR
```

- 所有状态转换在 API server exclusive lifecycle lock 下完成。
- checkpoint、suspend、resume 互斥。
- request lease 只有 READY 才能取得；RESUMING 返回 proxy 可识别的 backend-reloading SSE/JSON 响应。
- pressure-triggered suspend 与 idle suspend 使用同一实现，不再保留不同释放顺序。

## 9. 配置与可观测性

```text
FASTLLM_HOST_SUSPEND_CACHE=1
FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES=12884901888
FASTLLM_HOST_SUSPEND_MIN_FREE_BYTES=12884901888
FASTLLM_HOST_SUSPEND_STAGING_BYTES=268435456
FASTLLM_HOST_SUSPEND_PREFIX_MODE=opportunistic
```

`/props` 与 suspend/resume 响应增加：

- actual tier、budget/resident/peak host bytes；
- cached derived/ordinary/prefix bytes；
- source-evicted bytes；
- D2H/H2D/disk-read/materialize/warmup 毫秒；
- cache hit ratio（bytes）；
- fallback count/reason；
- source identity mismatch 和 checksum failure 计数。

日志只在状态转换和汇总时输出，不按 tensor 打大日志。详细 tensor 统计通过显式 debug 开关写有界 top-N。

## 10. 验证

### 单元与回归

- `Data` GPU->host->GPU round trip byte/hash 一致；每个稳定状态只有一个 manager-owned resident copy。
- budget 为 0、小于 must-cache、大于部分权重和足够全量四种路径。
- 每次 chunk 分配失败、D2H/H2D 失败、checksum 错误均进入定义好的回退路径。
- `GGUF_DIRECT`、transform、merge 和 model-generated plan 重建最终 tensor 后 hash 一致。
- 文件 size/mtime/fingerprint 改变使整份 image 失效。
- prefix RAM payload 被权重挤出后仍可从 SSD generation 正确恢复。
- concurrent checkpoint/suspend/resume/request 得到确定的 409/503，不死锁。

### V100 端到端

1. IQ6 256K Turbo3 KV MTP2 启动，记录完整磁盘启动、warmup 分段耗时和 greedy 输出 hash。
2. 用 12GiB shared cache suspend：GPU 回到冷基线，host 增量不超过 budget，`/dev/shm` 无模型文件。
3. resume：输出 hash 一致，READY 后 host weight cache 为 0，TTFT/恢复耗时低于磁盘基线。
4. 分别用 8GiB、1GiB、0GiB budget 重复；只比较 resume 的 materialize + warmup 段，缓存 bytes 增加不应增加该段耗时（允许 suspend 的 D2H 时间变长）；0GiB 必须等价于现有 disk tier。
5. 人为破坏 checksum、修改源 identity、注入分配/H2D 失败，确认自动完整磁盘恢复。
6. 连续执行带图请求、memory-tier suspend/resume、同图和不同图请求；验证 KV geometry、prefix restore、图片编码和短文本请求均不回归。
7. 生产 profile 只在以上 gates 通过后启用；默认代码路径保持关闭。

验收阈值：

- suspend 后 GPU used 回到约 1.5GB 冷基线；
- manager-owned host resident 始终不超过动态 budget；每次新分配前若观测到 `MemAvailable <= reserve`，必须停止增加缓存（其他进程并发用内存不由本 manager 保证）；
- READY 后 manager-owned host weight bytes 为 0；
- cached 和 source-rebuilt tensor 的最终 hash 与首次加载一致；
- 任意故障只造成恢复变慢，不造成错误 token、残留半模型或服务进程 OOM。
