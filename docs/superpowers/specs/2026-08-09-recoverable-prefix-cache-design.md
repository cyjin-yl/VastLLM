# FastLLM 可恢复三级 Prefix Cache 设计

**日期：** 2026-08-09  
**状态：** 已批准，待实现

## 1. 问题与目标

FastLLM 现有 prefix cache 已有 GPU resident page、CPU 压缩 payload、NVMe `pages.bin` 三层，但磁盘层是进程 session：`PagedPrefixCacheDiskStore` 启动时删除无锁旧 session，析构时删除自己的 session；trie、LRU/命中元数据和 Qwen3.5 linear-attention snapshot 只在内存中。因此 Thinking Proxy 卸载 FastLLM 子进程后，下一 epoch 无法复用此前 prefix。

本改动把**干净 drain 后的 prefix cache**保存为原子、可校验的磁盘 generation。新 FastLLM 进程恢复 trie 和 Qwen3.5 linear/MTP snapshot，但不在启动时占用 GPU page；实际命中时才从 NVMe 解压并 materialize 到 GPU。

### 目标

- Thinking Proxy 在 idle、显存压力和正常 shutdown 三条 owned-backend 卸载路径中，先 drain，再 checkpoint，再释放 FastLLM 显存。
- checkpoint 同时覆盖：
  - full-attention K/V 的 paged trie；
  - resident GPU page；
  - CPU/zstd tier payload；
  -已有 NVMe tier page；
  - Qwen3.5/3.6 GDN linear-attention snapshot；
  - MTP prefix snapshot 的 key/value（存在且当前 profile 需要时）。
- 新进程按稳定 model cache key 恢复；首个相同 prefix 请求产生真实 NVMe restore hit，并保持输出一致。
- 任意中途失败只能损失尚未提交的新 generation，不得破坏上一个已提交 generation，也不得改变生成正确性。
- 默认关闭；未启用时，现有 session-scoped 行为和性能不变。

### 非目标

- 不保存活动 request/context；checkpoint 只在 active lease 为零后执行。
- 不保证 SIGKILL、断电前尚未 checkpoint 的最新 prefix 可恢复；只能恢复最后一次已提交 generation。
- 不让多个 FastLLM 进程同时写同一个 cache key。
- 不把磁盘缓存当模型权重缓存；模型权重仍按原路径加载。
- 首版不做后台持续 write-through，也不在正常 decode 路径增加同步 NVMe I/O。

## 2. 配置与默认值

只有下列条件全部满足时启用：

```text
FASTLLM_PREFIX_CACHE_PERSIST=1
FASTLLM_PREFIX_CACHE_DISK_DIR=/absolute/cache/root
FASTLLM_PREFIX_CACHE_PERSIST_KEY=<stable-model-and-runtime-key>
```

`FASTLLM_PREFIX_CACHE_PERSIST_KEY` 必须由部署显式提供。它必须随会改变 cache 数值或布局的模型/运行时配置变化，例如模型 revision、量化、KV dtype、tensor-parallel 布局和 page length。FastLLM 仍会把实际 manager 几何写入 manifest 并逐项校验；key 不是绕过几何校验的开关。

Thinking Proxy owned 模式另使用：

```text
FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT=600
```

Proxy 每次 spawn 生成随机 `FASTLLM_PREFIX_CACHE_CONTROL_TOKEN` 并只传给自己的 child。外部 FastLLM 模式不生成 token、不调用 checkpoint endpoint，也不发送进程信号。

## 3. 生命周期与并发协议

### 3.1 卸载

1. `BackendLifecycleManager` 将状态置为 `DRAINING`，拒绝新 local lease。
2. 等待所有已有 lease 释放；流式请求的 lease 覆盖整个 backend response 生命周期。
3. Proxy 向 child 的 loopback endpoint 发送：

   ```text
   POST /internal/prefix-cache/checkpoint
   Authorization: Bearer <per-spawn-control-token>
   ```

4. FastLLM endpoint 再检查除控制请求自身外没有 active/queued generation request；否则返回 `409`，不写 generation。
5. checkpoint 成功返回 generation、manager/node/page/byte 计数和耗时。
6. Proxy 再向自己创建的 FastLLM 进程组发送 TERM，并按现有 timeout/KILL 策略回收。

checkpoint 失败时：

- 上一个 `CURRENT` generation 保持不变。
- idle、显存压力、proxy shutdown 都仍继续卸载；prefix cache 是性能状态，失败不能阻止显存回收或关机。
- lifecycle snapshot 记录 `last_checkpoint_error`、`last_checkpoint_generation`、耗时和成功/失败计数；下次 epoch 可恢复最后一次成功 generation。

### 3.2 恢复

1. FastLLM 只读打开 `<root>/<escaped-key>/CURRENT` 指向的 generation。
2. 校验 magic、schema version、文件长度、manifest checksum、cache key 和模型级布局。
3. 每个 `PagedCacheManager` 创建时，用 manager id/type/pageLen/dtype/pageBytes/TP layout 匹配 manifest 记录；不匹配的 manager 只丢弃自己的恢复记录，不影响服务启动。
4. 重建 trie node，所有恢复 node 初始 `pageId=-1`，持有只读 generation disk ref。
5. Qwen3.5 model hook 恢复 linear/MTP snapshot 到 CPU descriptor；只有完整记录通过校验才加入该 model 实例的 snapshot map。
6. Query 命中后沿用现有 `MaterializeTrieNode` 和 `RestorePagedPrefixCacheExtra`：按需读盘、校验、解压、复制到 GPU。

恢复失败必须 fail open：记录原因并从空 cache 正常服务，绝不把损坏 page 交给生成路径。

## 4. 原子 generation 存储

目录布局：

```text
<disk-dir>/persistent/<escaped-key>/
  LOCK
  CURRENT
  gen-<monotonic-id>/
    manifest.bin
    pages.bin
  .staging-<random-id>/
```

- `LOCK` 使用进程级独占文件锁。第二个 writer 无法取得锁时禁用 persistence 并记录明确错误；不得删除活跃 generation。
- checkpoint 总是写新的 `.staging-*`，不复用 `CURRENT` 中的 offset。
- 所有 page/snapshot payload 顺序写入 `pages.bin`。每条引用包含 offset、stored bytes、uncompressed bytes、codec 和 FNV-1a checksum。
- `pages.bin` 写完并 `fsync`；`manifest.bin` 写完、校验并 `fsync`；staging 目录 rename 为最终 generation；临时 `CURRENT.new` 写入 generation id、`fsync` 后 rename 为 `CURRENT`；最后 `fsync` 父目录。
- `CURRENT` 成功切换后才删除旧 generation。进程在任一步崩溃时，旧 `CURRENT` 仍指向完整 generation；下次启动清除未引用 staging。
- checkpoint 临时磁盘需求最多为旧 generation + 新 generation。开始前按预计新 payload 检查 `FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES` 和 `FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES`；空间不足直接失败，不触碰 `CURRENT`。

### 4.1 Manifest 内容

文件头：magic、endianness、schema version、generation id、cache key、创建时间、payload bytes、record counts、全文件 checksum。

每个 paged manager：

- stable manager id、cache type；
- page length、data type、uncompressed page bytes；
- device/TP 布局签名；
- trie node 列表：parent index、edge hash、完整 edge tokens、depth、max prefix depth、access count、last-access order；
- node payload disk ref；不持久化旧 GPU page id。

Qwen3.5 extra section：

- snapshot cached length、完整 prefix tokens、request/timestamp order；
- 每个 linear layer 的 first/second GDN state；
- single-device 或每个 TP local tensor 的 dims、dtype、transpose/layout metadata 和 payload ref；
- 可选 MTP key/value 和 token length。

完整 edge token 和 payload checksum 都必须验证，不能只信任 hash。

## 5. 代码边界

### FastLLM core

- `PagedCacheManager` 保存 stable manager id，并提供只在全局 quiescent checkpoint 下使用的 export/import。
- 新的 persistent archive 组件负责 binary format、generation lock、原子 commit、空间检查和只读 payload。
- `PagedPrefixCacheTierDiskRef` 区分现有 session store ref 与 immutable generation ref；generation ref 的释放不截断 generation 文件。
- 现有 ephemeral `PagedPrefixCacheDiskStore` 保持原用途。持久 checkpoint 从 resident page、CPU payload或 ephemeral disk ref 读取规范 payload，写入全新的 generation。
- 新的全局 checkpoint API 在固定 manager 锁顺序下做只读快照，避免交叉 manager 死锁。

### Model hook

`basellm` 增加可选 prefix-persistence hook，默认无 extra section。Qwen3.5/3.6 实现该 hook，序列化/恢复现有 `Qwen35LinearPrefixSnapshots`，包括需要的 MTP snapshot。这样 full-attention paged trie 与 GDN state 属于同一个原子 generation；缺任一部分时，该 prefix 不得报告可复用。

### API server

增加 token 保护的 checkpoint endpoint。endpoint 不接受客户端提供文件路径/key，不执行 shell，也不允许并发 generation request。`/props` 增加 persistence enabled、loaded generation、restore counts/bytes、checkpoint counts/bytes/duration/error。

### Thinking Proxy

owned child spawn 时注入随机 control token。`stop()` 的 drain 结束后、TERM 之前调用 checkpoint；external/adopted backend 路径完全不变。idle、pressure、shutdown 共享同一个 checkpoint-and-stop 实现，避免出现三套语义。

## 6. 错误处理与安全

- manifest 版本、key、manager geometry、TP layout、token length、offset 范围、codec 或 checksum 任一不匹配：拒绝对应恢复；不能越界读取。
- zstd 不可用时，不接受 zstd generation；从空 cache 启动。
- endpoint control token 用 constant-time compare；缺失/错误返回 `403`。
- 恢复文件只允许位于已配置 root + escaped key 下，不接受 HTTP 路径参数，防止路径穿越。
- checkpoint 在持有独占 generation lock 时串行执行；并发第二次请求返回 `409`。
- metrics 和错误日志不得输出 control token。
- checkpoint 失败、corrupt generation、cache miss 都只能影响性能，不能改变输出 token。

## 7. 验证与验收

### 单元/回归

- binary manifest round trip；截断、坏 checksum、越界 offset、未知 version、错误 key 和错误 manager geometry 全部 fail open。
- staging 任意写入步骤失败时，`CURRENT` 仍可恢复旧 generation。
- resident GPU/CPU payload/ephemeral NVMe 三种来源导出为 generation 后均可按需 materialize。
- `PagedPrefixCacheTierDiskRef` generation 生命周期不触发现有 session truncation。
- Qwen3.5 single-device linear state、TP local state和 MTP state round trip；缺任一 linear layer 时整条 prefix 不可用。
- checkpoint endpoint 的 token、method、active-request `409` 和成功统计。
- Proxy external mode 从不调用 checkpoint；owned idle/pressure/shutdown 均按 drain → checkpoint → signal 顺序；checkpoint 失败仍只 signal owned process group。

### V100 端到端

1. 用 Qwen3.5/3.6 + Turbo3 + persistence 启动 owned backend。
2. 运行一个 page-aligned 长 prefix 请求，记录输出 hash 和 `/props` tier 指标。
3. 等待 idle unload，确认 GPU 显存回到 cold 基线，`CURRENT` 已原子提交。
4. 再发完全相同请求，触发第二 epoch。
5. 验收：
   - generation restore 成功；
   - `prefix_cache_disk_hits`/read bytes 增加；
   - cold-prefill work 或 TTFT 明显低于无恢复 cold run；
   - greedy 输出 hash 完全一致；
   - 两次 unload/reload 无 OOM、死锁、残留 child 或损坏 generation。
6. 删除/截断最新 generation 后重启：服务从空 cache 正常生成，输出仍一致，并报告可诊断的 restore error。

## 8. Rollout

- 所有新行为默认关闭。
- 第一阶段只在 Thinking Proxy owned FastLLM profile 启用；保留现有 external production profile 作为回退。
- 只有 V100 端到端命中、输出一致、异常恢复和显存回收 gates 全部通过后，才把 persistence 加入推荐部署环境。
