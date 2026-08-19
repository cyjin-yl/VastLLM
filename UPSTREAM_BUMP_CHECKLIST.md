# 上游 bump 前必读

本仓有一批**反直觉的修复** —— 它们看起来像冗余代码或多余判断, 但每一处都对应
一个线上真实故障, 而且**故障都是静默的**(不报错、不崩溃, 只是悄悄输出垃圾)。
合并冲突时若"择优保留看起来更简洁的上游版本", 会把这些故障原样 bump 回来。

## 一条命令找出所有需要保护的位置

```bash
grep -rn "上游BUMP勿回退" src/ include/ example/ --include=*.cpp --include=*.h --include=*.cu
```

每处标记都写明了「回退会导致什么」。**冲突解决时以本仓版本为准。**

## 当前受保护的 22 处

| 文件 | 修复 | 回退的后果 |
|---|---|---|
| `src/model.cpp` | **读入 GGUF merges 排名做 BPE**（最危险的一处） | 分词退化成"从左到右贪心", 模型看到的 token 序列与训练时不同 -> 逐字复述走样, 表现为"胡言乱语、路径拼不对" |
| `src/tokenizer.cpp` + `src/model.cpp` + `include/utils/unicode_categories.h` | **预分词(pre-tokenizer)正则切分**(与上面同源的另一半) | BPE 在整段文本上合并, 能跨词/跨数字/跨标点 -> token 序列与 HF/llama.cpp 不同(合法但非规范) -> 同样是"路径拼不对、标识符掉字符" |
| `src/models/qwen3_5.cpp` | MTP exact acceptance | 逐字抄写被偶发替换(路径/函数名/commit hash/工具调用 JSON) |
| `src/models/basellm.cpp` | 工具名规范化 | 模型写 `Bash`、harness 声明 `bash` -> 掩码打爆 -> 静默退回自由采样 -> 调用失败无报错 |
| `src/models/basellm.cpp` | `FetchResponseTokens` 条件变量等待 | 并发活锁: 99.8% CPU 卡在 mutex、GPU 0%、metrics 冻结 |
| `src/fastllm.cpp` | `SanitizeLogitsForSampling` | NaN 劫持采样 -> 满屏 `!`; 且比较器不满足严格弱序 -> `std::sort` UB |
| `src/fastllm.cpp` | `zeroTemperature` 贪心分支 | `temperature=0` 除以零 -> 输出退化成同一字符长串 |
| `src/fastllm.cpp` | `Grow` 显存保留区 | 池子吃光显存 -> 激活分配 fatal 杀进程(实测 30 分钟崩 5 代) |
| `src/fastllm.cpp` | 预算内优先扩容 | 回收器空转 54924 次 / 471.5 秒在关键路径, 缓存被反复吃掉 |
| `example/apiserver/apiserver.cpp` | temperature 钳制 | 同上, 覆盖 CUDA 采样 kernel 那条路 |
| `src/devices/cuda/.../fastllm-turboquant-kv.cu` | TurboKV 打印改 env 开关 | 生产日志刷 13 万行 + 热路径 `fflush` |
| `src/models/qwen3_5.cpp` + `src/models/basellm.cpp`(x2) + `src/models/deepseekv4.cpp` | **`forwardLocker` 用 defer_lock 的 `unique_lock` 而非裸引用** | 批前向抛异常(Grow 显存不足)时跳过手工 `unlock()` -> 锁永久不释放 -> 下一轮 `lock()` 自死锁, 且线程攥着锁僵死 -> 所有客户端堵在 `FetchResponseTokens`。现场: running/pending 冻结、done 0 req、GPU 0% 而显存照占, 日志却写着 "process survives"。**间歇性**: 异常抛在 unlock 区间就不死(实测 6 次里前 5 次都活), 别当成玄学 |
| `src/fastllm.cpp` | **页字节数按 `dims[0]` 而非 `maxPages` 换算**(4 处) | `dims[0] <= maxPages` 恒成立(懒分配+Grow 被预算挡住)。恰好整除时 pageBytes 算小 -> 从错误 offset 抠页, "下放成功"但内容错; 上提用正确的 `dims[0]`, 尺寸永远对不上 -> **前缀缓存上提 100% 失败**, 表现为 `L1trie` 跳一下就归零 + `hits=0` 恒成立。另含 linear-attention 借用指针 stride 错(会指到别的请求的页上, 静默数据串扰) |
| `src/fastllm.cpp` | **`GetUnusedPageIndex` 里的投机性 `Grow` 包 try/catch + 退避** | 该处 `pageIndex` 通常已拿到手, 扩容只是提前垫水位; 裸调 `Grow` 抛异常会穿到 MTPLoop 的 catch, 把**所有**在飞请求 `isAbort`。实测 `prefill 63231 tok 87.94s / decode 0 tok` —— 跑完 88 秒 prefill 一个 token 没吐就被打掉 |
| `src/prefixcache_persistence.cpp` | **配额不足时先回收陈旧 generation 再判** | 原顺序把回收写在提交成功之后: 目录顶到配额线 -> 提交失败 -> 回收永不执行 -> **永久失败**。另: 旧顺序凭空多要一代余量(判定时需同时容纳三代) |
| `example/apiserver/apiserver.cpp` | **派发前 + 非流式循环的 `SocketPeerDisconnected` 探活** | 客户端早走了仍把请求算到底: 262K prefill 独占 GPU 几分钟、KV 页整段不释放把池顶到水位线诱发 `Grow` 失败、真请求被挤在队列后面。非流式路径原本用阻塞 `FetchResponseTokens`, 整个 prefill 期间零检测 |
| `example/apiserver/apiserver.cpp` | **轻量路由(`IsLightweightRoute` + `lightQ`)不占推理并发额度** | `maxActivateQueryNumber = min(256, --batch)`, 生产 `--batch 1` => 1; 闸门对所有路由一视同仁 -> 只要有请求在生成, `/health` `/version` 就排队到超时 -> **上游代理无法区分"在忙"与"已僵死"**, 本次死锁故障因此拖了一小时才被发现。注意 `/admin/*` 不能放进来(会 suspend/resume 模型, 必须串行) |
| `src/devices/cuda/fastllm-iq4xs-sm70.cu` | **设备能力查询缓存(`s70_device_caps`), 不再每次调用都 `cudaGetDeviceProperties`** | `FastllmCudaTrySm70Iq4XsMmq` 在**每一次 IQ4_XS 线性层调用**上都会先查设备属性, 而且排在 `n/m/k` 形状判定之前。本机实测 `cudaGetDeviceProperties` = **36.5 us/call**(对比 `cudaGetDevice` 0.027us、`cudaDeviceGetAttribute` 0.016us), 微基准见 `~/projects/EzraVastLLM/scripts/sm70-audit/bench_devprops.cu`。Qwen3.8-27B 每 decode 一个 token 约数百次线性调用 -> 约 **10+ ms/token 纯白花开销**, 而且 decode 的 n=1..3 必然在后面的 `n range` 上被拒, 这次查询 100% 无用。**只在 IQ4_XS 档位出现**(Q5_K_M 的权重不是 `GGML_TYPE_IQ4_XS`, 根本不进这个 trial), 所以表现成"换成更低位宽反而更慢", 极易被误判成"IQ 量化在 V100 上不行"而错误回退到 Q5 |
| `src/devices/cuda/fastllm-iq4xs-sm70.cu` + `src/fastllm-kernel-route.cpp` | **MMQ 形状资格判定收敛到纯函数 `Sm70Iq4XsMmqShapeRejectReason`** | `n in [8,64] / m%256 / k>=128` 这几个数字决定了这个 749 行 kernel 在生产里**到底有没有被用到**: decode 的 n=1..3、长 prefill 的 n=1024 两头都在区间外。判定若在 kernel 里再抄一份, 就会和 `test/kernelRouteCensusTest.cpp` 里用真实模型形状钉死的断言悄悄走岔, 测试随之失效 |
| `src/fastllm.cpp` | **KV 预算守卫的每页字节数走 `GetPagedPoolPageBytes`(打包感知), 不再用 `unitSize` 乘** | 打包 KV 类型(`Q8_0_KV`/`TURBO3_KV`/`TURBO4_KV`)的 `unitSize` 恒为 1, 是占位值不是每元素字节数。用 `pageLen*numHeads*headDim*unitSize` 估算, 在 head_dim=256 下给出 256B/行, 而真实是 q8_0 272B、turbo3 100B、turbo4 132B —— 对 K 少算 5.9%, 对 V 多算 156%, 32 个池合计**高估 1.376 倍**, 守卫于是把 `maxPages` 砍到实际能放下的 72.7%, **上下文容量凭空少 27%**。现场只留一句 `clamping maxPages A → B`, 会把人误导去调 `--tokens` / `--kv_cache_dtype`。注意 `Grow()` 一直用的是 `GetDataBytes`(正确), 只有这个守卫在用另一套账 |
| `src/models/basellm.cpp` + `include/models/basellm.h` + `example/apiserver/apiserver.cpp` | **`GetPagedCachePoolStats` 按物理页 `dims[0]` 统计水位, 逻辑预算 `maxPages` 单独由 `logicalMaxPagesOut` 带出; metrics 行加 `budget=` 字段** | 这是 af3830b1(页字节数 `dims[0]` vs `maxPages`)的**同胞, 当时漏掉的一处**。页池懒分配: `initialPages = min(128, maxPages)`, 之后靠 `Grow()` 追赶, 而 `freePages` 里只可能有物理页。旧式 `maxPages - FreePageCount()` 把"还没分配出来的页"全算成"正在使用"。线上实测(2026-08-20 空载, 只跑过 2 个请求): 32 个 manager 各 `maxPages=2048`/`dims[0]=128` 且全空闲, 打印成 **`kv_pool=61440/65536 pg (94%)`, 而真实占用是 0%**。一整天"页池顶满"的判断都建立在这个数上, 导致反复去动池预算和前缀缓存, 而池子其实是空的。`qwen3_5.cpp` 调度器里 `busyPages` 一直是对的(显式用 `dims[0]`), 两边口径必须一致 |
| `src/fastllm.cpp` + `src/models/qwen3_5.cpp` | **GDN 快照标称长度与递归状态位置不一致时计数(`extra-skip-unaligned`), 并提供 `FASTLLM_PREFIX_CACHE_STRICT_SNAPSHOT_ALIGN` 逃生开关** | `TryRecordPagedPrefixCacheExtra` 把 `currentLen` 向下取整到页边界, 但快照里的 GDN 递归状态对应的是取整**前**的 `currentLenRaw`。命中恢复时全注意力 KV 被恢复成正好 `cachedLen` 个 token, 而递归状态已经吃过更多, 区间 `[cachedLen, currentLenRaw)`(最多 127 个 token)被**卷进递归状态两次** —— 静默算错, 不报错不崩溃。**默认不拒绝**是刻意的: agent 负载的主导形态是单调增长的会话(第 N 轮 prompt 严格包含第 N-1 轮全部内容), 服务它的恰恰是生成结束时那个必然不对齐的快照, 拒绝掉会让每轮全量重算。真正的解法是"在 decode 的对齐点记录快照", 未验证前不上。回退掉计数器就再也看不到这个偏差有多频繁 |

## 新增: 算子路由普查(`include/fastllm-kernel-route.h`)

这一处不是"防回退", 是**防再次误判**。本仓在 V100 上给同一个逻辑算子准备了多份
实现, 靠一串 `if (...) return false;` 在运行期**静默**选路 —— 选错路不报错、结果
也对, 只是慢。因此"实际走了哪条 kernel"长期只能靠读代码猜, 而且已经猜错两次:

1. 生产 profile 里 `FASTLLM_CUDA_SM70_PAGED_XQA=0`, 而
   `fastllm-paged-attention-native.cuh:15` 写着 "Enabled by default", 于是被当成
   "我们把一个默认开启的优化关掉了, 打开就能提速"。
   **真相**: 该路径要求分页 K/V 都是 `FLOAT16`, 生产用 turbo3(K=q8_0, V=turbo3),
   开关取 0 还是 1 结果完全一样。同理 `FASTLLM_CUDA_SM70_FLASH_ATTN` 要求
   `FP8_E4M3`, 也进不去。
   连带后果: `v100-perfs/scripts/sweep_profiles.py` 把这两个开关绑成一个
   "SM70 开/关" 维度扫过, 两组的后端日志逐行相同, 这一维**什么都没测到**,
   却被写进 `v100-perfs/docs/EXPERIENCE.md` 当成选型依据。
2. 749 行的 SM70 IQ4_XS DP4A MMQ 被当成 "IQ4_XS 的 decode 快路径"。
   **真相**: 它只接受 `n in [8,64]`, decode 的 n 是 1..3, 长 prefill 的 n 是上千。

对应的三件事:

- `KernelRouteHit()` 常开计数(每次一个 relaxed 原子自增), 经 `/props` 的
  `kernel_routes` 暴露: `curl -s localhost:8002/props | jq .kernel_routes`
- `FASTLLM_KERNEL_ROUTE_STATS=1` 额外记录 `(route, ggml_type, n, m, k)` 明细,
  并让 15 秒的 `[metrics]` 行附带一行 `[kernel-route] ...`
- `FastllmReportSm70AttentionRouteOnce()`: 只要**显式设置过**那两个 SM70 环境变量,
  启动后第一次注意力就打印一次"该开关在当前 KV dtype 下不起作用"

**bump 时若这些被当成'调试代码'删掉, 上面两类误判会立刻重新可能发生。**

## 最危险的一处: `src/model.cpp` 的 token score

上游那一行长这样, 看起来**完全无害**:

```cpp
model->weight.AddTokenizerWord(it.string_value(), idx, 1.0f);
```

合并冲突时几乎必然会被当成"更简洁的版本"保留下来。但这个 `1.0f` 会让
248320 个 token 的 score 全相等, BPE 的合并优先级随之退化成"从左到右贪心",
完全不看 merges 排名 —— **那不是 BPE**。

判断是否被回退, 看启动日志有没有这一行:

```
Load tokenizer merges = 247587 (BPE 合并排名已启用)
```

没有这行就是被回退了。快速验证:

```bash
curl ... -d '{"messages":[{"role":"user","content":"逐字复述这一行, 只输出这一行:\nbrutalist"}],"temperature":0}'
# 正确: brutalist      被回退: "bru talist"
```

## 第二危险的一处: 从来没读过 `tokenizer.ggml.pre`

`grep -rn "tokenizer.ggml.pre" src/ include/` 曾经是**零命中** —— 上游的
`Tokenizer::Encode` 只按 special token 切分, 然后把**整段**剩余文本交给
`BytePairEncode`。而 HF / llama.cpp / vLLM 都先用 GGUF 里 `tokenizer.ggml.pre`
指定的预分词正则把文本切成块(单词 / 单个数字 / 一串标点 / 一段空白),
BPE 只在块内合并, **永远跨不出块边界**。

少了这一步, BPE 会跨词、跨数字、跨标点任意合并, 切出训练语料里根本不存在的
token 序列。它"合法但非规范": 解码回字符串一模一样, 所以任何字符串层面的自检
都发现不了, 只表现为模型逐字复述走样。

判断是否被回退, 看启动日志有没有这两行:

```
Load tokenizer model = gpt2
Load tokenizer pre = qwen35 (预分词正则已启用: qwen35)
```

没有这两行(或者变成 `Warning: ... 预分词切分未启用`)就是被回退了。

最容易漏的细节: qwen35 正则里 `\p{N}` 是**单个数字**成块, 数字逐位切分
(`12345` -> `1|2|3|4|5`)。写成 `\p{N}+` 一眼看不出错, 但所有长数字都会切错。

另外 `std::regex` **不支持** `\p{L}` 这类 Unicode 属性类, 别试图用它重写;
现在的实现是"码点类别表 + 手写状态机", 逐行对照 llama.cpp
`src/unicode.cpp` 的 `unicode_regex_split_custom_qwen35`。

## bump 后必须跑的回归

```bash
cmake -DUNIT_TEST=ON .. && cmake --build . --target testSamplingGuard testToolCallGrammar testPreTokenizer testPrefixCacheTier testPagedKvBudget
./testSamplingGuard      # 15/15   —— 采样路径(NaN / temperature=0 / top_k 两条分支)
./testToolCallGrammar    # ALL PASS —— 工具调用语法状态机
./testPreTokenizer       # 31/31   —— 预分词切分(不需要模型)
./testPrefixCacheTier    # 29/29   —— 前缀缓存三级下放/上提
./testPagedKvBudget      # 分页 KV 显存账本 + 页池水位口径 + GDN 快照对齐
FASTLLM_PREFIX_CACHE_STRICT_SNAPSHOT_ALIGN=1 ./testPagedKvBudget   # 严格对齐分支
./testPreTokenizer <gguf> --corpus /home/ezra/projects/EzraVastLLM/pretokenizer-corpus/corpus_list.txt
                         # 42/42, 45/45 篇与 llama-tokenize 逐 token 一致
```

这两个测试就是为了"bump 之后能立刻知道有没有把旧错误带回来"而写的。
**测试不过不要合并。**

## 已知的上游差异(不是冲突, 但要知道)

- 上游 `ztxz16/fastllm` 的 `pushurl = no_push`, 推不上去, 只能单向拉。
- 我们领先上游 70+ 个 commit, 落后约 10 个。历史冲突点只有
  `src/models/deepseekv4.cpp` 和 `test/ops/regressionOps.cpp` 两个文件,
  与上面 15 处修复**不重叠**。
