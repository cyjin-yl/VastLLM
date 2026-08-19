# 上游 bump 前必读

本仓有一批**反直觉的修复** —— 它们看起来像冗余代码或多余判断, 但每一处都对应
一个线上真实故障, 而且**故障都是静默的**(不报错、不崩溃, 只是悄悄输出垃圾)。
合并冲突时若"择优保留看起来更简洁的上游版本", 会把这些故障原样 bump 回来。

## 一条命令找出所有需要保护的位置

```bash
grep -rn "上游BUMP勿回退" src/ include/ example/ --include=*.cpp --include=*.h --include=*.cu
```

每处标记都写明了「回退会导致什么」。**冲突解决时以本仓版本为准。**

## 当前受保护的 10 处

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
cmake -DUNIT_TEST=ON .. && cmake --build . --target testSamplingGuard testToolCallGrammar testPreTokenizer
./testSamplingGuard      # 15/15   —— 采样路径(NaN / temperature=0 / top_k 两条分支)
./testToolCallGrammar    # ALL PASS —— 工具调用语法状态机
./testPreTokenizer       # 32/32   —— 预分词切分(不需要模型)
./testPreTokenizer <gguf> --corpus /home/ezra/projects/EzraVastLLM/pretokenizer-corpus/corpus_list.txt
                         # 42/42, 45/45 篇与 llama-tokenize 逐 token 一致
```

这两个测试就是为了"bump 之后能立刻知道有没有把旧错误带回来"而写的。
**测试不过不要合并。**

## 已知的上游差异(不是冲突, 但要知道)

- 上游 `ztxz16/fastllm` 的 `pushurl = no_push`, 推不上去, 只能单向拉。
- 我们领先上游 70+ 个 commit, 落后约 10 个。历史冲突点只有
  `src/models/deepseekv4.cpp` 和 `test/ops/regressionOps.cpp` 两个文件,
  与上面 9 处修复**不重叠**。
