# 上游 bump 前必读

本仓有一批**反直觉的修复** —— 它们看起来像冗余代码或多余判断, 但每一处都对应
一个线上真实故障, 而且**故障都是静默的**(不报错、不崩溃, 只是悄悄输出垃圾)。
合并冲突时若"择优保留看起来更简洁的上游版本", 会把这些故障原样 bump 回来。

## 一条命令找出所有需要保护的位置

```bash
grep -rn "上游BUMP勿回退" src/ include/ example/ --include=*.cpp --include=*.h --include=*.cu
```

每处标记都写明了「回退会导致什么」。**冲突解决时以本仓版本为准。**

## 当前受保护的 9 处

| 文件 | 修复 | 回退的后果 |
|---|---|---|
| `src/models/qwen3_5.cpp` | MTP exact acceptance | 逐字抄写被偶发替换(路径/函数名/commit hash/工具调用 JSON) |
| `src/models/basellm.cpp` | 工具名规范化 | 模型写 `Bash`、harness 声明 `bash` -> 掩码打爆 -> 静默退回自由采样 -> 调用失败无报错 |
| `src/models/basellm.cpp` | `FetchResponseTokens` 条件变量等待 | 并发活锁: 99.8% CPU 卡在 mutex、GPU 0%、metrics 冻结 |
| `src/fastllm.cpp` | `SanitizeLogitsForSampling` | NaN 劫持采样 -> 满屏 `!`; 且比较器不满足严格弱序 -> `std::sort` UB |
| `src/fastllm.cpp` | `zeroTemperature` 贪心分支 | `temperature=0` 除以零 -> 输出退化成同一字符长串 |
| `src/fastllm.cpp` | `Grow` 显存保留区 | 池子吃光显存 -> 激活分配 fatal 杀进程(实测 30 分钟崩 5 代) |
| `src/fastllm.cpp` | 预算内优先扩容 | 回收器空转 54924 次 / 471.5 秒在关键路径, 缓存被反复吃掉 |
| `example/apiserver/apiserver.cpp` | temperature 钳制 | 同上, 覆盖 CUDA 采样 kernel 那条路 |
| `src/devices/cuda/.../fastllm-turboquant-kv.cu` | TurboKV 打印改 env 开关 | 生产日志刷 13 万行 + 热路径 `fflush` |

## bump 后必须跑的回归

```bash
cmake -DUNIT_TEST=ON .. && cmake --build . --target testSamplingGuard testToolCallGrammar
./testSamplingGuard      # 15/15   —— 采样路径(NaN / temperature=0 / top_k 两条分支)
./testToolCallGrammar    # ALL PASS —— 工具调用语法状态机
```

这两个测试就是为了"bump 之后能立刻知道有没有把旧错误带回来"而写的。
**测试不过不要合并。**

## 已知的上游差异(不是冲突, 但要知道)

- 上游 `ztxz16/fastllm` 的 `pushurl = no_push`, 推不上去, 只能单向拉。
- 我们领先上游 70+ 个 commit, 落后约 10 个。历史冲突点只有
  `src/models/deepseekv4.cpp` 和 `test/ops/regressionOps.cpp` 两个文件,
  与上面 9 处修复**不重叠**。
