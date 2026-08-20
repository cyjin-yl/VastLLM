// 算子路由普查(kernel route census)。
//
// 为什么需要它:
//   本仓在 SM70(V100) 上给同一个逻辑算子准备了**多条**实现, 靠一串
//   `if (...) return false;` 的 trial/fallback 链在运行时选路:
//     线性层(GGUF 权重): SM70 IQ4_XS DP4A MMQ -> 反量化+cuBLAS GEMM -> DP4A MMVQ
//     分页注意力:        SM70 XQA -> SM70 flash prefill -> FlashInfer -> 原生 fallback
//   这些判定条件多达几十个(dtype/维度/stride/page 布局/算力/是否在 graph capture),
//   任何一条不满足就**静默**退到下一条 —— 不打日志、不报错、结果照样对, 只是慢。
//
//   代价是排查全靠猜, 而且已经猜错过两次:
//     1. 生产 profile 里 FASTLLM_CUDA_SM70_PAGED_XQA=0 被当成"我们关掉了一个
//        默认开启的优化"。实际上 XQA 要求分页 KV 是 FLOAT16, 而生产用
//        turbo3(K=q8_0, V=turbo3) —— 这条路**无论开关取什么值都进不去**,
//        设 0 和设 1 完全等价。
//     2. 由此 sweep 里的 "SM70 开/关" 这一维实际什么都没测到, 却被写进文档
//        当成选型依据(见 v100-perfs/docs/EXPERIENCE.md 关于 c1/c6 的结论)。
//
//   一句"实际走了哪条 kernel"的可观测输出, 能让上面两个误判在几秒钟内被否掉。
//
// 开销:
//   常开部分只有每次命中一次 relaxed 原子自增(约 1-2 ns), 相对于任何一个
//   kernel launch 都可以忽略。按形状分组的明细表默认关闭, 需要
//   FASTLLM_KERNEL_ROUTE_STATS=1 才记录(带锁, 只在排查时开)。
//
// 用法:
//   curl -s localhost:8002/props | jq .kernel_routes
//   FASTLLM_KERNEL_ROUTE_STATS=1 ./apiserver ...   # 额外记录 (n,m,k) 明细
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fastllm {

    // 路由枚举。新增时**只能往末尾追加**, 否则 /props 的历史数据对不上。
    enum KernelRoute {
        // ---- GGUF 线性层 ----
        KERNEL_ROUTE_GGUF_SM70_IQ4XS_MMQ = 0, // SM70 DP4A MMQ, 直接在量化权重上做整数点积
        KERNEL_ROUTE_GGUF_MMVQ,               // DP4A MMVQ(逐 token), n <= MMVQ_MAX_BATCH_SIZE
        KERNEL_ROUTE_GGUF_DEQUANT_FP16_GEMM,  // 反量化成 FP16 -> cuBLAS HGEMM(张量核)
        KERNEL_ROUTE_GGUF_DEQUANT_FP32_GEMM,  // 反量化成 FP32 -> cuBLAS SGEMM
        // ---- 分页注意力 ----
        KERNEL_ROUTE_ATTN_SM70_PAGED_XQA,     // SM70 decode specialization(要求 FP16 分页 KV)
        KERNEL_ROUTE_ATTN_SM70_FLASH_PREFILL, // SM70 短 query causal prefill(要求 FP8_E4M3 分页 KV)
        KERNEL_ROUTE_ATTN_FLASHINFER,         // FlashInfer(V100 上不可用)
        KERNEL_ROUTE_ATTN_NATIVE_FALLBACK,    // 原生 gather + chunked cublas
        // 融合式打包分页注意力(q8_0 K + turbo3 V 直读, 不物化 fp16 中间缓冲)。
        // 注意: 这一项与 attn.native_fallback **并存**不冲突 —— 后者在
        // FastllmCudaHalfPagedAttentionBatch 的分派层就已计数, 前者记录的是
        // 分派进去之后**实际执行**的 kernel。要判断融合路有没有生效, 看这一项是否非零,
        // 不要看 native_fallback 是否归零。
        KERNEL_ROUTE_ATTN_SM70_TURBO_XQA,
        KERNEL_ROUTE_COUNT
    };

    const char *GetKernelRouteName(KernelRoute route);

    // 常开: 只做原子计数。ggmlType < 0 表示"与 GGUF 类型无关"。
    void KernelRouteHit(KernelRoute route, int ggmlType, int n, int m, int k);

    struct KernelRouteTotals {
        uint64_t calls = 0;
        uint64_t tokens = 0; // n 的累加, 用来区分"调用多"和"token 多"
        int32_t minN = 0;
        int32_t maxN = 0;
    };

    struct KernelRouteShape {
        KernelRoute route = KERNEL_ROUTE_COUNT;
        int ggmlType = -1;
        int n = 0, m = 0, k = 0;
        uint64_t calls = 0;
    };

    std::vector<KernelRouteTotals> GetKernelRouteTotals();
    // 明细表; 只有 FASTLLM_KERNEL_ROUTE_STATS=1 时才非空。
    std::vector<KernelRouteShape> GetKernelRouteShapes();
    bool KernelRouteShapeStatsEnabled();
    void ResetKernelRouteCensus();
    // 人读格式, 用于日志/关机摘要。
    std::string FormatKernelRouteCensus();

    // SM70 IQ4_XS DP4A MMQ 的**纯形状**资格判定(不碰任何 CUDA API)。
    // 返回 nullptr = 合格; 否则返回拒绝原因, 字符串与 kernel 里 s70_log_rej
    // 用的原因一致。
    //
    // 为什么单独抽出来:
    //   这几条数字上限(n in [8,64] / m%256 / k>=128)决定了"这个 749 行的
    //   MMQ kernel 在生产里到底有没有被用到", 而生产的 n 是由 batch 与
    //   prefill chunk 决定的 —— 靠读代码推很容易推错。抽成纯函数后
    //   可以在**没有 GPU** 的单测里, 用真实模型形状直接把结论钉死
    //   (见 test/kernelRouteCensusTest.cpp)。
    //   kernel 侧必须调用同一个函数, 否则测试和实现会悄悄走岔。
    const char *Sm70Iq4XsMmqShapeRejectReason(int dataType, int n, int m, int k);

    // 上面判定用到的边界, 供测试与文档引用, 不要各处硬编码。
    static const int kSm70Iq4XsMmqMinN = 8;
    static const int kSm70Iq4XsMmqMaxN = 64;
    static const int kSm70Iq4XsMmqBlockK = 256; // IQ4_XS 的 QK_K
    static const int kSm70Iq4XsMmqTileRows = 128; // 每个 CTA 负责的权重行数

} // namespace fastllm
