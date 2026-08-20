//
// SM70 (Tesla V100) 融合式打包分页注意力 —— 直接在 q8_0 K + turbo3 V 上做注意力,
// 不再把量化 KV 反量化成 float16 连续缓冲。
//
// 结构来源 —— 从本仓已有的 FP16 快路径改写:
//   src/devices/cuda/attention/paged/fastllm-paged-attention-native.cu
//       -> FastllmSm70PagedXqaSplitKernel            (~1470-1642 行) 的整体骨架:
//          每个 4-warp block 处理一个 kv head, 一次 K/V 读喂 group 个 query head;
//          每 lane 持有 8 个连续维度; split-K over kvLen + phase2 combine。
//       -> FastllmPagedAttentionCombineGQAKernel     (~1917-1960 行) 的合并逻辑。
//   打包格式定义: include/devices/cuda/attention/fastllm-turboquant-kv-layout.h
//   反量化语义对照: src/devices/cuda/attention/paged/fastllm-turboquant-kv.cu
//       -> GatherQ8HeadRangeKernel      (~415-434 行)
//       -> GatherTurbo3HeadRangeKernel  (~435-457 行)
//       -> InverseWht128                (~169-177 行)
//
// ===========================================================================
// 【上游BUMP勿回退】核心手法: 逆 Walsh-Hadamard 变换被**整体挪到最后**
// ===========================================================================
//
// turbo3 的反量化不是逐元素的。按 GatherTurbo3HeadRangeKernel 的定义:
//
//     V_j = InverseWht128( centroid[idx_j] * correctedNorm_j )
//
// 而 InverseWht128 展开是
//
//     x *= kWhtSigns2;  Hadamard 蝶形 x7;  x *= (1/sqrt(128)) * kWhtSigns1
//
// 即 W^-1 = D1 · H · D2, 三个因子都是线性算子(两个对角符号阵 + Hadamard 阵),
// **整体是正交线性变换, 不含任何非线性**。注意力的 P·V 累加也是线性的, 于是
//
//     out = Σ_j p_j · W^-1(u_j) = W^-1( Σ_j p_j · u_j )     其中 u_j = norm_j · centroid[idx_j]
//
// 也就是说: 逆 WHT 可以只在**最后**做一次, 而不是每个 key 做一次。
//
// 这一步不是"顺手的优化", 而是整个 kernel 成立的前提。若每个 key 都做 128 点逆 WHT,
// 每 warp 每 token 要多付约 72 条 ALU + 32 条 warp shuffle, 内核会从访存瓶颈变成
// 计算瓶颈(手算等效带宽上限掉到 ~360 GB/s), 收益基本被吃光。挪到最后之后, V 侧
// 每 token 只剩「3 条 load + 8 次查表 + 8 次 FMA」, 比读 fp16 的 V 还便宜。
//
// 为什么和在线 softmax / split-K 合并不冲突:
//   - flash 在线 softmax 的修正是 acc *= corr, 标量乘, 与线性变换可交换;
//   - split-K 的 phase2 合并是 Σ_s acc_s · exp2(m_s - M) / L, 也全是标量加权;
//   - W^-1 在每个 128 维块内独立, headDim=256 恰好是两块, 互不串扰。
//   因此 acc 全程停留在「未变换域」(下称 u 域), 只在 combine kernel 末尾统一变换。
//
// 副作用是精度**变好**: 现路径每个 V 元素做完逆 WHT 要 __float2half_rn 落成 fp16 再喂
// cuBLAS, 且 score 缓冲(qk)也是 half; 本 kernel 全程 fp32。代价是与现路径**不会逐位一致**,
// 所以 test/ops/turboPagedAttentionTest.cpp 用的是 fp64 参考 + 容差, 不是 bit-exact 对拍。
//
// ===========================================================================
// 【上游BUMP勿回退】为什么必须支持 qoLen > 1, 不能只做 qoLen == 1
// ===========================================================================
//
// 本仓已经有两条**入口条件永远不满足**的 SM70 死代码:
//   FastllmCudaTrySm70PagedAttentionDecode  要求分页 K/V 都是 FLOAT16 —— 生产是 turbo3;
//   FastllmCudaTrySm70FlashAttentionPrefill 要求分页 K/V 都是 FP8_E4M3 —— 同样进不去。
// 两个开关在生产 profile 里取 0 还是 1 运行期完全一样(见 fastllm-attention.cu 里
// FastllmReportSm70AttentionRouteOnce 的长注释)。
//
// 第三个坑在 qoLen 上: 生产开着 MTP(FASTLLM_QWEN35_ENABLE_MTP=2, drafts_per_step=2),
// qwen3_5.cpp 的 isSpeculativeValidation 让稳态 target 前向 seqLen = 2 或 3,
// 于是 FastllmCudaHalfPagedAttentionBatchFastllmFallback 里的 isDecode 恒为 false。
// 只支持 qoLen==1 的 kernel 在生产里一次都不会被调用。
//
// 所以这里按 QO_LEN 模板实例化 1..4, 并用 GROUP_CHUNK 把 query head 拆进 grid,
// 保证寄存器里活跃的 (m, l, acc[8]) 组数恒为 QO_LEN*GROUP_CHUNK <= 6:
//     QO_LEN=1 -> GROUP_CHUNK=6 (K/V 每行读 1 次)
//     QO_LEN=2 -> GROUP_CHUNK=3 (读 2 次)
//     QO_LEN=3 -> GROUP_CHUNK=2 (读 3 次)
//     QO_LEN=4 -> GROUP_CHUNK=1 (读 4 次)
// 即便 qoLen=3 要把 K/V 读 3 遍(3*372 B/行), 相对现路径仍然少一个数量级。
//
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "attention/fastllm-turboquant-kv-layout.h"
#include "attention/fastllm-paged-attention-turbo-xqa.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>

#if !defined(USE_ROCM)

namespace {

using fastllm::turbokv::kQ8BlockBytes;
using fastllm::turbokv::kTurbo3BlockBytes;
using fastllm::turbokv::kTurbo4BlockBytes;

constexpr int kTurboHeadDim   = 256;
constexpr int kTurboQ8Row     = 272;   // = Q8RowBytes(256)
constexpr int kTurboT3Row     = 100;   // = Turbo3RowBytes(256)
constexpr int kTurboT4Row     = 132;   // = Turbo4RowBytes(256)
constexpr int kTurboWarps     = 4;
constexpr int kTurboThreads   = kTurboWarps * 32;
constexpr int kTurboDimsLane  = kTurboHeadDim / 32;   // 每 lane 8 个连续维度
constexpr int kTurboMaxSplits = 128;
constexpr int kTurboSplitTargetBlocks = 384;          // 与原生 GQA decode 的取值一致
constexpr int kTurboMaxQoLen  = 4;
constexpr int kTurboGroup     = 6;                    // Qwen3.5/3.8: 24 Q head / 4 KV head

static_assert(fastllm::turbokv::Q8RowBytes(kTurboHeadDim) == kTurboQ8Row, "q8_0 行字节数变了");
static_assert(fastllm::turbokv::Turbo3RowBytes(kTurboHeadDim) == kTurboT3Row, "turbo3 行字节数变了");
static_assert(fastllm::turbokv::Turbo4RowBytes(kTurboHeadDim) == kTurboT4Row, "turbo4 行字节数变了");

__device__ __constant__ float kTurboXqaCentroids[8] = FASTLLM_TURBOKV_TURBO3_CENTROIDS_INIT;
__device__ __constant__ float kTurbo4XqaCentroids[16] = FASTLLM_TURBOKV_TURBO4_CENTROIDS_INIT;
__device__ __constant__ float kTurboXqaWhtSigns1[128] = FASTLLM_TURBOKV_WHT_SIGNS1_INIT;
__device__ __constant__ float kTurboXqaWhtSigns2[128] = FASTLLM_TURBOKV_WHT_SIGNS2_INIT;

// 与 fastllm-turboquant-kv.cu 的 WhtStage / InverseWht128 逐行等价(那边在匿名 namespace 里,
// 跨 TU 用不了)。这里作用在长度 128 的共享内存半块上, 由 blockDim=256 的两半分别调用,
// 内部的 __syncthreads() 由全 block 统一到达。
__device__ __forceinline__ void TurboWhtStage(float *x, int lane, int h) {
    if ((lane % (2 * h)) < h) {
        float a = x[lane];
        float b = x[lane + h];
        x[lane] = a + b;
        x[lane + h] = a - b;
    }
    __syncthreads();
}

__device__ __forceinline__ void TurboInverseWht128(float *x, int lane) {
    x[lane] *= kTurboXqaWhtSigns2[lane];
    __syncthreads();
    TurboWhtStage(x, lane, 1);  TurboWhtStage(x, lane, 2);  TurboWhtStage(x, lane, 4);
    TurboWhtStage(x, lane, 8);  TurboWhtStage(x, lane, 16); TurboWhtStage(x, lane, 32);
    TurboWhtStage(x, lane, 64);
    x[lane] *= 0.08838834764831845f * kTurboXqaWhtSigns1[lane];
    __syncthreads();
}

// -------------------------------------------------------------------------
// phase1: 每个 block 负责 (kvHead, groupChunk, split), 输出 u 域的部分 softmax 状态。
// -------------------------------------------------------------------------
template <int QO_LEN, int GROUP_CHUNK>
__global__ void __launch_bounds__(kTurboThreads, 4)
FastllmSm70PagedTurboXqaSplitKernel(
    const half   * __restrict__ qd,
    const uint8_t * __restrict__ pagedK,
    const uint8_t * __restrict__ pagedV,
    float        * __restrict__ scratch,
    const int32_t * __restrict__ pageIndices,
    int numPages, int lastPageLen, int pageLenShift, int pageLenMask,
    int numKvHeads, int groupChunks,
    int qStrideH, int qStrideN, float scale, int splits) {
    constexpr int kSets = QO_LEN * GROUP_CHUNK;

    const int group = GROUP_CHUNK * groupChunks;
    const int hc = blockIdx.x;
    const int kvHead = hc / groupChunks;
    const int chunkId = hc - kvHead * groupChunks;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int pageLen = pageLenMask + 1;
    const int kvLen = (numPages - 1) * pageLen + lastPageLen;
    const int chunkSize = (kvLen + splits - 1) / splits;
    const int kvStart = split * chunkSize;
    const int kvEnd = min(kvStart + chunkSize, kvLen);

    __shared__ float sQ[kSets * kTurboHeadDim];
    __shared__ float sLut[8];
    __shared__ float sAcc[kTurboWarps * kTurboHeadDim];
    __shared__ float sM[kSets * kTurboWarps];
    __shared__ float sL[kSets * kTurboWarps];

    // Q 预乘 scale*log2(e): 点积出来直接就是 log2 域, 省掉每 (set, key) 一次乘法。
    const float scaleLog2 = scale * 1.4426950408889634f;
    for (int idx = tid; idx < kSets * kTurboHeadDim; idx += kTurboThreads) {
        const int s = idx / kTurboHeadDim;
        const int d = idx - s * kTurboHeadDim;
        const int t = s / GROUP_CHUNK;
        const int c = s - t * GROUP_CHUNK;
        const int h = kvHead * group + chunkId * GROUP_CHUNK + c;
        sQ[idx] = __half2float(qd[(size_t)h * qStrideH + (size_t)t * qStrideN + d]) * scaleLog2;
    }
    if (tid < 8) {
        sLut[tid] = kTurboXqaCentroids[tid];
    }
    __syncthreads();

    float m[kSets], l[kSets], acc[kSets * kTurboDimsLane];
    #pragma unroll
    for (int s = 0; s < kSets; s++) {
        m[s] = -1e30f;
        l[s] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < kSets * kTurboDimsLane; i++) {
        acc[i] = 0.0f;
    }

    // lane -> 维度切分。d0 = lane*8, 整 warp 覆盖 256 维。
    //   q8_0  : 32 值/块, 所以 lane 的 8 个维度必定落在同一个 q8 块内 (块号 lane/4)。
    //   turbo3: 128 值/块, lane 的 8 个维度落在同一个 turbo3 块内 (块号 lane/16),
    //           块内是第 (lane%16) 个 8 值组 -> low2 恰好 2 字节, high1 恰好 1 字节。
    const int d0 = lane * kTurboDimsLane;
    const int q8Block = lane >> 2;
    const int q8Off = (lane & 3) * 8;
    const int t3Block = lane >> 4;
    const int t3Grp = lane & 15;

    for (int j = kvStart + warp; j < kvEnd; j += kTurboWarps) {
        const int pageSlot = j >> pageLenShift;
        const int offInPage = j & pageLenMask;
        const int page = pageIndices[pageSlot];
        const size_t row = ((size_t)page * pageLen + offInPage) * numKvHeads + kvHead;

        // ---- K: q8_0, 逐元素反量化 ----
        const uint8_t *kBlk = pagedK + row * kTurboQ8Row + (size_t)q8Block * kQ8BlockBytes;
        const float kScale = __half2float(__ldg(reinterpret_cast<const half *>(kBlk)));
        const int8_t *kQs = reinterpret_cast<const int8_t *>(kBlk + 2 + q8Off);
        float kreg[kTurboDimsLane];
        #pragma unroll
        for (int i = 0; i < kTurboDimsLane; i++) {
            kreg[i] = kScale * (float)kQs[i];
        }

        // ---- V: turbo3, **只解码到 u 域, 不做逆 WHT**(见文件头) ----
        const uint8_t *vBlk = pagedV + row * kTurboT3Row + (size_t)t3Block * kTurbo3BlockBytes;
        const float vNorm = __half2float(__ldg(reinterpret_cast<const half *>(vBlk)));
        // low2 的 2 字节小端拼成 uint16 后, 第 i 个值的 2 bit 正好在 bit 2*i 处
        // (i<4 来自低字节 bit 0/2/4/6, i>=4 来自高字节 -> bit 8/10/12/14)。
        const uint32_t low2 = __ldg(reinterpret_cast<const uint16_t *>(vBlk + 2 + 2 * t3Grp));
        const uint32_t high1 = __ldg(vBlk + 2 + 32 + t3Grp);
        float vreg[kTurboDimsLane];
        #pragma unroll
        for (int i = 0; i < kTurboDimsLane; i++) {
            const int idx = (int)(((low2 >> (2 * i)) & 3u) | (((high1 >> i) & 1u) << 2));
            vreg[i] = vNorm * sLut[idx];
        }

        float partial[kSets];
        #pragma unroll
        for (int s = 0; s < kSets; s++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < kTurboDimsLane; i++) {
                dot += sQ[s * kTurboHeadDim + d0 + i] * kreg[i];
            }
            partial[s] = dot;
        }
        // group 个点积交错做 warp 归约, 关键路径只有一段 5 级 shuffle 链
        // (照搬 fastllm-paged-attention-native.cu:1385 注释里说明的手法)。
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int s = 0; s < kSets; s++) {
                partial[s] += __shfl_xor_sync(0xffffffffu, partial[s], off);
            }
        }

        #pragma unroll
        for (int s = 0; s < kSets; s++) {
            if (QO_LEN > 1) {
                // 因果: query token t 只能看到 key j <= kvLen - QO_LEN + t。
                // j 在整个 warp 内相同, 该分支 warp-uniform, 不产生 divergence。
                const int t = s / GROUP_CHUNK;
                if (j > kvLen - QO_LEN + t) {
                    continue;
                }
            }
            const float score = partial[s];
            if (score > m[s]) {
                const float corr = exp2f(m[s] - score);
                #pragma unroll
                for (int i = 0; i < kTurboDimsLane; i++) {
                    acc[s * kTurboDimsLane + i] *= corr;
                }
                l[s] *= corr;
                m[s] = score;
            }
            const float p = exp2f(score - m[s]);
            #pragma unroll
            for (int i = 0; i < kTurboDimsLane; i++) {
                acc[s * kTurboDimsLane + i] += p * vreg[i];
            }
            l[s] += p;
        }
    }

    // 跨 warp 归约 -> scratch。逐 set 复用同一块 sAcc(4KB), 避免 kSets*kWarps*256
    // 的共享内存把每 SM 驻留 block 数压下去; 循环编译期展开, acc 索引仍是常量, 不会 spill。
    #pragma unroll
    for (int s = 0; s < kSets; s++) {
        if (lane == 0) {
            sM[s * kTurboWarps + warp] = m[s];
            sL[s * kTurboWarps + warp] = l[s];
        }
        #pragma unroll
        for (int i = 0; i < kTurboDimsLane; i++) {
            sAcc[warp * kTurboHeadDim + d0 + i] = acc[s * kTurboDimsLane + i];
        }
        __syncthreads();

        const int t = s / GROUP_CHUNK;
        const int c = s - t * GROUP_CHUNK;
        const int h = kvHead * group + chunkId * GROUP_CHUNK + c;
        float *slot = scratch +
            ((size_t)(h * QO_LEN + t) * splits + split) * (kTurboHeadDim + 2);
        float maxScore = -1e30f;
        #pragma unroll
        for (int w = 0; w < kTurboWarps; w++) {
            maxScore = fmaxf(maxScore, sM[s * kTurboWarps + w]);
        }
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < kTurboWarps; w++) {
            sum += sL[s * kTurboWarps + w] * exp2f(sM[s * kTurboWarps + w] - maxScore);
        }
        for (int d = tid; d < kTurboHeadDim; d += kTurboThreads) {
            float value = 0.0f;
            #pragma unroll
            for (int w = 0; w < kTurboWarps; w++) {
                value += sAcc[w * kTurboHeadDim + d] * exp2f(sM[s * kTurboWarps + w] - maxScore);
            }
            slot[d] = value;
        }
        if (tid == 0) {
            slot[kTurboHeadDim] = maxScore;
            slot[kTurboHeadDim + 1] = sum;
        }
        __syncthreads();
    }
}

// -------------------------------------------------------------------------
// phase2: 合并 split, 然后**在这里**统一做那一次逆 WHT, 再写出 fp16。
// -------------------------------------------------------------------------
template <int QO_LEN>
__global__ void FastllmSm70PagedTurboXqaCombineKernel(
    const float * __restrict__ scratch,
    half        * __restrict__ od,
    int group, int splits, int outHeadStride, int outTokenStride) {
    const int kvh = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;                 // blockDim.x == kTurboHeadDim
    const int headDimPlus = kTurboHeadDim + 2;

    __shared__ float sMs[kTurboMaxSplits];
    __shared__ float sLs[kTurboMaxSplits];
    __shared__ float x[kTurboHeadDim];

    for (int g = 0; g < group; g++) {
        const int h = kvh * group + g;
        const float *base = scratch + (size_t)(h * QO_LEN + t) * splits * headDimPlus;
        for (int s = d; s < splits; s += kTurboHeadDim) {
            sMs[s] = base[(size_t)s * headDimPlus + kTurboHeadDim];
            sLs[s] = base[(size_t)s * headDimPlus + kTurboHeadDim + 1];
        }
        __syncthreads();

        float M = -1e30f;
        for (int s = 0; s < splits; s++) {
            M = fmaxf(M, sMs[s]);
        }
        float L = 0.0f;
        for (int s = 0; s < splits; s++) {
            L += sLs[s] * exp2f(sMs[s] - M);
        }
        float o = 0.0f;
        for (int s = 0; s < splits; s++) {
            o += base[(size_t)s * headDimPlus + d] * exp2f(sMs[s] - M);
        }
        // 到这里 o 仍在 u 域(未做逆 WHT)。除以 L 是标量, 与线性变换可交换。
        x[d] = (L > 0.0f) ? (o / L) : 0.0f;
        __syncthreads();

        // 【上游BUMP勿回退】全 kernel 唯一一次逆 WHT。headDim=256 = 两个独立的 128 块,
        // 由 threadIdx 的两半分别处理; TurboInverseWht128 内部的 __syncthreads()
        // 被全部 256 个线程统一到达, 两半严格同步推进。
        TurboInverseWht128(x + (d >> 7) * 128, d & 127);

        od[(size_t)t * outTokenStride + (size_t)h * outHeadStride + d] = __float2half_rn(x[d]);
        __syncthreads();
    }
}


template <bool TURBO4>
__global__ void __launch_bounds__(kTurboThreads, 4)
FastllmSm70PagedTurboPrefillSplitKernel(
    const half * __restrict__ qd,
    const uint8_t * __restrict__ pagedK,
    const uint8_t * __restrict__ pagedV,
    float * __restrict__ scratch,
    const int32_t * __restrict__ pageIndices,
    int numPages, int lastPageLen, int pageLenShift, int pageLenMask,
    int numKvHeads, int qoLen, int group,
    int qStrideH, int qStrideN, float scale, int splits) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int split = blockIdx.z;
    const int kvHead = h / group;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int pageLen = pageLenMask + 1;
    const int kvLen = (numPages - 1) * pageLen + lastPageLen;
    const int visible = kvLen - qoLen + t + 1;
    const int chunkSize = (kvLen + splits - 1) / splits;
    const int kvStart = split * chunkSize;
    const int kvEnd = min(min(kvStart + chunkSize, kvLen), visible);

    __shared__ float sQ[kTurboHeadDim];
    __shared__ float sLut[16];
    __shared__ float sAcc[kTurboWarps * kTurboHeadDim];
    __shared__ float sM[kTurboWarps];
    __shared__ float sL[kTurboWarps];
    for (int d = tid; d < kTurboHeadDim; d += kTurboThreads) {
        sQ[d] = __half2float(
            qd[(size_t)h * qStrideH + (size_t)t * qStrideN + d]) *
            scale * 1.4426950408889634f;
    }
    for (int i = tid; i < (TURBO4 ? 16 : 8); i += kTurboThreads) {
        sLut[i] = TURBO4 ? kTurbo4XqaCentroids[i] : kTurboXqaCentroids[i];
    }
    __syncthreads();

    float m = -1e30f, l = 0.0f;
    float acc[kTurboDimsLane] = {0.0f};
    const int d0 = lane * kTurboDimsLane;
    const int q8Block = lane >> 2;
    const int q8Off = (lane & 3) * 8;
    const int valueBlock = lane >> 4;
    const int valueGroup = lane & 15;
    for (int j = kvStart + warp; j < kvEnd; j += kTurboWarps) {
        const int pageSlot = j >> pageLenShift;
        const int offInPage = j & pageLenMask;
        const int page = pageIndices[pageSlot];
        const size_t row =
            ((size_t)page * pageLen + offInPage) * numKvHeads + kvHead;
        const uint8_t *kBlk = pagedK + row * kTurboQ8Row +
            (size_t)q8Block * kQ8BlockBytes;
        const float kScale =
            __half2float(__ldg(reinterpret_cast<const half *>(kBlk)));
        const int8_t *kQs =
            reinterpret_cast<const int8_t *>(kBlk + 2 + q8Off);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < kTurboDimsLane; i++) {
            dot += sQ[d0 + i] * (kScale * (float)kQs[i]);
        }
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }

        float vreg[kTurboDimsLane];
        if constexpr (TURBO4) {
            const uint8_t *vBlk = pagedV + row * kTurboT4Row +
                (size_t)valueBlock * kTurbo4BlockBytes;
            const float norm =
                __half2float(__ldg(reinterpret_cast<const half *>(vBlk)));
            const uint8_t *packedBytes =
                vBlk + 2 + 4 * valueGroup;
            const uint32_t packed =
                (uint32_t)__ldg(packedBytes) |
                ((uint32_t)__ldg(packedBytes + 1) << 8) |
                ((uint32_t)__ldg(packedBytes + 2) << 16) |
                ((uint32_t)__ldg(packedBytes + 3) << 24);
            #pragma unroll
            for (int i = 0; i < kTurboDimsLane; i++) {
                vreg[i] = norm * sLut[(packed >> (4 * i)) & 15u];
            }
        } else {
            const uint8_t *vBlk = pagedV + row * kTurboT3Row +
                (size_t)valueBlock * kTurbo3BlockBytes;
            const float norm =
                __half2float(__ldg(reinterpret_cast<const half *>(vBlk)));
            const uint32_t low2 = __ldg(
                reinterpret_cast<const uint16_t *>(
                    vBlk + 2 + 2 * valueGroup));
            const uint32_t high1 = __ldg(vBlk + 2 + 32 + valueGroup);
            #pragma unroll
            for (int i = 0; i < kTurboDimsLane; i++) {
                const int idx = (int)(((low2 >> (2 * i)) & 3u) |
                    (((high1 >> i) & 1u) << 2));
                vreg[i] = norm * sLut[idx];
            }
        }
        if (dot > m) {
            const float corr = exp2f(m - dot);
            #pragma unroll
            for (int i = 0; i < kTurboDimsLane; i++) acc[i] *= corr;
            l *= corr;
            m = dot;
        }
        const float p = exp2f(dot - m);
        #pragma unroll
        for (int i = 0; i < kTurboDimsLane; i++) acc[i] += p * vreg[i];
        l += p;
    }
    if (lane == 0) {
        sM[warp] = m;
        sL[warp] = l;
    }
    #pragma unroll
    for (int i = 0; i < kTurboDimsLane; i++) {
        sAcc[warp * kTurboHeadDim + d0 + i] = acc[i];
    }
    __syncthreads();
    float maxScore = -1e30f;
    for (int w = 0; w < kTurboWarps; w++) maxScore = fmaxf(maxScore, sM[w]);
    float sum = 0.0f;
    for (int w = 0; w < kTurboWarps; w++) {
        sum += sL[w] * exp2f(sM[w] - maxScore);
    }
    float *slot = scratch +
        ((size_t)(h * qoLen + t) * splits + split) *
        (kTurboHeadDim + 2);
    for (int d = tid; d < kTurboHeadDim; d += kTurboThreads) {
        float value = 0.0f;
        for (int w = 0; w < kTurboWarps; w++) {
            value += sAcc[w * kTurboHeadDim + d] *
                exp2f(sM[w] - maxScore);
        }
        slot[d] = value;
    }
    if (tid == 0) {
        slot[kTurboHeadDim] = maxScore;
        slot[kTurboHeadDim + 1] = sum;
    }
}

__global__ void FastllmSm70PagedTurboPrefillCombineKernel(
    const float * __restrict__ scratch, half * __restrict__ od,
    int qoLen, int group, int splits,
    int outHeadStride, int outTokenStride) {
    const int kvh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    __shared__ float sMs[kTurboMaxSplits], sLs[kTurboMaxSplits];
    __shared__ float x[kTurboHeadDim];
    for (int g = 0; g < group; g++) {
        const int h = kvh * group + g;
        const float *base = scratch +
            (size_t)(h * qoLen + t) * splits * (kTurboHeadDim + 2);
        for (int s = d; s < splits; s += kTurboHeadDim) {
            sMs[s] = base[(size_t)s * (kTurboHeadDim + 2) + kTurboHeadDim];
            sLs[s] = base[(size_t)s * (kTurboHeadDim + 2) + kTurboHeadDim + 1];
        }
        __syncthreads();
        float M = -1e30f;
        for (int s = 0; s < splits; s++) M = fmaxf(M, sMs[s]);
        float L = 0.0f, value = 0.0f;
        for (int s = 0; s < splits; s++) {
            const float factor = exp2f(sMs[s] - M);
            L += sLs[s] * factor;
            value += base[(size_t)s * (kTurboHeadDim + 2) + d] * factor;
        }
        x[d] = L > 0.0f ? value / L : 0.0f;
        __syncthreads();
        TurboInverseWht128(x + (d >> 7) * 128, d & 127);
        od[(size_t)t * outTokenStride + (size_t)h * outHeadStride + d] =
            __float2half_rn(x[d]);
        __syncthreads();
    }
}
int TurboXqaSplits(int numKvHeads, int groupChunks) {
    const int parallel = numKvHeads * groupChunks;
    if (parallel <= 0) {
        return 1;
    }
    int s = (kTurboSplitTargetBlocks + parallel - 1) / parallel;
    return std::max(1, std::min(s, kTurboMaxSplits));
}

// 进程级持久 scratch。按最坏情况一次分配, 之后地址固定 —— 与 native 侧同样的理由:
// 已捕获的 CUDA graph 不能持有会被 realloc 掉的指针。
float *TurboXqaScratch(int device, int H, size_t &capacitySlots) {
    struct Entry { float *ptr = nullptr; size_t slots = 0; };
    static thread_local std::map<int64_t, Entry> cache;
    Entry &entry = cache[((int64_t)device << 20) | (int64_t)H];
    if (entry.ptr == nullptr) {
        // 槽位 = H * QO_LEN * splits, 对 QO_LEN=1..4 取最大。
        size_t worst = 0;
        for (int qo = 1; qo <= kTurboMaxQoLen; qo++) {
            const int groupChunk = (qo <= 3) ? (kTurboGroup / qo) : 1;
            const int groupChunks = kTurboGroup / groupChunk;
            const int splits = TurboXqaSplits(4, groupChunks);
            worst = std::max(worst, (size_t)H * qo * splits);
            const int splits32 = TurboXqaSplits(32, groupChunks);
            worst = std::max(worst, (size_t)H * qo * splits32);
        }
        entry.ptr = (float *)FastllmCudaMalloc(worst * (size_t)(kTurboHeadDim + 2) * sizeof(float));
        entry.slots = entry.ptr != nullptr ? worst : 0;
    }
    capacitySlots = entry.slots;
    return entry.ptr;
}

bool TurboXqaEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_SM70_TURBO_XQA");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 &&
           strcmp(value, "off") != 0 && strcmp(value, "OFF") != 0;
}

template <int QO_LEN, int GROUP_CHUNK>
void LaunchTurboXqa(const half *qd, const uint8_t *pagedK, const uint8_t *pagedV,
                    float *scratch, const int32_t *pageIndices,
                    int numPages, int lastPageLen, int pageLenShift, int pageLenMask,
                    int numKvHeads, int groupChunks, int qStrideH, int qStrideN,
                    float scale, int splits, half *od,
                    int outHeadStride, int outTokenStride) {
    dim3 grid1((unsigned)(numKvHeads * groupChunks), (unsigned)splits, 1);
    FastllmSm70PagedTurboXqaSplitKernel<QO_LEN, GROUP_CHUNK>
        <<<grid1, kTurboThreads, 0, cudaStreamPerThread>>>(
            qd, pagedK, pagedV, scratch, pageIndices, numPages, lastPageLen,
            pageLenShift, pageLenMask, numKvHeads, groupChunks,
            qStrideH, qStrideN, scale, splits);
    dim3 grid2((unsigned)numKvHeads, (unsigned)QO_LEN, 1);
    FastllmSm70PagedTurboXqaCombineKernel<QO_LEN>
        <<<grid2, kTurboHeadDim, 0, cudaStreamPerThread>>>(
            scratch, od, GROUP_CHUNK * groupChunks, splits,
            outHeadStride, outTokenStride);
}


float *TurboPrefillScratch(int device, size_t needSlots, size_t &capacity) {
    struct Entry { float *ptr = nullptr; size_t slots = 0; };
    static thread_local std::map<int, Entry> cache;
    Entry &entry = cache[device];
    if (entry.slots < needSlots) {
        float *next = (float *)FastllmCudaMalloc(
            needSlots * (size_t)(kTurboHeadDim + 2) * sizeof(float));
        if (next == nullptr) {
            capacity = entry.slots;
            return entry.ptr;
        }
        if (entry.ptr != nullptr) {
            printf("[Fastllm][turbo-prefill] scratch grow: keep old %p, "
                   "slots=%zu -> %zu\n", entry.ptr, entry.slots, needSlots);
        }
        entry.ptr = next;
        entry.slots = needSlots;
    }
    capacity = entry.slots;
    return entry.ptr;
}

bool TurboPrefillEnabled() {
    const char *value =
        std::getenv("FASTLLM_CUDA_SM70_TURBO_PREFILL");
    return value == nullptr || (strcmp(value, "0") != 0 &&
        strcmp(value, "false") != 0 && strcmp(value, "off") != 0);
}

int TurboPrefillMaxQoLen() {
    const char *value =
        std::getenv("FASTLLM_CUDA_SM70_TURBO_PREFILL_MAX_Q");
    if (value == nullptr || value[0] == '\0') return 32;
    return std::max(5, std::min(1024, std::atoi(value)));
}
}  // namespace

bool FastllmCudaTrySm70PagedTurboXqa(
    void *qData, fastllm::DataType qType,
    int H, int qoLen, int qDim, int qHeadStride, int qTokenStride,
    const int32_t *pageIndicesGpu, int numPages, int lastPageLen,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int pageLen, int numKvHeads, int headDim,
    void *outData, fastllm::DataType outType, int outHeadStride, int outTokenStride,
    int group, float scale) {
    // ---- 资格判定。任何一条不满足直接返回 false, 不写 outData。 ----
    if (!TurboXqaEnabled()) {
        return false;
    }
    if (qData == nullptr || outData == nullptr || pageIndicesGpu == nullptr ||
        pagedKVCacheK == nullptr || pagedKVCacheV == nullptr) {
        return false;
    }
    if (qType != fastllm::DataType::FLOAT16 || outType != fastllm::DataType::FLOAT16) {
        return false;
    }
    if (pagedKVCacheK->dataType != fastllm::DataType::Q8_0_KV ||
        pagedKVCacheV->dataType != fastllm::DataType::TURBO3_KV) {
        return false;
    }
    if (pagedKVCacheK->cudaData == nullptr || pagedKVCacheV->cudaData == nullptr) {
        return false;
    }
    if (headDim != kTurboHeadDim || qDim != kTurboHeadDim || group != kTurboGroup ||
        numKvHeads <= 0 || H != group * numKvHeads) {
        return false;
    }
    if (qoLen < 1 || qoLen > kTurboMaxQoLen || numPages <= 0 || lastPageLen <= 0) {
        return false;
    }
    // pageLen 必须是 2 的幂: 热循环里用移位/掩码定位页, 避免每个 key 一次整数除法。
    if (pageLen <= 0 || (pageLen & (pageLen - 1)) != 0 || lastPageLen > pageLen) {
        return false;
    }
    const int kvLen = (numPages - 1) * pageLen + lastPageLen;
    if (kvLen < qoLen) {
        return false;
    }
    if (qHeadStride <= 0 || qTokenStride < headDim ||
        outHeadStride < headDim || outTokenStride < headDim) {
        return false;
    }
    int device = -1, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
        major != 7 || minor != 0) {
        cudaGetLastError();
        return false;
    }

    const int groupChunk = (qoLen <= 3) ? (kTurboGroup / qoLen) : 1;
    const int groupChunks = kTurboGroup / groupChunk;
    const int splits = TurboXqaSplits(numKvHeads, groupChunks);

    size_t capacitySlots = 0;
    float *scratch = TurboXqaScratch(device, H, capacitySlots);
    if (scratch == nullptr || capacitySlots < (size_t)H * qoLen * splits) {
        return false;
    }

    int pageLenShift = 0;
    while ((1 << pageLenShift) < pageLen) {
        pageLenShift++;
    }

    const half *qd = (const half *)qData;
    half *od = (half *)outData;
    const uint8_t *pagedK = (const uint8_t *)pagedKVCacheK->cudaData;
    const uint8_t *pagedV = (const uint8_t *)pagedKVCacheV->cudaData;

    switch (qoLen) {
        case 1:
            LaunchTurboXqa<1, 6>(qd, pagedK, pagedV, scratch, pageIndicesGpu, numPages,
                                 lastPageLen, pageLenShift, pageLen - 1, numKvHeads,
                                 groupChunks, qHeadStride, qTokenStride, scale, splits,
                                 od, outHeadStride, outTokenStride);
            break;
        case 2:
            LaunchTurboXqa<2, 3>(qd, pagedK, pagedV, scratch, pageIndicesGpu, numPages,
                                 lastPageLen, pageLenShift, pageLen - 1, numKvHeads,
                                 groupChunks, qHeadStride, qTokenStride, scale, splits,
                                 od, outHeadStride, outTokenStride);
            break;
        case 3:
            LaunchTurboXqa<3, 2>(qd, pagedK, pagedV, scratch, pageIndicesGpu, numPages,
                                 lastPageLen, pageLenShift, pageLen - 1, numKvHeads,
                                 groupChunks, qHeadStride, qTokenStride, scale, splits,
                                 od, outHeadStride, outTokenStride);
            break;
        case 4:
            LaunchTurboXqa<4, 1>(qd, pagedK, pagedV, scratch, pageIndicesGpu, numPages,
                                 lastPageLen, pageLenShift, pageLen - 1, numKvHeads,
                                 groupChunks, qHeadStride, qTokenStride, scale, splits,
                                 od, outHeadStride, outTokenStride);
            break;
        default:
            return false;
    }
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    return true;
}

bool FastllmCudaTrySm70PagedTurboPrefill(
    void *qData, fastllm::DataType qType,
    int H, int qoLen, int qDim, int qHeadStride, int qTokenStride,
    const int32_t *pageIndicesGpu, int numPages, int lastPageLen,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int pageLen, int numKvHeads, int headDim,
    void *outData, fastllm::DataType outType,
    int outHeadStride, int outTokenStride, int group, float scale) {
    if (!TurboPrefillEnabled() || qoLen <= kTurboMaxQoLen ||
        qoLen > TurboPrefillMaxQoLen() ||
        qData == nullptr || outData == nullptr || pageIndicesGpu == nullptr ||
        pagedKVCacheK == nullptr || pagedKVCacheV == nullptr ||
        qType != fastllm::DataType::FLOAT16 ||
        outType != fastllm::DataType::FLOAT16 ||
        pagedKVCacheK->dataType != fastllm::DataType::Q8_0_KV ||
        (pagedKVCacheV->dataType != fastllm::DataType::TURBO3_KV &&
         pagedKVCacheV->dataType != fastllm::DataType::TURBO4_KV) ||
        pagedKVCacheK->cudaData == nullptr ||
        pagedKVCacheV->cudaData == nullptr ||
        headDim != kTurboHeadDim || qDim != kTurboHeadDim ||
        group != kTurboGroup || numKvHeads <= 0 ||
        H != group * numKvHeads || numPages <= 0 ||
        lastPageLen <= 0 || pageLen <= 0 ||
        (pageLen & (pageLen - 1)) != 0 || lastPageLen > pageLen ||
        qHeadStride <= 0 || qTokenStride < headDim ||
        outHeadStride < headDim || outTokenStride < headDim) {
        return false;
    }
    const int kvLen = (numPages - 1) * pageLen + lastPageLen;
    if (kvLen < qoLen) return false;
    int device = -1, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess ||
        major != 7 || minor != 0) {
        cudaGetLastError();
        return false;
    }
    const int parallel = H * qoLen;
    const int splits = std::max(
        1, std::min(8, (kTurboSplitTargetBlocks + parallel - 1) / parallel));
    const size_t needSlots = (size_t)H * qoLen * splits;
    size_t capacity = 0;
    float *scratch = TurboPrefillScratch(device, needSlots, capacity);
    if (scratch == nullptr || capacity < needSlots) return false;
    int pageLenShift = 0;
    while ((1 << pageLenShift) < pageLen) pageLenShift++;
    dim3 splitGrid((unsigned)H, (unsigned)qoLen, (unsigned)splits);
    const half *qd = (const half *)qData;
    const uint8_t *pagedK = (const uint8_t *)pagedKVCacheK->cudaData;
    const uint8_t *pagedV = (const uint8_t *)pagedKVCacheV->cudaData;
    if (pagedKVCacheV->dataType == fastllm::DataType::TURBO4_KV) {
        FastllmSm70PagedTurboPrefillSplitKernel<true>
            <<<splitGrid, kTurboThreads, 0, cudaStreamPerThread>>>(
                qd, pagedK, pagedV, scratch, pageIndicesGpu,
                numPages, lastPageLen, pageLenShift, pageLen - 1,
                numKvHeads, qoLen, group, qHeadStride, qTokenStride,
                scale, splits);
    } else {
        FastllmSm70PagedTurboPrefillSplitKernel<false>
            <<<splitGrid, kTurboThreads, 0, cudaStreamPerThread>>>(
                qd, pagedK, pagedV, scratch, pageIndicesGpu,
                numPages, lastPageLen, pageLenShift, pageLen - 1,
                numKvHeads, qoLen, group, qHeadStride, qTokenStride,
                scale, splits);
    }
    dim3 combineGrid((unsigned)numKvHeads, (unsigned)qoLen, 1);
    FastllmSm70PagedTurboPrefillCombineKernel
        <<<combineGrid, kTurboHeadDim, 0, cudaStreamPerThread>>>(
            scratch, (half *)outData, qoLen, group, splits,
            outHeadStride, outTokenStride);
    return cudaGetLastError() == cudaSuccess;
}

#else   // USE_ROCM

bool FastllmCudaTrySm70PagedTurboXqa(
    void *, fastllm::DataType, int, int, int, int, int,
    const int32_t *, int, int, fastllm::Data *, fastllm::Data *,
    int, int, int, void *, fastllm::DataType, int, int, int, float) {
    return false;
}

bool FastllmCudaTrySm70PagedTurboPrefill(
    void *, fastllm::DataType, int, int, int, int, int,
    const int32_t *, int, int, fastllm::Data *, fastllm::Data *,
    int, int, int, void *, fastllm::DataType, int, int, int, float) {
    return false;
}

#endif
