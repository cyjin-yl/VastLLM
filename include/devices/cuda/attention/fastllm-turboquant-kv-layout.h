//
// TurboQuant 打包 KV 缓存的**布局单一真相源**(host / device 通用, 不含 CUDA 语法)。
//
// 来源算法: llama.cpp-turboquant (MIT), GGML_TYPE_TURBO3_0 / GGML_TYPE_Q8_0。
// 本文件只描述 Qwen3.5/3.6/3.8 在 SM70 上实际使用的闭包:
//   K: q8_0   —— 每 32 个值一块 = fp16 scale + 32 个 int8            (headDim=256 -> 272 B/行)
//   V: turbo3 —— 每 128 个值一块 = fp16 corrected norm + 3-bit 索引   (headDim=256 -> 100 B/行)
// 「行」= 一个 (page, token, head) 三元组, 列 = headDim。
//
// 【上游BUMP勿回退】为什么要有这个头文件, 而不是各写各的:
//
//   这些常量(块大小 / 块字节数 / 码本 / 随机化 Hadamard 的两张符号表)原先只存在于
//   src/devices/cuda/attention/paged/fastllm-turboquant-kv.cu 的匿名 namespace 里。
//   一旦有第二处代码需要**直接读打包字节**(融合注意力 kernel、host 端 fp64 对拍参考),
//   就只能复制一份。复制出来的副本和原件会**静默走岔**:
//     - 码本改一个小数位, 两边不一致 -> 反量化结果偏一点点, 不报错, 只是模型变笨;
//     - 符号表抄错一个 ±1 -> 逆 WHT 出来的那一维符号翻转, 同样不报错。
//   本仓已经吃过一次「测试和实现悄悄走岔」的亏(见 include/fastllm-kernel-route.h 里
//   把 Sm70Iq4XsMmqShapeRejectReason 抽成纯函数的那段说明)。
//
//   所以: 码本与符号表用**宏初始化列表**给出, device 侧的 __constant__ 数组和 host 侧的
//   constexpr 数组都从同一个宏展开 —— 想改就必须改这一处, 物理上无法只改一边。
//
#pragma once

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// 分块几何
// ---------------------------------------------------------------------------
#define FASTLLM_TURBOKV_Q8_BLOCK_VALUES     32
#define FASTLLM_TURBOKV_Q8_BLOCK_BYTES      34
#define FASTLLM_TURBOKV_TURBO3_BLOCK_VALUES 128
#define FASTLLM_TURBOKV_TURBO3_BLOCK_BYTES  50
#define FASTLLM_TURBOKV_TURBO4_BLOCK_VALUES 128
#define FASTLLM_TURBOKV_TURBO4_BLOCK_BYTES  66

namespace fastllm {
namespace turbokv {

constexpr int kQ8BlockValues     = FASTLLM_TURBOKV_Q8_BLOCK_VALUES;
constexpr int kQ8BlockBytes      = FASTLLM_TURBOKV_Q8_BLOCK_BYTES;
constexpr int kTurbo3BlockValues = FASTLLM_TURBOKV_TURBO3_BLOCK_VALUES;
constexpr int kTurbo3BlockBytes  = FASTLLM_TURBOKV_TURBO3_BLOCK_BYTES;
constexpr int kTurbo4BlockValues = FASTLLM_TURBOKV_TURBO4_BLOCK_VALUES;
constexpr int kTurbo4BlockBytes  = FASTLLM_TURBOKV_TURBO4_BLOCK_BYTES;

// 逆 WHT 的归一化因子 1/sqrt(128)。与 fastllm-turboquant-kv.cu 中字面量一致。
constexpr double kWhtScale = 0.08838834764831845;

constexpr size_t Q8RowBytes(size_t columns) {
    return ((columns + kQ8BlockValues - 1) / kQ8BlockValues) * (size_t)kQ8BlockBytes;
}
constexpr size_t Turbo3RowBytes(size_t columns) {
    return ((columns + kTurbo3BlockValues - 1) / kTurbo3BlockValues) * (size_t)kTurbo3BlockBytes;
}
constexpr size_t Turbo4RowBytes(size_t columns) {
    return ((columns + kTurbo4BlockValues - 1) / kTurbo4BlockValues) * (size_t)kTurbo4BlockBytes;
}

// 与 fastllm.cpp 的 GetKVCacheRowBytes 必须一致; headDim=256 是生产形状。
static_assert(Q8RowBytes(256) == 272,     "q8_0 KV 每行必须是 272 B (headDim=256)");
static_assert(Turbo3RowBytes(256) == 100, "turbo3 KV 每行必须是 100 B (headDim=256)");
static_assert(Turbo4RowBytes(256) == 132, "turbo4 KV 每行必须是 132 B (headDim=256)");

// ---------------------------------------------------------------------------
// 块内存布局(POD, host/device 通用)
// ---------------------------------------------------------------------------
struct Q8KvBlock {
    uint16_t scale;                     // fp16 位模式
    int8_t   values[kQ8BlockValues];
};
static_assert(sizeof(Q8KvBlock) == kQ8BlockBytes, "Q8 KV 块布局变了");

struct Turbo3KvBlock {
    uint16_t norm;                                  // fp16 位模式, corrected norm
    uint8_t  low2[kTurbo3BlockValues / 4];          // 每字节 4 个值的低 2 bit
    uint8_t  high1[kTurbo3BlockValues / 8];         // 每字节 8 个值的第 3 bit
};
static_assert(sizeof(Turbo3KvBlock) == kTurbo3BlockBytes, "Turbo3 KV 必须是 50 B / 128 值");

struct Turbo4KvBlock {
    uint16_t norm;
    uint8_t  qs[kTurbo4BlockValues / 2];
};
static_assert(sizeof(Turbo4KvBlock) == kTurbo4BlockBytes, "Turbo4 KV 必须是 66 B / 128 值");

} // namespace turbokv
} // namespace fastllm

// ---------------------------------------------------------------------------
// 码本与随机化 Hadamard 符号表 —— 宏初始化列表(device __constant__ 与 host constexpr 共用)
// ---------------------------------------------------------------------------
#define FASTLLM_TURBOKV_TURBO3_CENTROIDS_INIT { \
    -0.190207f, -0.118786f, -0.066822f, -0.021663f, \
     0.021663f,  0.066822f,  0.118786f,  0.190207f }

#define FASTLLM_TURBOKV_TURBO3_MIDPOINTS_INIT { \
    -0.154496f, -0.092804f, -0.044243f, 0.0f, \
     0.044243f,  0.092804f,  0.154496f }

#define FASTLLM_TURBOKV_TURBO4_CENTROIDS_INIT { \
    -0.241529f, -0.182877f, -0.143016f, -0.111036f, \
    -0.083292f, -0.058050f, -0.034299f, -0.011349f, \
     0.011349f,  0.034299f,  0.058050f,  0.083292f, \
     0.111036f,  0.143016f,  0.182877f,  0.241529f }

#define FASTLLM_TURBOKV_TURBO4_MIDPOINTS_INIT { \
    -0.212203f, -0.162947f, -0.127026f, -0.097164f, \
    -0.070671f, -0.046174f, -0.022824f,  0.000000f, \
     0.022824f,  0.046174f,  0.070671f,  0.097164f, \
     0.127026f,  0.162947f,  0.212203f }

#define FASTLLM_TURBOKV_WHT_SIGNS1_INIT { \
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1, \
    1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1, \
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1, \
    1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1, \
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1, \
    1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1, \
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1, \
    1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1 }

#define FASTLLM_TURBOKV_WHT_SIGNS2_INIT { \
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1, \
    1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1, \
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1, \
    1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1, \
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1, \
    -1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1, \
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1, \
    -1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1 }

namespace fastllm {
namespace turbokv {

// host 侧副本, 供 fp64 对拍参考使用。与 device __constant__ 从同一个宏展开。
constexpr float kTurbo3CentroidsHost[8]  = FASTLLM_TURBOKV_TURBO3_CENTROIDS_INIT;
constexpr float kTurbo4CentroidsHost[16] = FASTLLM_TURBOKV_TURBO4_CENTROIDS_INIT;
constexpr float kWhtSigns1Host[128]      = FASTLLM_TURBOKV_WHT_SIGNS1_INIT;
constexpr float kWhtSigns2Host[128]      = FASTLLM_TURBOKV_WHT_SIGNS2_INIT;

} // namespace turbokv
} // namespace fastllm
