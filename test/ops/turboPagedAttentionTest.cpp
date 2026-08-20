//
// 打包分页 KV (K=q8_0, V=turbo3) 注意力的**数值对拍**。
//
// 这个测试要回答三个问题, 每个都是踩过坑之后才加的:
//
// 1) `FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA=1` 与默认(关闭) 的输出是否等价?
//    这个 gate 在 fastllm-paged-attention-native.cu 里决定 chunked-cuBLAS 路径是
//    「逐 query head 循环调 group 次 cublasHgemm」还是「一次 StridedBatched」。
//    docs/qwen35_v100_local_stack.md 把它列进「不启用负结果」, 但那一轮度量的是
//    **prefill** 形状(qoLen 数百); decode / MTP 校验形状(qoLen=1..3)下 GEMM 退化成
//    GEMV, 两条路的访存量差 6 倍, 是完全不同的 regime。要重新 A/B 就必须先证明
//    两条路**算的是同一个东西**, 否则性能对比毫无意义。
//
// 2) 打包 KV 的注意力到底离精确解有多远? 现路径把反量化结果落成 fp16、把 score 也
//    落成 fp16(qk 缓冲是 half), 误差来源不止量化本身。要评价任何新 kernel, 必须先有
//    一个**与实现无关的 fp64 参考**: 直接从打包字节按格式定义在 double 下反量化,
//    再在 double 下做注意力。
//
// 3) (后续)融合 kernel 的输出是否比现路径更接近 fp64 参考。
//    「差异在 1e-3 以内」不构成结论 —— 现路径也可能在 1e-3 以内但偏向另一侧。
//
// 参考值的定义: 打包缓存里存的就是「反量化后的 K/V」, 其数值由格式唯一确定
// (q8_0: scale*int8; turbo3: InverseWht128(centroid[idx]*correctedNorm))。
// 本测试在 host 上用 double 复现这两个反量化, 得到的 K/V 即为该 KV 缓存的精确内容;
// 在此之上做 double 注意力, 就是「给定这份量化 KV, 正确答案是什么」。
//
#include "fastllm.h"
#include "model.h"

#if defined(USE_CUDA) && !defined(USE_ROCM)
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/attention/fastllm-paged-attention-native.cuh"
#include "attention/fastllm-turboquant-kv-layout.h"
#include "attention/fastllm-paged-attention-turbo-xqa.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <functional>
#include <vector>

namespace {

int gFailures = 0;
int gChecks = 0;

void Expect(bool condition, const std::string &message) {
    gChecks++;
    if (!condition) {
        gFailures++;
        std::cout << "  [FAIL] " << message << "\n";
    }
}

#if defined(USE_CUDA) && !defined(USE_ROCM)

int SetTestEnv(const std::string &name, const char *value) {
    return value == nullptr ? unsetenv(name.c_str()) : setenv(name.c_str(), value, 1);
}

// 作用域内临时改环境变量, 离开时精确还原(包括「原本就没设」这种情况)。
class ScopedEnvVar {
public:
    ScopedEnvVar(const std::string &name, const char *value) : name(name) {
        const char *current = std::getenv(name.c_str());
        hadValue = current != nullptr;
        if (hadValue) {
            oldValue = current;
        }
        SetTestEnv(name, value);
    }
    ~ScopedEnvVar() { SetTestEnv(name, hadValue ? oldValue.c_str() : nullptr); }
    ScopedEnvVar(const ScopedEnvVar &) = delete;
    ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;
private:
    std::string name;
    std::string oldValue;
    bool hadValue = false;
};

fastllm::Data MakeCudaTensor(fastllm::DataType dataType, const std::vector<int> &dims,
                             const std::vector<float> &values) {
    fastllm::Data data(dataType, dims, values);
    data.ToDevice(fastllm::DataDevice::CUDA);
    return data;
}

fastllm::Data MakeIntTensor(const std::vector<int> &dims, const std::vector<int32_t> &values) {
    int count = 1;
    for (int dim : dims) {
        count *= dim;
    }
    fastllm::Data data(fastllm::DataType::INT32, dims);
    data.Allocate();
    data.cpuIntDatas.assign(values.begin(), values.end());
    if (count > 0) {
        std::memcpy(data.cpuData, values.data(), (size_t)count * sizeof(int32_t));
    }
    data.ToDevice(fastllm::DataDevice::CUDA);
    return data;
}

std::vector<float> ToFloatVector(fastllm::Data data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    if (data.dataType != fastllm::DataType::FLOAT32) {
        fastllm::ToDataTypeForceCPU(data, fastllm::DataType::FLOAT32);
    }
    int count = (int)data.Count(0);
    std::vector<float> values(count);
    if (count > 0) {
        std::memcpy(values.data(), data.cpuData, (size_t)count * sizeof(float));
    }
    return values;
}

// 伪随机但完全确定的测试数据(不用 rand(), 保证跨机器可复现)。
std::vector<float> MakeValues(size_t count, double seed, double scale) {
    std::vector<float> out(count);
    double x = seed;
    for (size_t i = 0; i < count; i++) {
        x = std::fmod(x * 1103515245.0 + 12345.0, 2147483648.0);
        out[i] = (float)((x / 2147483648.0 - 0.5) * 2.0 * scale);
    }
    return out;
}

// --------------------------------------------------------------------------
// host 端**精确**反量化(double)。逐行对照 fastllm-turboquant-kv.cu 的
// GatherQ8HeadRangeKernel / GatherTurbo3HeadRangeKernel + InverseWht128。
// --------------------------------------------------------------------------
double HalfBitsToDouble(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return (double)__half2float(h);
}

void InverseWht128Host(double *x) {
    using namespace fastllm::turbokv;
    for (int i = 0; i < 128; i++) {
        x[i] *= (double)kWhtSigns2Host[i];
    }
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i++) {
            if ((i % (2 * h)) < h) {
                double a = x[i], b = x[i + h];
                x[i] = a + b;
                x[i + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; i++) {
        x[i] *= kWhtScale * (double)kWhtSigns1Host[i];
    }
}

void DequantQ8RowHost(const uint8_t *row, int headDim, double *out) {
    using namespace fastllm::turbokv;
    const Q8KvBlock *blocks = reinterpret_cast<const Q8KvBlock *>(row);
    for (int d = 0; d < headDim; d++) {
        const Q8KvBlock &b = blocks[d / kQ8BlockValues];
        out[d] = HalfBitsToDouble(b.scale) * (double)b.values[d % kQ8BlockValues];
    }
}

void DequantTurbo3RowHost(const uint8_t *row, int headDim, double *out) {
    using namespace fastllm::turbokv;
    const Turbo3KvBlock *blocks = reinterpret_cast<const Turbo3KvBlock *>(row);
    for (int g = 0; g < headDim / kTurbo3BlockValues; g++) {
        const Turbo3KvBlock &b = blocks[g];
        double norm = HalfBitsToDouble(b.norm);
        double x[128];
        for (int i = 0; i < 128; i++) {
            int idx = (b.low2[i / 4] >> (2 * (i & 3))) & 3;
            idx |= ((b.high1[i / 8] >> (i & 7)) & 1) << 2;
            x[i] = (double)kTurbo3CentroidsHost[idx] * norm;
        }
        InverseWht128Host(x);
        for (int i = 0; i < 128; i++) {
            out[g * 128 + i] = x[i];
        }
    }
}

struct ErrorStats {
    double maxAbs = 0.0;
    double rms = 0.0;
    double refMaxAbs = 0.0;
    double MaxRel() const { return refMaxAbs > 0 ? maxAbs / refMaxAbs : 0.0; }
};

ErrorStats CompareToReference(const std::vector<double> &reference,
                              const std::vector<float> &actual) {
    ErrorStats stats;
    double sumSq = 0.0;
    size_t n = std::min(reference.size(), actual.size());
    for (size_t i = 0; i < n; i++) {
        double diff = std::fabs((double)actual[i] - reference[i]);
        stats.maxAbs = std::max(stats.maxAbs, diff);
        stats.refMaxAbs = std::max(stats.refMaxAbs, std::fabs(reference[i]));
        sumSq += diff * diff;
    }
    stats.rms = n > 0 ? std::sqrt(sumSq / (double)n) : 0.0;
    return stats;
}

// --------------------------------------------------------------------------
// 测试夹具: 建立一份打包分页 KV, 抓出其精确内容, 给出 fp64 参考输出。
// --------------------------------------------------------------------------
struct Fixture {
    int numKvHeads = 4;
    int group = 6;
    int headDim = 256;
    int pageLen = 128;
    int H = 24;
    int kvLen = 0;
    float scale = 0.0625f;   // 1/sqrt(256), fp16 可精确表示

    std::vector<int32_t> physicalPages;
    fastllm::PagedCacheManager *pagedK = nullptr;
    fastllm::PagedCacheManager *pagedV = nullptr;
    fastllm::Data kCaches{fastllm::DataType::FLOAT16};
    fastllm::Data vCaches{fastllm::DataType::FLOAT16};

    // 精确反量化后的 KV: [numKvHeads][kvLen][headDim]
    std::vector<double> exactK, exactV;
};

void BuildPackedFixture(Fixture &f, int layerIdBase,
                        const std::vector<int32_t> &physicalPages, int lastPageLen) {
    using namespace fastllm;
    f.physicalPages = physicalPages;
    int numPages = (int)physicalPages.size();
    f.kvLen = (numPages - 1) * f.pageLen + lastPageLen;

    int maxPhysical = 0;
    for (int32_t p : physicalPages) {
        maxPhysical = std::max(maxPhysical, (int)p);
    }
    int poolPages = maxPhysical + 1;

    Data kDesc(DataType::Q8_0_KV);
    kDesc.Resize({f.numKvHeads, 1, f.headDim});
    kDesc.dataDevice = DataDevice::CUDA;
    Data vDesc(DataType::TURBO3_KV);
    vDesc.Resize({f.numKvHeads, 1, f.headDim});
    vDesc.dataDevice = DataDevice::CUDA;

    f.pagedK = AllocatePagedCacheManager(
        layerIdBase, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
        kDesc, f.pageLen, poolPages);
    f.pagedV = AllocatePagedCacheManager(
        layerIdBase + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
        vDesc, f.pageLen, poolPages);

    // 源 KV: [numKvHeads, kvLen, headDim] fp16 (LaunchCopy 的 batchLayout=false 布局)
    size_t srcCount = (size_t)f.numKvHeads * f.kvLen * f.headDim;
    Data srcK = MakeCudaTensor(DataType::FLOAT16,
                               {f.numKvHeads, f.kvLen, f.headDim},
                               MakeValues(srcCount, 12345.0, 1.0));
    Data srcV = MakeCudaTensor(DataType::FLOAT16,
                               {f.numKvHeads, f.kvLen, f.headDim},
                               MakeValues(srcCount, 54321.0, 1.0));

    for (int p = 0; p < numPages; p++) {
        int len = (p == numPages - 1) ? lastPageLen : f.pageLen;
        Expect(FastllmCudaPackedKVCacheCopy(
                   (uint8_t *)f.pagedK->cudaData, physicalPages[p], f.pageLen,
                   f.numKvHeads, f.headDim, DataType::Q8_0_KV,
                   (uint8_t *)srcK.cudaData, DataType::FLOAT16,
                   f.kvLen, p * f.pageLen, len, 0),
               "q8_0 打包写入失败");
        Expect(FastllmCudaPackedKVCacheCopy(
                   (uint8_t *)f.pagedV->cudaData, physicalPages[p], f.pageLen,
                   f.numKvHeads, f.headDim, DataType::TURBO3_KV,
                   (uint8_t *)srcV.cudaData, DataType::FLOAT16,
                   f.kvLen, p * f.pageLen, len, 0),
               "turbo3 打包写入失败");
    }
    FastllmCudaSyncCurrentThreadStream();

    f.kCaches.dataType = DataType::Q8_0_KV;
    f.vCaches.dataType = DataType::TURBO3_KV;
    f.kCaches.Resize({f.numKvHeads, 1, f.headDim});
    f.vCaches.Resize({f.numKvHeads, 1, f.headDim});
    f.kCaches.isKVCache = f.vCaches.isKVCache = true;
    f.kCaches.isPagedKVCache = f.vCaches.isPagedKVCache = true;
    f.kCaches.pageLen = f.vCaches.pageLen = f.pageLen;
    f.kCaches.pagedKVCacheData = f.pagedK;
    f.vCaches.pagedKVCacheData = f.pagedV;

    // 把打包字节抓回 host, 按格式定义在 double 下精确反量化。
    size_t kBytes = ((fastllm::Data *)f.pagedK)->GetBytes();
    size_t vBytes = ((fastllm::Data *)f.pagedV)->GetBytes();
    std::vector<uint8_t> kHost(kBytes), vHost(vBytes);
    FastllmCudaCopyFromDeviceToHost(kHost.data(), f.pagedK->cudaData, kBytes);
    FastllmCudaCopyFromDeviceToHost(vHost.data(), f.pagedV->cudaData, vBytes);

    const size_t kRowBytes = fastllm::turbokv::Q8RowBytes(f.headDim);
    const size_t vRowBytes = fastllm::turbokv::Turbo3RowBytes(f.headDim);
    f.exactK.assign((size_t)f.numKvHeads * f.kvLen * f.headDim, 0.0);
    f.exactV.assign((size_t)f.numKvHeads * f.kvLen * f.headDim, 0.0);
    for (int j = 0; j < f.kvLen; j++) {
        int pageSlot = j / f.pageLen;
        int offsetInPage = j % f.pageLen;
        int page = physicalPages[pageSlot];
        for (int kvh = 0; kvh < f.numKvHeads; kvh++) {
            size_t row = ((size_t)page * f.pageLen + offsetInPage) * f.numKvHeads + kvh;
            size_t dst = ((size_t)kvh * f.kvLen + j) * f.headDim;
            DequantQ8RowHost(kHost.data() + row * kRowBytes, f.headDim, f.exactK.data() + dst);
            DequantTurbo3RowHost(vHost.data() + row * vRowBytes, f.headDim, f.exactV.data() + dst);
        }
    }
}

// fp64 参考注意力。因果: query token t 可见 key j <= kvLen - qoLen + t。
std::vector<double> ReferenceAttention(const Fixture &f, const std::vector<float> &qHost,
                                       int qoLen) {
    std::vector<double> out((size_t)qoLen * f.H * f.headDim, 0.0);
    std::vector<double> scores(f.kvLen);
    for (int t = 0; t < qoLen; t++) {
        int visibleEnd = f.kvLen - qoLen + t;    // 闭区间
        for (int h = 0; h < f.H; h++) {
            int kvh = h / f.group;
            const float *qv = qHost.data() + ((size_t)h * qoLen + t) * f.headDim;
            double maxScore = -1e300;
            for (int j = 0; j <= visibleEnd; j++) {
                const double *kv = f.exactK.data() + ((size_t)kvh * f.kvLen + j) * f.headDim;
                double dot = 0.0;
                for (int d = 0; d < f.headDim; d++) {
                    dot += (double)qv[d] * kv[d];
                }
                scores[j] = dot * (double)f.scale;
                maxScore = std::max(maxScore, scores[j]);
            }
            double sum = 0.0;
            for (int j = 0; j <= visibleEnd; j++) {
                scores[j] = std::exp(scores[j] - maxScore);
                sum += scores[j];
            }
            double inv = sum > 0 ? 1.0 / sum : 0.0;
            double *dst = out.data() + ((size_t)t * f.H + h) * f.headDim;
            for (int j = 0; j <= visibleEnd; j++) {
                double p = scores[j] * inv;
                const double *vv = f.exactV.data() + ((size_t)kvh * f.kvLen + j) * f.headDim;
                for (int d = 0; d < f.headDim; d++) {
                    dst[d] += p * vv[d];
                }
            }
        }
    }
    return out;
}

struct RunResult {
    bool ok = false;
    std::vector<float> output;
};

RunResult RunChunkedCublas(Fixture &f, fastllm::Data &q, int qoLen, bool batchGqa) {
    using namespace fastllm;
    RunResult result;
    Data qSizes = MakeIntTensor({2}, {0, qoLen});
    Data pageSizes = MakeIntTensor({2}, {0, (int32_t)f.physicalPages.size()});
    Data pageIndexs = MakeIntTensor({(int)f.physicalPages.size()}, f.physicalPages);
    int lastPageLen = f.kvLen - ((int)f.physicalPages.size() - 1) * f.pageLen;
    Data lastPageLens = MakeIntTensor({1}, {lastPageLen});

    Data output = MakeCudaTensor(DataType::FLOAT16, {qoLen, f.H, f.headDim},
                                 std::vector<float>((size_t)qoLen * f.H * f.headDim, 0.0f));
    ScopedEnvVar gate("FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA", batchGqa ? "1" : "0");
    // 【关键】必须显式关掉融合路。它已经挂在 chunked-cuBLAS 入口处试探, 不关掉的话
    // 这里测到的根本不是 chunked-cuBLAS 基线, 而是融合 kernel 自己跟自己比。
    ScopedEnvVar noFused("FASTLLM_CUDA_SM70_TURBO_XQA", "0");
    result.ok = FastllmCudaHalfPagedAttentionBatchFastllmFallback(
        q, f.kCaches, f.vCaches, qSizes, pageSizes, pageIndexs, lastPageLens,
        output, f.group, f.scale);
    FastllmCudaSyncCurrentThreadStream();
    result.output = ToFloatVector(output);
    return result;
}

// 直接调用融合 kernel(不经过 Data 层路由), 参数与 chunked-cuBLAS 的 raw 入口一一对应。
RunResult RunFusedTurboXqa(Fixture &f, fastllm::Data &q, int qoLen) {
    using namespace fastllm;
    RunResult result;
    Data pageIndexs = MakeIntTensor({(int)f.physicalPages.size()}, f.physicalPages);
    int lastPageLen = f.kvLen - ((int)f.physicalPages.size() - 1) * f.pageLen;
    Data output = MakeCudaTensor(DataType::FLOAT16, {qoLen, f.H, f.headDim},
                                 std::vector<float>((size_t)qoLen * f.H * f.headDim, 0.0f));
    result.ok = FastllmCudaTrySm70PagedTurboXqa(
        q.cudaData, DataType::FLOAT16, f.H, qoLen, f.headDim,
        (int)q.strides[0], (int)q.strides[1],
        (const int32_t *)pageIndexs.cudaData, (int)f.physicalPages.size(), lastPageLen,
        (Data *)f.pagedK, (Data *)f.pagedV, f.pageLen, f.numKvHeads, f.headDim,
        output.cudaData, DataType::FLOAT16, f.headDim, f.H * f.headDim,
        f.group, f.scale);
    FastllmCudaSyncCurrentThreadStream();
    result.output = ToFloatVector(output);
    return result;
}

// 走与生产同一条 Data 层入口, 融合路保持默认(开启), 用来验证 hook 真的接上了。
RunResult RunDispatchDefault(Fixture &f, fastllm::Data &q, int qoLen) {
    using namespace fastllm;
    RunResult result;
    Data qSizes = MakeIntTensor({2}, {0, qoLen});
    Data pageSizes = MakeIntTensor({2}, {0, (int32_t)f.physicalPages.size()});
    Data pageIndexs = MakeIntTensor({(int)f.physicalPages.size()}, f.physicalPages);
    int lastPageLen = f.kvLen - ((int)f.physicalPages.size() - 1) * f.pageLen;
    Data lastPageLens = MakeIntTensor({1}, {lastPageLen});
    Data output = MakeCudaTensor(DataType::FLOAT16, {qoLen, f.H, f.headDim},
                                 std::vector<float>((size_t)qoLen * f.H * f.headDim, 0.0f));
    ScopedEnvVar defaultFused("FASTLLM_CUDA_SM70_TURBO_XQA", nullptr);
    result.ok = FastllmCudaHalfPagedAttentionBatchFastllmFallback(
        q, f.kCaches, f.vCaches, qSizes, pageSizes, pageIndexs, lastPageLens,
        output, f.group, f.scale);
    FastllmCudaSyncCurrentThreadStream();
    result.output = ToFloatVector(output);
    return result;
}

void ReportStats(const std::string &label, const ErrorStats &s) {
    std::cout << "    " << label
              << "  maxAbs=" << s.maxAbs
              << "  rms=" << s.rms
              << "  maxAbs/refMax=" << s.MaxRel() << "\n";
}

// --------------------------------------------------------------------------
// 性能模式 (--bench)。
//
// 为什么不像 tools/prefill_dequant_bench 那样另写一个自包含 bench:
// 那个先例度量的是**逐字复制过来的** kernel, 复制体和真身可以各自演化。这里要比的
// baseline 是**生产真正在跑的那条路**(gather + chunked cuBLAS, 带 workspace 分块、
// cuBLAS 内核选择、逐 head 循环), 复制一份必然走样。所以直接驱动库里的真实入口,
// 与上面的对拍共用同一套夹具构造, 保证"测的就是跑的"。
//
// 显存: kvLen=131072 时页池约 195 MB (1024 页 x 128 x 4 head x 372 B) + 约 8.5 MB
// 注意力 workspace。数字要干净必须独占 GPU。
struct BenchFixture {
    Fixture f;
    std::vector<int32_t> pages;
};

void BuildBenchFixture(Fixture &f, int layerIdBase, int kvLen) {
    using namespace fastllm;
    const int numPages = (kvLen + f.pageLen - 1) / f.pageLen;
    f.kvLen = kvLen;
    f.physicalPages.resize(numPages);
    for (int i = 0; i < numPages; i++) {
        f.physicalPages[i] = i;
    }
    Data kDesc(DataType::Q8_0_KV);
    kDesc.Resize({f.numKvHeads, 1, f.headDim});
    kDesc.dataDevice = DataDevice::CUDA;
    Data vDesc(DataType::TURBO3_KV);
    vDesc.Resize({f.numKvHeads, 1, f.headDim});
    vDesc.dataDevice = DataDevice::CUDA;

    // 【必须】AllocatePagedCacheManager 默认只**物理**分配 min(128, maxPages) 页
    // (src/fastllm.cpp: `initialPages = preallocateMax ? maxPages : min(128, maxPages)`),
    // 剩下的靠运行期 GetUnusedPageIndex -> Grow() 按需追加。
    //
    // 本夹具绕开了取页路径, 直接往物理页号 0..numPages-1 上写。kvLen > 128*pageLen
    // (=16384 token) 时, 页号 >=128 全部落在池子外面 —— 量化 kernel 会**越界写显存**,
    // 报 cudaErrorIllegalAddress(700), 而且这个错是粘性的: 之后每一次 CUDA 调用都报同一个错,
    // 于是崩溃点看起来在后面某次无关的 H2D 拷贝上, 极易被误判成 kernel 有越界。
    // (2026-08-20 就是这么炸的: 8192=64 页正常, 32768=256 页崩。)
    ScopedEnvVar preallocate("FASTLLM_PAGED_CACHE_PREALLOCATE_MAX", "1");
    f.pagedK = AllocatePagedCacheManager(layerIdBase,
        PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kDesc, f.pageLen, numPages);
    f.pagedV = AllocatePagedCacheManager(layerIdBase + 1,
        PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vDesc, f.pageLen, numPages);
    // 硬断言: 宁可当场失败, 也不要再把"池子比页号小"变成显存踩踏。
    // 预算守卫(kvBudgetBytes)也可能把 maxPages 砍小, 所以这里查的是**实际** dims[0]。
    for (auto *mgr : {f.pagedK, f.pagedV}) {
        int physical = (int)((Data *)mgr)->dims[0];
        if (physical < numPages) {
            throw std::runtime_error(
                "分页池物理页数不足: dims[0]=" + std::to_string(physical) +
                " < 需要 " + std::to_string(numPages) +
                " 页 (kvLen=" + std::to_string(kvLen) + ")");
        }
    }

    // 按页填充, 避免为 131072 token 一次性造 537 MB 的 host 缓冲。
    const size_t pageElems = (size_t)f.numKvHeads * f.pageLen * f.headDim;
    for (int p = 0; p < numPages; p++) {
        const int len = std::min(f.pageLen, kvLen - p * f.pageLen);
        Data src = MakeCudaTensor(DataType::FLOAT16,
                                  {f.numKvHeads, f.pageLen, f.headDim},
                                  MakeValues(pageElems, 1000.0 + p, 1.0));
        FastllmCudaPackedKVCacheCopy((uint8_t *)f.pagedK->cudaData, p, f.pageLen,
                                     f.numKvHeads, f.headDim, DataType::Q8_0_KV,
                                     (uint8_t *)src.cudaData, DataType::FLOAT16,
                                     f.pageLen, 0, len, 0);
        FastllmCudaPackedKVCacheCopy((uint8_t *)f.pagedV->cudaData, p, f.pageLen,
                                     f.numKvHeads, f.headDim, DataType::TURBO3_KV,
                                     (uint8_t *)src.cudaData, DataType::FLOAT16,
                                     f.pageLen, 0, len, 0);
    }
    FastllmCudaSyncCurrentThreadStream();
    f.kCaches.dataType = DataType::Q8_0_KV;
    f.vCaches.dataType = DataType::TURBO3_KV;
    f.kCaches.Resize({f.numKvHeads, 1, f.headDim});
    f.vCaches.Resize({f.numKvHeads, 1, f.headDim});
    f.kCaches.isKVCache = f.vCaches.isKVCache = true;
    f.kCaches.isPagedKVCache = f.vCaches.isPagedKVCache = true;
    f.kCaches.pageLen = f.vCaches.pageLen = f.pageLen;
    f.kCaches.pagedKVCacheData = f.pagedK;
    f.vCaches.pagedKVCacheData = f.pagedV;
}

double TimeMs(const std::function<void()> &body, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        body();
    }
    FastllmCudaSyncCurrentThreadStream();
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg, cudaStreamPerThread);
    for (int i = 0; i < iters; i++) {
        body();
    }
    cudaEventRecord(end, cudaStreamPerThread);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return ms / iters;
}

// 大形状 fp64 对拍。
//
// 为什么必须单独有这一档: 上面那组对拍的 kvLen <= 273, splits=96 时每段只有 3 个 token,
// 绝大多数 split 是空的 —— 那是"分段几乎不工作"的退化情形。真实生产是 kvLen 数万,
// 每段几百个 token 全满, 跨段合并、在线 softmax 的 max 修正、u 域累加的幅度都完全不同。
// compute-sanitizer 只能证明**没越界**, 证明不了**算得对**, 两者都要。
//
// host 峰值内存: 逐 kv head 流式反量化, 只驻留一个 head 的 K/V
// (kvLen=32768 时 2 * 32768 * 256 * 8 B = 134 MB), 不是全部 4 个 head。
void RunLargeCorrectness(int kvLen, int layerIdBase, const std::vector<int> &qoLens) {
    using namespace fastllm;
    Fixture f;
    BuildBenchFixture(f, layerIdBase, kvLen);
    std::cout << "  大形状夹具: kvLen=" << kvLen
              << " 页数=" << f.physicalPages.size() << "\n";

    const size_t kBytes = ((Data *)f.pagedK)->GetBytes();
    const size_t vBytes = ((Data *)f.pagedV)->GetBytes();
    std::vector<uint8_t> kHost(kBytes), vHost(vBytes);
    FastllmCudaCopyFromDeviceToHost(kHost.data(), f.pagedK->cudaData, kBytes);
    FastllmCudaCopyFromDeviceToHost(vHost.data(), f.pagedV->cudaData, vBytes);
    const size_t kRowBytes = fastllm::turbokv::Q8RowBytes(f.headDim);
    const size_t vRowBytes = fastllm::turbokv::Turbo3RowBytes(f.headDim);

    for (int qoLen : qoLens) {
        Data q = MakeCudaTensor(DataType::FLOAT16, {f.H, qoLen, f.headDim},
                                MakeValues((size_t)f.H * qoLen * f.headDim, 4242.0 + qoLen, 1.0));
        std::vector<float> qHost = ToFloatVector(q);
        q.ToDevice(DataDevice::CUDA);

        std::vector<double> reference((size_t)qoLen * f.H * f.headDim, 0.0);
        std::vector<double> kSlice((size_t)kvLen * f.headDim);
        std::vector<double> vSlice((size_t)kvLen * f.headDim);
        std::vector<double> scores(kvLen);
        for (int kvh = 0; kvh < f.numKvHeads; kvh++) {
            for (int j = 0; j < kvLen; j++) {
                const int page = f.physicalPages[j / f.pageLen];
                const int off = j % f.pageLen;
                const size_t row = ((size_t)page * f.pageLen + off) * f.numKvHeads + kvh;
                DequantQ8RowHost(kHost.data() + row * kRowBytes, f.headDim,
                                 kSlice.data() + (size_t)j * f.headDim);
                DequantTurbo3RowHost(vHost.data() + row * vRowBytes, f.headDim,
                                     vSlice.data() + (size_t)j * f.headDim);
            }
            for (int g = 0; g < f.group; g++) {
                const int h = kvh * f.group + g;
                for (int t = 0; t < qoLen; t++) {
                    const int visibleEnd = kvLen - qoLen + t;
                    const float *qv = qHost.data() + ((size_t)h * qoLen + t) * f.headDim;
                    double maxScore = -1e300;
                    for (int j = 0; j <= visibleEnd; j++) {
                        const double *kv = kSlice.data() + (size_t)j * f.headDim;
                        double dot = 0.0;
                        for (int d = 0; d < f.headDim; d++) {
                            dot += (double)qv[d] * kv[d];
                        }
                        scores[j] = dot * (double)f.scale;
                        maxScore = std::max(maxScore, scores[j]);
                    }
                    double sum = 0.0;
                    for (int j = 0; j <= visibleEnd; j++) {
                        scores[j] = std::exp(scores[j] - maxScore);
                        sum += scores[j];
                    }
                    const double inv = sum > 0 ? 1.0 / sum : 0.0;
                    double *dst = reference.data() + ((size_t)t * f.H + h) * f.headDim;
                    for (int j = 0; j <= visibleEnd; j++) {
                        const double pr = scores[j] * inv;
                        const double *vv = vSlice.data() + (size_t)j * f.headDim;
                        for (int d = 0; d < f.headDim; d++) {
                            dst[d] += pr * vv[d];
                        }
                    }
                }
            }
        }

        RunResult base = RunChunkedCublas(f, q, qoLen, false);
        RunResult fused = RunFusedTurboXqa(f, q, qoLen);
        Expect(base.ok, "大形状 kvLen=" + std::to_string(kvLen) +
               " qoLen=" + std::to_string(qoLen) + " 基线返回 false");
        Expect(fused.ok, "大形状 kvLen=" + std::to_string(kvLen) +
               " qoLen=" + std::to_string(qoLen) + " 融合 kernel 拒绝了合法形状");
        if (!base.ok || !fused.ok) {
            continue;
        }
        ErrorStats baseStats = CompareToReference(reference, base.output);
        ErrorStats fusedStats = CompareToReference(reference, fused.output);
        std::cout << "  qoLen=" << qoLen << "\n";
        ReportStats("chunked-cuBLAS 默认(逐 head)  vs fp64", baseStats);
        ReportStats("融合 turbo-XQA(延后逆WHT)     vs fp64", fusedStats);
        Expect(fusedStats.MaxRel() < 5e-3,
               "大形状 kvLen=" + std::to_string(kvLen) + " qoLen=" + std::to_string(qoLen) +
               " 融合 kernel 相对误差过大 (" + std::to_string(fusedStats.MaxRel()) + ")");
        Expect(fusedStats.rms <= baseStats.rms,
               "大形状 kvLen=" + std::to_string(kvLen) + " qoLen=" + std::to_string(qoLen) +
               " 融合 kernel RMS 反而更大 (" + std::to_string(fusedStats.rms) +
               " vs " + std::to_string(baseStats.rms) + ")");
    }
}

void RunBench(int kvLen, int layerIdBase, const std::vector<int> &qoLens) {
    using namespace fastllm;
    Fixture f;
    BuildBenchFixture(f, layerIdBase, kvLen);
    const int numPages = (int)f.physicalPages.size();
    const int lastPageLen = kvLen - (numPages - 1) * f.pageLen;

    for (int qoLen : qoLens) {
        Data q = MakeCudaTensor(DataType::FLOAT16, {f.H, qoLen, f.headDim},
                                MakeValues((size_t)f.H * qoLen * f.headDim, 31.0, 1.0));
        Data qSizes = MakeIntTensor({2}, {0, qoLen});
        Data pageSizes = MakeIntTensor({2}, {0, numPages});
        Data pageIndexs = MakeIntTensor({numPages}, f.physicalPages);
        Data lastPageLens = MakeIntTensor({1}, {lastPageLen});
        Data output = MakeCudaTensor(DataType::FLOAT16, {qoLen, f.H, f.headDim},
                                     std::vector<float>((size_t)qoLen * f.H * f.headDim, 0.0f));

        auto viaDispatch = [&]() {
            FastllmCudaHalfPagedAttentionBatchFastllmFallback(
                q, f.kCaches, f.vCaches, qSizes, pageSizes, pageIndexs, lastPageLens,
                output, f.group, f.scale);
        };
        auto viaFused = [&]() {
            FastllmCudaTrySm70PagedTurboXqa(
                q.cudaData, DataType::FLOAT16, f.H, qoLen, f.headDim,
                (int)q.strides[0], (int)q.strides[1],
                (const int32_t *)pageIndexs.cudaData, numPages, lastPageLen,
                (Data *)f.pagedK, (Data *)f.pagedV, f.pageLen, f.numKvHeads, f.headDim,
                output.cudaData, DataType::FLOAT16, f.headDim, f.H * f.headDim,
                f.group, f.scale);
        };

        double base, gqa, fused;
        {
            ScopedEnvVar noFused("FASTLLM_CUDA_SM70_TURBO_XQA", "0");
            ScopedEnvVar gate("FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA", "0");
            base = TimeMs(viaDispatch, 3, 20);
        }
        {
            ScopedEnvVar noFused("FASTLLM_CUDA_SM70_TURBO_XQA", "0");
            ScopedEnvVar gate("FASTLLM_CUDA_PAGED_CUBLAS_BATCH_GQA", "1");
            gqa = TimeMs(viaDispatch, 3, 20);
        }
        fused = TimeMs(viaFused, 3, 20);

        // 每 (token, kvHead) 行的等效访存量, 用于和 372 B/行 的理论下限对照。
        const double rows = (double)kvLen * f.numKvHeads;
        auto bytesPerRow = [&](double ms) { return ms * 1e-3 * 750e9 / rows; };
        std::cout << "  kvLen=" << kvLen << " qoLen=" << qoLen << "\n"
                  << "    chunked-cuBLAS 默认      " << base  << " ms"
                  << "  (等效 " << bytesPerRow(base) << " B/行 @750GB/s)\n"
                  << "    chunked-cuBLAS BATCH_GQA " << gqa   << " ms"
                  << "  (等效 " << bytesPerRow(gqa) << " B/行)"
                  << "  相对默认 " << (base / gqa) << "x\n"
                  << "    融合 turbo-XQA           " << fused << " ms"
                  << "  (等效 " << bytesPerRow(fused) << " B/行)"
                  << "  相对默认 " << (base / fused) << "x"
                  << "  相对 BATCH_GQA " << (gqa / fused) << "x\n";
    }
}

void RunFixture(const std::string &name, int layerIdBase,
                const std::vector<int32_t> &physicalPages, int lastPageLen,
                const std::vector<int> &qoLens) {
    Fixture f;
    BuildPackedFixture(f, layerIdBase, physicalPages, lastPageLen);
    std::cout << "  夹具 " << name << ": kvLen=" << f.kvLen
              << " 物理页=";
    for (size_t i = 0; i < physicalPages.size(); i++) {
        std::cout << (i ? "," : "") << physicalPages[i];
    }
    std::cout << "\n";

    for (int qoLen : qoLens) {
        fastllm::Data q = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {f.H, qoLen, f.headDim},
            MakeValues((size_t)f.H * qoLen * f.headDim, 777.0 + qoLen, 1.0));
        std::vector<float> qHost = ToFloatVector(q);
        q.ToDevice(fastllm::DataDevice::CUDA);

        std::vector<double> reference = ReferenceAttention(f, qHost, qoLen);

        RunResult off = RunChunkedCublas(f, q, qoLen, false);
        RunResult on = RunChunkedCublas(f, q, qoLen, true);
        Expect(off.ok, name + " qoLen=" + std::to_string(qoLen) + " 默认路径返回 false");
        Expect(on.ok, name + " qoLen=" + std::to_string(qoLen) + " BATCH_GQA 路径返回 false");
        if (!off.ok || !on.ok) {
            continue;
        }

        ErrorStats offStats = CompareToReference(reference, off.output);
        ErrorStats onStats = CompareToReference(reference, on.output);
        // 两条 cuBLAS 路径互相之间的差
        double maxPathDiff = 0.0, outMax = 0.0;
        for (size_t i = 0; i < off.output.size(); i++) {
            maxPathDiff = std::max(maxPathDiff, std::fabs((double)off.output[i] - (double)on.output[i]));
            outMax = std::max(outMax, std::fabs((double)off.output[i]));
        }
        int bitIdentical = std::memcmp(off.output.data(), on.output.data(),
                                       off.output.size() * sizeof(float)) == 0;

        std::cout << "  qoLen=" << qoLen << "\n";
        ReportStats("chunked-cuBLAS 默认(逐 head)  vs fp64", offStats);
        ReportStats("chunked-cuBLAS BATCH_GQA=1    vs fp64", onStats);
        std::cout << "    两条路径互差 maxAbs=" << maxPathDiff
                  << " (输出幅度 " << outMax << ")"
                  << "  逐位一致=" << (bitIdentical ? "是" : "否") << "\n";

        // 等价性判据: 两条路径都是「同一数学式的不同 GEMM 分组」, 差异只应来自 cuBLAS
        // 内核选择带来的累加次序变化, 量级必须与各自对 fp64 的误差同阶, 不能更大。
        double tolerance = std::max(3.0 * std::max(offStats.maxAbs, onStats.maxAbs), 1e-4);
        Expect(maxPathDiff <= tolerance,
               name + " qoLen=" + std::to_string(qoLen) +
               " BATCH_GQA 与默认路径差异超出容差 (" + std::to_string(maxPathDiff) +
               " > " + std::to_string(tolerance) + ")");

        // ---- 融合 kernel(延后逆 WHT) ----
        RunResult fused = RunFusedTurboXqa(f, q, qoLen);
        Expect(fused.ok, name + " qoLen=" + std::to_string(qoLen) + " 融合 kernel 拒绝了合法形状");
        if (fused.ok) {
            ErrorStats fusedStats = CompareToReference(reference, fused.output);
            ReportStats("融合 turbo-XQA(延后逆WHT)     vs fp64", fusedStats);
            // 判据一: 必须真的算对了 —— 相对幅度误差在 fp16 输出的分辨率量级。
            Expect(fusedStats.MaxRel() < 5e-3,
                   name + " qoLen=" + std::to_string(qoLen) +
                   " 融合 kernel 相对误差过大 (" + std::to_string(fusedStats.MaxRel()) + ")");
            // 判据二: 延后逆 WHT + 全程 fp32 不应比现路径更差。现路径把反量化结果和
            // score 都落成 fp16, 融合版没有这两次舍入, 预期 RMS 误差更小。
            Expect(fusedStats.rms <= offStats.rms,
                   name + " qoLen=" + std::to_string(qoLen) +
                   " 融合 kernel 的 RMS 误差反而比 chunked-cuBLAS 大 (" +
                   std::to_string(fusedStats.rms) + " vs " + std::to_string(offStats.rms) + ")");

            // 判据三: Data 层默认分派必须真的走进融合 kernel。只比误差不够 ——
            // 本仓已有两条「资格判定永远不满足」的 SM70 死路径, 必须钉死 hook 有效。
            RunResult routed = RunDispatchDefault(f, q, qoLen);
            Expect(routed.ok, name + " qoLen=" + std::to_string(qoLen) + " 默认分派失败");
            if (routed.ok) {
                bool same = std::memcmp(routed.output.data(), fused.output.data(),
                                        fused.output.size() * sizeof(float)) == 0;
                Expect(same, name + " qoLen=" + std::to_string(qoLen) +
                       " 默认分派没有走融合 kernel(输出与直调不一致)");
            }
        }
    }
}

#endif  // USE_CUDA && !USE_ROCM

}  // namespace

int main(int argc, char **argv) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    if (getCudaInfos() == nullptr) {
        std::cout << "打包分页注意力对拍: SKIP (无 CUDA 设备)\n";
        return 0;
    }
    // --bench [kvLen ...]  不给 kvLen 时用生产关心的 8K/32K/128K 三档。
    // 显式给一个小 kvLen 可以在**不独占 GPU** 的情况下冒烟验证 bench 通路本身,
    // 免得真拿到独占窗口时才发现 harness 崩了。
    bool benchMode = false;
    bool largeMode = false;
    std::vector<int> benchKvLens;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bench") {
            benchMode = true;
        } else if (arg == "--large") {
            largeMode = true;
        } else if (benchMode || largeMode) {
            int v = std::atoi(arg.c_str());
            if (v > 0) {
                benchKvLens.push_back(v);
            }
        }
    }
    if (largeMode) {
        if (benchKvLens.empty()) {
            benchKvLens = {8192, 32768};
        }
        std::cout << "打包分页注意力大形状 fp64 对拍\n";
        try {
            int layerId = 60300;
            for (int kvLen : benchKvLens) {
                RunLargeCorrectness(kvLen, layerId, {1, 3});
                layerId += 4;
            }
        } catch (const std::exception &ex) {
            std::cout << "  [FAIL] 异常: " << ex.what() << "\n";
            gFailures++;
        }
        fastllm::ClearAllPagedCacheManagers();
        std::cout << (gFailures == 0 ? "ALL PASS" : "FAILED")
                  << "  (" << (gChecks - gFailures) << "/" << gChecks << ")\n";
        return gFailures == 0 ? 0 : 1;
    }
    if (benchMode) {
        if (benchKvLens.empty()) {
            benchKvLens = {8192, 32768, 131072};
        }
        std::cout << "打包分页注意力性能对照 (需要独占 GPU 才有干净数字)\n";
        try {
            int layerId = 60200;
            for (int kvLen : benchKvLens) {
                RunBench(kvLen, layerId, {1, 3});
                layerId += 4;
            }
        } catch (const std::exception &ex) {
            std::cout << "  bench 异常: " << ex.what() << "\n";
        }
        fastllm::ClearAllPagedCacheManagers();
        return 0;
    }
    std::cout << "打包分页注意力数值对拍 (K=q8_0, V=turbo3, headDim=256, GQA6)\n";
    try {
        // 单页 / 多页 / 非线性页表 / 非整页尾部 —— 覆盖页表边界与 lastPageLen。
        RunFixture("单页尾部 40", 60100, {2}, 40, {1, 2, 3, 4});
        RunFixture("三页乱序尾部 17", 60102, {2, 0, 3}, 17, {1, 2, 3, 4});
        RunFixture("两页整页", 60104, {1, 4}, 128, {1, 3});
    } catch (const std::exception &ex) {
        std::cout << "  [FAIL] 异常: " << ex.what() << "\n";
        gFailures++;
    }
    fastllm::ClearAllPagedCacheManagers();
    std::cout << (gFailures == 0 ? "ALL PASS" : "FAILED")
              << "  (" << (gChecks - gFailures) << "/" << gChecks << ")\n";
    return gFailures == 0 ? 0 : 1;
#else
    std::cout << "打包分页注意力对拍: SKIP (未启用 CUDA)\n";
    return 0;
#endif
}
