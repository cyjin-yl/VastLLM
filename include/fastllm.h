//
// Created by huangyuyang on 5/11/23.
//

#ifndef TEST_FASTLLM_H
#define TEST_FASTLLM_H

#define _USE_MATH_DEFINES
#include <vector>
#include <cstdint>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>
#include <memory>
#include <mutex>
#include <locale>
#include <codecvt>
#include "devices/cpu/alivethreadpool.h"
#include "json11.hpp"
#include "prefixcache_persistence.h"

#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

namespace fastllm {
    class Data;
    class HostCacheReservation;

    class FastllmEnv {
    public:
        FastllmEnv();

        bool activateNuma = false;
        int numaThreads = -1;
        int numas = -1;
        bool cudaSync = false;
        bool printLogits = false;
        bool printProfile = false;
        bool skipWarmup = false;
        bool cudaGraph = false;
        bool cudaMemCheck = false;
        bool cudaTriton = false;
        bool useFusedTransferAttn = true;
        bool useFusedGdnPrefill = true;
        std::string debugTokenId;
    };

    const FastllmEnv &GetFastllmEnv();

    struct ModelLoadProgress {
        std::string stage;
        uint64_t current = 0;
        uint64_t total = 0;
        uint64_t completedBytes = 0;
        uint64_t totalBytes = 0;
    };

    using ModelLoadProgressCallback = std::function<void(const ModelLoadProgress &)>;

    void SetModelLoadProgressCallback(const ModelLoadProgressCallback &callback);
    void ClearModelLoadProgressCallback();
    void ReportModelLoadProgress(const std::string &stage,
                                 uint64_t current = 0,
                                 uint64_t total = 0,
                                 uint64_t completedBytes = 0,
                                 uint64_t totalBytes = 0);

    void SetDeviceMap(const std::map <std::string, int> &deviceMap);
    void SetMoeDeviceMap(const std::map <std::string, int> &moeDeviceMap);
    void SetLayeredMoeDeviceMap(const std::map <std::string, int> &moeDeviceMap);
    void SetMoeDeviceLayers(int layers);

    std::map <std::string, int> GetDeviceMap();
    std::map <std::string, int> GetMoeDeviceMap();
    std::map <std::string, int> GetLayeredMoeDeviceMap();
    int GetMoeDeviceLayers();
    std::string SelectDeviceFromMap(const std::map <std::string, int> &deviceMap, int current, int total);

    Data *GetEmptyData();
    void PrintInstructionInfo();
    void SetThreads(int t);
    void SetLowMemMode(bool m);
    void SetKVCacheInCPU(bool kvCacheInCPU);
    void SetCudaSharedExpert(bool v);
    bool GetCudaSharedExpert();
    void SetHistoryCacheInCPU(bool v);
    bool GetLowMemMode();
    void SetCudaEmbedding(bool v);
    bool GetCudaEmbedding();
    bool GetCudaEmbeddingRequested();
    void SetCudaSlabMB(int mb);
    int GetCudaSlabMB();
    // 释放 CUDA 内存池中所有空闲缓冲区，把显存归还给驱动；无 CUDA 构建为 no-op。
    void ReleaseCudaIdlePoolMemory();
    int GetThreads();
    bool GetKVCacheInCPU();
    bool GetHistoryCacheInCPU();
    void EnableAMX(bool enable);
    bool GetEnableAMX();
    void SetMaxTokens(int maxTokens);
    int GetMaxTokens();
    void SetPageLen(int pageLen);
    int GetPageLen();
    void SetGpuMemRatio(float ratio);
    float GetGpuMemRatio();
    AliveThreadPool *GetAlivePool();

    template<typename T, std::size_t Alignment>
    class alignedAllocator {
    public:
        using value_type = T;
        
        T* allocate(std::size_t n) {
            std::size_t size = n * sizeof(T);
            
            // 分配额外的内存用于对齐和存储原始指针
            std::size_t total_size = size + Alignment - 1 + sizeof(void*);
            void* raw_ptr = std::malloc(total_size);
            
            if (!raw_ptr) throw std::bad_alloc();
            
            // 计算对齐后的地址
            void* aligned_ptr = reinterpret_cast<void*>(
                (reinterpret_cast<std::uintptr_t>(raw_ptr) + sizeof(void*) + Alignment - 1) 
                & ~(Alignment - 1)
            );
            
            // 在对齐地址前存储原始指针
            *(reinterpret_cast<void**>(aligned_ptr) - 1) = raw_ptr;
            
            return static_cast<T*>(aligned_ptr);
        }
        
        void deallocate(T* p, std::size_t) noexcept {
            if (p) {
                // 获取原始指针并释放
                void* raw_ptr = *(reinterpret_cast<void**>(p) - 1);
                std::free(raw_ptr);
            }
        }
        
        template<typename U>
        struct rebind {
            using other = alignedAllocator<U, Alignment>;
        };
    };

    struct GenerationConfig {
        int output_token_limit = -1; // 最多输出多少, <= 0代表无限制
        int output_token_least = 0; // 最低输出的多少
        int input_token_length = 0;
        int last_n = 64; // 末尾last_n个token计入重复惩罚
        float repeat_penalty = 1.0f; // 重复惩罚系数，1.0代表不惩罚
        bool do_sample = false; // false表示贪心解码，保留前端的采样语义
        int top_k = 1; // top_k采样
        float top_p = 1.0; // top_p采样
        float temperature = 1.0; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性
        bool output_logits = false; // 是否返回logits
        bool enable_hash_id = false; // 给会话添加hash id
        bool add_special_tokens = true; // prompt添加special tokens（chatglm模型生效）
        std::multiset <int> stop_token_ids;
        std::vector<std::vector<int>> stop_token_sequences;
        std::vector<std::string> stop_strings;
        bool tool_call_name_constraint_enabled = false;
        std::vector <std::string> tool_call_allowed_names;
        std::vector <std::string> tool_call_invoke_name_prefixes;
        std::string tool_call_name_terminator = "\"";
        bool tool_call_parameter_name_constraint_enabled = false;
        std::map <std::string, std::vector <std::string> > tool_call_allowed_parameter_names;
        std::vector <std::string> tool_call_parameter_name_prefixes;
        // 强制 required 参数块: <function=name> 完成后, 该工具 schema 中
        // 尚有未闭合的必填参数名时, 约束只允许生成 "<parameter=" 前缀链,
        // 且参数名位置只放行缺失的必填名(强制 required-first),
        // 堵死量化损伤导致的"整块跳过参数"/"发可选漏必填"空调用。
        bool tool_call_required_parameter_constraint_enabled = false;
        std::map <std::string, std::vector <std::string> > tool_call_required_parameter_names;
        std::vector <int> tool_call_allowed_token_ids;
        // ROOT CAUSE #4: 黑名单通道 —— S4 空值时屏蔽 </parameter> 等闭合序列的
        // 起始 token(让空值不可表达)。与 allowed 独立判定, 两者可叠加。
        std::vector <int> tool_call_blocked_token_ids;
        bool tool_call_content_sampling_enabled = false;
        // Set on the per-step config after Kimi-K3 has drained DSpark's
        // scheduler-ahead queue. DSpark then samples from its batched target
        // verification logits while keeping target and draft caches aligned.
        bool tool_call_content_sampling_active = false;
        int tool_call_content_top_k = 1;
        float tool_call_content_top_p = 1.0f;
        float tool_call_content_temperature = 1.0f;

        bool IsSimpleGreedy() const {
            if (!tool_call_allowed_token_ids.empty() ||
                !tool_call_blocked_token_ids.empty()) {
                return false;
            }
            if (fabs(repeat_penalty - 1) > 1e-8) {
                return false;
            }
            if (top_k > 1) {
                return false;
            }
            return true;
        }
    };

    struct LastTokensUnit {
        int tot = 0;
        std::multiset <int> tokenSet;
        std::queue <int> tokenQueue;

        LastTokensUnit () {}

        LastTokensUnit (int tot) {
            Init(tot);
        }

        void Init(int tot) {
            this->tot = tot;
            tokenSet.clear();
            while (tokenQueue.size() > 0) {
                tokenQueue.pop();
            }
        }

        void Push(int id) {
            if (tokenQueue.size() == tot && tot > 0) {
                tokenSet.erase(tokenSet.find(tokenQueue.front()));
                tokenQueue.pop();
            }
            tokenQueue.push(id);
            tokenSet.insert(id);
        }
    };

    struct LastTokensManager {
        std::vector <LastTokensUnit> units;

        LastTokensManager () {}

        LastTokensManager (int batch, int lastN) {
            units.resize(batch);
            for (int i = 0; i < batch; i++) {
                units[i].Init(lastN);
            }
        }
    };

    struct LowBitConfig {
        int bit;
        float min, max;
        uint8_t zeroPoint;
        float scale;
        int type; // 0: 有zero点 1: 不需要zero点

        LowBitConfig(float min, float max, int bit, int type) {
            this->min = min;
            this->max = max;
            this->bit = bit;
            this->type = type;
            Reset();
        }

        LowBitConfig () {

        }

        void Reset() {
            /*if (type == 1) {
                this->scale = (max - min) / 15.0;
                return;
            }*/
            /*if (type == 1) {
                this->scale = std::max(fabs(max), fabs(min)) / 7.0;
                this->min = this->scale * (-7.0);
                return;
            }*/
            min = std::min(min, 0.f);
            max = std::max(max, 0.f);

            const float qmin = 0;
            const float qmax = (1 << bit) - 1;
            scale = (max - min) / (qmax - qmin);
            const float initial_zero_point = qmin - min / scale;
            zeroPoint = 0;
            if (initial_zero_point < qmin) {
                zeroPoint = qmin;
            } else if (initial_zero_point > qmax) {
                zeroPoint = qmax;
            } else {
                zeroPoint = static_cast<uint8_t>(std::round(initial_zero_point));
            }

            if (type == 1) {
                this->min = -this->scale * zeroPoint;
                return;
            }
        }

        uint8_t quantization(const float &realNumber) const {
            if (type == 0) {
                return (uint8_t) (std::min((double) ((1 << bit) - 1),
                                           (double) std::max(realNumber / scale + zeroPoint + 0.5, 0.0)));
            } else {
                return (uint8_t) (std::max(0.f, std::min(15.f, (realNumber - min) / scale + 0.5f)));
            }
        }

        float invQuantization(const uint8_t &qNumber) const {
            if (type == 0) {
                return (scale * ((float) qNumber - (float) zeroPoint));
            } else {
                return min + scale * qNumber;
            }
        }
    };

    enum DataType {
        FLOAT32 = 0, BFLOAT16 = 1, INT16 = 2, INT8 = 3, INT4 = 4, INT2 = 5, BIT = 6, FLOAT16 = 7,
        INT4_NOZERO = 8, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale
        INT4_GROUP = 9, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale, 且使用分组量化
        FP8_E4M3 = 10,
        INT2_GROUP = 11, // 不用zeroPoint的int2, floatValue = min + uint2Value * scale, 且使用分组量化
        BASE3_GROUP = 12, // 三元量化，-1 0 1
        INT32 = 13, // int32
        NVFP4 = 14, // packed fp4 e2m1 + compact e8m0 block scales
        Q8_0_KV = 15, // KV cache: one fp16 scale + 32 int8 values per 32-value block
        TURBO3_KV = 16, // TurboQuant KV: fp16 norm + packed 3-bit indices per 128-value block
        TURBO4_KV = 17, // TurboQuant KV: fp16 norm + packed 4-bit indices per 128-value block
        INT32PARAM = 100, // int32的参数，这种类型的数据永远存在CPU上
        FP8_E4M3_BLOCK_128 = 1000, // fp8e4m3, block = 128
        AWQ_4BIT_128 = 1001, // awq, bits = 4, group = 128
        INT4_PERCHANNEL = 1002, // int4, per channel量化
        FP8_E4M3_PERCHANNEL = 1003, // fp8, per channel量化
        INT4_GROUP128 = 1004, // int4, per group量化，group = 128
        INT8_PERCHANNEL = 1005, // int8, per channel量化
        NVFP4_BLOCK_16 = 1006, // packed fp4 e2m1, blockM = 16, inline float scale per block
        NVFP4_BLOCK_16_E8M0 = 1007, // packed fp4 e2m1, blockM = 16, inline e8m0 scale per block
        // Symmetric group-32 INT4. Four groups form one compact block:
        // [up to 4 * 16 packed INT4 bytes] [the corresponding BF16 scales].
        // The final partial block has no padding. The implicit zero point is 8.
        INT4_GROUP32 = 1008,
        // Internal NUMA layout for NVFP4 blockM=32 weights:
        // [16 packed fp4 bytes] [one inline E8M0 scale byte].
        NVFP4_BLOCK_32_E8M0 = 1009,
        INF_INT8_PERCHANNEL = 2000, // 推理用的int8, per channel量化
        INF_INT8_GROUP128 = 2001, // 推理用的int8, per group量化，group = 128
        INF_INT8_GROUP32 = 2002, // 推理用的int8, per group量化，group = 32
        DATA_GGUF_FORMAT = 9999, DATA_GGUF_FORMAT_END = 19999, // [DATA_GGUF_FORMAT, DATA_GGUF_FORMAT_END]之间为GGUF格式的数据，ggml_type = type - DATA_FFUF_FORMAT
        DATA_AUTO_NONE = 99999, DATA_AUTO_LINEAR, DATA_AUTO_EMBEDDING, DATA_AUTO_CONV,
        DATA_AUTO_SOURCE // auto keeps scaled FP8 source weights, otherwise uses FLOAT16
    };

    std::string GetDataTypeName(DataType type);

    size_t GetDataBytes(DataType type, size_t rows, size_t columns);
    bool IsPackedKVCacheDataType(DataType type);
    size_t GetKVCacheRowBytes(DataType type, size_t columns);

    // 【上游BUMP勿回退】分页 KV 池"一页占多少字节"的唯一口径。
    // 打包 KV 类型(Q8_0_KV/TURBO3_KV/TURBO4_KV)的 unitSize 恒为 1, 是占位值而非
    // 每元素字节数, 任何地方都不许用 pageLen*numHeads*headDim*unitSize 去估页大小。
    // 详见 src/fastllm.cpp 中 AllocatePagedCacheManager 里预算守卫处的长注释。
    size_t GetPagedPoolPageBytes(
        DataType type, int pageLen, int numHeads, int headDim);

    // 【上游BUMP勿回退】GDN/线性注意力递归状态快照的"标称长度"合法性判据。
    // 递归状态不能截断: 它对应的 token 数(currentLenRaw)必须恰好等于快照标称的
    // 页对齐长度, 否则命中恢复时会把 [alignedLen, currentLenRaw) 这段 token 卷进
    // 递归状态两次(静默算错)。详见 src/models/qwen3_5.cpp 中
    // TryRecordPagedPrefixCacheExtra 的长注释。
    // 返回 true 表示可安全记录, *alignedLen 写回对齐后的长度。
    bool PagedPrefixSnapshotLengthUsable(
        int currentLenRaw, int pageLen, int *alignedLen);
    // FASTLLM_PREFIX_CACHE_STRICT_SNAPSHOT_ALIGN=1 时才真的拒绝未对齐快照。
    // 默认 false: 见 src/fastllm.cpp 里该函数的权衡说明。
    bool PagedPrefixSnapshotStrictAlign();

    // 【上游BUMP勿回退】分层准入的纯判据。做成纯函数是为了让单测能把
    // "什么时候该走存储、什么时候该重算"钉死 —— 这个判断错了不会报错,
    // 只会悄悄变慢(在机械盘上把请求拖死, 或在 NVMe 上白白放弃一整层)。
    //   restore = seekSeconds * max(1,inflight) + storedBytes/readMiBps [+解压]
    //   recompute = recomputeTokens / recomputeTps
    // 返回 true = 从存储取回比重算划算(且至少快 margin 这么多)。
    bool PagedPrefixCacheStorageWinsPure(
        size_t storedBytes, size_t uncompressedBytes, bool zstdCompressed,
        size_t recomputeTokens, double readMiBps, double decompressMiBps,
        double recomputeTps, double seekSeconds, int inflight, double margin);

    // L3 介质画像。**只读 sysfs 推导, 零 I/O 零写入**, 不做探针,
    // 不写校准文件 —— 探针会消耗 SSD 的 TBW 寿命和机械盘的机械磨损,
    // 而这是用户设备上长期生效的行为。详见 src/fastllm.cpp 里的长注释。
    struct PagedPrefixCacheDiskProfileInfo {
        bool resolved = false;
        bool rotational = true;
        double seekSeconds = 0.012;
        double readMiBPerSecond = 60.0;
        std::string deviceKey = "unknown";
    };
    PagedPrefixCacheDiskProfileInfo GetPagedPrefixCacheDiskProfileInfo();
    constexpr size_t INT4_GROUP32_GROUP_SIZE = 32;
    constexpr size_t INT4_GROUP32_PACKED_BYTES = 16;
    constexpr size_t INT4_GROUP32_BLOCK_GROUPS = 4;

    // INT4_GROUP32 keeps four packed groups together before their four scales.
    // These helpers also handle the final 1-3 group block without padding.
    inline size_t GetInt4Group32DataOffset(size_t group, size_t groups) {
        (void)groups;
        const size_t block = group / INT4_GROUP32_BLOCK_GROUPS;
        const size_t inBlock = group % INT4_GROUP32_BLOCK_GROUPS;
        return block * INT4_GROUP32_BLOCK_GROUPS *
                   (INT4_GROUP32_PACKED_BYTES + sizeof(uint16_t)) +
               inBlock * INT4_GROUP32_PACKED_BYTES;
    }

    inline size_t GetInt4Group32ScaleOffset(size_t group, size_t groups) {
        const size_t block = group / INT4_GROUP32_BLOCK_GROUPS;
        const size_t inBlock = group % INT4_GROUP32_BLOCK_GROUPS;
        const size_t blockBegin = block * INT4_GROUP32_BLOCK_GROUPS;
        const size_t blockGroups = std::min(
            INT4_GROUP32_BLOCK_GROUPS, groups - blockBegin);
        return block * INT4_GROUP32_BLOCK_GROUPS *
                   (INT4_GROUP32_PACKED_BYTES + sizeof(uint16_t)) +
               blockGroups * INT4_GROUP32_PACKED_BYTES +
               inBlock * sizeof(uint16_t);
    }

    size_t GetNVFP4WeightBytes(size_t rows, size_t columns);
    size_t GetNVFP4ScaleBytes(size_t rows, size_t columns, int blockK, int blockM);
    size_t GetNVFP4StorageBytes(size_t rows, size_t columns, int blockK, int blockM);
    uint8_t *GetNVFP4ScaleData(Data &data);
    const uint8_t *GetNVFP4ScaleData(const Data &data);
    float NVFP4E8M0ScaleToFloat(uint8_t v);

    enum DataDevice {
        CPU = 0, CUDA = 1
    };

    struct DiskWeightPart {
        std::string fileName;
        long long fileOffset = 0;
        uint64_t bytes = 0;
        DataType sourceDataType = DataType::FLOAT32;
        std::vector <int> dims;
        bool isScalePart = false;
        uint64_t scaleOffset = 0;
    };

    enum WeightType {
        NONE = 0, LINEAR = 1, EMBEDDING = 2, CONV2D = 3, CONV1D = 4, AUTO = 99999
    };

    enum TensorParallelLayoutType {
        TP_LAYOUT_NONE = 0,       // 不使用 tensor parallel 布局，按普通单份张量处理
        TP_LAYOUT_REPLICATED = 1, // 多卡各持有一份完整副本
        TP_LAYOUT_SHARDED = 2     // 多卡沿 tpAxis 切分，每卡只持有部分数据
    };

    enum TensorParallelLinearType {
        TP_LINEAR_NONE = 0,
        TP_LINEAR_ROW = 1,
        TP_LINEAR_COLUMN = 2
    };

    enum TensorParallelPackType {
        TP_PACK_NONE = 0,
        TP_PACK_GATEUP = 1,
        TP_PACK_QKV = 2
    };

    struct FileMmap {
    public:
        FileMmap(const std::string &path);
        ~FileMmap();

        char *data;
        size_t size;
    };

    struct ModelLoader {
        ModelLoader(const char *buffer, size_t size) : data(buffer), size(size), ptr(buffer) {}

        int64_t tell() const { return ptr - data; }

        void seek(int64_t offset, int whence);

        template <typename T>
        T read_basic() {
            T obj = *(T *)ptr;
            ptr += sizeof(T);
            return obj;
        }

        std::string ReadString();
        int ReadInt();
        float ReadFloat();
        uint8_t* ReadBytes(uint64_t bytes);

        const char *const data;
        size_t size;
        const char *ptr;
    };

    class PagedCacheManager;
    struct DataOffloadRecord {
        DataDevice originalDevice = DataDevice::CPU;
        int originalDeviceId = 0;
        uint64_t bytes = 0;
        uint64_t checksum = 0;
    };


    class Data {
    public:
        bool isFake = false; // 没有创建空间，指向别的data（无需销毁）

        long long cacheUid = 0; // 用来标注Cache id
        bool isKVCache = false; // 是否是KV Cache TODO: 做一些KVCache的管理
        bool isLinearAttention = false; // 是否是线性attention的缓存（永远保持同样的形状）
        bool isLinearAttentionTransposed = false; // 线性attention recurrent state是否物理存成[V,K]

        // Paged KV Cache的相关信息
        // 当isKVCache = true且isPagedKVCache = true时，下面这些信息才有意义
        bool isPagedKVCache = false; // 是否是分片的KV Cache
        int pageLen = 128; // 每个page的长度（token数）
        PagedCacheManager *pagedKVCacheData = nullptr; // 存储kv cached的数据，shape为 [maxPages, pageLen, numHeads, headDim]
        std::vector <int> pageIndex; // 目前使用的Index编号
        int lastPageLen = 0; // 最后一个Page中使用了多少长度

        bool lockInCPU = false; // 如果lock在CPU上，那么不允许移动到其余设备
        WeightType weightType = WeightType::NONE; // 权重类型，NONE代表非权重（或未知权重）

        DataType dataType = DataType::FLOAT32; // 数据类型
        int unitSize, unitSizeDiv = 1; // 单个元素的字节数 = unitSIze / unitSizeDiv

        std::vector <int> dims; // 数据形状
        std::vector <uint64_t> strides; // 跨度

        uint64_t expansionSize = 0; // 扩容后的尺寸
        uint64_t expansionBytes = 0; // 扩容后的字节数
        std::vector <int> expansionDims; // 预扩容的形状
        uint8_t *cpuData = nullptr; // 数据指针

	    void *cudaData = nullptr;
        bool cudaDataBorrowed = false; // cudaData points into another owner and should not be freed directly
        std::vector <void*> extraCudaData;
        std::vector <void*> extraCudaHalfData;

        void *deviceData = nullptr;
        std::vector <void*> extraDeviceData;

        DataDevice dataDevice = DataDevice::CPU;
        std::vector <int> dataDeviceIds;

        // 以下参数用于量化，对FLOAT数据不适用
        int perChannelAxis = -1; // 沿哪个轴分通道量化，-1代表没有分通道
        int group = -1, groupCnt = -1; // 分组量化，group代表组数，groupCnt代表每组有多少个元素，-1代表不使用分组量化

        // FP8的分组量化， [blockK, blockM]的小矩阵为一组
        int blockK = -1, blockM = -1;

        // 以下为每个通道/分组的量化参数
        // 1. 若不使用分通道量化，那么总组数 = 1
        // 2. 若使用分通道量化，那么总组数 = 通道数
        // 3. 若使用分组量化，那么总组数 = 通道数 * 组数(group)
        std::vector <LowBitConfig> perChannelsConfigs; // perChannelsConfigs[i]代表第i个通道的min, max; 如果没有分通道，perChannelsConfigs[0]代表全局min, max
        std::vector <float> scales, mins;
        std::vector <int> zeros;
        std::vector <int> weightSum; // 作为权重时，有时候需要存一些和加速计算

        std::vector <uint16_t> halfScales; // 某些量化方式使用float16的scales

        bool isModelWeight = false; // 是否是模型权重
        std::string name; // weightName
        std::string fileName;
        long long filePos;
        std::shared_ptr<FileMmap> mapFile;
        bool isDiskWeight = false; // 权重仅保留磁盘位置，计算时按需读取
        std::vector <DiskWeightPart> diskWeightParts;

        bool directMemory = false; // 直接分配/释放Memory，不经过缓存

        bool multiDeviceData = false;
        std::map <int, Data*> multiDeviceDatas;

        TensorParallelLayoutType tpLayout = TP_LAYOUT_NONE;
        int tpAxis = -1;
        std::vector <int> tpGlobalDims;
        std::map <int, std::vector <std::pair <int, int> > > tpRanges;
        TensorParallelLinearType tpLinearType = TP_LINEAR_NONE;
        TensorParallelPackType tpPackType = TP_PACK_NONE;
        int tpQHeads = 0;
        int tpKVHeads = 0;
        int tpHeadDim = 0;
        int tpSplitUnit = 0;

        int weightId;
        bool isRegistered = false;

        bool isGGUFData = false; // gguf格式的数据
        void *ggmlTensor = nullptr;
        int ggmlType = -1;
        bool IsRepacked = false;
        bool disableGGUFRepack = false;
        bool forceGGUFFp32Dequant = false;

        std::vector <uint8_t*> numasData; // numa数据
        bool isPinned = false; // 是否使用pinned memory (page-locked)

        std::vector <int> cpuIntDatas; // 锁定在cpu上的int数据
        
        Data () {};

        Data (DataType type);

        Data (DataType type, const std::vector <int> &dims); // 构造函数

        Data (DataType type, int ggmlType, const std::vector <int> &dims); // ggml类型

        Data (DataType type, const std::vector <int> &dims, DataDevice device, void *ptr); // 构造函数，使用已有数据地址的Fake data

        // 构造函数，创建好之后从data复制数据
        // data中是原始数据，如果type不是float那么需要量化
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        ~Data(); // 析构函数

        Data (const Data &ori); // 深拷贝

        void CreateFromOriData(WeightType weightType, DataType oriDataType, uint8_t *oriData, float *oriMins, float *oriScales, 
                int groupCnt = -1, int blockK = -1, int blockM = -1); // 从oriData中创建

        void CopyFrom(const Data &ori); // 复制

        void FakeFrom(const Data &ori, size_t offset); // 将data指针指向ori的data + offset，delete时不销毁

        uint64_t GetBytes() const; // 获取总字节数

        void Allocate(); // 分配内存

        void Allocate(bool zero); // 分配内存，zero=false 时跳过清零

        void Allocate(float v); // 分配内存并初始化

        void Expansion(const std::vector <int> &dims); // 预扩容到相应尺寸

        void MallocSpace(uint64_t size, bool zero = true); // 在设备上分配

        void FreeSpace(); // 回收设备上的内存

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸

        void Reshape(const std::vector <int> &dims); // 更改尺寸,但不修改数据

        uint64_t Count(int i) const; // dims[i] * strides[i]

        void PrintShape() const; // 输出形状

        std::vector<int> Shape() const; 

        void Print(const std::string &name = "") const; // 输出

        void CalcWeightSum(); // 计算WeightSum

        void ToDevice(DataDevice device, bool copyData = true); // 移动到指定device

        void ToDevice(DataDevice device, const std::vector <int> &deviceIds, bool copyData = true); // 移动到指定device

        void ToDevice(void *device, bool copyData = true);
        bool MoveCudaStorageToHost(DataOffloadRecord &record, std::string *error = nullptr);
        bool RestoreCudaStorageFromHost(const DataOffloadRecord &record, std::string *error = nullptr);
        bool ReleaseCudaAuxiliaryStorage(std::string *error = nullptr);
        bool ReleaseCudaStorageWithoutHostCopy(std::string *error = nullptr);


        void ToCudaTemporary(const std::vector <int> &deviceIds, bool copyData, void *stream = nullptr); // 临时移动到cuda

        void FreeCudaTemporary(const std::vector <int> &deviceIds, bool copyData); // 销毁临时移动到cuda的数据

        void Repack(); // 重新打包数据，便于计算

        void SetMapFile(std::shared_ptr<FileMmap> file) {
        	mapFile = file;
        }

        void SetKVCache();

        // 计算形成Fastllm格式需要多少Bytes
        uint64_t GetFastllmFormateBytes();

        // 导出成Fastllm格式
        void ExportFastllmFormat(uint8_t *bytes);

        // 从Fastllm格式中创建
        void CreateFromFastllmFormat(uint8_t *datas, uint64_t len);

        // 普通类型：直接返回dataType, GGUF类型：返回dataType + ggmltype
        DataType GetDataType();

        // 当前权重作为linear的weight时，输入应该是什么类型
        DataType GetLinearActDataType(int batchSize);

        bool IsTensorParallel() const;
        bool IsTensorParallelReplicated() const;
        bool IsTensorParallelSharded() const;
        void ClearTensorParallelLayout();
        void ResetMultiDeviceState();
    };

    struct PagedPrefixCacheTierPayload {
        std::vector<uint8_t> bytes;
        size_t uncompressedBytes = 0;
        bool zstdCompressed = false;
        uint64_t checksum = 0;

        std::shared_ptr<HostCacheReservation> budgetReservation;
        bool accounted = false;
        ~PagedPrefixCacheTierPayload();
    };

    struct PagedPrefixCacheTierDiskRef {
        uint64_t offset = 0;
        size_t storedBytes = 0;
        size_t uncompressedBytes = 0;
        bool zstdCompressed = false;
        uint64_t checksum = 0;
        bool persistentArchive = false;
        std::string persistentRoot;
        uint64_t persistentGeneration = 0;
        PersistentPayloadRef persistentRef;
    };

    struct CacheTrieNode {
        int pageId = -1;
        long long timestamp = 0;
        uint64_t accessCount = 0;
        long long lastAccessTimestamp = 0;
        int depthPages = 0;
        int maxPrefixDepthPages = 0;
        std::unordered_map<uint64_t, CacheTrieNode*> children;
        CacheTrieNode *parent = nullptr;
        uint64_t edgeHash = 0;
        std::vector<int> edgeTokens;
        // 【上游BUMP勿回退】这个字段是"最小驻留时间"滞回的状态位, 删掉会退回
        // 抖动版本。含义: 该节点的物理页最近一次"成为 L1 常驻"的时刻
        // (steady_clock 毫秒, 0 表示未知)。Record() 与 MaterializeTrieNode()
        // 各自在写 pageId 时打点。
        // 为什么需要: 池子顶到预算天花板后, 取页路径每缺一页就淘汰一页, 刚从
        // L2/L3 上提回来的页可能在下一次分配就被重新踢下去 —— 上提做的 H2D +
        // 解压全白费, 表现为 CPU/磁盘层反复读写而命中率不涨。淘汰候选里跳过
        // "刚常驻不久"的页即可打断这个循环。
        long long residentSinceMs = 0;
        std::shared_ptr<PagedPrefixCacheTierPayload> tierPayload;
        std::shared_ptr<PagedPrefixCacheTierDiskRef> tierDisk;
    };

    // 一个带PageCache功能的Data，可以管理多个PageCache
    class PagedCacheManager : public Data {
        public:
            ~PagedCacheManager();
            enum PagedCacheManagerType {
                PAGED_CACHE_MANAGER_TYPE_KV_CACHE = 0,
                PAGED_CACHE_MANAGER_TYPE_MLP_CACHE = 1
            };

            int persistentId = -1;
            bool persistentRestoreAttempted = false;

            // 类型
            PagedCacheManagerType type;

            // 页长
            int pageLen;

            // 最大页数
            int maxPages;

            // 空闲页双池：freePages 不在 Trie 中，triePages 在 Trie 中但未被引用
            std::vector<int> freePages;
            std::vector<int> triePages;
            std::unordered_set<int> freePagesSet;
            std::unordered_set<int> triePagesSet;
            std::mutex pageIndexLocker;

            int FreePageCount() const { return (int)freePages.size() + (int)triePages.size(); }

            // 每个页面的使用时间戳
            std::vector<long long> pageTimestamp;
            long long currentTimestamp = 0;

            // 每个页面的引用计数
            std::vector<int> pageRefCount;

            // 【上游BUMP勿回退】滞回状态: 上一轮"水位批量回收"发生的时刻
            // (steady_clock 毫秒)。用于两轮回收之间的冷却期, 避免在天花板附近
            // 高频重复扫描 trie + D2H + zstd。freePages 真的见底时会绕过冷却。
            long long lastRecycleMs = 0;

            // 【上游BUMP勿回退】上一次"水位扩容"失败(池预算/显存不足)的时刻
            // (steady_clock 毫秒, 0 = 从未失败)。撞到天花板后进入退避期, 期间
            // 直接走回收路径, 不再每次取页都去撞同一堵墙。
            long long lastGrowFailureMs = 0;


            // Trie树缓存管理
            CacheTrieNode *trieRoot = nullptr;
            std::unordered_map<int, CacheTrieNode*> pageToTrieNode;

            void SetMaxPages(int maxPages);
            // 按需扩大页池：分配更大的底层缓冲区、拷贝旧页并释放旧缓冲，
            // 物理页号保持不变（pageIndex 无需迁移），新页加入空闲池。
            // 与 SetMaxPages 不同，Grow 保留已用页的引用/Trie 状态。
            void Grow(int newMaxPages);
            int GetUnusedPageIndex(bool pick);
            void EvictTrieSubtree(CacheTrieNode *node);
            int GetUnusedPageIndexLocked(
                bool pick,
                const std::unordered_set<int> *protectedPages);
            // 淘汰一个最冷的前缀页进 freePages(下沉优先),
            // 供取页路径与水位批量回收共用。调用方须持 pageIndexLocker。
            int EvictOneColdPageLocked(
                const std::unordered_set<int> *protectedPages,
                bool ignoreResidencyGuard = false);
            // 批量淘汰: 一次补够 wantPages 页, 而不是"缺一页放一页"。
            // 返回实际淘汰到 freePages 的页数。调用方须持 pageIndexLocker。
            int EvictColdPagesLocked(
                const std::unordered_set<int> *protectedPages,
                int wantPages);
            bool PageOutTrieNode(CacheTrieNode *node);
            // L2(CPU 内存) -> L3(磁盘) 轮转: 把 CPU 层最冷的载荷写盘并释放其
            // 内存, 直到腾出 bytesNeeded 字节。allowDrop=true 时, 写盘失败的
            // 载荷才允许直接丢弃(仅用于硬内存压力)。
            // 调用方须持 pageIndexLocker。
            uint64_t RotateCpuTierToDiskLocked(
                uint64_t bytesNeeded, bool allowDrop);
            uint64_t EvictCpuTierPayloads(uint64_t bytesNeeded);
            bool MaterializeTrieNode(
                CacheTrieNode *node,
                const std::unordered_set<int> &protectedPages);
            void ReleasePageIndex(int pageIndex);
            void ReleasePageIndices(const std::vector<int> &pageIndices);
            void Pick(std::vector<int> &pageIds);

            static uint64_t HashTokenPage(const int *tokens, int len);
            void Record(const std::vector<int> &tokens, const std::vector<int> &pages);
            void Query(const std::vector<int> &tokens, std::vector<int> &cachedPageIds);

            bool ExportPersistentRecords(
                std::vector<PersistentPayloadRecord> &records,
                uint64_t &pages,
                std::string *error);
            bool ImportPersistentRecords(
                const std::filesystem::path &root,
                uint64_t generation,
                const std::vector<uint8_t> &trieBytes,
                const std::vector<PersistentPayloadRef> &refs,
                std::string *error);
    };
    // Return an absolute physical-page target that covers the current
    // deficit while limiting small incremental growth to 128 pages.
    int GetPagedCacheGrowthTarget(
        int physicalPages, int maxPages, int additionalPages);

#ifdef USE_CUDA
    // CUDA graphs retain paged-cache base pointers. Graph launch and pool
    // relocation share this mutex; the version changes after every relocation.
    std::mutex &GetPagedCacheCudaStorageMutex();
    uint64_t GetPagedCacheCudaStorageVersion();
#endif


    bool PagedPrefixCacheCpuTierEnabled();
    bool PagedPrefixCacheDiskTierEnabled();
    uint64_t GetPagedPrefixCacheGpuHitPages();
    uint64_t GetPagedPrefixCacheCpuHitPages();
    uint64_t GetPagedPrefixCacheCpuTierBytes();
    uint64_t GetPagedPrefixCacheDiskWriteBytes();
    uint64_t GetPagedPrefixCacheDiskLiveBytes();
    uint64_t GetPagedPrefixCacheDiskReadBytes();
    uint64_t GetPagedPrefixCacheDiskHitCount();
    double GetPagedPrefixCacheDiskReadMegabytesPerSecond();
    void ObservePagedPrefixCacheRecompute(
        size_t tokens, double seconds);
    double GetPagedPrefixCacheRecomputeTokensPerSecond();
    double GetPagedPrefixCacheZstdDecompressMegabytesPerSecond();
    uint64_t GetPagedPrefixCacheZstdCompressCalls();
    uint64_t GetPagedPrefixCacheZstdCompressInputBytes();
    uint64_t GetPagedPrefixCacheZstdCompressOutputBytes();
    double GetPagedPrefixCacheZstdCompressSeconds();
    uint64_t GetPagedPrefixCacheZstdDecompressCalls();
    uint64_t GetPagedPrefixCacheZstdDecompressInputBytes();
    uint64_t GetPagedPrefixCacheZstdDecompressOutputBytes();
    double GetPagedPrefixCacheZstdDecompressSeconds();
    struct CacheSnapshotMetadata {
        long long cacheUid = 0;
        bool isKVCache = false;
        bool isLinearAttention = false;
        bool isLinearAttentionTransposed = false;
        int pageLen = 0;
        int lastPageLen = 0;
        DataType dataType = DataType::FLOAT32;
        std::vector<int> dims;
        std::vector<uint64_t> strides;
        uint64_t expansionSize = 0;
        uint64_t expansionBytes = 0;
        std::vector<int> expansionDims;
        DataDevice dataDevice = DataDevice::CPU;
        std::vector<int> dataDeviceIds;
        TensorParallelLayoutType tpLayout = TP_LAYOUT_NONE;
        int tpAxis = -1;
        std::vector<int> tpGlobalDims;
        std::map<int, std::vector<std::pair<int, int> > > tpRanges;
        TensorParallelLinearType tpLinearType = TP_LINEAR_NONE;
        TensorParallelPackType tpPackType = TP_PACK_NONE;
        int tpQHeads = 0;
        int tpKVHeads = 0;
        int tpHeadDim = 0;
    };

    // Shared ownership keeps copied snapshot descriptors from unlinking a
    // spill file while another descriptor can still restore it.
    struct PagedCacheCpuSnapshotDiskFile {
        std::string path;
        size_t bytes = 0;
        uint64_t checksum = 0;

        bool ownsPath = false;
        ~PagedCacheCpuSnapshotDiskFile();
    };

    struct PagedCacheCpuSnapshotPart {
        bool valid = false;
        CacheSnapshotMetadata metadata;
        PagedCacheManager *manager = nullptr;
        int pageCount = 0;
        size_t pageBytes = 0;
        size_t uncompressedBytes = 0;
        bool zstdCompressed = false;
        bool zstdContentChecksum = false;
        std::vector<uint8_t> compressedBytes;
        std::vector<uint8_t> bytes;
        std::shared_ptr<PagedCacheCpuSnapshotDiskFile> diskFile;

    };

    struct PagedCacheCpuSnapshot {
        bool valid = false;
        bool multiDeviceData = false;
        PagedCacheCpuSnapshotPart single;
    };
    CacheSnapshotMetadata CaptureCacheSnapshotMetadata(
        const Data &cache);

    // Copies the physical bytes of a single-device paged cache to pageable
    // host storage without changing page ownership. Multi-device roots are
    // rejected so callers can retain the resident cache as a safe fallback.
    bool SnapshotPagedCacheToCpu(
        const Data &cache,
        PagedCacheCpuSnapshot &snapshot,
        std::string *error = nullptr);

    // Compresses each physical-page byte stream only when zstd produces a
    // smaller representation. Unsupported builds and incompressible input
    // remain byte-for-byte raw and restorable.
    bool TryCompressPagedCacheCpuSnapshot(
        PagedCacheCpuSnapshot &snapshot,
        int compressionLevel = 1,
        std::string *error = nullptr);
    struct PagedCacheCpuSnapshotZstdMetrics {
        uint64_t compressCalls = 0;
        uint64_t compressInputBytes = 0;
        uint64_t compressOutputBytes = 0;
        uint64_t compressNanoseconds = 0;
        uint64_t decompressCalls = 0;
        uint64_t decompressInputBytes = 0;
        uint64_t decompressOutputBytes = 0;
        uint64_t decompressNanoseconds = 0;
    };
    PagedCacheCpuSnapshotZstdMetrics
        GetPagedCacheCpuSnapshotZstdMetrics();
    bool PagedCacheCpuSnapshotZstdAvailable();
    size_t GetPagedCacheCpuSnapshotStoredBytes(
        const PagedCacheCpuSnapshot &snapshot);

    // ---- 前缀缓存可观测性(FASTLLM_PREFIX_CACHE_STATS=1 启用, 默认关)----
    // 纯测量, 不改变任何缓存行为。请求级打印每请求命中/未命中;
    // 每 64 次请求事件节流打印一层占用/逐出/记录被拒汇总;
    // 累计计数器经 GetPrefixCacheStatsSnapshot() 暴露给 /props。
    struct PrefixCacheStats {
        // 请求级
        uint64_t requests = 0;            // 做了前缀缓存查询的请求数
        uint64_t hitRequests = 0;         // 有任何命中的请求数
        uint64_t queryTokens = 0;         // 参与查询的 token 总数
        uint64_t hitTokens = 0;           // 命中 token 总数
        uint64_t hitTokensMemTrie = 0;    // 命中来源分层
        uint64_t hitTokensCpuTier = 0;
        uint64_t hitTokensDisk = 0;
        // 未命中原因分布(请求数)
        uint64_t missNoRecord = 0;        // trie 中无该前缀(未记录过)
        uint64_t missEvicted = 0;         // 记录过但页/载荷已被逐出
        uint64_t missBelowThreshold = 0;  // MIN_TOKENS/MIN_HITS 门槛不满足
        uint64_t missGenerationMismatch = 0;  // generation/布局不匹配
        uint64_t missRestoreFailed = 0;   // 有命中但恢复(paged/extra)失败
        uint64_t missOther = 0;
        // 记录被拒原因分布(PageOutTrieNode / Record 路径)
        uint64_t recordAccepted = 0;
        uint64_t recordRejectedMinHitsTokens = 0;  // accessCount<minHits && tokens<minTokens
        uint64_t recordRejectedCapacity = 0;       // 层容量不足
        uint64_t recordRejectedNoSpace = 0;        // 分配/空闲空间失败
        uint64_t recordRejectedOther = 0;
        // 逐出
        uint64_t evictTrieNodes = 0;      // EvictTrieSubtree 逐出的节点数
        uint64_t evictCpuTierCalls = 0;   // EvictCpuTierPayloads 调用次数
        uint64_t evictCpuTierBytes = 0;   // CPU 层逐出释放字节
        // 层占用快照(调用 Get 时采样, 非累计)
        uint64_t memTrieResidentBytes = 0;
        uint64_t cpuTierResidentBytes = 0;
        uint64_t diskResidentBytes = 0;
        // ---- 记录路径(TryRecordPagedCache 链)跳过原因 ----
        uint64_t recordCalls = 0;            // TryRecordPagedCache 被调次数
        uint64_t recordSkipLinearBounded = 0;// linear/bounded 且 extra 未成功
        uint64_t recordSkipNoPagedLen = 0;   // 无任何层有 paged 长度
        uint64_t recordSkipNoUnbounded = 0;  // 无 unbounded 层 -> reusable=0
        uint64_t recordSkipBoundedShort = 0; // bounded 层页链不足 reusable
        uint64_t recordSkipManagerInvalid = 0;// manager null/非 KV 类型/页空
        uint64_t recordLayersOk = 0;         // Record() 实际执行层数
        uint64_t recordManagerNoPages = 0;   // Record 内 numPages=0
        // qwen3_5 linear-attention extra 子路径
        uint64_t extraCalls = 0;
        uint64_t extraOk = 0;
        uint64_t extraSkipDisabled = 0;      // 前缀缓存关/无 linear 层
        uint64_t extraSkipLenMisaligned = 0; // len<=0|>allTokens|未页对齐
        uint64_t extraSkipNoProgress = 0;    // len<=lastSnapshotLen
        uint64_t extraSkipInterval = 0;      // %snapshotInterval!=0
        uint64_t extraSkipSnapshotCopy = 0;  // linear 层快照拷贝失败
        uint64_t extraSkipMtp = 0;           // requireMtp 但 MTP cache 缺/不配
        // 【上游BUMP勿回退】GDN 递归状态位置与快照标称长度不一致而丢弃的次数。
        // 这个计数**不为 0 是正常的**(生成结束时的记录点几乎必然不页对齐),
        // 它存在的意义是: 一旦 extraOk 长期为 0 而这一项很大, 说明快照只在
        // 非对齐点被尝试记录 —— 即分块 prefill 的对齐记录钩子没生效。
        uint64_t extraSkipUnaligned = 0;     // GDN 状态位置未页对齐, 丢弃以免错位
    };
    void PrefixCacheStatsObserveRecordPath(const char *event);
    bool PrefixCacheStatsEnabled();
    void PrefixCacheStatsObserveRequest(
        int totalTokens, int hitTokens,
        const char *hitLayer,       // "mem-trie" | "cpu" | "disk" | nullptr(未命中)
        const char *missReason);    // "no-record" | "evicted" | "below-threshold" | "generation" | "restore-failed" | "other" | nullptr(有命中)
    void PrefixCacheStatsObserveRecord(bool accepted, const char *rejectReason);
    void PrefixCacheStatsObserveEviction(const char *kind, uint64_t nodesOrBytes);
    PrefixCacheStats GetPrefixCacheStatsSnapshot();

    // ---- 工具调用语法约束(ROOT CAUSE #3, 完整状态机) ----
    // 总开关: FASTLLM_TOOLCALL_GRAMMAR (默认开, =0 关 -> 退回打点式约束)
    bool ToolCallGrammarEnabled();
    // 逐步 trace: FASTLLM_TOOLCALL_TRACE=1 (打印状态/allowedIds 规模)
    bool ToolCallTraceEnabled();
    // 破损块 dump 目录: FASTLLM_TOOLCALL_TRACE_DIR (默认 EzraVastLLM/logs)
    struct ToolCallGrammarStats {
        // 解析器侧: 完整块 ParseBlock 尝试 / 失败(裸文本回退) / Flush 修复成功
        uint64_t blocksTotal = 0;
        uint64_t malformedTotal = 0;
        uint64_t repairedTotal = 0;
        // 引擎侧: 约束激活步数 / 被 mask 掉的候选 token 总数
        uint64_t constraintSteps = 0;
        uint64_t maskedTokens = 0;
    };
    // 解析器回调: parsedOk=false 计入 malformed; repaired=true 计入 repaired
    void ToolCallGrammarStatsObserveParse(bool parsedOk, bool repaired);
    void ToolCallGrammarStatsObserveConstraint(size_t allowedCount, size_t vocabSize);
    ToolCallGrammarStats GetToolCallGrammarStatsSnapshot();
    // 破损/修复时 dump 原始块到 trace 目录(jsonl, 追加); reason: "malformed"/"repaired"
    void ToolCallTraceDumpBlock(const char *reason, const std::string &block);

    // Atomically writes the currently stored raw or zstd stream, records a
    // checksum, and releases its resident vector. The shared disk descriptor
    // unlinks the transient file after the last snapshot owner is destroyed.
    bool SpillPagedCacheCpuSnapshotPartToDisk(
        PagedCacheCpuSnapshotPart &part,
        const std::string &path,
        std::string *error = nullptr);
    bool LoadPagedCacheCpuSnapshotPartFromDisk(
        const PagedCacheCpuSnapshotPart &part,
        std::vector<uint8_t> &storedBytes,
        std::string *error = nullptr);

    // Allocates fresh physical pages in the original manager and restores the
    // logical page order and packed bytes captured by SnapshotPagedCacheToCpu.
    // The destination must not own live storage.
    bool RestorePagedCacheFromCpu(
        const PagedCacheCpuSnapshot &snapshot,
        Data &cache,
        std::string *error = nullptr);

    // Releases every paged reference reachable from cache exactly once, then
    // resets it to an empty, non-owning cache descriptor.
    void ReleasePagedCacheStorage(Data &cache);

    struct PartitionLinkNode {
        std::pair <int, int> *cur = nullptr;
        PartitionLinkNode *next = nullptr;
        PartitionLinkNode *prev = nullptr;
        int id = -1;

        PartitionLinkNode *Skip(int t) {
            PartitionLinkNode *ret = this;
            while (t--) {
                if (ret != nullptr) {
                    ret = ret->next;
                }
            }
            return ret;
        }
    };

    struct Tokenizer {
        enum TokenizerType {
            BPE = 0,
            NORMAL = 1,
            QWEN = 2,
            GLM = 3,
            BERT = 4,
            UNIGRAM = 5
        };

        struct TrieNode {
            int tokenId;
            float score;
            std::map <int, TrieNode*> next;
            TrieNode();
        };
        struct Symbol {
            TrieNode *node;
            char *s;
            int pos, len;
            int prev, next;
            int fixId;

            Symbol (Tokenizer::TrieNode *node,
                    char *s, int pos, int len,
                    int prev, int next, int fixId) {
                this->node = node;
                this->s = s;
                this->pos = pos;
                this->len = len;
                this->prev = prev;
                this->next = next;
                this->fixId = fixId;
            }
        };
        struct SymbolPairs {
            float score;
            int l, r, size;

            SymbolPairs(float score, int l, int r, int size) {
                this->score = score;
                this->l = l;
                this->r = r;
                this->size = size;
            }
        };

        friend bool operator < (const SymbolPairs &a, const SymbolPairs &b) {
            return a.score < b.score || (a.score == b.score && a.l > b.l);
        }

        json11::Json tokenizerConfig;
        std::string chatTemplate = "";

        TrieNode *root;

        TrieNode *specialRoot = nullptr;

        TokenizerType type = TokenizerType::BPE;

        // 【上游BUMP勿回退】下面这组"预分词(pre-tokenizer)"成员不能删。
        //
        // 线上故障(与 merges 那处同源, 表现同样是"胡言乱语 / 路径抄不对"):
        //   /home/ezra/Documents/Proto-UI  ->  /home/eze/Documents/PotouI
        //
        // 根因: 上游 Tokenizer::Encode 只按 special token 切分, 剩下的**整段**
        // 文本直接丢进 BytePairEncode。而 HF / llama.cpp / vLLM 都是先用
        // GGUF 里 `tokenizer.ggml.pre` 指定的**预分词正则**把文本切成小块,
        // BPE 只在块内合并, 永远跨不出块边界。少了这一步, BPE 就能跨词、
        // 跨数字、跨标点任意合并, 产生训练时根本不存在的 token 序列 ——
        // 合法但**非规范**, 模型在分布外推理, 逐字复述必然走样。
        //
        // 现场特征: 单词级别正确率恢复(merges 修好之后)但整句 exact match 上不去;
        // 出错的地方集中在"字母紧挨数字/标点/连字符"以及长数字上。
        //
        // 正确做法: 读 GGUF 的 `tokenizer.ggml.pre`, 按对应正则切块再逐块 BPE。
        // qwen35 的正则(取自 llama.cpp src/llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35):
        //   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
        //   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
        //   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
        // 注意 `\p{N}` 是**单独一位**成块 —— 数字逐位切, 这点最容易漏。
        //
        // std::regex 不支持 \p{L} 这类 Unicode 属性, 所以实现方式与 llama.cpp
        // 一致: 查码点类别表 + 手写状态机, 见 src/tokenizer.cpp 的 PreTokenizeSplit
        // 和 include/utils/unicode_categories.h。
        //
        // 未知/缺失的 pre 值必须**保持不切分的旧行为并打印提示**, 绝不静默改行为。
        enum PreTokenizerType {
            PRE_TOKENIZER_NONE = 0,    // 不切分 —— 上游原始行为
            PRE_TOKENIZER_QWEN2 = 1,   // \p{L}+        (qwen2 / deepseek-r1-qwen ...)
            PRE_TOKENIZER_QWEN35 = 2   // [\p{L}\p{M}]+ (qwen35)
        };

        PreTokenizerType preTokenizerType = PreTokenizerType::PRE_TOKENIZER_NONE;
        std::string preTokenizerName = "";   // GGUF 里 tokenizer.ggml.pre 的原值, 仅用于日志

        int blankRepeatCount = 0;     // 重复空格替换数量，0表示不替换
        bool addDummyPrefix = true;   // 是否在首位添加空格
        bool removeExtraWhitespaces = true;   // 是否将多个空格合并为一个
        bool byteAsChar = false;  // 是否将byte变为展示字符

        std::unordered_map <int, std::string> tokenToStringDict;
        std::unordered_map <int, float> tokenToScoreDict;
        std::unordered_map <std::string, int> stringToTokenDict;
        std::vector <std::string> specialTokens;

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::unordered_map <wchar_t, wchar_t> byteCharDict;
        std::unordered_map <wchar_t, wchar_t> charByteDict;
#ifdef USE_SENTENCEPIECE
        std::unique_ptr<sentencepiece::SentencePieceProcessor> spProcessor;
#endif

        Tokenizer ();

        ~Tokenizer();

        void Clear(); // 清空分词器

        void TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q); // 插入备选symbol

        int GetRank(std::vector <Symbol> &symbols, PartitionLinkNode *cur, int skip);

        int GetRank(std::vector<Symbol> &symbols,  std::vector<std::pair<int, int>> &partitions, int idx, int skip);

        void Insert(const std::string &s, int tokenId, float score = 1.0f); // 插入一个token

        void SetSpecialTokens(const std::map <std::string, int> &specialTokens); // 设置需要优先处理的特殊token

        void SetTokenizerConfig(const json11::Json &config);

        std::string Normalize(const std::string &ori, const bool addDummyPrefix=true); // 字符规范化

        // 【上游BUMP勿回退】设置 GGUF 的 tokenizer.ggml.pre。
        // 认识的值 -> 启用对应预分词正则; 空值/不认识的值 -> 保持"不切分"的旧行为,
        // 并打印一行提示(不要改成静默忽略: 静默 = 线上悄悄输出垃圾)。
        void SetPreTokenizer(const std::string &pre);

        // 【上游BUMP勿回退】按预分词正则把文本切成块, 每块之后各自独立走 BPE。
        // 未启用预分词时原样返回 {s}, 与上游行为逐字节一致。
        std::vector <std::string> PreTokenizeSplit(const std::string &s) const;

        Data Encode(const std::string &s); // 编码

        std::string Decode(const Data &data); // 解码

        std::string DecodeTokens(const std::vector <int> &tokens); // 解码

        int GetTokenId(const std::string &s); // 获取s对应的tokenid

        std::string GetToken(int id); // 获取id对应的token
    private:
        std::vector<float> BytePairEncode(const std::string &s);

        std::vector<float> UnigramEncode(const std::string &s);
    };

    std::string GetModelTypeFromFile(const std::string &fileName);

    struct WeightMap {
        int versionId = 2;

        Tokenizer tokenizer;

        std::map <std::string, std::string> dicts;

        std::unordered_map <std::string, Data> weight;

        std::map <std::string, std::map <std::string, std::string>> peftDict;

        std::set <std::string> embeddingNames;

        std::set <std::string> linearNames;

        void LoadFromFile(const std::string &fileName); // 从文件读取

        void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型, bit = 0代表直接存

        void AddTokenizerWord(const std::string &key, int value, float score); // 增加一个词

        void AddDict(const std::string &key, const std::string &value); // 插入一个词条

        void AddAdapterDict(const std::string &name, const std::string &key, const std::string &value);

        void AddEmptyWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType);

        void AddEmptyGGMLWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType, int ggmlType);

        void AddWeight(const std::string &key, const std::vector <int> &dims,
                       DataType dataType, WeightType weightType, DataType oriDataType, uint8_t *oriData,
                       int groupCnt = -1); // 插入一个权重

        void ReleaseWeight(); // 释放所有权重占用的空间

        void AddQLinearWeight(const std::string &key, const std::vector <int> &dims,
                              int bit, float *scales, uint8_t *oriData); // 插入一个Qlinear层的权重，量化规则为float value = scales * oriData

        WeightType GetWeightType(const std::string &key); // 获取某个权重的类型（若未判断出来，则为None)

        Data &operator [] (const std::string &key);
    };

    void *GetExecutor();

    void SetCurrentThreadExecutor(void *executor);

    bool HasDeviceType(const std::string &deviceType);

    void ClearProfiler();

    void PrintProfiler();

    void ApplyDeviceMap(const std::map <std::string, int> &deviceMap, int current, int total); // 执行到了current, 一共total，使用deviceMap切换设备

    // 采样前把 NaN/Inf 压成 -1e30f, 返回被修正的个数; where 只用于日志。
    // 不做这一步的话 NaN 会劫持比较器, 输出退化成 token 0 ('!') 的长串。
    // 全部 layer 的 paged KV 池当前占用的显存字节(用于与 nvidia-smi 对账)
    uint64_t GetPagedPoolCudaBytes();

    // 显存对账明细。nvidia-smi 的"已用"应当能被 pool + allocBusy + allocFree
    // 解释, 剩下的就是 other(线上曾有 7.6GB 无主, 只能靠猜)。
    // 注意 cudaMemGetInfo 的 free **不含** allocFree —— fastllm 自己的缓存
    // 分配器攥着的空闲块对 CUDA 而言仍是"已用"。
    struct VramBreakdown {
        uint64_t usedBytes = 0;
        uint64_t totalBytes = 0;
        uint64_t pagedPoolBytes = 0;
        uint64_t allocBusyBytes = 0;
        uint64_t allocFreeBytes = 0;
        uint64_t graphPinnedBytes = 0;
    };
    bool GetVramBreakdown(VramBreakdown &out);

    int SanitizeLogitsForSampling(float *base, int count, const char *where);
    long long GetNonFiniteLogitSteps();

    int LLMSamplingOnly(Data &logits, int outerOffset, const GenerationConfig &config);

    int LLMSampling(Data &logits, int outerOffset,
                    const GenerationConfig &config, const LastTokensUnit &tokens); // 对logits里[outerOffset * vocabSize, (outerOffset + 1) * vocabSize]做Sampling

    void ToDataType(const Data &input, DataType dataType);
    void ToDataType(const Data &input, Data &output, DataType dataType);

    // 与 ToDataType(input, dataType) 行为相同，但强制只在 CPU device 上完成转换。
    // 适用于希望权重保留在 CPU 上、避免被算子派发逻辑迁移到 GPU 的场景（例如不开 cuda_embedding 时的 embedding 权重）。
    void ToDataTypeForceCPU(const Data &input, DataType dataType);

    void CopyKVCache(Data &oldCache, Data &newCache, int oldBsStart, int newBsStart, int bs, int offset);

    bool CanRunMergeMOE(const Data &input, std::vector <Data*> &biass);
    bool CanRunMergeMOE(const Data &input, std::vector <Data*> &weights,
                        std::vector <Data*> &biass);
    enum MoeGateType {
        MoeGateSwiglu = 0,
        MoeGateGeglu = 1
    };
    void MergeMOE(const Data &input, const Data &index, const Data &score, std::vector <Data*> &weights, std::vector <Data*> &biass,
                Data &w1, Data &w2, Data &w3, Data &curInput, Data &curOutput,
                float sharedScale, Data &output, int layer = 0, MoeGateType gateType = MoeGateSwiglu,
                bool expertParallel = false, float swigluLimit = 0.0f,
                bool deepSeekV4Mode = false,
                Data *pairedReduceInput = nullptr,
                const Data *allowedExpertMask = nullptr);

    void FusedMOE(const Data &input, const Data &index, const Data &score,
                Data &gate, Data &up, Data &down, Data &w1,
                Data &output, int layer = 0, MoeGateType gateType = MoeGateSwiglu, float swigluLimit = 0.0f);
    
    void MergeMLA(Data &qNope, Data &qPe, Data &kvCache, Data &peCache, const Data &mask, Data &output, float softmaxScale);

    // MLA with paged KV cache: kvCache (kpe) and peCache (ckv) are stored in paged form (isPagedKVCache, pageIndex, lastPageLen, pagedKVCacheData).
    void MergeMLAPaged(Data &qNope, Data &qPe, Data &kvCachePaged,
                       Data &peCachePaged, Data &output,
                       float softmaxScale, int kvLen = -1);

    void Attention(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output,
                   int group, float scale, int attentionType);

    void AttentionBatch(std::vector <Data*> &q, std::vector <Data*> &k, std::vector <Data*> &v,
                        std::vector <Data*> &mask, std::vector <Data*> &output,
                        int group, float scale, int attentionType);
    
    void Conv1DPerChannel(const Data &input, Data &weight, Data &bias, int inputChannels, int outputChannels, 
            int kernel, int stride, int pad, Data &output);

    void Conv2D(const Data &input, Data &weight, Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, Data &output);

    void Embedding(const Data &input, Data &weight, Data &output);

    void EmbeddingDirect(const Data &input, Data &weight, Data &output);

    void RMSNorm(const Data &input, const Data &weight, float eps, Data &output);

    void RMSNormPart(const Data &input, const Data &weight, float eps, int start, int end, Data &output);

    // Kimi-K3 operators.  These are dispatched through the regular FastLLM
    // executor; the CPU backend is the first implementation.
    void KimiK3RMSNorm(const Data &input, const Data &weight, float eps,
                       Data &output);

    void KimiK3CausalConv1D(const Data &input, const Data &weight,
                           int kernelSize, Data &output);

    void KimiK3CausalConv1D(const Data &input, const Data &weight,
                           int kernelSize, Data &cache, Data &output);

    // Updates the packed Q/K/V short-convolution cache from a prefix of the
    // projected inputs without evaluating convolution outputs.
    void KimiK3UpdatePackedConvCache(
            const Data &q, const Data &k, const Data &v,
            int history, int tokens, Data &cache);

    void KimiK3L2Norm(const Data &input, float eps, Data &output);

    void KimiK3RecurrentKDA(
            const Data &q, const Data &k, const Data &v,
            const Data &rawGate, const Data &rawBeta,
            const Data &aLog, const Data &dtBias, float lowerBound,
            Data &state, Data &output, Data &decay, Data &beta);

    // Inference only consumes the recurrent output and updated state.  Avoid
    // materializing the full-sequence float32 decay/beta diagnostics on that
    // path while retaining KimiK3RecurrentKDA for validation and tooling.
    void KimiK3RecurrentKDAOutputOnly(
            const Data &q, const Data &k, const Data &v,
            const Data &rawGate, const Data &rawBeta,
            const Data &aLog, const Data &dtBias, float lowerBound,
            Data &state, Data &output);

    // Replays only the recurrent-state transition for the first `tokens`
    // rows of a captured verification batch.
    void KimiK3RecurrentKDAUpdateState(
            const Data &k, const Data &v,
            const Data &rawGate, const Data &rawBeta,
            const Data &aLog, const Data &dtBias, float lowerBound,
            int tokens, Data &state);

    void KimiK3RMSNormSigmoidGate(
            const Data &input, const Data &gate, const Data &weight,
            float eps, Data &output);

    void KimiK3AttnRes(
            const Data &prefixSum, const Data &blockResidual,
            const Data &projection, const Data &norm, float eps,
            Data &output);

    void KimiK3SiTUAndMul(
            const Data &gate, const Data &up, float beta,
            float linearBeta, Data &output);

    void KimiK3RoutedExperts(
            const Data &input, const Data &index, const Data &score,
            std::vector<Data*> &w1s, std::vector<Data*> &w2s,
            std::vector<Data*> &w3s, float beta, float linearBeta,
            Data &output);

    void KimiK3CausalAttention(
            const Data &q, const Data &k, const Data &v,
            float scale, Data &output);

    void LayerNorm(Data &input, Data &gamma, Data &beta, int axis, Data &output);

    void Linear(Data &input, Data &weight, const Data &bias, Data &output, bool keepTpReplicated = false);

    void LinearAdd(const Data &input, const Data &weight, const Data &bias, Data &middle, Data &output);

    bool CanRunLinearAdd(const Data &input, const Data &weight, const Data &bias, const Data &output);

    void SwigluLinearAdd(const Data &input, const Data &weight, const Data &bias, Data &middle, Data &output);

    bool CanRunSwigluLinearAdd(const Data &input, const Data &weight, const Data &bias, const Data &output);

    void LinearSwiglu(const Data &input, const Data &weight, const Data &bias, Data &middle, Data &output);

    bool CanRunLinearSwiglu(const Data &input, const Data &weight);

    enum LinearExType {
        ExTypeNone = 0,
        ExSwiglu = 1,
        ExGelu = 2,
        ExSilu = 3
    };
    
    bool CanRunLinearEx(LinearExType exType);

    bool CanRunMergeAttention();
    
    void MergeAttention(Data &input, Data &weight0, Data &bias0, Data &weight1, Data &bias1, 
        bool doQKNorm, Data &qNorm, Data &kNorm, float eps,
        Data &qkv, Data &q, Data &k, Data &v,
        int qNum, int kvNum, int headDim, int rotDim, float attentionScale,
        const Data &positionIds, Data &sinData, Data &cosData,
        std::vector <Data*> &keys, std::vector <Data*> &values, std::vector <Data*> &masks, 
        Data &output);

    bool CanRunMLP();

    void MLP(Data &input, Data &weight0, const Data &bias0, Data &weight1, const Data &bias1, 
            Data &w1, Data &w2, Data &w3, Data &output); // mlp

    void LinearEx(Data &input, Data &weight, const Data &bias, Data &output,
                    LinearExType exType); // 扩展Linear，可以接后续操作

    void Split(const Data &input, int axis, int start, int end, Data &output);

    void Repeat(const Data &input, int axis, int repeatTimes, Data &output);

    void Copy(const Data &input, Data &output);

    void DeepSeekV4HcPre(const Data &input, Data &hcFn, Data &hcScale, Data &hcBase,
                         int hcMult, int sinkhornIters, float eps, float normEps,
                         Data &output, Data &post, Data &comb);

    void DeepSeekV4HcPost(const Data &input, const Data &residual, const Data &post, const Data &comb, Data &output);

    void ScaleQRatory(Data &q, float eps, int ropeDim, float ropeBase, int startPos,
                      int originalSeqLen, float ropeFactor, int betaFast, int betaSlow);

    void DeepSeekV4RotaryQuant(Data &x, int ropeDim, float ropeBase, int startPos,
                               int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                               int quantDim, int blockSize, int posStep = 1);

    void DeepSeekV4SparseAttention(const Data &q, const Data &kv, Data &attnSink,
                                   int windowSize, int ropeDim, float ropeBase,
                                   int startPos, float softmaxScale, Data &output,
                                   int compressRatio, int originalSeqLen,
                                   float ropeFactor, int betaFast, int betaSlow,
                                   int prefixLen,
                                   const Data *compressedTopK = nullptr);

    void DeepSeekV4SparseAttentionDecodeCached(
            const Data &q, const Data &windowKV, const Data &compressedKV,
            Data &attnSink, int windowSize, int startPos,
            int compressedCount, int ropeDim, float ropeBase,
            float softmaxScale, Data &output, int originalSeqLen,
            float ropeFactor, int betaFast, int betaSlow,
            const Data *compressedTopK = nullptr);

    void DeepSeekV4IndexerTopK(const Data &q, const Data &weights,
                               const Data &compressedKV, int topK,
                               int compressRatio, int ropeDim, float ropeBase,
                               int startPos, int originalSeqLen,
                               float ropeFactor, int betaFast, int betaSlow,
                               Data &output);

    void DeepSeekV4WoA(Data &o, Data &woA, int groups, int oRank, Data &output);

    void DeepSeekV4BuildCompressedKVFromRaw(const Data &kv, const Data &score,
                                            Data &ape, Data &normWeight,
                                            int rawTokenBase, int rawLen,
                                            int blockStart, int blockCount,
                                            int compressRatio, int headDim,
                                            int ropeDim, float ropeBase,
                                            float ropeFactor, int betaFast,
                                            int betaSlow, int originalSeqLen,
                                            bool overlap, bool preferCudaOutput,
                                            Data &cache, bool indexer = false);

    void Cat(const Data &input0, const Data &input1, int axis, Data &output);

    void Pad(const Data &input, int axis, int padSize, Data &output);

    void CatDirect(Data &input0, const Data &input1, int axis); // 直接把input1的数据拷贝到input0后面（需要input0提前扩容了足够的空间）

    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha = 1.0, int group = 1);

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha = 1.0, int group = 1);

    void Softmax(const Data &input, Data &output, int axis);

    void Silu(const fastllm::Data &input, fastllm::Data &output);

    void TanH(const Data &input, Data &output);

    void Relu(const Data &input, Data &output);

    void Sigmoid(const Data &input, Data &output);

    void Normalize(const Data &input, Data &output, int axis);

    void Exp(const Data &input, Data &output);

    void Gelu(const Data &input, Data &output);
    
    void GeluNew(const Data &input, Data &output);

    void Geglu(const fastllm::Data &input, fastllm::Data &output);

    void Swiglu(const fastllm::Data &input, fastllm::Data &output);

    void SwigluGptOss(const fastllm::Data &input, fastllm::Data &output);

    void MambaSoftplus(const Data &input, Data &aLog, Data &dtBias, Data &output);

    void SigmoidMambaSoftplus(Data &sigmoidInputOutput, const Data &softplusInput, Data &aLog, Data &dtBias, Data &softplusOutput);

    void Mul(const Data &input, float v, Data &output);

    void MulTo(Data &input0, const Data &input1); // input0 *= input1

    void CausalMask(Data &input, int base, float maskValue);

    void TransferAttn(Data &input);

    void GatedDeltaRulePrepareAttn(const Data &at, const Data &decayMask,
                                   Data &attn);
    void GatedDeltaRuleBuildDecay(Data &g, Data &decayMask);

    void GatedDeltaRuleApplyDecayMask(Data &attn, const Data &decayMask,
                                      int causalBase);

    void RecurrentGatedDeltaRule(Data &q, Data &k, Data &v, Data &g, Data &b, 
                                Data &last_recurrent_state, Data &core_attn_out, float qScale = 1.0f);

    void ChunkGatedDeltaRulePrefill(Data &q, Data &k, Data &v, Data &g,
                                Data &attn, Data &k_cumdecay,
                                Data &last_recurrent_state, Data &core_attn_out);

    void AddTo(Data &input0, const Data &input1, float alpha = 1.0); // input0 += input1 * alpha

    void AttentionMask(Data &input, const Data &mask, float maskValue); // 把input里对应位置mask中为1的部分变成maskValue

    void AttentionExtendedMask(Data &input, const Data &mask); // bert中的extended mask

    void AlibiMask(Data &input, const Data &mask, float maskValue); // alibi mask

    void Permute(const Data &input, const std::vector<int> &axis, Data &output); // 转置

    void PermuteSelf(const Data &input, const std::vector<int> &axis); // 转置

    void TopK(const Data &input, Data &output, int topK); // 求topk

    void SelectExpert(const Data &logits, Data &index, Data &score, int topk, bool needNorm = false, float routeScale = 1.0f, const Data *gateBias = nullptr); // MOE专家选择

    void RotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position

    void NearlyRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim, int positionStride = 1); // 2D position embedding, 相邻的维度旋转

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position embedding for llama，前后各一半的维度旋转

    void LlamaRotatePosition2DPart(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim, int part); // 2D position embedding for llama，前后各一半的维度旋转

    void RopeEncoding(Data &input, const Data &positionIds, int rotaryDim, float ropeTheta, float ropeScale); // RoPE encoding，直接用rope_theta和rope_scale计算，无需sin/cos缓存

    void Llama3RopeEncoding(Data &input, const Data &positionIds, int rotaryDim, float ropeTheta,
                            float factor, float originalMaxPosition,
                            float lowFreqFactor, float highFreqFactor);

    // YaRN RoPE encoding computed directly from positions, without a sin/cos cache.
    void YarnRopeEncoding(Data &input, const Data &positionIds, int rotaryDim, float ropeTheta,
                          float factor, float originalMaxPosition,
                          float betaFast, float betaSlow, float attentionFactor);

    void Qwen35InterleavedRope(Data &input, const Data &positionIds, int rotaryDim,
                               int sectionT, int sectionH, int sectionW,
                               float ropeTheta, float ropeScale,
                               int useYarn = 0, float yarnFactor = 2.0f,
                               float yarnAttentionFactor = 1.0f,
                               float yarnCorrectionLow = 0.0f,
                               float yarnCorrectionHigh = 1.0f); // Qwen3.5 interleaved MRoPE (可选 YaRN)

    // 在 qkv 拼接张量上融合执行 RMSNorm + RoPE（仅对 q 和 k 部分），v 不处理
    void QKVRMSNormRope(Data &qkv, Data &qNormWeight, Data &kNormWeight,
                        const Data &positionIds, int q_heads, int k_heads, int head_dim,
                        int rotaryDim, float eps, float ropeTheta, float ropeScale);

    // 融合 QKVRMSNormRope + Split KV + AppendPagedCacheBatch（K/V直接写入paged cache，Q单独输出）
    // qkv: [bs, seqlen, (q_heads + k_heads + v_heads) * head_dim]
    // qOutput: 输出Q，布局为 [bs * q_heads, seqlen, head_dim]（已做Permute）
    // pagedKCacheData / pagedVCacheData: paged cache manager (作为Data传入)
    // insertIndexs / insertPositions: 每个batch对应的page idx和page offset
    // batch: 逻辑batch数（= insertIndexs长度，decode时每个token对应一个batch）
    void QKVRMSNormRopeSplitAppendPagedCache(
        Data &qkv, Data &qNormWeight, Data &kNormWeight,
        const Data &positionIds, 
        Data &qOutput,
        Data &pagedKCacheData, Data &pagedVCacheData,
        Data &insertIndexs, Data &insertPositions,
        int q_heads, int k_heads, int head_dim,
        int rotaryDim, float eps, float ropeTheta, float ropeScale,
        int pageLen, int batch, bool doQKNorm = true, Data *lastPageLens = nullptr);

    void Step3p5QKVRMSNormRopeSplitAppendPagedCache(
        Data &qkv, Data &qNormWeight, Data &kNormWeight,
        const Data &positionIds,
        Data &qOutput,
        Data &pagedKCacheData, Data &pagedVCacheData,
        Data &insertIndexs, Data &insertPositions,
        int q_heads, int k_heads, int head_dim,
        int rotaryDim, float eps, float ropeTheta,
        bool useLlama3, float llama3Factor,
        float llama3OriginalMaxPosition,
        float llama3LowFreqFactor,
        float llama3HighFreqFactor,
        int pageLen, int batch, Data *lastPageLens = nullptr);

    void RepeatPenalty(Data &input, const Data &penalty, const Data &penaltyScale); // 重复惩罚

    void ApplyLognAttn(Data &input, const Data &lognAttn, const Data &positionIds);

    void CumSumLastDim(Data &input);

    void MakeDecayMask(Data &input, Data &output);

    void ApplyChunkDecayByLastLogG(Data &input, const Data &g);

    void MulBatch(std::vector <Data*> &input, float v, std::vector <Data*> &output);

    void SplitBatch(const Data &input, int axis, int part, std::vector <Data*> &outputs); // 将input沿着axis轴切开，每份axis上的尺寸为1，放到outputs里

    void CatBatch(std::vector <Data*> &input, int axis, Data &outputs); // 将input沿着axis轴合起来，每份axis上的尺寸为1，放到output里

    void MatMulBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, std::vector <Data*> &output, float alpha = 1.0);

    void MatMulTransBBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, std::vector <Data*> &output, float alpha = 1.0);

    void SoftmaxBatch(std::vector <Data*> &input, std::vector <Data*> &output, int axis);

    void CatDirectBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, int axis);

    void AppendKVCacheBatch(std::vector <Data*> &cache, const Data &input);

    void LoraLayer(Data &input, Data &weight, Data &loraA, Data &loraB, const Data &bias, Data &output, 
                   std::map <std::string, std::string> loraConfig);

    void IA3Layer(Data &input, Data &weight, Data &ia3_l, Data &bias, Data &output,
                  std::map <std::string, std::string> ia3Config);

    PagedCacheManager* AllocatePagedCacheManager(int layerIndex, 
        PagedCacheManager::PagedCacheManagerType type, 
        const Data &cacheData, 
        int pageLen =  -1, 
        int maxPages = -1);

    PagedCacheManager* GetPagedCacheManager(int layerIndex);

    void ClearAllPagedCacheManagers();

    bool ExportPersistentPagedCacheRecords(
        std::vector<PersistentPayloadRecord> &records,
        uint64_t &pages,
        std::string *error);
    void AttachPreparedPersistentPrefixCacheManagers();

    void AppendPagedCache(PagedCacheManager &pagedCacheManager, Data &cache, const Data &input);
    
    // 从batch个pastKey中生成AppendPagedCacheBatch所需要的insertIndexs和insertPositions
    // pastKeys: batch个pastKey的列表，每个元素是一个Data*
    // batch: 批量大小
    // insertIndexs: 是INT32PARAM，长度为(batch), 第i个询问的插入的page id为insertIndexs[i]
    // insertPositions: 是INT32PARAM，长度为(batch), 第i个询问的插入位置为insertPositions[i]
    void GenerateAppendPagedCacheBatchParams(PagedCacheManager &pagedCacheManager, 
        const std::vector<Data*> &pastKeys, int batch, 
        Data &insertIndexs, Data &insertPositions);

    // 将input中的数据插入到pagedCacheManager中, 用于decode，每个batch的seqlen都是1
    // pagedCacheManager: PagedCacheManager
    // currentCaches: batch个caches的列表，每个元素是一个Data*
    // input: 输入数据，维度为[batch, num_heads, head_dim]
    // insertIndexs: 是INT32PARAM，长度为(batch), 第i个询问的插入的page id为insertIndexs[i]
    // insertPositions: 是INT32PARAM，长度为(batch), 第i个询问的插入位置为insertPositions[i]
    void AppendPagedCacheBatch(PagedCacheManager &pagedCacheManager, const std::vector<Data*> &currentCaches, const Data &input, 
        Data &insertIndexs, Data &insertPositions);

    void AttentionPaged(const Data &q, const Data &k, const Data &v, Data &output,
        int group, float scale, int attentionType, bool inited = false);

    // 这里一般都是Decode部分，q中所有batch的seqlen都是1
    // kCaches, vCaches: 总的PagedKVCache
    // qSizes: 是INT32PARAM，长度为(batch + 1), 第i个询问位于q的[qSizes[i], qSizes[i+1])范围内
    // pageSizes: 是INT32PARAM，长度为(batch + 1), 第i个询问缓存于pageIndexs[pageSizes[i] : pageSizes[i + 1]]
    // pageIndexs: 是INT32PARAM，长度为所有询问使用的pages数目之和
    // lastPageLens: 是INT32PARAM，长度为(batch), 第i个询问的最后一个page的长度为lastPageLens[i]
    void AttentionPagedBatch(const Data &q, const Data &kCaches, const Data &vCaches, 
        const Data &qSizes, const Data &pageSizes, const Data &pageIndexs, const Data &lastPageLens, 
        Data &output, int group, float scale, int attentionType, bool inited = false, bool sync = true);

    // 从batch个pastKey中生成AttentionPagedBatch所需要的qSizes, pageSizes, pageIndexs, lastPageLens
    // pastKeys: batch个pastKey的列表，每个元素是一个Data*
    // q: query数据，维度为[num_heads, batch, head_dim]
    // batch: 批量大小
    // qSizes, pageSizes, pageIndexs, lastPageLens: 输出的参数
    // seqLens: 可选，每个batch的seqLen（prefill时使用）。为空时每个batch的seqLen默认为1（decode）
    void GeneratePagedBatchParams(const Data &q, const std::vector<Data*> &pastKeys, 
        int batch, Data &qSizes, Data &pageSizes, Data &pageIndexs, Data &lastPageLens,
        const std::vector<int> &seqLens = {}, bool lastPageLensOnDevice = false);
}

#endif //TEST_FASTLLM_H
