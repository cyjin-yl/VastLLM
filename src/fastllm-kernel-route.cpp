// 算子路由普查的实现。设计说明见 include/fastllm-kernel-route.h。
#include "fastllm-kernel-route.h"
#include "fastllm.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

namespace fastllm {

    namespace {

        struct RouteCounters {
            std::atomic<uint64_t> calls {0};
            std::atomic<uint64_t> tokens {0};
            std::atomic<int32_t> minN {0};
            std::atomic<int32_t> maxN {0};
        };

        RouteCounters &Counters(int route) {
            static RouteCounters table[KERNEL_ROUTE_COUNT];
            return table[route];
        }

        using ShapeKey = std::tuple<int, int, int, int, int>; // route, ggmlType, n, m, k

        std::mutex &ShapeMutex() {
            static std::mutex mutex;
            return mutex;
        }

        std::map<ShapeKey, uint64_t> &ShapeTable() {
            static std::map<ShapeKey, uint64_t> table;
            return table;
        }

        // 明细表的硬上限。形状本来就是有限集(每层几种), 但 n 会随批次变化,
        // 极端情况下可能爆表; 加上限保证排查开关不会把内存吃掉。
        const size_t kMaxShapeEntries = 4096;

        bool ParseEnvFlag(const char *name) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return false;
            }
            return std::strcmp(value, "0") != 0 &&
                   std::strcmp(value, "false") != 0 &&
                   std::strcmp(value, "off") != 0;
        }

    } // namespace

    const char *GetKernelRouteName(KernelRoute route) {
        switch (route) {
            case KERNEL_ROUTE_GGUF_SM70_IQ4XS_MMQ:     return "gguf.sm70_iq4xs_mmq";
            case KERNEL_ROUTE_GGUF_MMVQ:               return "gguf.mmvq";
            case KERNEL_ROUTE_GGUF_DEQUANT_FP16_GEMM:  return "gguf.dequant_fp16_gemm";
            case KERNEL_ROUTE_GGUF_DEQUANT_FP32_GEMM:  return "gguf.dequant_fp32_gemm";
            case KERNEL_ROUTE_ATTN_SM70_PAGED_XQA:     return "attn.sm70_paged_xqa";
            case KERNEL_ROUTE_ATTN_SM70_FLASH_PREFILL: return "attn.sm70_flash_prefill";
            case KERNEL_ROUTE_ATTN_FLASHINFER:         return "attn.flashinfer";
            case KERNEL_ROUTE_ATTN_NATIVE_FALLBACK:    return "attn.native_fallback";
            default:                                   return "unknown";
        }
    }

    bool KernelRouteShapeStatsEnabled() {
        // static 只求一次值: 这个判定在热路径上, 不能每次都 getenv。
        static const bool enabled = ParseEnvFlag("FASTLLM_KERNEL_ROUTE_STATS");
        return enabled;
    }

    void KernelRouteHit(KernelRoute route, int ggmlType, int n, int m, int k) {
        if (route < 0 || route >= KERNEL_ROUTE_COUNT) {
            return;
        }
        RouteCounters &counters = Counters(route);
        // relaxed: 这些计数只用于事后观察, 不参与任何同步决策。
        counters.calls.fetch_add(1, std::memory_order_relaxed);
        counters.tokens.fetch_add((uint64_t)(n > 0 ? n : 0),
                                  std::memory_order_relaxed);
        if (n > 0) {
            int32_t prevMin = counters.minN.load(std::memory_order_relaxed);
            while ((prevMin == 0 || n < prevMin) &&
                   !counters.minN.compare_exchange_weak(
                       prevMin, n, std::memory_order_relaxed)) {
            }
            int32_t prevMax = counters.maxN.load(std::memory_order_relaxed);
            while (n > prevMax &&
                   !counters.maxN.compare_exchange_weak(
                       prevMax, n, std::memory_order_relaxed)) {
            }
        }

        if (!KernelRouteShapeStatsEnabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(ShapeMutex());
        auto &table = ShapeTable();
        ShapeKey key {(int)route, ggmlType, n, m, k};
        auto it = table.find(key);
        if (it != table.end()) {
            it->second++;
        } else if (table.size() < kMaxShapeEntries) {
            table.emplace(key, 1);
        }
    }

    std::vector<KernelRouteTotals> GetKernelRouteTotals() {
        std::vector<KernelRouteTotals> out((size_t)KERNEL_ROUTE_COUNT);
        for (int i = 0; i < KERNEL_ROUTE_COUNT; i++) {
            RouteCounters &counters = Counters(i);
            out[i].calls = counters.calls.load(std::memory_order_relaxed);
            out[i].tokens = counters.tokens.load(std::memory_order_relaxed);
            out[i].minN = counters.minN.load(std::memory_order_relaxed);
            out[i].maxN = counters.maxN.load(std::memory_order_relaxed);
        }
        return out;
    }

    std::vector<KernelRouteShape> GetKernelRouteShapes() {
        std::vector<KernelRouteShape> out;
        std::lock_guard<std::mutex> lock(ShapeMutex());
        for (const auto &entry : ShapeTable()) {
            KernelRouteShape shape;
            shape.route = (KernelRoute)std::get<0>(entry.first);
            shape.ggmlType = std::get<1>(entry.first);
            shape.n = std::get<2>(entry.first);
            shape.m = std::get<3>(entry.first);
            shape.k = std::get<4>(entry.first);
            shape.calls = entry.second;
            out.push_back(shape);
        }
        return out;
    }

    void ResetKernelRouteCensus() {
        for (int i = 0; i < KERNEL_ROUTE_COUNT; i++) {
            RouteCounters &counters = Counters(i);
            counters.calls.store(0, std::memory_order_relaxed);
            counters.tokens.store(0, std::memory_order_relaxed);
            counters.minN.store(0, std::memory_order_relaxed);
            counters.maxN.store(0, std::memory_order_relaxed);
        }
        std::lock_guard<std::mutex> lock(ShapeMutex());
        ShapeTable().clear();
    }

    const char *Sm70Iq4XsMmqShapeRejectReason(int dataType, int n, int m, int k) {
        if (dataType != (int)DataType::FLOAT32 &&
            dataType != (int)DataType::FLOAT16 &&
            dataType != (int)DataType::BFLOAT16) {
            return "dtype";
        }
        // n 是本次矩阵乘的 token 数。上界 64 不是偷懒:
        //   n < 8  时 DP4A MMVQ(逐 token)更快, 不必付 tile 的代价;
        //   n 很大时该走张量核 —— V100 的 DP4A 约 62 TOPS, 而 FP16 张量核
        //   约 125 TFLOPS, 反量化+HGEMM 反而更快。
        // 也就是说这个 kernel 天然只覆盖中间那一段, 不是"应该覆盖全部却漏了"。
        if (n < kSm70Iq4XsMmqMinN || n > kSm70Iq4XsMmqMaxN) {
            return "n range";
        }
        if (m <= 0 || (m % kSm70Iq4XsMmqBlockK) != 0) {
            return "m%256";
        }
        if (k < kSm70Iq4XsMmqTileRows) {
            return "k<128";
        }
        return nullptr;
    }

    std::string FormatKernelRouteCensus() {
        std::string out = "[kernel-route] ";
        std::vector<KernelRouteTotals> totals = GetKernelRouteTotals();
        bool any = false;
        for (int i = 0; i < KERNEL_ROUTE_COUNT; i++) {
            if (totals[i].calls == 0) {
                continue;
            }
            any = true;
            char buffer[256];
            snprintf(buffer, sizeof(buffer),
                     "%s=%llu calls/%llu tok (n=%d..%d) ",
                     GetKernelRouteName((KernelRoute)i),
                     (unsigned long long)totals[i].calls,
                     (unsigned long long)totals[i].tokens,
                     (int)totals[i].minN, (int)totals[i].maxN);
            out += buffer;
        }
        if (!any) {
            out += "(无记录: 还没跑过任何被普查的算子)";
        }
        return out;
    }

} // namespace fastllm
