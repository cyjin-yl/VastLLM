// 算子路由普查 + SM70 IQ4_XS MMQ 形状资格的回归。
//
// 为什么需要它:
//   本仓在 V100 上给同一个逻辑算子准备了多份实现, 靠一串 `if (...) return false;`
//   在运行期静默选路。选错路不会报错、结果也对, 只是慢 —— 于是"到底走了哪条"
//   长期只能靠读代码猜, 而且已经猜错过两次:
//
//     1) 生产 profile 里 FASTLLM_CUDA_SM70_PAGED_XQA=0, 而代码注释写着
//        "Enabled by default", 于是被当成"我们把一个默认开启的优化关掉了"。
//        真相是 XQA 要求分页 KV 为 float16, 生产用 turbo3(K=q8_0,V=turbo3),
//        这条路无论开关取 0 还是 1 都进不去 —— 设 0 是 no-op。
//        (对应的运行期诊断在 fastllm-attention.cu 的
//         FastllmReportSm70AttentionRouteOnce)
//
//     2) 749 行的 SM70 IQ4_XS DP4A MMQ kernel 被当成"IQ4_XS 的快路径", 以为
//        decode 会走它。真相是它只接受 n in [8,64], 而 decode 的 n 是 1..3,
//        长 prefill 的 n 是上千 —— 两头都不在区间内。
//
//   这两条都属于"数字边界决定了整块代码有没有被用到", 靠人读判定链很容易读错,
//   所以固化成断言。本测试**不需要 GPU**: 形状判定是纯函数, 计数器是纯 C++。
//
// 构建: cmake -DUNIT_TEST=ON .. && cmake --build . --target testKernelRouteCensus
// 运行: ./testKernelRouteCensus   (退出码非 0 = 有用例失败)

#include "fastllm.h"
#include "fastllm-kernel-route.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string &what) {
    g_checks++;
    if (ok) {
        printf("  ok   %s\n", what.c_str());
    } else {
        printf("  FAIL %s\n", what.c_str());
        g_failures++;
    }
}

const char *Reason(int n, int m, int k) {
    return fastllm::Sm70Iq4XsMmqShapeRejectReason(
        (int)fastllm::DataType::FLOAT16, n, m, k);
}

std::string ReasonText(int n, int m, int k) {
    const char *r = Reason(n, m, k);
    return r == nullptr ? std::string("<合格>") : std::string(r);
}

// -------------------------------------------------------------------------
// 1) 生产模型的真实线性层形状。
//
// 取自 Qwen3.8-27B-Uncensored-Cyber 的 GGUF 张量表(embed=5120, ffn=17408,
// head=24x256, kv=4x256, ssm_inner=6144, vocab=248320):
//   名称                  m(输入维)  k(输出维)
//   attn_qkv               5120      10240
//   attn_q                 5120      12288
//   attn_k / attn_v        5120       1024
//   attn_gate              5120       6144
//   attn_output            6144       5120
//   ffn_gate / ffn_up      5120      17408
//   ffn_down              17408       5120
//   ssm_out                6144       5120
//   ssm_alpha / ssm_beta   5120         48   <- k=48, 小投影
//   nextn.eh_proj         10240       5120   <- MTP draft 头
//   output(lm_head)        5120     248320
// -------------------------------------------------------------------------
struct LinearShape {
    const char *name;
    int m;
    int k;
};

const LinearShape kProdShapes[] = {
    {"attn_qkv",      5120,  10240},
    {"attn_q",        5120,  12288},
    {"attn_k",        5120,   1024},
    {"attn_v",        5120,   1024},
    {"attn_gate",     5120,   6144},
    {"attn_output",   6144,   5120},
    {"ffn_gate",      5120,  17408},
    {"ffn_up",        5120,  17408},
    {"ffn_down",     17408,   5120},
    {"ssm_out",       6144,   5120},
    {"nextn.eh_proj",10240,   5120},
    {"output",        5120, 248320},
};

void TestProductionShapesNeverReachMmqInDecode() {
    printf("[用例] 生产形状 x 生产 n: decode 与长 prefill 都进不了 MMQ\n");

    // decode: batch=1, MTP=2 -> 验证步最多 3 个 token; 就算 batch 开到 2 也才 6。
    const int decodeN[] = {1, 2, 3, 4, 6};
    for (int n : decodeN) {
        for (const auto &shape : kProdShapes) {
            const char *why = Reason(n, shape.m, shape.k);
            Check(why != nullptr && std::strcmp(why, "n range") == 0,
                  std::string("decode n=") + std::to_string(n) + " " +
                  shape.name + " 应因 n range 被拒, 实际: " +
                  ReasonText(n, shape.m, shape.k));
        }
    }

    // 长 prefill: FASTLLM_QWEN35_PREFILL_CHUNK_CAP 默认 1024。
    const int prefillN[] = {65, 128, 256, 512, 1024};
    for (int n : prefillN) {
        const auto &shape = kProdShapes[6]; // ffn_gate, 最热的一层
        const char *why = Reason(n, shape.m, shape.k);
        Check(why != nullptr && std::strcmp(why, "n range") == 0,
              std::string("prefill n=") + std::to_string(n) +
              " ffn_gate 应因 n range 被拒, 实际: " +
              ReasonText(n, shape.m, shape.k));
    }
}

void TestProductionShapesInsideMmqBand() {
    printf("[用例] n 落在 [8,64] 时, 生产形状确实合格(说明 kernel 本身没写死)\n");
    for (int n : {8, 16, 33, 64}) {
        for (const auto &shape : kProdShapes) {
            const char *why = Reason(n, shape.m, shape.k);
            // ssm_alpha/beta 的 k=48 单独测, 这里的形状 k 都 >= 1024
            Check(why == nullptr,
                  std::string("n=") + std::to_string(n) + " " + shape.name +
                  " 应合格, 实际: " + ReasonText(n, shape.m, shape.k));
        }
    }
}

void TestSmallProjectionRejected() {
    printf("[用例] ssm_alpha/ssm_beta 的 k=48 因 k<128 被拒(有意为之)\n");
    // 128 行的 MMQ tile 在 k=48 上比老的 MMVQ 慢, 所以刻意不接。
    Check(Reason(16, 5120, 48) != nullptr &&
          std::strcmp(Reason(16, 5120, 48), "k<128") == 0,
          "k=48 应因 k<128 被拒, 实际: " + ReasonText(16, 5120, 48));
    Check(Reason(16, 5120, 127) != nullptr,
          "k=127 应被拒, 实际: " + ReasonText(16, 5120, 127));
    Check(Reason(16, 5120, 128) == nullptr,
          "k=128 应合格, 实际: " + ReasonText(16, 5120, 128));
}

void TestBlockAlignment() {
    printf("[用例] m 必须是 IQ4_XS 块大小 256 的整数倍\n");
    Check(Reason(16, 255, 1024) != nullptr &&
          std::strcmp(Reason(16, 255, 1024), "m%256") == 0,
          "m=255 应因 m%256 被拒, 实际: " + ReasonText(16, 255, 1024));
    Check(Reason(16, 0, 1024) != nullptr,
          "m=0 应被拒, 实际: " + ReasonText(16, 0, 1024));
    // 生产的每个 m 都对齐, 顺手确认一遍(不对齐的话 GGUF 根本存不下)
    for (const auto &shape : kProdShapes) {
        Check(shape.m % fastllm::kSm70Iq4XsMmqBlockK == 0,
              std::string(shape.name) + " 的 m=" + std::to_string(shape.m) +
              " 应是 256 的整数倍");
    }
}

void TestDtypeGate() {
    printf("[用例] 只接受 fp32/fp16/bf16 激活\n");
    Check(fastllm::Sm70Iq4XsMmqShapeRejectReason(
              (int)fastllm::DataType::FLOAT32, 16, 5120, 1024) == nullptr,
          "float32 激活应合格");
    Check(fastllm::Sm70Iq4XsMmqShapeRejectReason(
              (int)fastllm::DataType::BFLOAT16, 16, 5120, 1024) == nullptr,
          "bfloat16 激活应合格");
    const char *why = fastllm::Sm70Iq4XsMmqShapeRejectReason(
        (int)fastllm::DataType::INT8, 16, 5120, 1024);
    Check(why != nullptr && std::strcmp(why, "dtype") == 0,
          "int8 激活应因 dtype 被拒");
}

void TestCensusCounters() {
    printf("[用例] 普查计数器: 累加 / min-max n / 复位\n");
    fastllm::ResetKernelRouteCensus();
    auto totals = fastllm::GetKernelRouteTotals();
    Check((int)totals.size() == (int)fastllm::KERNEL_ROUTE_COUNT,
          "totals 长度应等于路由数");
    Check(totals[fastllm::KERNEL_ROUTE_GGUF_MMVQ].calls == 0,
          "复位后 calls 应为 0");

    fastllm::KernelRouteHit(fastllm::KERNEL_ROUTE_GGUF_MMVQ, 23, 3, 5120, 17408);
    fastllm::KernelRouteHit(fastllm::KERNEL_ROUTE_GGUF_MMVQ, 23, 1, 5120, 17408);
    fastllm::KernelRouteHit(fastllm::KERNEL_ROUTE_GGUF_MMVQ, 23, 8, 5120, 17408);
    fastllm::KernelRouteHit(fastllm::KERNEL_ROUTE_ATTN_NATIVE_FALLBACK, -1, 1, 0, 0);

    totals = fastllm::GetKernelRouteTotals();
    const auto &mmvq = totals[fastllm::KERNEL_ROUTE_GGUF_MMVQ];
    Check(mmvq.calls == 3, "MMVQ calls 应为 3, 实际 " + std::to_string(mmvq.calls));
    Check(mmvq.tokens == 12, "MMVQ tokens 应为 3+1+8=12, 实际 " +
          std::to_string(mmvq.tokens));
    Check(mmvq.minN == 1, "minN 应为 1, 实际 " + std::to_string((int)mmvq.minN));
    Check(mmvq.maxN == 8, "maxN 应为 8, 实际 " + std::to_string((int)mmvq.maxN));
    Check(totals[fastllm::KERNEL_ROUTE_ATTN_NATIVE_FALLBACK].calls == 1,
          "注意力 fallback calls 应为 1");
    Check(totals[fastllm::KERNEL_ROUTE_GGUF_SM70_IQ4XS_MMQ].calls == 0,
          "没命中过的路由应保持 0");

    const std::string text = fastllm::FormatKernelRouteCensus();
    Check(text.find("gguf.mmvq=3 calls/12 tok (n=1..8)") != std::string::npos,
          "文本报告应含 MMVQ 明细, 实际: " + text);
    Check(text.find("gguf.sm70_iq4xs_mmq") == std::string::npos,
          "文本报告不应列出 0 次的路由");

    fastllm::ResetKernelRouteCensus();
    Check(fastllm::GetKernelRouteTotals()[fastllm::KERNEL_ROUTE_GGUF_MMVQ]
              .calls == 0,
          "复位应清零");
}

void TestRouteNames() {
    printf("[用例] 路由名齐全且互不重复\n");
    std::vector<std::string> names;
    for (int i = 0; i < (int)fastllm::KERNEL_ROUTE_COUNT; i++) {
        std::string name =
            fastllm::GetKernelRouteName((fastllm::KernelRoute)i);
        Check(name != "unknown",
              "路由 " + std::to_string(i) + " 应有名字");
        for (const auto &prev : names) {
            Check(prev != name, "路由名 " + name + " 应与其它路由不重名");
        }
        names.push_back(name);
    }
    Check(std::string(fastllm::GetKernelRouteName(
              (fastllm::KernelRoute)fastllm::KERNEL_ROUTE_COUNT)) == "unknown",
          "越界枚举应返回 unknown");
}

void TestShapeTableGatedByEnv() {
    printf("[用例] (n,m,k) 明细表默认关闭, 不为热路径付锁的代价\n");
    // 本进程没设 FASTLLM_KERNEL_ROUTE_STATS -> 明细表应为空。
    // 注意 KernelRouteShapeStatsEnabled 用 static 只求一次值, 所以这里
    // 不能靠 setenv 翻转它 —— 那正是热路径需要的性质。
    const bool enabled = fastllm::KernelRouteShapeStatsEnabled();
    fastllm::ResetKernelRouteCensus();
    fastllm::KernelRouteHit(fastllm::KERNEL_ROUTE_GGUF_MMVQ, 23, 3, 5120, 17408);
    const auto shapes = fastllm::GetKernelRouteShapes();
    if (enabled) {
        Check(shapes.size() == 1,
              "开了 FASTLLM_KERNEL_ROUTE_STATS 时应记录 1 条明细, 实际 " +
              std::to_string(shapes.size()));
        Check(shapes[0].n == 3 && shapes[0].m == 5120 && shapes[0].k == 17408,
              "明细的 n/m/k 应与调用一致");
    } else {
        Check(shapes.empty(),
              "默认(未设 FASTLLM_KERNEL_ROUTE_STATS)明细表应为空, 实际 " +
              std::to_string(shapes.size()) + " 条");
    }
    fastllm::ResetKernelRouteCensus();
}

} // namespace

int main() {
    printf("=== 算子路由普查 / SM70 IQ4_XS MMQ 形状资格 回归 ===\n");
    TestProductionShapesNeverReachMmqInDecode();
    TestProductionShapesInsideMmqBand();
    TestSmallProjectionRejected();
    TestBlockAlignment();
    TestDtypeGate();
    TestCensusCounters();
    TestRouteNames();
    TestShapeTableGatedByEnv();
    printf("=== %d 项检查, %d 项失败 ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
