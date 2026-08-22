// 采样路径的正确性回归测试。
//
// 为什么需要它: 这一版改了推理引擎最核心的采样逻辑, 而这些 bug 的共同特征是
// **静默** —— 不报错、不崩溃、只是悄悄输出垃圾。"能编译 + 线上没崩"完全
// 判断不出对错, 必须把失败条件固化成可执行的断言。
//
// 覆盖的三个真实线上故障:
//   1. temperature=0 除以零   -> 输出退化成同一个字符的长串(实测 "44444...")
//   2. NaN logits 无防护      -> 输出退化成 token 0, 本词表 token 0 是 '!',
//                                 表现为满屏 "!!!!!!"
//   3. 含 NaN 的比较器        -> 不满足严格弱序, std::sort 是 UB(潜在越界写)
//
// 构建: cmake -DUNIT_TEST=ON && cmake --build . --target testSamplingGuard
// 运行: ./testSamplingGuard   (退出码非 0 = 有用例失败)

#include "fastllm.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const char *what) {
    g_checks++;
    if (ok) {
        printf("  ok   %s\n", what);
    } else {
        printf("  FAIL %s\n", what);
        g_failures++;
    }
}

// 造一份 logits: 除了 peak 位置是最大值, 其余递减且互不相同,
// 这样 argmax 唯一, 断言不会因并列而抖。
std::vector<float> MakeLogits(int vocab, int peak, float peakValue = 20.0f) {
    std::vector<float> v(vocab);
    for (int i = 0; i < vocab; i++) {
        v[i] = -1.0f - (float)i * 0.01f;
    }
    v[peak] = peakValue;
    return v;
}

fastllm::GenerationConfig NonGreedyConfig(int topK) {
    fastllm::GenerationConfig cfg;
    // repeat_penalty != 1 让 IsSimpleGreedy() 为 false, 从而走 LLMSampling
    // 这条 CPU 采样路径 —— 生产上正是这条(proxy 默认带 frequency_penalty)。
    cfg.repeat_penalty = 1.05f;
    cfg.top_k = topK;
    cfg.top_p = 1.0f;
    cfg.last_n = 64;
    return cfg;
}

// ---------------------------------------------------------------- 用例

void TestSanitizeReplacesNonFinite() {
    printf("[1] SanitizeLogitsForSampling: 非有限值被压掉, 有限值不动\n");
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> v = {1.0f, nan, 3.0f, inf, -inf, 2.0f};

    const int bad = fastllm::SanitizeLogitsForSampling(
        v.data(), (int)v.size(), "unit-test");

    Check(bad == 3, "识别出 3 个非有限值(NaN / +Inf / -Inf)");
    Check(v[0] == 1.0f && v[2] == 3.0f && v[5] == 2.0f,
          "有限值原样保留");
    bool allFinite = true;
    for (float x : v) {
        if (!std::isfinite(x)) {
            allFinite = false;
        }
    }
    Check(allFinite, "净化后不再有非有限值");
    // 被压掉的必须排在真实值后面, 否则 top-k 会把垃圾选出来
    Check(v[1] < v[0] && v[3] < v[2] && v[4] < v[5],
          "被压掉的值排在有限值之后(不会被 top-k 选中)");
}

void TestSanitizeAllNonFinite() {
    printf("[2] SanitizeLogitsForSampling: 整层全废也不能留下非有限值\n");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> v(16, nan);
    const int bad = fastllm::SanitizeLogitsForSampling(
        v.data(), (int)v.size(), "unit-test-all-nan");
    Check(bad == (int)v.size(), "全部计入 bad 计数(用于分流病因)");
    bool allFinite = true;
    for (float x : v) {
        if (!std::isfinite(x)) {
            allFinite = false;
        }
    }
    Check(allFinite, "整层全废时仍然不留非有限值");
}

void TestZeroTemperatureIsArgmax() {
    printf("[3] temperature=0 必须等于贪心 argmax(不是除以零)\n");
    // 线上故障: 1.0f/0.0f == +inf -> base[i]*invTemp 全 ±inf ->
    // expf(inf-inf)==NaN -> 概率链全 NaN -> 兜底返回固定低位 token,
    // 输出变成 "44444444..."。
    const int vocab = 512;
    const int peak = 321;
    std::vector<float> values = MakeLogits(vocab, peak);

    fastllm::Data logits(fastllm::DataType::FLOAT32, {1, vocab}, values);
    fastllm::GenerationConfig cfg = NonGreedyConfig(/*topK=*/40);
    cfg.temperature = 0.0f;
    fastllm::LastTokensUnit unit;

    const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
    Check(tok == peak, "temperature=0 返回 argmax");
    Check(tok != 0, "temperature=0 不退化成 token 0");
}

void TestNegativeAndNanTemperature() {
    printf("[4] temperature 为负 / NaN 也不能炸\n");
    const int vocab = 256;
    const int peak = 77;
    std::vector<float> values = MakeLogits(vocab, peak);
    fastllm::LastTokensUnit unit;

    {
        fastllm::Data logits(fastllm::DataType::FLOAT32, {1, vocab}, values);
        fastllm::GenerationConfig cfg = NonGreedyConfig(40);
        cfg.temperature = -1.0f;
        const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
        Check(tok == peak, "temperature<0 按贪心处理");
    }
    {
        fastllm::Data logits(fastllm::DataType::FLOAT32, {1, vocab}, values);
        fastllm::GenerationConfig cfg = NonGreedyConfig(40);
        cfg.temperature = std::numeric_limits<float>::quiet_NaN();
        const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
        // 判据写成 !(x > 0) 就是为了把 NaN 一起挡住
        Check(tok == peak, "temperature=NaN 按贪心处理");
    }
}

void TestNanLogitsDoNotCollapseToTokenZero() {
    printf("[5] logits 含 NaN 时不能退化成 token 0(满屏感叹号那个)\n");
    // 线上故障: 比较器对 NaN 恒 false, 前 topk 次无条件入堆的 NaN
    // (下标恰好 0..topk-1)再也踢不出去, 最终返回低位下标。
    // 本词表 token 0 == '!', 所以表现为 "!!!!!!"。
    const int vocab = 512;
    const int peak = 400;
    std::vector<float> values = MakeLogits(vocab, peak);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // 故意污染前若干个下标 —— 正是原 bug 会挑中的那一段
    for (int i = 0; i < 8; i++) {
        values[i] = nan;
    }

    fastllm::Data logits(fastllm::DataType::FLOAT32, {1, vocab}, values);
    fastllm::GenerationConfig cfg = NonGreedyConfig(/*topK=*/40);
    cfg.temperature = 0.0f;   // 贪心, 断言唯一
    fastllm::LastTokensUnit unit;

    const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
    Check(tok == peak, "NaN 不劫持采样, 仍返回真实 argmax");
    Check(tok >= 8, "返回值不落在被 NaN 污染的低位下标区间");
}

void TestLargeTopKPathAlsoGuarded() {
    printf("[6] top_k>64 走 partial_sort 分支, 同样不能被 NaN 带偏\n");
    // topk<=64 走堆, >64 走 partial_sort —— 两条分支都要覆盖,
    // 因为含 NaN 的比较器在 partial_sort 下是 UB(可能越界写)。
    const int vocab = 2048;
    const int peak = 1500;
    std::vector<float> values = MakeLogits(vocab, peak);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < 32; i++) {
        values[i] = nan;
    }
    values[7] = std::numeric_limits<float>::infinity();

    fastllm::Data logits(fastllm::DataType::FLOAT32, {1, vocab}, values);
    fastllm::GenerationConfig cfg = NonGreedyConfig(/*topK=*/128);
    cfg.temperature = 0.0f;
    fastllm::LastTokensUnit unit;

    const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
    Check(tok == peak, "大 top_k 分支同样返回真实 argmax");
    Check(tok != 7, "+Inf 不会被当成最大值选中");
}

void TestNormalSamplingStillWorks() {
    printf("[7] 正常温度下采样仍然只在 top-k 内取值(没被防护改坏)\n");
    const int vocab = 256;
    const int peak = 100;
    std::vector<float> values = MakeLogits(vocab, peak);
    // 造第二、第三高, 让 top-3 是确定的集合
    values[101] = 19.0f;
    values[102] = 18.0f;

    fastllm::GenerationConfig cfg = NonGreedyConfig(/*topK=*/3);
    cfg.temperature = 0.8f;
    fastllm::LastTokensUnit unit;

    bool allInTopK = true;
    for (int trial = 0; trial < 64; trial++) {
        fastllm::Data logits(fastllm::DataType::FLOAT32, {1, vocab}, values);
        const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
        if (tok != 100 && tok != 101 && tok != 102) {
            allInTopK = false;
            printf("       越界取值: %d\n", tok);
            break;
        }
    }
    Check(allInTopK, "64 次采样全部落在 top-3 集合内");
}

void TestFullLogitsToolCallMaskEmptyFallbackIsCounted() {
    printf("[8] 全词表工具调用候选为空时回退可观测\n");
    const int vocab = 64;
    const int peak = 37;
    const std::vector<float> values = MakeLogits(vocab, peak);
    fastllm::Data logits(
        fastllm::DataType::FLOAT32, {1, vocab}, values);
    fastllm::GenerationConfig cfg = NonGreedyConfig(/*topK=*/3);
    cfg.temperature = 0.0f;
    cfg.tool_call_allowed_token_ids = {999};
    fastllm::LastTokensUnit unit;

    const long long before = fastllm::GetToolCallMaskEmptiedCount();
    const int tok = fastllm::LLMSampling(logits, 0, cfg, unit);
    const long long after = fastllm::GetToolCallMaskEmptiedCount();

    Check(tok == peak, "全词表候选全被排除时仍返回无约束最佳候选");
    Check(after - before == 1, "全词表掩码为空时计数器恰好增加 1");
}

void TestToolCallMaskEmptyFallbackIsCounted() {
    printf("[9] top-k 工具调用候选为空时回退可观测\n");
    // LLMSamplingOnly 接受已经按分数降序排列的 (tokenId, logit) 交错 top-k 张量。
    const std::vector<float> interleaved = {
        41.0f, 9.0f,
        42.0f, 8.0f,
        43.0f, 7.0f
    };
    fastllm::Data logits(
        fastllm::DataType::FLOAT32, {1, (int)interleaved.size()}, interleaved);
    fastllm::GenerationConfig cfg;
    cfg.top_k = 3;
    cfg.temperature = 0.0f;
    cfg.tool_call_allowed_token_ids = {999};

    const long long before = fastllm::GetToolCallMaskEmptiedCount();
    const int tok = fastllm::LLMSamplingOnly(logits, 0, cfg);
    const long long after = fastllm::GetToolCallMaskEmptiedCount();

    Check(tok == 41, "候选全被 allow-list 排除时仍返回无约束最佳候选");
    Check(after - before == 1, "候选掩码为空时计数器恰好增加 1");
}

void TestPresencePenaltyUsesGeneratedHistory() {
    printf("[10] presence penalty 对历史中出现过的 token 固定扣减\n");
    fastllm::Data logits(
        fastllm::DataType::FLOAT32, {1, 3}, {0.0f, 10.0f, 9.5f});
    fastllm::GenerationConfig cfg;
    cfg.temperature = 0.0f;
    cfg.presence_penalty = 1.0f;
    fastllm::LastTokensUnit history(16);
    history.Push(1);

    Check(fastllm::LLMSampling(logits, 0, cfg, history) == 2,
          "出现过一次的最高 logit token 被固定惩罚后让位");
}

void TestFrequencyPenaltyScalesWithOccurrenceCount() {
    printf("[11] frequency penalty 按历史出现次数线性扣减\n");
    fastllm::GenerationConfig cfg;
    cfg.temperature = 0.0f;
    cfg.frequency_penalty = 0.3f;

    fastllm::LastTokensUnit once(16);
    once.Push(1);
    fastllm::Data logitsOnce(
        fastllm::DataType::FLOAT32, {1, 3}, {0.0f, 10.0f, 9.5f});
    Check(fastllm::LLMSampling(logitsOnce, 0, cfg, once) == 1,
          "出现一次仅扣 0.3, 原最高 token 仍胜出");

    fastllm::LastTokensUnit twice(16);
    twice.Push(1);
    twice.Push(1);
    fastllm::Data logitsTwice(
        fastllm::DataType::FLOAT32, {1, 3}, {0.0f, 10.0f, 9.5f});
    Check(fastllm::LLMSampling(logitsTwice, 0, cfg, twice) == 2,
          "出现两次累计扣 0.6, 次高 token 胜出");
}

void TestRepeatPenaltyLastNZeroPreservesUniqueTokenSemantics() {
    printf("[12] repeat penalty 在 last_n<=0 时每个历史 token 只作用一次\n");
    fastllm::GenerationConfig cfg;
    cfg.temperature = 0.0f;
    cfg.repeat_penalty = 2.0f;
    cfg.last_n = 0;
    fastllm::LastTokensUnit history(16);
    history.Push(1);
    history.Push(1);
    fastllm::Data logits(
        fastllm::DataType::FLOAT32, {1, 3}, {0.0f, 10.0f, 4.0f});

    Check(fastllm::LLMSampling(logits, 0, cfg, history) == 1,
          "重复出现两次仍只除以 2, 不应错误除以 4");
}


}  // namespace


int main() {
    printf("== 采样路径正确性回归 ==\n");
    TestSanitizeReplacesNonFinite();
    TestSanitizeAllNonFinite();
    TestZeroTemperatureIsArgmax();
    TestNegativeAndNanTemperature();
    TestNanLogitsDoNotCollapseToTokenZero();
    TestLargeTopKPathAlsoGuarded();
    TestNormalSamplingStillWorks();
    TestFullLogitsToolCallMaskEmptyFallbackIsCounted();

    TestToolCallMaskEmptyFallbackIsCounted();
    TestPresencePenaltyUsesGeneratedHistory();
    TestFrequencyPenaltyScalesWithOccurrenceCount();
    TestRepeatPenaltyLastNZeroPreservesUniqueTokenSemantics();

    printf("\n%d/%d 通过\n", g_checks - g_failures, g_checks);
    if (g_failures > 0) {
        printf("有 %d 个断言失败\n", g_failures);
        return 1;
    }
    return 0;
}
