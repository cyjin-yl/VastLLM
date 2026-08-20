// 多模态请求命中前缀缓存之后, mRoPE 位置到底对不对 —— 逐元素对拍。
//
// 为什么要它: 这类错误**完全静默**。命中路径把位置整体平移一点点, 输出看起来
// 仍然是通顺的句子, 端到端"结果看着对"根本证明不了位置没错位。只有把
// "在完整序列上算完再切后缀"当作参考、逐元素比对, 才能判定。
//
// 被测的两条路(见 qwen3_5.cpp Qwen3_5Model::ForwardMultimodal 开头的守卫):
//   * 参考路: BuildMultimodalPositionData(完整 allTokens) -> 切 [cacheLen, total)
//   * 命中路: FillLLMInputs 给出**纯顺序**位置 [cacheLen, total),
//             再由 AdjustPositionIdsWithDelta 整体加上 mrope_position_delta
// 结论(本用例固化):
//   1. 残段里**没有**图像 token 时, 两者逐元素相等 -> 命中路是对的, 前提是
//      delta 拿得到;
//   2. delta 拿不到(退化成 0)时, 两者相差恰好 |delta| -> 这就是
//      "mrope_position_delta 只在视觉分支里产生, 命中时视觉分支没跑过"的后果;
//   3. 残段里**有**图像 token 时, 无论 delta 对不对, 命中路都还原不了 ——
//      图像块内 mRoPE 的三行 (t,h,w) 互不相同, 而整体平移只能给出三行相同的值。
//      这一档必须靠"在完整序列上重算再切"来修, 顺带那些图还得真的过视觉塔。
//
// 构建: cmake -DUNIT_TEST=ON .. && cmake --build . --target testQwen35MultimodalPrefixPosition
// 运行: ./testQwen35MultimodalPrefixPosition   (退出码非 0 = 有用例失败)

#include "fastllm.h"
#include "models/qwen3_5.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const char *what) {
    g_checks++;
    printf(ok ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!ok) {
        g_failures++;
    }
}

// BuildMultimodalPositionData 是 protected 的, 用一个派生类把它暴露出来。
// 只用到三个纯 int 成员, 不需要权重, 因此完全跑在 CPU 上。
class TestQwen35 : public fastllm::Qwen3_5Model {
public:
    using fastllm::Qwen3_5Model::BuildMultimodalPositionData;
    using fastllm::Qwen3_5Model::image_token_id;
    using fastllm::Qwen3_5Model::video_token_id;
    using fastllm::Qwen3_5Model::vision_spatial_merge_size;
};

const int kImageToken = 151655;
const int kVideoToken = 151656;
const int kTextBase = 1000;

struct Fixture {
    std::vector<float> ids;              // 完整 prompt 的 token id
    std::vector<std::vector<int>> grids; // 每张图的 grid_thw(patch 单位)
    std::vector<int> imageStarts;        // 每个图像块的起始下标
    int total = 0;
};

// [text 40][image 16][text 40][image 16][text 40] = 152
Fixture MakeFixture(int mergeSize) {
    Fixture f;
    auto addText = [&](int n) {
        for (int i = 0; i < n; i++) {
            f.ids.push_back((float)(kTextBase + (int)f.ids.size()));
        }
    };
    auto addImage = [&](int gridH, int gridW) {
        f.grids.push_back({1, gridH, gridW});
        const int pads = 1 * (gridH / mergeSize) * (gridW / mergeSize);
        f.imageStarts.push_back((int)f.ids.size());
        for (int i = 0; i < pads; i++) {
            f.ids.push_back((float)kImageToken);
        }
    };
    addText(40);
    addImage(8, 8);   // -> 4x4 = 16 个 pad
    addText(40);
    addImage(8, 8);
    addText(40);
    f.total = (int)f.ids.size();
    return f;
}

// 命中路实际会喂给模型的位置: 纯顺序 [cacheLen, total) 整体加 delta, 三行相同。
std::vector<float> ContinuationPositions(int cacheLen, int total, float delta) {
    const int n = total - cacheLen;
    std::vector<float> out((size_t)3 * n);
    for (int row = 0; row < 3; row++) {
        for (int i = 0; i < n; i++) {
            out[(size_t)row * n + i] = (float)(cacheLen + i) + delta;
        }
    }
    return out;
}

// 参考路: 完整序列上的 mRoPE 位置, 切出 [cacheLen, total)
std::vector<float> ReferenceSlice(const fastllm::Data &fullMrope,
                                  int cacheLen, int total) {
    const int n = total - cacheLen;
    std::vector<float> out((size_t)3 * n);
    const float *src = (const float*)fullMrope.cpuData;
    for (int row = 0; row < 3; row++) {
        for (int i = 0; i < n; i++) {
            out[(size_t)row * n + i] = src[(size_t)row * total + cacheLen + i];
        }
    }
    return out;
}

float MaxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// 复刻 F10 (b) 的守卫谓词: 残段里还有没有 image/video token。
// 与 qwen3_5.cpp 里 prepareMultimodalRestore 的判断必须保持一致。
bool RemainderHasMedia(const Fixture &f, int cacheLen) {
    for (int i = cacheLen; i < f.total; i++) {
        const int id = (int)f.ids[i];
        if (id == kImageToken || id == kVideoToken) {
            return true;
        }
    }
    return false;
}

int CountImageTokens(const Fixture &f, int from, int to) {
    int n = 0;
    for (int i = from; i < to; i++) {
        if ((int)f.ids[i] == kImageToken) {
            n++;
        }
    }
    return n;
}

}  // namespace

int main() {
    printf("== 多模态前缀命中后的 mRoPE 位置对拍 ==\n");

    TestQwen35 model;
    model.image_token_id = kImageToken;
    model.video_token_id = kVideoToken;
    const int mergeSize = model.vision_spatial_merge_size;

    Fixture f = MakeFixture(mergeSize);
    fastllm::Data fullIds(fastllm::DataType::FLOAT32, {1, f.total}, f.ids);
    fastllm::Data mmTypes, mrope, deltaData;
    model.BuildMultimodalPositionData(fullIds, f.grids, {}, mmTypes, mrope, deltaData);

    mrope.ToDevice(fastllm::DataDevice::CPU);
    deltaData.ToDevice(fastllm::DataDevice::CPU);
    const float delta = ((const float*)deltaData.cpuData)[0];
    printf("  fixture: total=%d 图像块=%zu delta=%.1f mrope.dims={%d,%d}\n",
           f.total, f.grids.size(), delta,
           mrope.dims.size() > 0 ? mrope.dims[0] : -1,
           mrope.dims.size() > 1 ? mrope.dims[1] : -1);

    // 防空转: delta 必须非 0, 否则用例 2 变成"0 和 0 相等", 什么都没测到。
    Check(std::fabs(delta) > 0.5f,
          "fixture 的 delta 非 0(否则后面的用例是空转)");
    Check(mrope.dims.size() == 2 && mrope.dims[0] == 3 && mrope.dims[1] == f.total,
          "BuildMultimodalPositionData 产出 {3, total}");

    // ---- 用例 1: 残段里没有图像 -> 命中路(带 delta)与参考路逐元素相等 ----
    {
        const int cacheLen = 120;   // 第二个图像块结束于 112
        Check(CountImageTokens(f, cacheLen, f.total) == 0,
              "[1] 切点选在所有图像之后(残段内无 image token)");
        auto ref = ReferenceSlice(mrope, cacheLen, f.total);
        auto got = ContinuationPositions(cacheLen, f.total, delta);
        const float d = MaxAbsDiff(ref, got);
        printf("       maxAbsDiff = %.3f\n", d);
        Check(d < 1e-3f,
              "[1] 残段无图像时, 顺序位置+delta == 完整序列切片 "
              "-> 命中路正确(**前提是 delta 拿得到**)");
    }

    // ---- 用例 2: 修复后 —— delta 由"完整 ids + grid"补算, 不碰视觉塔 ----
    // 这正是 qwen3_5.cpp prepareMultimodalRestore 做的事: 命中路径上视觉分支
    // 不会跑, 所以 delta 取不到; 补算只需要 token id 和 image_grid_thw
    // (EncodeVisualItems 的 grid 也是直接读它, 两边同源)。
    {
        const int cacheLen = 120;
        TestQwen35 recompute;
        recompute.image_token_id = kImageToken;
        recompute.video_token_id = kVideoToken;
        fastllm::Data ids2(fastllm::DataType::FLOAT32, {1, f.total}, f.ids);
        fastllm::Data t2, m2, d2;
        recompute.BuildMultimodalPositionData(ids2, f.grids, {}, t2, m2, d2);
        d2.ToDevice(fastllm::DataDevice::CPU);
        const float recomputedDelta = ((const float*)d2.cpuData)[0];
        Check(std::fabs(recomputedDelta - delta) < 1e-3f,
              "[2] 补算出来的 delta 与视觉分支产出的 delta 一致");

        auto ref = ReferenceSlice(mrope, cacheLen, f.total);
        auto got = ContinuationPositions(cacheLen, f.total, recomputedDelta);
        const float d = MaxAbsDiff(ref, got);
        printf("       maxAbsDiff = %.3f (修复前 delta 取不到时会是 %.3f)\n",
               d, std::fabs(delta));
        Check(d < 1e-3f,
              "[2] 修复后位置逐元素相等(0.000) "
              "-> delta 在命中路上可得, 不再有 |delta| 的整体错位");
    }

    // ---- 用例 3: 残段里含图像 -> 整体平移无论如何都还原不了 ----
    {
        const int cacheLen = 48;    // 落在第一个图像块(40..56)内部
        const int imgTokens = CountImageTokens(f, cacheLen, f.total);
        Check(imgTokens > 0,
              "[3] 切点落在图像块内(残段仍含 image token)");
        auto ref = ReferenceSlice(mrope, cacheLen, f.total);
        auto got = ContinuationPositions(cacheLen, f.total, delta);
        const float d = MaxAbsDiff(ref, got);
        printf("       残段内 image token = %d, maxAbsDiff = %.3f\n", imgTokens, d);
        Check(d > 0.5f,
              "[3] 残段含图像时, 顺序位置+delta != 完整序列切片 "
              "-> 整体平移修不了(图像块内 mRoPE 三行互不相同)");
        // 修复后这种输入**根本走不到**续算路径: prepareMultimodalRestore 会
        // 以 remainder-has-image 拒绝复用并计数, 请求退回全量 prefill。
        Check(RemainderHasMedia(f, cacheLen),
              "[3] 守卫谓词命中 -> 该输入被 remainder-has-image 拒绝, 不会复用");
        Check(!RemainderHasMedia(f, 120),
              "[3] 反向对照: 切点在所有图像之后时守卫不误伤");
    }

    printf("== %d checks, %d failures ==\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
