// 前缀缓存"记录侧 / 查询侧必须落在同一个 PagedCacheManager 上"的回归。
//
// 为什么需要它: 这类故障**完全静默** —— 不报错、不崩溃、所有记录路径计数器
// 都在正常增长、L1trie/resident 指标看着满满当当, 唯一的症状是命中率恒为 0。
// 生产上真实发生过: Qwen3.5 的多模态(vision)前向走
// ForwardFromHiddenStates, 那里把分页 KV manager 的全局层号硬编码成 i*2
// (基址 0), 而纯文本前向走 ForwardGPU, 用的是 threadTpPagedCacheBase + i
// (启动时被定成 3000000+)。前缀缓存**查询侧**(GetPagedKVCacheManagers)只认
// threadTpPagedCacheBase 那一套。结果:
//   * vision 请求把前缀链记进了一组永远不会被查询的 manager;
//   * 查询又落在一组从没记录过这条前缀的 manager 上;
//   * Query() 恒返回 0 页 -> 每个 vision 请求都报 miss=no-record。
// 实测生产日志: 同一个后端, 纯文本 agent 循环稳定命中 94%
// (hit=1024/1095), 而所有带图请求(48K~85K token)无一例外 hit=0 ——
// 而这台机器上 agent 流量 100% 带图, 于是"29 万 token 的 prefill 全白算"。
//
// 靠"能编译 + 服务没挂"完全判断不出对错, 必须固化成可执行断言。
//
// 构建: cmake -DUNIT_TEST=ON .. && cmake --build . --target testPrefixCacheRouting
// 运行: ./testPrefixCacheRouting   (退出码非 0 = 有用例失败)

#include "fastllm.h"

#include <cstdio>
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

// 一个 CPU 上的分页 KV manager, 几何随便但足够走完 Record/Query。
class RoutingTestManager {
public:
    RoutingTestManager(int physicalPages, int pageLen,
                       int numHeads, int headDim)
            : pageLenValue(pageLen) {
        fastllm::Data *data = (fastllm::Data*)&manager;
        data->dataType = fastllm::DataType::FLOAT32;
        data->UpdateUnitSize();
        data->Resize({physicalPages, pageLen, numHeads, headDim});
        data->Allocate();
        manager.pageLen = pageLen;
        manager.type =
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        manager.SetMaxPages(physicalPages);
    }

    fastllm::PagedCacheManager &Get() { return manager; }

    std::vector<int> MakeTokens(int pages, int salt) const {
        std::vector<int> tokens;
        for (int p = 0; p < pages; p++) {
            for (int i = 0; i < pageLenValue; i++) {
                tokens.push_back(salt * 100000 + p * 100 + i + 1);
            }
        }
        return tokens;
    }

private:
    fastllm::PagedCacheManager manager;
    int pageLenValue;
};

// [1] 基线: 记录一条前缀链后, 用"严格前缀延长"的 token 序列去查,
//     必须命中全部已记录的整页。这是 agent 循环的真实形态。
void TestRecordThenQueryStrictExtension() {
    printf("[1] Record 之后, 严格前缀延长的查询必须命中全部整页\n");
    RoutingTestManager env(/*physicalPages=*/16, /*pageLen=*/4,
                           /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();

    std::vector<int> tokens = env.MakeTokens(/*pages=*/4, /*salt=*/1);
    std::vector<int> pages = {0, 1, 2, 3};
    m.Record(tokens, pages);

    // 第二轮: 上一轮的全部内容 + 新增一页
    std::vector<int> extended = tokens;
    for (int i = 0; i < 4; i++) {
        extended.push_back(777000 + i);
    }

    std::vector<int> hitPages;
    m.Query(extended, hitPages);
    Check(hitPages.size() == 4,
          "严格前缀延长的查询命中 4 页(全部已记录的整页)");
    bool same = hitPages.size() == pages.size();
    for (size_t i = 0; same && i < hitPages.size(); i++) {
        same = hitPages[i] == pages[i];
    }
    Check(same, "命中的物理页号与记录时一致");
}

// [2] 事故机制本身: 记录进 manager A, 却去 manager B 查 —— 必然 0 页,
//     而且没有任何错误信号。前向路径与查询路径的层号基址一旦不一致,
//     线上看到的就是这个: 记录一切正常, 命中恒为 0。
void TestQueryOnAnotherManagerAlwaysMisses() {
    printf("[2] 记录进 A / 查询落在 B -> 恒 0 页(vision 事故的机制)\n");
    RoutingTestManager envA(16, 4, 2, 3);
    RoutingTestManager envB(16, 4, 2, 3);

    std::vector<int> tokens = envA.MakeTokens(4, 1);
    std::vector<int> pages = {0, 1, 2, 3};
    envA.Get().Record(tokens, pages);

    std::vector<int> hitA, hitB;
    envA.Get().Query(tokens, hitA);
    envB.Get().Query(tokens, hitB);
    Check(hitA.size() == 4, "同一个 manager 上查得到");
    Check(hitB.empty(),
          "另一个 manager 上查不到(且没有任何报错 -> 必须靠断言兜住)");
}

// [3] manager 注册表是按**全局层号**索引的: 两条前向路径只要层号基址不同,
//     拿到的就是两个不同的池子。这条断言把"基址必须一致"变成可执行条件。
void TestRegistryIsKeyedByGlobalLayerIndex() {
    printf("[3] 注册表按全局层号索引: 基址不同 = 两个互不相交的池子\n");
    fastllm::Data descriptor(fastllm::DataType::FLOAT32);
    descriptor.dims = {2, 1, 3};
    descriptor.UpdateUnitSize();

    const int layer = 5;
    const int base = 3000000;
    const int lookupIndex = (base + layer) * 2;   // 查询侧/ForwardGPU 的算法
    const int legacyIndex = layer * 2;            // 曾经硬编码的 i*2

    fastllm::PagedCacheManager *viaLookup =
        fastllm::AllocatePagedCacheManager(
            lookupIndex,
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            descriptor, /*pageLen=*/4, /*maxPages=*/16);
    fastllm::PagedCacheManager *viaLegacy =
        fastllm::AllocatePagedCacheManager(
            legacyIndex,
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            descriptor, /*pageLen=*/4, /*maxPages=*/16);

    Check(viaLookup != nullptr && viaLegacy != nullptr,
          "两个层号都能分配出 manager");
    Check(viaLookup != viaLegacy,
          "不同全局层号 -> 不同 manager 对象(基址错了就是两个池子)");
    Check(fastllm::GetPagedCacheManager(lookupIndex) == viaLookup,
          "GetPagedCacheManager 按同一个层号取回同一个对象");

    // 把"记录进 legacy 池子, 查询落在 lookup 池子"完整走一遍
    std::vector<int> tokens;
    for (int p = 0; p < 3; p++) {
        for (int i = 0; i < 4; i++) {
            tokens.push_back(p * 100 + i + 1);
        }
    }
    std::vector<int> pages = {0, 1, 2};
    viaLegacy->Record(tokens, pages);

    std::vector<int> hitLegacy, hitLookup;
    viaLegacy->Query(tokens, hitLegacy);
    viaLookup->Query(tokens, hitLookup);
    Check(hitLegacy.size() == 3, "记录侧池子里查得到 3 页");
    Check(hitLookup.empty(),
          "查询侧池子里 0 页 -> 命中率恒为 0(这就是线上的 miss=no-record)");
}

// [4] numPages == 0 的记录(token 不足一页)不应该往 trie 里塞任何东西,
//     否则短请求会在 trie 根上留下垃圾节点。
void TestRecordShorterThanOnePageIsNoop() {
    printf("[4] 不足一页的记录是 no-op\n");
    RoutingTestManager env(16, 4, 2, 3);
    std::vector<int> tokens = {1, 2, 3};   // < pageLen
    std::vector<int> pages = {0};
    env.Get().Record(tokens, pages);
    Check(env.Get().pageToTrieNode.empty(),
          "trie 里没有任何页(不足一页不记录)");
}

}  // namespace

int main() {
    printf("== 前缀缓存记录/查询路由回归 ==\n");
    TestRecordThenQueryStrictExtension();
    TestQueryOnAnotherManagerAlwaysMisses();
    TestRegistryIsKeyedByGlobalLayerIndex();
    TestRecordShorterThanOnePageIsNoop();
    printf("== %d checks, %d failures ==\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
