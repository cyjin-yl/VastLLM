// 前缀缓存三级下放/上提(L1 显存 -> L2 CPU 内存 -> L3 磁盘)的正确性回归。
//
// 为什么需要它: 这一层的所有故障都是**静默**的 —— 不报错、不崩溃, 只是缓存
// 命中率恒为 0, 而后端看起来一切正常。线上表现是 metrics 里长期的
//     kv_pool=54144/65536 pg (83%) L1trie=0 pg (~0 tok) ... ckpt=0 hits=0
// (偶尔跳到 L1trie=2080 pg 又立刻回落到 0)。靠"能编译 + 服务没挂"完全判断
// 不出对错, 必须把失败条件固化成可执行断言。
//
// 覆盖的四类真实故障:
//   1. 页字节数除以 maxPages(逻辑预算)而不是 dims[0](物理页数)
//      -> 池子顶到 FASTLLM_PAGED_POOL_MAX_MB 天花板后 dims[0] < maxPages,
//         下放要么被判成 "manager-state" 直接拒绝, 要么按错误的 offset/长度
//         抠一段存下去(上提永远对不上) -> L2/L3 恒为空, 冷页只能被丢弃。
//   2. 下放顺序反了: "够热就直接写盘, 否则才进内存"
//      -> 越热的前缀被放到越慢的一层; 且 CPU 层没有向下出口。
//   3. CPU 层满了直接 reset() 丢弃载荷
//      -> 每次内存紧张都把 L2 整层抹掉, 磁盘层永远拿不到运行期数据。
//   4. 没有滞回: 每缺一页就淘汰一页, 且不看"这一页在 L1 待了多久"
//      -> 刚上提回来的页立刻被踢下去, 上提/下放来回抖动。
//
// 构建: cmake -DUNIT_TEST=ON .. && cmake --build . --target testPrefixCacheTier
// 运行: ./testPrefixCacheTier   (退出码非 0 = 有用例失败)

#include "fastllm.h"
#include "prefixcache_persistence.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <filesystem>
#include <string>
#include <unordered_set>
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

void SetEnv(const char *name, const char *value) {
    setenv(name, value, 1);
}

// 复现线上几何: 物理页数(dims[0]) < 逻辑预算(maxPages)。
// 生产里这是常态 —— AllocatePagedCacheManager 只先分配
// initialPages = min(128, maxPages), 之后靠 Grow() 追赶; 而 Grow 一旦被
// FASTLLM_PAGED_POOL_MAX_MB 挡住(实测 pool=6590MB / budget=6600MB),
// dims[0] 就永远追不上 maxPages。
class TierTestManager {
public:
    TierTestManager(int physicalPages, int logicalMaxPages,
                    int pageLen, int numHeads, int headDim)
            : pageLenValue(pageLen) {
        fastllm::Data *data = (fastllm::Data*)&manager;
        data->dataType = fastllm::DataType::FLOAT32;
        data->UpdateUnitSize();
        data->Resize({physicalPages, pageLen, numHeads, headDim});
        data->Allocate();
        manager.pageLen = pageLen;
        manager.type =
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        // SetMaxPages 会按物理页数建好 freePages / pageTimestamp / trieRoot
        manager.SetMaxPages(physicalPages);
        // 再把逻辑预算抬上去, 制造 dims[0] < maxPages
        manager.maxPages = logicalMaxPages;
    }

    fastllm::PagedCacheManager &Get() { return manager; }

    size_t RealPageBytes() const {
        return (size_t)manager.GetBytes() / (size_t)manager.dims[0];
    }

    uint8_t *PageData(int pageIndex) {
        return manager.cpuData + (size_t)pageIndex * RealPageBytes();
    }

    // 给某一页填一个可辨认的字节模式
    void FillPage(int pageIndex, uint8_t seed) {
        uint8_t *p = PageData(pageIndex);
        const size_t n = RealPageBytes();
        for (size_t i = 0; i < n; i++) {
            p[i] = (uint8_t)(seed + (uint8_t)(i * 7));
        }
    }

    std::vector<uint8_t> SnapshotPage(int pageIndex) {
        uint8_t *p = PageData(pageIndex);
        return std::vector<uint8_t>(p, p + RealPageBytes());
    }

    // 长度为 pages*pageLen 的 token 序列, 前缀由 prefixSeed 决定
    std::vector<int> MakeTokens(int pages, int tailSeed) {
        std::vector<int> tokens;
        for (int p = 0; p < pages; p++) {
            for (int i = 0; i < pageLenValue; i++) {
                // 最后一页用 tailSeed 区分, 前面的页保持相同 -> 共享前缀
                tokens.push_back(
                    p + 1 == pages
                        ? tailSeed * 1000 + i
                        : p * 100 + i + 1);
            }
        }
        return tokens;
    }

private:
    fastllm::PagedCacheManager manager;
    int pageLenValue;
};

fastllm::CacheTrieNode *NodeOfPage(
        fastllm::PagedCacheManager &manager, int pageIndex) {
    auto it = manager.pageToTrieNode.find(pageIndex);
    return it == manager.pageToTrieNode.end() ? nullptr : it->second;
}

// -------------------------------------------------------------- 用例

// [1] 几何(静默损坏型): dims[0] 整除 maxPages 的情况。
// 旧代码 pageBytes = GetBytes()/maxPages 会把 96 字节算成 12 字节,
// 于是"下放成功"了但内容是错的, 上提永远对不上。
void TestGeometrySilentCorruption() {
    printf("[1] 下放的页字节数必须按 dims[0] 换算(静默损坏型几何)\n");
    // pageBytes = 4(pageLen) * 2(heads) * 3(dim) * 4(float32) = 96
    // dims[0]=8 -> GetBytes()=768; 768 % 64(maxPages) == 0 -> 旧代码不报错,
    // 只是把 pageBytes 算成 768/64 = 12。
    TierTestManager env(/*physicalPages=*/8, /*logicalMaxPages=*/64,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();
    Check(env.RealPageBytes() == 96, "真实页字节数 = 96");
    Check(m.dims[0] == 8 && m.maxPages == 64,
          "几何: dims[0]=8 < maxPages=64 (复现线上顶天花板的状态)");

    env.FillPage(0, 0x11);
    env.FillPage(1, 0x77);
    std::vector<int> tokens = env.MakeTokens(2, 5);
    m.Record(tokens, {0, 1});

    fastllm::CacheTrieNode *leaf = NodeOfPage(m, 1);
    Check(leaf != nullptr, "第 2 页在 trie 里有节点");
    if (leaf == nullptr) {
        return;
    }
    const bool ok = m.PageOutTrieNode(leaf);
    Check(ok, "PageOutTrieNode 下放成功");
    Check(leaf->tierPayload != nullptr,
          "落在 L2(CPU 内存)层");
    if (leaf->tierPayload != nullptr) {
        Check(leaf->tierPayload->uncompressedBytes == 96,
              "记录的未压缩页大小 = 96 (旧代码会记成 12)");
    }
}

// [2] 几何(直接拒绝型): dims[0] 不整除 maxPages。
// 旧代码走 managerBytes % maxPages != 0 -> "manager-state" -> 下放全灭。
void TestGeometryRejectMode() {
    printf("[2] 下放的页字节数必须按 dims[0] 换算(直接拒绝型几何)\n");
    // dims[0]=5 -> GetBytes()=480; 480 % 64 = 32 != 0 -> 旧代码直接拒绝
    TierTestManager env(/*physicalPages=*/5, /*logicalMaxPages=*/64,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();
    Check(m.GetBytes() % (uint64_t)m.maxPages != 0,
          "构造出 GetBytes() 不整除 maxPages 的几何");

    env.FillPage(0, 0x22);
    env.FillPage(1, 0x33);
    std::vector<int> tokens = env.MakeTokens(2, 6);
    m.Record(tokens, {0, 1});

    fastllm::CacheTrieNode *leaf = NodeOfPage(m, 1);
    Check(leaf != nullptr, "第 2 页在 trie 里有节点");
    if (leaf == nullptr) {
        return;
    }
    Check(m.PageOutTrieNode(leaf),
          "PageOutTrieNode 下放成功(旧代码在这里返回 false)");
}

// [3] 下放 -> 上提 的内容必须逐字节一致(走真实的淘汰路径)
void TestDemotePromoteRoundTrip() {
    printf("[3] 下放再上提: 页内容逐字节一致\n");
    TierTestManager env(/*physicalPages=*/8, /*logicalMaxPages=*/64,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();

    env.FillPage(0, 0x41);
    env.FillPage(1, 0x59);
    std::vector<int> tokens = env.MakeTokens(2, 7);
    m.Record(tokens, {0, 1});
    const std::vector<uint8_t> expected = env.SnapshotPage(1);

    fastllm::CacheTrieNode *leaf = NodeOfPage(m, 1);
    if (leaf == nullptr) {
        Check(false, "第 2 页在 trie 里有节点");
        return;
    }
    // 走真实的淘汰路径(会调用 PageOutTrieNode 并把页还给 freePages)
    const int evicted = m.EvictOneColdPageLocked(nullptr, true);
    Check(evicted == 1, "淘汰选中了第 2 页(唯一叶子)");
    Check(leaf->pageId == -1,
          "节点仍在 trie 里, 只是不再常驻 L1 (没有被删掉)");
    Check(leaf->tierPayload != nullptr || leaf->tierDisk != nullptr,
          "页已下放到 L2 或 L3, 而不是被丢弃");

    // 把原来那块显存/内存涂掉, 确保上提读的是下层副本而不是残留
    env.FillPage(1, 0xEE);

    std::unordered_set<int> protectedPages;
    const bool promoted = m.MaterializeTrieNode(leaf, protectedPages);
    Check(promoted, "MaterializeTrieNode 上提成功");
    if (!promoted || leaf->pageId < 0) {
        return;
    }
    const std::vector<uint8_t> restored = env.SnapshotPage(leaf->pageId);
    Check(restored.size() == expected.size() &&
              std::memcmp(restored.data(), expected.data(),
                          expected.size()) == 0,
          "上提回来的页内容与下放前逐字节一致");
}

// [4] 下放顺序: 先 L2(内存), 不能因为"够热"就直接跳到 L3(磁盘)
void TestCpuTierBeforeDisk() {
    printf("[4] 下放顺序: 热前缀也要先进 L2, 不能直接落 L3\n");
    SetEnv("FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES", "1048576");
    TierTestManager env(/*physicalPages=*/8, /*logicalMaxPages=*/64,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();
    env.FillPage(0, 0x01);
    env.FillPage(1, 0x02);
    std::vector<int> tokens = env.MakeTokens(2, 8);
    m.Record(tokens, {0, 1});

    fastllm::CacheTrieNode *leaf = NodeOfPage(m, 1);
    if (leaf == nullptr) {
        Check(false, "第 2 页在 trie 里有节点");
        return;
    }
    // 造一个"很热"的节点: 旧代码在 accessCount >= DISK_MIN_HITS 时会
    // 直接把它写到磁盘, 完全跳过内存层。
    leaf->accessCount = 100;
    Check(m.PageOutTrieNode(leaf), "下放成功");
    Check(leaf->tierPayload != nullptr,
          "热前缀落在 L2(CPU 内存), 而不是被直接甩到磁盘");
    Check(leaf->tierDisk == nullptr,
          "L2 装得下时不应该同时占用 L3");
}

// [5] L2 满 -> 轮转到 L3, 而不是把载荷丢掉
void TestCpuOverflowRotatesToDisk() {
    printf("[5] L2 满了要轮转到 L3, 不能直接丢弃\n");
    TierTestManager env(/*physicalPages=*/8, /*logicalMaxPages=*/64,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();
    for (int i = 0; i < 4; i++) {
        env.FillPage(i, (uint8_t)(0x60 + i));
    }
    // 两条共享首页的分支 -> root -> p0 -> {p1, p2}, 两个叶子
    m.Record(env.MakeTokens(2, 11), {0, 1});
    m.Record(env.MakeTokens(2, 12), {0, 2});

    fastllm::CacheTrieNode *leafA = NodeOfPage(m, 1);
    fastllm::CacheTrieNode *leafB = NodeOfPage(m, 2);
    if (leafA == nullptr || leafB == nullptr) {
        Check(false, "两个叶子节点都建立了");
        return;
    }
    // 把 CPU 层额度压到"只够一份载荷"
    const uint64_t before = fastllm::GetPagedPrefixCacheCpuTierBytes();
    char budget[64];
    std::snprintf(budget, sizeof(budget), "%llu",
                  (unsigned long long)(before + 96));
    SetEnv("FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES", budget);
    SetEnv("FASTLLM_PREFIX_CACHE_ZSTD", "0");
    // 本用例验证的是"已满足 L3 准入时的 L2 -> L3 轮转"。4-token
    // 合成页按机械盘默认寻道成本本来不值得落盘；降低重算速度，让存储
    // 明确胜出，避免把经济性门槛误测成轮转失败。
    SetEnv("FASTLLM_PREFIX_CACHE_RECOMPUTE_TPS", "1");

    const bool okA = m.PageOutTrieNode(leafA);
    const bool okB = m.PageOutTrieNode(leafB);
    Check(okA && okB, "两页都成功下放(没有一页被丢弃)");
    Check(leafA->tierPayload != nullptr || leafA->tierDisk != nullptr,
          "第一页仍有下层副本");
    Check(leafB->tierPayload != nullptr || leafB->tierDisk != nullptr,
          "第二页仍有下层副本");
    Check(leafA->tierDisk != nullptr || leafB->tierDisk != nullptr,
          "额度只够一份时, 至少有一页轮转到了 L3(磁盘)");
    SetEnv("FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES", "1048576");
    SetEnv("FASTLLM_PREFIX_CACHE_ZSTD", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_RECOMPUTE_TPS", "800");
}

// [6] 滞回: 最小驻留时间内的页不做淘汰候选
void TestMinResidencyHysteresis() {
    printf("[6] 滞回: 刚常驻的页不能立刻被踢下去\n");
    SetEnv("FASTLLM_KV_MIN_RESIDENCY_MS", "600000");
    TierTestManager env(/*physicalPages=*/8, /*logicalMaxPages=*/64,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();
    for (int i = 0; i < 4; i++) {
        env.FillPage(i, (uint8_t)(0x90 + i));
    }
    m.Record(env.MakeTokens(2, 21), {0, 1});
    m.Record(env.MakeTokens(2, 22), {0, 2});

    fastllm::CacheTrieNode *leafA = NodeOfPage(m, 1);
    fastllm::CacheTrieNode *leafB = NodeOfPage(m, 2);
    if (leafA == nullptr || leafB == nullptr) {
        Check(false, "两个叶子节点都建立了");
        return;
    }
    // A: 更热(accessCount 大), 但已经常驻很久 -> 该被淘汰
    // B: 更冷, 但刚刚才常驻(比如刚从 L2/L3 上提回来) -> 本轮必须保住
    leafA->accessCount = 50;
    leafA->residentSinceMs = 1;
    leafB->accessCount = 1;
    // leafB->residentSinceMs 保持 Record 刚打的点(= 现在), 即"刚常驻"

    const int victim = m.EvictOneColdPageLocked(nullptr, false);
    Check(victim == 1,
          "保护期内: 选中常驻已久的 A(页 1), 而不是更冷但刚常驻的 B(页 2)");

    // A 已经被淘汰走了, trie 里剩下的候选(页 0 / 页 2)全部处在保护期内
    const int victim2 = m.EvictOneColdPageLocked(nullptr, false);
    Check(victim2 < 0,
          "剩下的候选全在保护期内 -> 带保护的淘汰返回 -1(不硬吃缓存)");

    const int victim3 = m.EvictOneColdPageLocked(nullptr, true);
    Check(victim3 >= 0,
          "放宽保护期后仍能淘汰出页(保护期不是硬承诺, 不会把缺页变成硬错误)");

    SetEnv("FASTLLM_KV_MIN_RESIDENCY_MS", "0");
}

// [7] 滞回: 批量补页, 不是"缺一页放一页"
void TestBatchRefillNotOnePageAtATime() {
    printf("[7] 滞回: 取页缺页时批量补, 不是缺一页放一页\n");
    SetEnv("FASTLLM_KV_MIN_RESIDENCY_MS", "0");
    // 这里必须让 dims[0] == maxPages(池子已经顶到天花板, 扩不动了)。
    // 池子还能扩容时, 取页路径只淘汰 1 页救急 —— "预算内优先扩容"优先级更高,
    // 逐出一页冷前缀 = 丢掉一段已经算好的 KV, 比申请显存贵得多。
    // 批量补页正是为"扩不动了"这个状态准备的, 也正是线上的状态。
    TierTestManager env(/*physicalPages=*/8, /*logicalMaxPages=*/8,
                        /*pageLen=*/4, /*numHeads=*/2, /*headDim=*/3);
    fastllm::PagedCacheManager &m = env.Get();
    for (int i = 0; i < 8; i++) {
        env.FillPage(i, (uint8_t)(0xA0 + i));
    }
    // 8 条共享首页的分支, 把 8 个物理页全部记进 trie
    m.Record(env.MakeTokens(2, 31), {0, 1});
    for (int i = 2; i < 8; i++) {
        m.Record(env.MakeTokens(2, 30 + i), {0, i});
    }
    Check(m.freePages.empty(), "所有物理页都进了 trie, freePages 见底");

    const int page = m.GetUnusedPageIndexLocked(true, nullptr);
    Check(page >= 0, "缺页时仍能取到页");
    // FASTLLM_KV_RECYCLE_BATCH_PAGES=4 -> 一次淘汰 4 页, 取走 1 页后还剩 3
    Check((int)m.freePages.size() == 3,
          "一次补够一批(4 页), 取走 1 页后 freePages 还剩 3 —— "
          "旧代码只淘汰 1 页, 这里会是 0");
}

// [8] L3 配额: 不够时先回收陈旧 generation, 不能直接拒写把自己锁死
void TestPersistentDiskQuotaReclaims() {
    printf("[8] L3 磁盘配额: 写不下时先回收历史 generation, 不能自锁\n");
    std::error_code ec;
    const std::filesystem::path root =
        std::filesystem::temp_directory_path(ec) /
        ("fastllm-quota-" + std::to_string((long long)getpid()));
    std::filesystem::remove_all(root, ec);

    // 每代 200KiB, 配额 512KiB。提交后的既有回收保留 2 代(当前 + 1 代),
    // 于是稳态占用 400KiB —— 再写第 3 代时 400KiB > 512-200=312KiB,
    // 旧代码在这里直接判超限返回失败, 而"回收"写在提交成功之后, 永远轮不到,
    // 目录再也缩不回去 = 自锁。
    //
    // 注意: 线上那次"连续 6 次 shutdown checkpoint failed"**不是**这个自锁,
    // 而是配额单纯定得太小(活跃 root 1884.81MB / 配额 2GiB, 余量仅 163MB,
    // 装不下 262K 上下文的 checkpoint), 现已把配额提到 32GiB。
    // 这个用例守的是另一件事: 配额无论多大, 一旦真的顶到线, 旧顺序只会
    // **永久失败**而不是回收后降级 —— 这里把"能自己走出来"固化成断言。
    const size_t payloadBytes = 200 * 1024;
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES", "524288");
    unsetenv("FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES");

    // 必须是不可压缩的数据, 否则 zstd 把它压没了就复现不出配额压力
    std::vector<uint8_t> bytes(payloadBytes);
    uint64_t lcg = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < bytes.size(); i++) {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        bytes[i] = (uint8_t)(lcg >> 33);
    }

    int committed = 0;
    std::string firstError;
    for (uint64_t generation = 1; generation <= 5; generation++) {
        fastllm::PersistedPrefixCacheGeneration source;
        source.generation = generation;
        source.cacheKey = "tier-test";
        fastllm::PersistentPayloadRecord record;
        record.kind = fastllm::PersistentPayloadKind::PAGED_CACHE_PAGE;
        record.name = "page-" + std::to_string(generation);
        record.bytes = bytes;
        source.payloads.push_back(record);

        std::vector<fastllm::PersistentPayloadRef> refs;
        std::string error;
        if (fastllm::CommitPersistentPrefixCacheGeneration(
                root, source, refs, &error)) {
            committed++;
        } else if (firstError.empty()) {
            firstError = "gen " + std::to_string(generation) + ": " + error;
        }
    }
    if (!firstError.empty()) {
        printf("       首个失败: %s\n", firstError.c_str());
    }
    Check(committed == 5,
          "连续 5 代都提交成功(旧代码在第 3 代开始永久失败)");

    // 回收确实发生了: 目录不该无限膨胀
    uint64_t total = 0;
    for (std::filesystem::recursive_directory_iterator it(root, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            total += (uint64_t)it->file_size(ec);
        }
    }
    printf("       目录实际占用 = %llu B (配额 524288 B)\n",
           (unsigned long long)total);
    Check(total <= 524288, "回收之后目录占用没有超过配额");

    std::filesystem::remove_all(root, ec);
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES", "67108864");
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES", "1024");
}

}  // namespace

int main() {
    printf("== 前缀缓存三级下放/上提 正确性回归 ==\n");

    // 注意: GetUnusedPageIndexLocked 里的 batch 页数、批量回收的冷却期都是
    // 函数内 static(只读一次 env), 因此这些开关必须在任何用例之前设好。
    SetEnv("FASTLLM_KV_RECYCLE_BATCH_PAGES", "4");
    SetEnv("FASTLLM_KV_RECYCLE_COOLDOWN_MS", "0");
    SetEnv("FASTLLM_KV_MIN_RESIDENCY_MS", "0");

    SetEnv("FASTLLM_PREFIX_CACHE_CPU_TIER", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES", "1048576");
    SetEnv("FASTLLM_PREFIX_CACHE_MIN_HITS", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_MIN_TOKENS", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MIN_HITS", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MIN_TOKENS", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES", "1024");
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES", "67108864");
    // 共享 host 预算默认关; 显式关掉避免受外部 profile 影响
    unsetenv("FASTLLM_HOST_SUSPEND_CACHE");

    std::error_code ec;
    const std::filesystem::path diskDir =
        std::filesystem::temp_directory_path(ec) /
        ("fastllm-tiertest-" + std::to_string((long long)getpid()));
    std::filesystem::create_directories(diskDir, ec);
    SetEnv("FASTLLM_PREFIX_CACHE_DISK_DIR", diskDir.string().c_str());

    TestGeometrySilentCorruption();
    TestGeometryRejectMode();
    TestDemotePromoteRoundTrip();
    TestCpuTierBeforeDisk();
    TestCpuOverflowRotatesToDisk();
    TestMinResidencyHysteresis();
    TestBatchRefillNotOnePageAtATime();
    TestPersistentDiskQuotaReclaims();

    std::filesystem::remove_all(diskDir, ec);

    printf("\n%d/%d 通过\n", g_checks - g_failures, g_checks);
    if (g_failures > 0) {
        printf("有 %d 个断言失败\n", g_failures);
        return 1;
    }
    return 0;
}
