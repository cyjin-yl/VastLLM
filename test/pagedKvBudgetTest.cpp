// 分页 KV 显存账本与 GDN 快照对齐的正确性回归。
//
// 为什么需要它: 这两处的错误都是**静默**的 —— 不报错、不崩溃, 只是上下文容量
// 凭空少一截 / 命中后输出悄悄变差。靠"能编译 + 服务没挂"完全判断不出对错。
//
// 覆盖两类真实故障:
//
//   1. 预算守卫用 unitSize 估算页字节数。
//      打包 KV 类型(Q8_0_KV / TURBO3_KV / TURBO4_KV)的 UpdateUnitSize() 一律把
//      unitSize 设成 1 —— 它是**占位值**, 不是每元素字节数。真实字节数由
//      GetKVCacheRowBytes 的分块布局决定。用 unitSize 估算的结果对 q8_0 少算
//      5.9%、对 turbo3 多算 156%, 合计高估 1.376 倍, 于是
//      AllocatePagedCacheManager 的预算守卫把 maxPages 砍到实际能放下的 72.7%。
//      现场表现只有一句 "clamping maxPages A -> B", 会把人误导去调 --tokens。
//
//   2. GDN(线性注意力)递归状态快照标称长度与状态真实位置不一致。
//      递归状态无法截断到页边界。把标称长度向下取整、却不动状态本身, 命中恢复
//      时区间 [alignedLen, currentLenRaw) 的 token 会被卷进递归状态两次。
//      最大 127 个 token, 静默算错, 无任何日志。
//
// 构建: cmake -DUNIT_TEST=ON .. && cmake --build . --target testPagedKvBudget
// 运行: ./testPagedKvBudget   (退出码非 0 = 有用例失败)

#include "fastllm.h"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

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

void CheckEq(long long got, long long want, const char *what) {
    g_checks++;
    if (got == want) {
        printf("  ok   %s (= %lld)\n", what, got);
    } else {
        printf("  FAIL %s: got %lld, want %lld\n", what, got, want);
        g_failures++;
    }
}

// 生产几何(Qwen3.8-27B on V100): pageLen=128, head_count_kv=4, head_dim=256。
constexpr int kPageLen = 128;
constexpr int kNumHeads = 4;
constexpr int kHeadDim = 256;

// 全注意力层数: block_count=65, full_attention_interval=4 => 16 层。
// 每层有 K / V 两个 PagedCacheManager, 合计 32 个池。
constexpr int kFullAttnLayers = 16;

long long PageBytes(fastllm::DataType type) {
    return (long long)fastllm::GetPagedPoolPageBytes(
        type, kPageLen, kNumHeads, kHeadDim);
}

// 被替换掉的旧算法, 只在测试里保留, 用来把"回退就会失败"固化成断言。
long long LegacyPageBytesFromUnitSize(int unitSize) {
    return (long long)kPageLen * kNumHeads * kHeadDim * unitSize;
}

void TestPackedPageBytes() {
    printf("[1] 分页 KV 每页字节数(打包布局)\n");

    // q8_0: 256 列 / 32 = 8 块, 每块 2(fp16 scale) + 32(int8) = 34 字节 => 272 B/行
    //       一页 = 128 token * 4 head = 512 行 => 512 * 272 = 139264 B
    CheckEq(PageBytes(fastllm::DataType::Q8_0_KV), 139264,
            "q8_0 每页字节数");
    // turbo3: 256 列 / 128 = 2 块, 每块 2 + 32 + 16 = 50 字节 => 100 B/行
    CheckEq(PageBytes(fastllm::DataType::TURBO3_KV), 51200,
            "turbo3 每页字节数");
    // turbo4: 每块 2 + 64 = 66 字节 => 132 B/行
    CheckEq(PageBytes(fastllm::DataType::TURBO4_KV), 67584,
            "turbo4 每页字节数");
    // 非打包类型: 直接 行*列*每元素字节
    CheckEq(PageBytes(fastllm::DataType::FP8_E4M3), 131072,
            "fp8_e4m3 每页字节数");
    CheckEq(PageBytes(fastllm::DataType::FLOAT16), 262144,
            "float16 每页字节数");

    // 边界: 非法几何返回 0, 不能返回负数或炸掉。
    CheckEq((long long)fastllm::GetPagedPoolPageBytes(
                fastllm::DataType::TURBO3_KV, 0, kNumHeads, kHeadDim), 0,
            "pageLen=0 返回 0");
    CheckEq((long long)fastllm::GetPagedPoolPageBytes(
                fastllm::DataType::TURBO3_KV, kPageLen, -1, kHeadDim), 0,
            "numHeads<0 返回 0");
}

void TestLegacyFormulaWouldRegress() {
    printf("[2] 反向验证: 回退成 unitSize 估算必然算错\n");

    // 打包 KV 类型的 unitSize 恒为 1(见 Data::UpdateUnitSize 的
    // IsPackedKVCacheDataType 分支)。旧算法因此对所有打包类型给出同一个数。
    const long long legacy = LegacyPageBytesFromUnitSize(1);
    CheckEq(legacy, 131072, "旧算法(unitSize=1)对任何打包类型都给出同一个数");

    // 这三条断言就是"改回去必然失败"的固化: 只要有人把预算守卫改回
    // pageLen*numHeads*headDim*unitSize, 下面三条立刻红。
    Check(PageBytes(fastllm::DataType::Q8_0_KV) != legacy,
          "q8_0 真实值 != 旧算法(旧算法少算 5.9%)");
    Check(PageBytes(fastllm::DataType::TURBO3_KV) != legacy,
          "turbo3 真实值 != 旧算法(旧算法多算 156%)");
    Check(PageBytes(fastllm::DataType::TURBO4_KV) != legacy,
          "turbo4 真实值 != 旧算法");

    // 打包类型必须被识别成打包, 否则 GetDataBytes 会走 unitSize 分支。
    Check(fastllm::IsPackedKVCacheDataType(fastllm::DataType::Q8_0_KV),
          "Q8_0_KV 被识别为打包 KV");
    Check(fastllm::IsPackedKVCacheDataType(fastllm::DataType::TURBO3_KV),
          "TURBO3_KV 被识别为打包 KV");
    Check(!fastllm::IsPackedKVCacheDataType(fastllm::DataType::FP8_E4M3),
          "FP8_E4M3 不是打包 KV");
}

void TestWholePoolAccounting() {
    printf("[3] 整池账本: 16 个全注意力层 x (K + V)\n");

    // 生产配置: K=q8_0, V=turbo3。
    const long long realPerPage =
        (long long)kFullAttnLayers *
        (PageBytes(fastllm::DataType::Q8_0_KV) +
         PageBytes(fastllm::DataType::TURBO3_KV));
    CheckEq(realPerPage, 3047424, "K=q8_0 + V=turbo3 全池每页字节数");

    // 旧算法: 32 个池 x 131072
    const long long legacyPerPage =
        (long long)kFullAttnLayers * 2 * LegacyPageBytesFromUnitSize(1);
    CheckEq(legacyPerPage, 4194304, "旧算法全池每页字节数");

    // 高估倍率 1.376 => 预算守卫把 maxPages 砍到 72.7%。
    // 用整数比较避免浮点抖动: legacy/real 应落在 1.37~1.38 之间。
    Check(legacyPerPage * 1000 / realPerPage == 1376,
          "旧算法整体高估 1.376 倍(即容量凭空少 27%)");

    // 每 token 字节数: 全池每页 / pageLen
    CheckEq(realPerPage / kPageLen, 23808, "每 token 字节数(K=q8_0 + V=turbo3)");

    // 262144 上下文的真实 KV 占用。这是"一条满长上下文就吃掉整个池"的依据:
    // 生产 FASTLLM_PAGED_POOL_MAX_MB=6600 (MiB), 而这里是 5952 MiB。
    const long long pages262k = 262144 / kPageLen;   // 2048 页
    const long long bytes262k = pages262k * realPerPage;
    CheckEq(bytes262k / 1048576, 5952, "262144 上下文的 KV 占用(MiB)");
}

void TestFp8IsNotCheaper() {
    printf("[4] 方向判断固化: fp8_e4m3 KV 并不比 q8_0+turbo3 省\n");

    // 线索#1 要回答的问题: 打通 MTP + fp8_e4m3 有没有显存收益。
    // 答案是没有 —— fp8 反而更费。把这个结论固化成断言, 免得以后有人
    // 又花时间去"修" MTP/fp8 不兼容, 以为那是省显存的路。
    const long long current =
        PageBytes(fastllm::DataType::Q8_0_KV) +
        PageBytes(fastllm::DataType::TURBO3_KV);          // 190464
    const long long fp8Both = PageBytes(fastllm::DataType::FP8_E4M3) * 2;  // 262144

    Check(fp8Both > current,
          "fp8_e4m3 双边比 q8_0+turbo3 更费显存(不是更省)");
    Check(fp8Both * 100 / current == 137,
          "fp8_e4m3 双边多占 37%");

    // 真正省显存的方向在 K 侧: K 现在是 8.5 bit/值, 占 KV 的 73%;
    // V 已经压到 3.125 bit/值。把 K 换成 turbo4 才是量级动作。
    Check(PageBytes(fastllm::DataType::Q8_0_KV) * 100 / current == 73,
          "K(q8_0) 占当前 KV 字节的 73%");
    const long long turbo4K =
        PageBytes(fastllm::DataType::TURBO4_KV) +
        PageBytes(fastllm::DataType::TURBO3_KV);
    Check(turbo4K * 100 / current == 62,
          "K 换成 turbo4 可降到 62%(省 38%)");
}

void TestSnapshotAlignment() {
    printf("[5] GDN 快照标称长度: 未对齐必须拒绝\n");

    int aligned = -1;

    // 分块 prefill 的记录点: chunk 由 GetChunkedPrefillSize() 保证是 pageLen
    // 的整数倍, 天然对齐 -> 必须接受。
    Check(fastllm::PagedPrefixSnapshotLengthUsable(1024, kPageLen, &aligned),
          "1024(chunk 边界)可记录");
    CheckEq(aligned, 1024, "1024 对齐后仍是 1024");

    Check(fastllm::PagedPrefixSnapshotLengthUsable(kPageLen, kPageLen, &aligned),
          "正好一页可记录");
    CheckEq(aligned, kPageLen, "一页对齐后是 128");

    Check(fastllm::PagedPrefixSnapshotLengthUsable(262144, kPageLen, &aligned),
          "满长上下文可记录");

    // 生成结束时的记录点: currentLenRaw = promptLen + 已生成, 几乎必然不对齐。
    // 此时 GDN 递归状态吃过 1000 个 token, 而标称长度只有 896 ——
    // 命中恢复后 [896, 1000) 这 104 个 token 会被卷进递归状态两次。
    //
    // 默认**不拒绝**(保住单调增长会话的命中), 严格模式才拒绝。
    // 两种模式下 alignedLen 的写回都必须正确。
    const bool strict = fastllm::PagedPrefixSnapshotStrictAlign();
    printf("  (严格对齐模式 = %s)\n", strict ? "开" : "关");

    CheckEq(fastllm::PagedPrefixSnapshotLengthUsable(1000, kPageLen, &aligned) ? 1 : 0,
            strict ? 0 : 1,
            "1000(生成结束, 未对齐): 严格模式拒绝, 默认接受");
    CheckEq(aligned, 896, "无论接受与否都写回对齐长度 896, 供统计/调试");

    CheckEq(fastllm::PagedPrefixSnapshotLengthUsable(1023, kPageLen, &aligned) ? 1 : 0,
            strict ? 0 : 1, "1023(差一个 token)");
    CheckEq(aligned, 896, "1023 对齐到 896");
    CheckEq(fastllm::PagedPrefixSnapshotLengthUsable(1025, kPageLen, &aligned) ? 1 : 0,
            strict ? 0 : 1, "1025(多一个 token)");
    CheckEq(aligned, 1024, "1025 对齐到 1024");

    // 不足一页: 没有任何可记录的前缀。
    Check(!fastllm::PagedPrefixSnapshotLengthUsable(127, kPageLen, &aligned),
          "127(不足一页)拒绝");
    CheckEq(aligned, 0, "不足一页时对齐长度为 0");

    // 退化输入不能崩。
    Check(!fastllm::PagedPrefixSnapshotLengthUsable(0, kPageLen, &aligned),
          "0 拒绝");
    Check(!fastllm::PagedPrefixSnapshotLengthUsable(-5, kPageLen, &aligned),
          "负数拒绝");

    // pageLen<=0 是"不分页"的退化配置, 此时不存在对齐问题, 恒接受。
    Check(fastllm::PagedPrefixSnapshotLengthUsable(1000, 0, &aligned),
          "pageLen=0 时不做对齐约束");
    CheckEq(aligned, 1000, "pageLen=0 时对齐长度即原长度");

    // 允许不关心 alignedLen(不能崩)。
    (void)fastllm::PagedPrefixSnapshotLengthUsable(1000, kPageLen, nullptr);
    Check(true, "alignedLen 传 nullptr 不崩");
    Check(!fastllm::PagedPrefixSnapshotLengthUsable(100, kPageLen, nullptr),
          "不足一页 + nullptr 仍恒拒绝");
}

// 复现生产几何: 物理页数(dims[0]) 远小于逻辑预算(maxPages)。
// 页池懒分配 —— AllocatePagedCacheManager 只先分配
// initialPages = min(128, maxPages), 之后靠 Grow() 追赶。
class PoolTestManager {
public:
    PoolTestManager(int physicalPages, int logicalMaxPages) {
        fastllm::Data *data = (fastllm::Data*)&manager;
        data->dataType = fastllm::DataType::FLOAT32;
        data->UpdateUnitSize();
        data->Resize({physicalPages, kPageLen, kNumHeads, kHeadDim});
        data->Allocate();
        manager.pageLen = kPageLen;
        manager.type =
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        manager.SetMaxPages(physicalPages);
        manager.maxPages = logicalMaxPages;   // 制造 dims[0] < maxPages
    }
    fastllm::PagedCacheManager &Get() { return manager; }
private:
    fastllm::PagedCacheManager manager;
};

void TestPoolWatermark() {
    printf("[6] 页池水位: 懒分配下必须按物理页算\n");

    // 线上实测的形状: maxPages=2048(--tokens 262144 / pageLen 128),
    // dims[0]=128(initialPages, 还没 Grow 过), 全部空闲。
    PoolTestManager mgr(128, 2048);
    fastllm::PagedCacheManager &m = mgr.Get();

    CheckEq(m.dims[0], 128, "物理页数 dims[0]");
    CheckEq(m.maxPages, 2048, "逻辑预算 maxPages");
    CheckEq(m.FreePageCount(), 128, "空闲页数(全空闲)");

    // 正确口径: 物理页 - 空闲页
    const long long correctUsed =
        (long long)m.dims[0] - (long long)m.FreePageCount();
    CheckEq(correctUsed, 0, "正确口径: 空载时已用 0 页");

    // 错误口径(被替换掉的那版): 逻辑预算 - 空闲页
    const long long legacyUsed =
        (long long)m.maxPages - (long long)m.FreePageCount();
    CheckEq(legacyUsed, 1920, "错误口径: 空载时报 1920 页已用");

    // 反向验证: 两者必须不同, 否则说明有人把口径改回去了。
    Check(correctUsed != legacyUsed,
          "两种口径必须不同(改回 maxPages 口径则此断言红)");

    // 32 个 manager 聚合后就是线上那条 metrics: 61440/65536 pg (94%)。
    // 修正后应该是 0/4096 pg (0%)。
    const int kManagers = kFullAttnLayers * 2;
    CheckEq(legacyUsed * kManagers, 61440,
            "错误口径聚合 = 线上实测的 61440");
    CheckEq((long long)m.maxPages * kManagers, 65536,
            "错误口径分母 = 线上实测的 65536");
    CheckEq(correctUsed * kManagers, 0, "正确口径聚合 = 0 页已用");
    CheckEq((long long)m.dims[0] * kManagers, 4096, "正确口径分母 = 4096 页");

    // 占用一半物理页后, 正确口径要跟着动。
    std::vector<int> picked;
    for (int i = 0; i < 64; i++) {
        picked.push_back(i);
    }
    m.Pick(picked);
    CheckEq((long long)m.dims[0] - (long long)m.FreePageCount(), 64,
            "取走 64 页后正确口径报 64 页已用");
    CheckEq((long long)m.dims[0], 128, "取页不改变物理池大小");
}

void TestTierAdmission() {
    printf("[7] 分层准入: 取回 vs 重算(寻道 + 并发 + margin)\n");

    // 本机实测常数(coordinator 提供): prefill 643 tok/s。
    // 每 token 字节数用**本文件已验证的真实值** 23808 B, 不是 33.9KB ——
    // 那个数来自 unitSize 估算(见用例[2]), 会把每 token 算成 32768 B。
    const double kRecomputeTps = 643.0;
    const double kNoDecompress = 0.0;
    const double kMargin = 0.2;

    // 一页 = 128 token 的全池字节数(K=q8_0 + V=turbo3)
    const size_t kPageStored = 3047424;          // 2.9 MiB
    const size_t kPageTokens = 128;

    // --- 机械盘(本机 /dev/sda ST16000NM000J, rotational=1) ---
    const double kHddSeek = 0.024;   // 实测中位 24ms
    const double kHddBw   = 41.0;    // 实测随机读 41 MB/s

    // 单请求: 一整页 2.9MiB, 重算 128/643 = 199ms; 取回 24 + 71 = 95ms -> 划算
    Check(fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored, kPageStored, false, kPageTokens,
              kHddBw, kNoDecompress, kRecomputeTps, kHddSeek, 1, kMargin),
          "机械盘 + 单请求 + 整页: 取回划算");

    // 8 并发: 寻道串行化 24*8 = 192ms, 加带宽 71ms = 263ms > 199ms -> 不划算
    Check(!fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored, kPageStored, false, kPageTokens,
              kHddBw, kNoDecompress, kRecomputeTps, kHddSeek, 8, kMargin),
          "机械盘 + 8 并发 + 整页: 取回反而亏(寻道串行化)");

    // 小节点(16 token): 重算只要 25ms, 而一次寻道就 24ms -> 永远不划算
    Check(!fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored / 8, kPageStored / 8, false, 16,
              kHddBw, kNoDecompress, kRecomputeTps, kHddSeek, 1, kMargin),
          "机械盘 + 小节点: 寻道就吃掉全部收益");

    // --- NVMe(本机 nvme0n1 KIOXIA, rotational=0) ---
    const double kNvmeSeek = 0.00008;
    const double kNvmeBw   = 3000.0;

    Check(fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored, kPageStored, false, kPageTokens,
              kNvmeBw, kNoDecompress, kRecomputeTps, kNvmeSeek, 1, kMargin),
          "NVMe + 单请求: 取回划算");
    Check(fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored, kPageStored, false, kPageTokens,
              kNvmeBw, kNoDecompress, kRecomputeTps, kNvmeSeek, 8, kMargin),
          "NVMe + 8 并发: 仍然划算(寻道可忽略)");
    // 同样的小节点, 在 NVMe 上就划算 —— 这正是"阈值不能写死"的证据
    Check(fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored / 8, kPageStored / 8, false, 16,
              kNvmeBw, kNoDecompress, kRecomputeTps, kNvmeSeek, 1, kMargin),
          "NVMe + 小节点: 划算(与机械盘结论相反)");

    // --- 退化输入不能误判成"划算" ---
    Check(!fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored, kPageStored, false, 0,
              kNvmeBw, kNoDecompress, kRecomputeTps, kNvmeSeek, 1, kMargin),
          "重算 0 个 token: 没有收益可言, 不划算");
    Check(!fastllm::PagedPrefixCacheStorageWinsPure(
              kPageStored, kPageStored, false, kPageTokens,
              0.0, kNoDecompress, kRecomputeTps, kNvmeSeek, 1, kMargin),
          "带宽为 0(未知设备): 不划算");

    // margin 必须真的起作用: 打平的场景不能算划算
    // 取回 100ms, 重算 100ms -> margin=0.2 要求取回 < 80ms
    Check(!fastllm::PagedPrefixCacheStorageWinsPure(
              (size_t)(100.0 * 1048576.0 * 0.1), 0, false, 100,
              100.0, kNoDecompress, 1000.0, 0.0, 1, 0.2),
          "打平不算划算(margin=0.2 生效)");
    Check(fastllm::PagedPrefixCacheStorageWinsPure(
              (size_t)(100.0 * 1048576.0 * 0.1), 0, false, 100,
              100.0, kNoDecompress, 1000.0, 0.0, 1, 0.0),
          "同一场景 margin=0 时算划算(证明差异确实来自 margin)");

    // --- 本机实际解析出的介质画像 ---
    const auto prof = fastllm::GetPagedPrefixCacheDiskProfileInfo();
    printf("  介质画像: resolved=%d rotational=%d seek=%.4fs bw=%.0fMiB/s dev=%s\n",
           prof.resolved ? 1 : 0, prof.rotational ? 1 : 0,
           prof.seekSeconds, prof.readMiBPerSecond, prof.deviceKey.c_str());
    Check(prof.seekSeconds >= 0.0 && prof.readMiBPerSecond > 0.0,
          "介质画像给出可用的正数参数");
    // 未解析出设备时必须落到**保守**(机械盘)默认, 不能乐观
    if (!prof.resolved) {
        Check(prof.rotational && prof.seekSeconds >= 0.01,
              "未识别设备时落到保守默认(按机械盘)");
    }
}

}  // namespace

int main() {
    printf("=== 分页 KV 显存账本 / GDN 快照对齐回归 ===\n");
    TestPackedPageBytes();
    TestLegacyFormulaWouldRegress();
    TestWholePoolAccounting();
    TestFp8IsNotCheaper();
    TestSnapshotAlignment();
    TestPoolWatermark();
    TestTierAdmission();
    printf("=== %d/%d 通过 ===\n", g_checks - g_failures, g_checks);
    if (g_failures != 0) {
        printf("有 %d 个用例失败\n", g_failures);
        return 1;
    }
    return 0;
}
