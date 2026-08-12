#include "host_offload.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr uint64_t KiB = UINT64_C(1024);
constexpr uint64_t MiB = KiB * KiB;
constexpr uint64_t GiB = MiB * KiB;

void Expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void TestDynamicLimitAndRaiiRelease() {
    uint64_t available = 24 * GiB;
    fastllm::HostCacheBudget budget(
        12 * GiB, 12 * GiB, [&available]() { return available; });

    auto weight = budget.TryReserve(
        6 * GiB, fastllm::HostCacheClass::MODEL_WEIGHT);
    auto prefix = budget.TryReserve(
        3 * GiB, fastllm::HostCacheClass::PREFIX_KV);
    Expect((bool)weight && (bool)prefix,
           "reservations within the dynamic limit must succeed");
    Expect(budget.CurrentLimitBytes() == 12 * GiB,
           "dynamic limit must use min(hard max, MemAvailable - reserve)");
    Expect(budget.ResidentBytes() == 9 * GiB,
           "resident accounting must equal live reservations");
    Expect(budget.Bytes(fastllm::HostCacheClass::MODEL_WEIGHT) == 6 * GiB,
           "weight accounting must be classified");
    Expect(budget.Bytes(fastllm::HostCacheClass::PREFIX_KV) == 3 * GiB,
           "prefix accounting must be classified");

    available = 21 * GiB;
    Expect(budget.CurrentLimitBytes() == 9 * GiB,
           "dynamic limit must shrink when MemAvailable shrinks");
    Expect(!budget.TryReserve(1, fastllm::HostCacheClass::PREFIX_KV),
           "a shrinking dynamic limit must forbid new reservations");

    weight.Reset();
    Expect(budget.ResidentBytes() == 3 * GiB,
           "Reset must release exactly one reservation");
    weight.Reset();
    Expect(budget.ResidentBytes() == 3 * GiB,
           "Reset must be idempotent");

    auto moved = std::move(prefix);
    Expect(!(bool)prefix && (bool)moved,
           "reservation move must transfer ownership");
    moved.Reset();
    Expect(budget.ResidentBytes() == 0,
           "destroying the final reservation must clear accounting");
    Expect(budget.PeakResidentBytes() == 9 * GiB,
           "peak accounting must retain the observed high water mark");
}

void TestZeroAndSaturatingLimits() {
    uint64_t available = 12 * GiB;
    fastllm::HostCacheBudget noHeadroom(
        12 * GiB, 12 * GiB, [&available]() { return available; });
    Expect(noHeadroom.CurrentLimitBytes() == 0,
           "MemAvailable equal to reserve must yield zero capacity");
    Expect(!noHeadroom.TryReserve(1, fastllm::HostCacheClass::MODEL_WEIGHT),
           "zero dynamic capacity must reject reservations");

    available = 1;
    Expect(noHeadroom.CurrentLimitBytes() == 0,
           "MemAvailable below reserve must saturate at zero");

    fastllm::HostCacheBudget disabled(
        0, 0, []() { return UINT64_MAX; });
    Expect(disabled.CurrentLimitBytes() == 0,
           "hard max zero must disable host caching");
    Expect(!disabled.TryReserve(1, fastllm::HostCacheClass::DERIVED_WEIGHT),
           "disabled budget must reject every cache class");
}

void TestWeightReservationEvictsPrefixFirst() {
    uint64_t available = 24 * GiB;
    fastllm::HostCacheBudget budget(
        12 * GiB, 12 * GiB, [&available]() { return available; });
    auto ordinary = budget.TryReserve(
        8 * GiB, fastllm::HostCacheClass::MODEL_WEIGHT);
    auto prefix = budget.TryReserve(
        4 * GiB, fastllm::HostCacheClass::PREFIX_KV);
    Expect((bool)ordinary && (bool)prefix,
           "setup reservations must fill the budget");

    uint64_t requestedEviction = 0;
    int evictCalls = 0;
    budget.RegisterPrefixEvictor(
        [&](uint64_t bytesNeeded) {
            ++evictCalls;
            requestedEviction = bytesNeeded;
            const uint64_t bytes = prefix.Bytes();
            prefix.Reset();
            return bytes;
        });

    auto derived = budget.TryReserve(
        2 * GiB, fastllm::HostCacheClass::DERIVED_WEIGHT);
    Expect((bool)derived,
           "weight reservation must retry after prefix eviction");
    Expect(evictCalls == 1 && requestedEviction == 2 * GiB,
           "evictor must receive the exact deficit");
    Expect(!(bool)prefix,
           "evicted prefix reservation must no longer own budget");
    Expect(budget.Bytes(fastllm::HostCacheClass::PREFIX_KV) == 0,
           "prefix accounting must be zero after eviction");
    Expect(budget.ResidentBytes() == 10 * GiB,
           "weight reservations must remain after prefix eviction");
}

void TestFailedWorkAfterReservationRollsBackAccounting() {
    fastllm::HostCacheBudget budget(
        64 * MiB, 0, []() { return 64 * MiB; });
    try {
        auto reservation = budget.TryReserve(
            32 * MiB, fastllm::HostCacheClass::MODEL_WEIGHT);
        Expect((bool)reservation, "setup reservation must succeed");
        throw std::runtime_error("injected allocation failure");
    } catch (const std::runtime_error &) {
    }
    Expect(budget.ResidentBytes() == 0,
           "reservation destruction must roll back failed allocation work");
    Expect(budget.PeakResidentBytes() == 32 * MiB,
           "failed work must not erase the observed peak");
}

void TestPagedPrefixPayloadSharesGlobalBudget() {
    ::setenv("FASTLLM_HOST_SUSPEND_CACHE", "1", 1);
    ::setenv("FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES", "12582912", 1);
    ::setenv("FASTLLM_HOST_SUSPEND_MIN_FREE_BYTES", "0", 1);
    ::setenv("FASTLLM_PREFIX_CACHE_CPU_TIER", "1", 1);
    ::setenv("FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES", "1073741824", 1);
    ::setenv("FASTLLM_PREFIX_CACHE_ZSTD", "0", 1);
    ::unsetenv("FASTLLM_PREFIX_CACHE_DISK_DIR");

    std::vector<float> values(1, 0.25f);
    fastllm::Data cacheData(
        fastllm::DataType::FLOAT32, {1, 1, 1}, values);
    fastllm::PagedCacheManager *manager =
        fastllm::AllocatePagedCacheManager(
            900001,
            fastllm::PagedCacheManager::
                PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            cacheData, 1, 2);
    Expect(manager != nullptr && manager->cpuData != nullptr,
           "CPU paged manager fixture must allocate storage");

    const int firstPage = manager->GetUnusedPageIndex(true);
    manager->cpuData[
        (size_t)firstPage * manager->GetBytes() / manager->maxPages] =
        0x5a;
    manager->Record({17}, {firstPage});
    manager->ReleasePageIndex(firstPage);
    fastllm::CacheTrieNode *firstNode =
        manager->trieRoot->children.begin()->second;
    firstNode->accessCount = 2;
    Expect(manager->PageOutTrieNode(firstNode),
           "prefix page-out must create a CPU-tier payload");
    Expect(firstNode->tierPayload != nullptr,
           "prefix page-out must retain compressed bytes");

    fastllm::HostCacheBudget &budget =
        fastllm::HostCacheBudget::Global();
    const uint64_t prefixBytes =
        (uint64_t)firstNode->tierPayload->bytes.size();
    Expect(prefixBytes > 0 &&
               budget.Bytes(fastllm::HostCacheClass::PREFIX_KV) ==
                   prefixBytes,
           "real prefix payload must reserve the shared budget");

    firstNode->tierDisk =
        std::make_shared<fastllm::PagedPrefixCacheTierDiskRef>();
    firstNode->tierDisk->persistentArchive = true;
    firstNode->tierDisk->persistentGeneration = 7;
    auto weight = budget.TryReserve(
        12 * MiB, fastllm::HostCacheClass::MODEL_WEIGHT);
    Expect((bool)weight,
           "model reservation must evict prefix RAM before failing");
    Expect(firstNode->tierPayload == nullptr,
           "registered evictor must release prefix RAM");
    Expect(firstNode->tierDisk != nullptr &&
               firstNode->tierDisk->persistentGeneration == 7,
           "prefix RAM eviction must preserve the SSD generation");

    const int secondToken = 23;
    const int secondPage = manager->GetUnusedPageIndex(true);
    manager->cpuData[
        (size_t)secondPage * manager->GetBytes() / manager->maxPages] =
        0xa5;
    manager->Record({secondToken}, {secondPage});
    manager->ReleasePageIndex(secondPage);
    auto secondIt = manager->trieRoot->children.find(
        fastllm::PagedCacheManager::HashTokenPage(&secondToken, 1));
    Expect(secondIt != manager->trieRoot->children.end(),
           "second prefix fixture must exist");
    secondIt->second->accessCount = 2;
    Expect(!manager->PageOutTrieNode(secondIt->second),
           "legacy prefix max must not exceed the full shared budget");
    Expect(budget.ResidentBytes() == 12 * MiB,
           "shared resident bytes must remain at the hard cap");

    weight.Reset();
    fastllm::ClearAllPagedCacheManagers();
    Expect(budget.ResidentBytes() == 0,
           "test teardown must release the global shared budget");
}

} // namespace

int main() {
    try {
        TestDynamicLimitAndRaiiRelease();
        TestZeroAndSaturatingLimits();
        TestWeightReservationEvictsPrefixFirst();
        TestFailedWorkAfterReservationRollsBackAccounting();
        TestPagedPrefixPayloadSharesGlobalBudget();
        std::cout << "host cache budget: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "host cache budget: FAIL: " << error.what() << "\n";
        return 1;
    }
}
