#ifndef FASTLLM_HOST_OFFLOAD_H
#define FASTLLM_HOST_OFFLOAD_H

#include "fastllm.h"
#include "weight_materialization.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastllm {

    enum class HostCacheClass : uint8_t {
        DERIVED_WEIGHT = 0,
        MODEL_WEIGHT = 1,
        PREFIX_KV = 2,
    };

    class HostCacheBudget;

    class HostCacheReservation {
    public:
        HostCacheReservation() = default;
        HostCacheReservation(const HostCacheReservation &) = delete;
        HostCacheReservation &operator=(const HostCacheReservation &) = delete;
        HostCacheReservation(HostCacheReservation &&other) noexcept;
        HostCacheReservation &operator=(HostCacheReservation &&other) noexcept;
        ~HostCacheReservation();

        explicit operator bool() const;
        uint64_t Bytes() const;
        HostCacheClass CacheClass() const;
        void Reset();

    private:
        friend class HostCacheBudget;
        HostCacheReservation(HostCacheBudget *budget, uint64_t bytes,
                             HostCacheClass cacheClass);

        HostCacheBudget *budget = nullptr;
        uint64_t bytes = 0;
        HostCacheClass cacheClass = HostCacheClass::MODEL_WEIGHT;
    };

    uint64_t ReadLinuxMemAvailableBytes();

    class HostCacheBudget {
    public:
        using AvailableBytesReader = std::function<uint64_t()>;
        using PrefixEvictor = std::function<uint64_t(uint64_t)>;

        HostCacheBudget(uint64_t hardMaxBytes, uint64_t minFreeBytes,
                        AvailableBytesReader reader =
                            ReadLinuxMemAvailableBytes);
        HostCacheBudget(const HostCacheBudget &) = delete;
        HostCacheBudget &operator=(const HostCacheBudget &) = delete;

        static HostCacheBudget &Global();
        static bool SharedBudgetEnabled();

        HostCacheReservation TryReserve(uint64_t bytes,
                                        HostCacheClass cacheClass);
        void RegisterPrefixEvictor(PrefixEvictor evictor);

        uint64_t CurrentLimitBytes() const;
        uint64_t ResidentBytes() const;
        uint64_t PeakResidentBytes() const;
        uint64_t Bytes(HostCacheClass cacheClass) const;
        uint64_t HardMaxBytes() const;
        uint64_t MinFreeBytes() const;

    private:
        friend class HostCacheReservation;
        static size_t ClassIndex(HostCacheClass cacheClass);
        uint64_t CurrentLimitBytesUnlocked() const;
        void Release(uint64_t bytes, HostCacheClass cacheClass);

        const uint64_t hardMaxBytes;
        const uint64_t minFreeBytes;
        AvailableBytesReader availableBytesReader;
        mutable std::mutex mutex;
        std::array<uint64_t, 3> classBytes {{0, 0, 0}};
        uint64_t residentBytes = 0;
        uint64_t peakResidentBytes = 0;
        PrefixEvictor prefixEvictor;
    };

    enum class HostOffloadWeightClass : uint8_t {
        EXCLUDED = 0,
        MUST_CACHE = 1,
        DERIVED = 2,
        ORDINARY = 3,
    };

    enum class HostOffloadOutcome : uint8_t {
        SUSPENDED_HOST = 0,
        READY = 1,
        DISK_FALLBACK_REQUIRED = 2,
        FAILED = 3,
    };

    struct HostOffloadTransitionResult {
        HostOffloadOutcome outcome = HostOffloadOutcome::FAILED;
        uint64_t generation = 0;
        uint64_t cachedBytes = 0;
        uint64_t rebuiltBytes = 0;
        uint64_t sourceEvictedBytes = 0;
        uint64_t d2hMilliseconds = 0;
        uint64_t h2dMilliseconds = 0;
        uint64_t materializeMilliseconds = 0;
        double cacheHitRatio = 0.0;
        std::string reason;
    };

    class HostOffloadManager {
    public:
        using ReloadWeights = std::function<bool(
            const std::vector<std::string> &, std::string *)>;
        using WeightClassifier = std::function<HostOffloadWeightClass(
            const std::string &, const Data &,
            const WeightMaterializationRecipe *)>;

        HostOffloadManager(
            std::unordered_map<std::string, Data> &weights,
            const WeightMaterializationPlan &materializationPlan,
            HostCacheBudget &budget,
            ReloadWeights reloadWeights,
            WeightClassifier classifier = WeightClassifier());
        HostOffloadManager(const HostOffloadManager &) = delete;
        HostOffloadManager &operator=(const HostOffloadManager &) = delete;

        HostOffloadTransitionResult Suspend(uint64_t generation);
        HostOffloadTransitionResult Resume(uint64_t generation);
        void Reset();
        bool IsSuspended() const;
        uint64_t Generation() const;
        uint64_t CachedWeightBytes() const;
        std::vector<std::string> SourceEvictedWeights() const;

        static HostOffloadWeightClass DefaultClassify(
            const std::string &name, const Data &data,
            const WeightMaterializationRecipe *recipe);

    private:
        struct CachedTensor {
            Data *data = nullptr;
            DataOffloadRecord record;
            HostCacheReservation reservation;
        };
        struct EvictedTensor {
            std::string name;
            int originalDeviceId = 0;
        };

        HostOffloadTransitionResult RequireDiskFallback(
            uint64_t generation, const std::string &reason);
        bool RollbackCached(std::string *error);

        std::unordered_map<std::string, Data> &weights;
        const WeightMaterializationPlan &materializationPlan;
        HostCacheBudget &budget;
        ReloadWeights reloadWeights;
        WeightClassifier classifier;
        std::map<std::string, CachedTensor> cached;
        std::vector<EvictedTensor> sourceEvicted;
        uint64_t suspendedGeneration = 0;
        uint64_t sourceEvictedBytes = 0;
        bool suspended = false;
    };

} // namespace fastllm

#endif
