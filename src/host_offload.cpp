#include "host_offload.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace fastllm {
namespace {

    constexpr uint64_t KiB = UINT64_C(1024);

    uint64_t SaturatingMultiply(uint64_t left, uint64_t right) {
        if (left == 0 || right == 0) {
            return 0;
        }
        if (left > std::numeric_limits<uint64_t>::max() / right) {
            return std::numeric_limits<uint64_t>::max();
        }
        return left * right;
    }

    uint64_t ReadEnvironmentBytes(const char *name, uint64_t fallback) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        char *end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0') {
            return fallback;
        }
        return static_cast<uint64_t>(parsed);
    }

    bool EnvironmentFlagEnabled(const char *name) {
        const char *value = std::getenv(name);
        if (value == nullptr) {
            return false;
        }
        std::string normalized(value);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return normalized == "1" || normalized == "true" ||
               normalized == "yes" || normalized == "on";
    }


} // namespace

HostCacheReservation::HostCacheReservation(
        HostCacheBudget *budget, uint64_t bytes,
        HostCacheClass cacheClass)
    : budget(budget), bytes(bytes), cacheClass(cacheClass) {
}

HostCacheReservation::HostCacheReservation(
        HostCacheReservation &&other) noexcept
    : budget(other.budget), bytes(other.bytes), cacheClass(other.cacheClass) {
    other.budget = nullptr;
    other.bytes = 0;
}

HostCacheReservation &HostCacheReservation::operator=(
        HostCacheReservation &&other) noexcept {
    if (this != &other) {
        Reset();
        budget = other.budget;
        bytes = other.bytes;
        cacheClass = other.cacheClass;
        other.budget = nullptr;
        other.bytes = 0;
    }
    return *this;
}

HostCacheReservation::~HostCacheReservation() {
    Reset();
}

HostCacheReservation::operator bool() const {
    return budget != nullptr;
}

uint64_t HostCacheReservation::Bytes() const {
    return bytes;
}

HostCacheClass HostCacheReservation::CacheClass() const {
    return cacheClass;
}

void HostCacheReservation::Reset() {
    HostCacheBudget *owner = budget;
    const uint64_t ownedBytes = bytes;
    const HostCacheClass ownedClass = cacheClass;
    budget = nullptr;
    bytes = 0;
    if (owner != nullptr) {
        owner->Release(ownedBytes, ownedClass);
    }
}

uint64_t ReadLinuxMemAvailableBytes() {
#if defined(__linux__)
    std::ifstream input("/proc/meminfo");
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (input >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return unit == "kB" ? SaturatingMultiply(value, KiB) : value;
        }
    }
#endif
    return 0;
}

HostCacheBudget::HostCacheBudget(
        uint64_t hardMaxBytes, uint64_t minFreeBytes,
        AvailableBytesReader reader)
    : hardMaxBytes(hardMaxBytes),
      minFreeBytes(minFreeBytes),
      availableBytesReader(reader ? std::move(reader) :
                                    AvailableBytesReader(
                                        ReadLinuxMemAvailableBytes)) {
}

HostCacheBudget &HostCacheBudget::Global() {
    static HostCacheBudget budget(
        SharedBudgetEnabled()
            ? ReadEnvironmentBytes("FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES",
                                   UINT64_C(12884901888))
            : 0,
        ReadEnvironmentBytes("FASTLLM_HOST_SUSPEND_MIN_FREE_BYTES",
                             UINT64_C(12884901888)));
    return budget;
}

bool HostCacheBudget::SharedBudgetEnabled() {
    return EnvironmentFlagEnabled("FASTLLM_HOST_SUSPEND_CACHE");
}


size_t HostCacheBudget::ClassIndex(HostCacheClass cacheClass) {
    return static_cast<size_t>(cacheClass);
}

uint64_t HostCacheBudget::CurrentLimitBytesUnlocked() const {
    if (hardMaxBytes == 0) {
        return 0;
    }
    const uint64_t available = availableBytesReader();
    const uint64_t dynamicLimit = available > minFreeBytes ?
        available - minFreeBytes : 0;
    return std::min(hardMaxBytes, dynamicLimit);
}

HostCacheReservation HostCacheBudget::TryReserve(
        uint64_t bytes, HostCacheClass cacheClass) {
    if (bytes == 0) {
        return HostCacheReservation();
    }

    PrefixEvictor evictor;
    uint64_t deficit = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const uint64_t limit = CurrentLimitBytesUnlocked();
        if (residentBytes <= limit && bytes <= limit - residentBytes) {
            residentBytes += bytes;
            classBytes[ClassIndex(cacheClass)] += bytes;
            peakResidentBytes = std::max(peakResidentBytes, residentBytes);
            return HostCacheReservation(this, bytes, cacheClass);
        }
        if (cacheClass != HostCacheClass::PREFIX_KV && prefixEvictor) {
            const uint64_t freeBytes = residentBytes < limit ?
                limit - residentBytes : 0;
            deficit = bytes > freeBytes ? bytes - freeBytes : 0;
            evictor = prefixEvictor;
        }
    }

    if (!evictor || deficit == 0) {
        return HostCacheReservation();
    }
    evictor(deficit);

    std::lock_guard<std::mutex> lock(mutex);
    const uint64_t limit = CurrentLimitBytesUnlocked();
    if (residentBytes > limit || bytes > limit - residentBytes) {
        return HostCacheReservation();
    }
    residentBytes += bytes;
    classBytes[ClassIndex(cacheClass)] += bytes;
    peakResidentBytes = std::max(peakResidentBytes, residentBytes);
    return HostCacheReservation(this, bytes, cacheClass);
}

void HostCacheBudget::RegisterPrefixEvictor(PrefixEvictor evictor) {
    std::lock_guard<std::mutex> lock(mutex);
    prefixEvictor = std::move(evictor);
}

uint64_t HostCacheBudget::CurrentLimitBytes() const {
    std::lock_guard<std::mutex> lock(mutex);
    return CurrentLimitBytesUnlocked();
}

uint64_t HostCacheBudget::ResidentBytes() const {
    std::lock_guard<std::mutex> lock(mutex);
    return residentBytes;
}

uint64_t HostCacheBudget::PeakResidentBytes() const {
    std::lock_guard<std::mutex> lock(mutex);
    return peakResidentBytes;
}

uint64_t HostCacheBudget::Bytes(HostCacheClass cacheClass) const {
    std::lock_guard<std::mutex> lock(mutex);
    return classBytes[ClassIndex(cacheClass)];
}

uint64_t HostCacheBudget::HardMaxBytes() const {
    return hardMaxBytes;
}

uint64_t HostCacheBudget::MinFreeBytes() const {
    return minFreeBytes;
}

void HostCacheBudget::Release(uint64_t bytes, HostCacheClass cacheClass) {
    std::lock_guard<std::mutex> lock(mutex);
    const size_t index = ClassIndex(cacheClass);
    const uint64_t released = std::min(bytes, classBytes[index]);
    classBytes[index] -= released;
    residentBytes -= std::min(released, residentBytes);
}

HostOffloadManager::HostOffloadManager(
        std::unordered_map<std::string, Data> &weights,
        const WeightMaterializationPlan &materializationPlan,
        HostCacheBudget &budget,
        ReloadWeights reloadWeights,
        WeightClassifier classifier)
    : weights(weights),
      materializationPlan(materializationPlan),
      budget(budget),
      reloadWeights(std::move(reloadWeights)),
      classifier(classifier ? std::move(classifier)
                            : WeightClassifier(DefaultClassify)) {
}

HostOffloadWeightClass HostOffloadManager::DefaultClassify(
        const std::string &, const Data &data,
        const WeightMaterializationRecipe *recipe) {
    if (!data.isModelWeight || data.cudaData == nullptr) {
        return HostOffloadWeightClass::EXCLUDED;
    }
    if (recipe == nullptr) {
        return HostOffloadWeightClass::MUST_CACHE;
    }
    return recipe->kind == WeightRecipeKind::MERGE
        ? HostOffloadWeightClass::DERIVED
        : HostOffloadWeightClass::ORDINARY;
}

HostOffloadTransitionResult HostOffloadManager::RequireDiskFallback(
        uint64_t generation, const std::string &reason) {
    HostOffloadTransitionResult result;
    result.outcome = HostOffloadOutcome::DISK_FALLBACK_REQUIRED;
    result.generation = generation;
    result.cachedBytes = CachedWeightBytes();
    result.sourceEvictedBytes = sourceEvictedBytes;
    result.reason = reason;
    return result;
}

bool HostOffloadManager::RollbackCached(std::string *error) {
    bool restoredAll = true;
    std::string firstError;
    for (auto &entry : cached) {
        std::string restoreError;
        if (!entry.second.data->RestoreCudaStorageFromHost(
                entry.second.record, &restoreError)) {
            restoredAll = false;
            if (firstError.empty()) {
                firstError = entry.first + ": " + restoreError;
            }
        } else {
            entry.second.reservation.Reset();
        }
    }
    if (restoredAll) {
        cached.clear();
    }
    if (error != nullptr) {
        *error = firstError;
    }
    return restoredAll;
}

HostOffloadTransitionResult HostOffloadManager::Suspend(uint64_t generation) {
    using Clock = std::chrono::steady_clock;
    struct Candidate {
        std::string name;
        Data *data = nullptr;
        const WeightMaterializationRecipe *recipe = nullptr;
        HostOffloadWeightClass weightClass = HostOffloadWeightClass::EXCLUDED;
        HostCacheReservation reservation;
        bool sourceEvicted = false;
    };

    if (suspended || !cached.empty()) {
        return RequireDiskFallback(generation, "model is already host-suspended");
    }

    std::vector<Candidate> candidates;
    candidates.reserve(weights.size());
    for (auto &entry : weights) {
        Data &data = entry.second;
        const WeightMaterializationRecipe *recipe =
            materializationPlan.FindByOutput(entry.first);
        const HostOffloadWeightClass weightClass =
            classifier(entry.first, data, recipe);
        if (weightClass == HostOffloadWeightClass::EXCLUDED) {
            continue;
        }
        if (data.isGGUFData &&
            (!data.extraCudaData.empty() ||
             !data.extraCudaHalfData.empty())) {
            std::string auxiliaryError;
            if (!data.ReleaseCudaAuxiliaryStorage(
                    &auxiliaryError)) {
                return RequireDiskFallback(
                    generation,
                    "GGUF auxiliary CUDA cache teardown failed for " +
                    entry.first + ": " + auxiliaryError);
            }
        }
        if (data.isFake || data.cudaDataBorrowed || data.multiDeviceData ||
            data.IsTensorParallel() || data.isPagedKVCache ||
            !data.extraCudaData.empty() || !data.extraCudaHalfData.empty()) {
            return RequireDiskFallback(
                generation, "unsupported CUDA ownership for weight " + entry.first);
        }
        if (data.cudaData == nullptr || data.expansionBytes == 0) {
            return RequireDiskFallback(
                generation, "weight has no owned CUDA payload: " + entry.first);
        }
        if (weightClass != HostOffloadWeightClass::MUST_CACHE) {
            if (recipe == nullptr) {
                return RequireDiskFallback(
                    generation, "source-rebuildable weight has no recipe: " + entry.first);
            }
            std::string closureError;
            const auto closure = materializationPlan.BuildReloadClosure(
                {entry.first}, &closureError);
            if (closure.empty()) {
                return RequireDiskFallback(
                    generation, "invalid materialization closure for " +
                                    entry.first + ": " + closureError);
            }
            for (const WeightMaterializationRecipe *dependency : closure) {
                if (dependency->kind != WeightRecipeKind::GGUF_DIRECT) {
                    continue;
                }
                std::string identityError;
                if (!ValidateWeightSourceIdentity(
                        dependency->source,
                        dependency->source.tensorTableFingerprint,
                        &identityError)) {
                    return RequireDiskFallback(
                        generation, "source identity mismatch for " +
                                        entry.first + ": " + identityError);
                }
            }
        }
        Candidate candidate;
        candidate.name = entry.first;
        candidate.data = &data;
        candidate.recipe = recipe;
        candidate.weightClass = weightClass;
        candidates.push_back(std::move(candidate));
    }

    auto priority = [](HostOffloadWeightClass value) {
        switch (value) {
            case HostOffloadWeightClass::MUST_CACHE: return 0;
            case HostOffloadWeightClass::DERIVED: return 1;
            case HostOffloadWeightClass::ORDINARY: return 2;
            default: return 3;
        }
    };
    std::sort(candidates.begin(), candidates.end(),
              [&](const Candidate &left, const Candidate &right) {
                  const int leftPriority = priority(left.weightClass);
                  const int rightPriority = priority(right.weightClass);
                  if (leftPriority != rightPriority) {
                      return leftPriority < rightPriority;
                  }
                  if (left.data->expansionBytes !=
                      right.data->expansionBytes) {
                      return left.data->expansionBytes <
                          right.data->expansionBytes;
                  }
                  return left.name < right.name;
              });

    for (Candidate &candidate : candidates) {
        const HostCacheClass cacheClass =
            candidate.weightClass == HostOffloadWeightClass::DERIVED
                ? HostCacheClass::DERIVED_WEIGHT
                : HostCacheClass::MODEL_WEIGHT;
        candidate.reservation =
            budget.TryReserve(candidate.data->expansionBytes, cacheClass);
        if (!candidate.reservation) {
            if (candidate.weightClass == HostOffloadWeightClass::MUST_CACHE) {
                return RequireDiskFallback(
                    generation, "host budget cannot fit required weight " +
                                    candidate.name);
            }
            candidate.sourceEvicted = true;
        }
    }

    const auto d2hStart = Clock::now();
    uint64_t cachedBytes = 0;
    for (Candidate &candidate : candidates) {
        if (candidate.sourceEvicted) {
            continue;
        }
        CachedTensor stored;
        stored.data = candidate.data;
        std::string migrationError;
        if (!candidate.data->MoveCudaStorageToHost(
                stored.record, &migrationError)) {
            std::string rollbackError;
            const bool rollbackOk = RollbackCached(&rollbackError);
            return RequireDiskFallback(
                generation,
                "D2H failed for " + candidate.name + ": " + migrationError +
                (rollbackOk ? "" : "; rollback failed: " + rollbackError));
        }
        cachedBytes += stored.record.bytes;
        stored.reservation = std::move(candidate.reservation);
        cached.emplace(candidate.name, std::move(stored));
    }
    const auto d2hEnd = Clock::now();

    sourceEvicted.clear();
    sourceEvictedBytes = 0;
    for (Candidate &candidate : candidates) {
        if (!candidate.sourceEvicted) {
            continue;
        }
        EvictedTensor evicted;
        evicted.name = candidate.name;
        evicted.originalDeviceId =
            candidate.data->dataDeviceIds.empty()
                ? 0
                : candidate.data->dataDeviceIds.front();
        std::string releaseError;
        if (!candidate.data->ReleaseCudaStorageWithoutHostCopy(&releaseError)) {
            std::string rollbackError;
            RollbackCached(&rollbackError);
            return RequireDiskFallback(
                generation, "source eviction failed for " + candidate.name +
                                ": " + releaseError);
        }
        sourceEvicted.push_back(std::move(evicted));
        sourceEvictedBytes += candidate.data->expansionBytes;
    }

    suspended = true;
    suspendedGeneration = generation;
    HostOffloadTransitionResult result;
    result.outcome = HostOffloadOutcome::SUSPENDED_HOST;
    result.generation = generation;
    result.cachedBytes = cachedBytes;
    result.sourceEvictedBytes = sourceEvictedBytes;
    result.d2hMilliseconds = std::chrono::duration_cast<
        std::chrono::milliseconds>(d2hEnd - d2hStart).count();
    const uint64_t total = cachedBytes + sourceEvictedBytes;
    result.cacheHitRatio = total == 0
        ? 1.0
        : static_cast<double>(cachedBytes) / static_cast<double>(total);
    return result;
}

HostOffloadTransitionResult HostOffloadManager::Resume(uint64_t generation) {
    using Clock = std::chrono::steady_clock;
    if (!suspended) {
        return RequireDiskFallback(generation, "model is not host-suspended");
    }
    if (generation != suspendedGeneration) {
        return RequireDiskFallback(generation, "host cache generation mismatch");
    }

    HostOffloadTransitionResult result;
    result.generation = generation;
    result.cachedBytes = CachedWeightBytes();
    result.sourceEvictedBytes = sourceEvictedBytes;
    const uint64_t totalBytes = result.cachedBytes + sourceEvictedBytes;
    result.cacheHitRatio = totalBytes == 0
        ? 1.0
        : static_cast<double>(result.cachedBytes) /
              static_cast<double>(totalBytes);

    const auto h2dStart = Clock::now();
    for (auto it = cached.begin(); it != cached.end();) {
        std::string restoreError;
        if (!it->second.data->RestoreCudaStorageFromHost(
                it->second.record, &restoreError)) {
            return RequireDiskFallback(
                generation, "H2D failed for " + it->first + ": " + restoreError);
        }
        it->second.reservation.Reset();
        it = cached.erase(it);
    }
    const auto h2dEnd = Clock::now();
    result.h2dMilliseconds = std::chrono::duration_cast<
        std::chrono::milliseconds>(h2dEnd - h2dStart).count();

    if (!sourceEvicted.empty()) {
        if (!reloadWeights) {
            return RequireDiskFallback(
                generation, "no partial weight reloader is configured");
        }
        std::vector<std::string> evictedNames;
        evictedNames.reserve(sourceEvicted.size());
        for (const EvictedTensor &evicted : sourceEvicted) {
            evictedNames.push_back(evicted.name);
        }
        const auto materializeStart = Clock::now();
        std::string reloadError;
        if (!reloadWeights(evictedNames, &reloadError)) {
            return RequireDiskFallback(
                generation, "partial weight reload failed: " + reloadError);
        }
        for (const EvictedTensor &evicted : sourceEvicted) {
            auto weight = weights.find(evicted.name);
            if (weight == weights.end()) {
                return RequireDiskFallback(
                    generation,
                    "partial reload lost model weight " + evicted.name);
            }
            Data &data = weight->second;
            if (data.cudaData == nullptr ||
                data.dataDevice != DataDevice::CUDA) {
                if (data.dataDevice != DataDevice::CPU ||
                    data.cpuData == nullptr) {
                    return RequireDiskFallback(
                        generation,
                        "partial reload produced no payload for " +
                            evicted.name);
                }
                data.ToDevice(
                    DataDevice::CUDA,
                    {evicted.originalDeviceId},
                    true);
            }
            if (data.cudaData == nullptr ||
                data.dataDevice != DataDevice::CUDA) {
                return RequireDiskFallback(
                    generation,
                    "partial reload did not restore CUDA weight " +
                        evicted.name);
            }
        }
        const auto materializeEnd = Clock::now();
        result.materializeMilliseconds = std::chrono::duration_cast<
            std::chrono::milliseconds>(
                materializeEnd - materializeStart).count();
        result.rebuiltBytes = sourceEvictedBytes;
    }

    sourceEvicted.clear();
    sourceEvictedBytes = 0;
    suspended = false;
    suspendedGeneration = 0;
    result.outcome = HostOffloadOutcome::READY;
    return result;
}

void HostOffloadManager::Reset() {
    cached.clear();
    sourceEvicted.clear();
    sourceEvictedBytes = 0;
    suspendedGeneration = 0;
    suspended = false;
}

bool HostOffloadManager::IsSuspended() const {
    return suspended;
}

uint64_t HostOffloadManager::Generation() const {
    return suspendedGeneration;
}

uint64_t HostOffloadManager::CachedWeightBytes() const {
    uint64_t bytes = 0;
    for (const auto &entry : cached) {
        bytes += entry.second.record.bytes;
    }
    return bytes;
}

std::vector<std::string>
HostOffloadManager::SourceEvictedWeights() const {
    std::vector<std::string> names;
    names.reserve(sourceEvicted.size());
    for (const EvictedTensor &evicted : sourceEvicted) {
        names.push_back(evicted.name);
    }
    return names;
}

} // namespace fastllm
