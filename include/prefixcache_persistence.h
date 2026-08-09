#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fastllm {
    class basellm;
    class Data;
    class PagedCacheManager;

    enum class PersistentPayloadKind : uint32_t {
        PAGED_CACHE_TRIE = 1,
        PAGED_CACHE_PAGE = 2,
        MODEL_EXTRA = 3,
    };

    struct PersistentPayloadRecord {
        PersistentPayloadKind kind = PersistentPayloadKind::MODEL_EXTRA;
        std::string name;
        uint32_t dataType = 0;
        std::vector<int64_t> dimensions;
        std::vector<uint8_t> bytes;
    };

    struct PersistentPayloadRef {
        PersistentPayloadKind kind = PersistentPayloadKind::MODEL_EXTRA;
        std::string name;
        uint32_t dataType = 0;
        std::vector<int64_t> dimensions;
        uint64_t fileOffset = 0;
        uint64_t storedBytes = 0;
        uint64_t uncompressedBytes = 0;
        uint64_t checksum = 0;
        bool zstdCompressed = false;
    };

    struct PersistedPrefixCacheGeneration {
        uint64_t generation = 0;
        std::string cacheKey;
        std::vector<PersistentPayloadRecord> payloads;
    };

    enum class PersistentPrefixCommitFailpoint : uint32_t {
        NONE = 0,
        BEFORE_CURRENT_RENAME = 1,
    };

    uint64_t PrefixCacheChecksum(const uint8_t *data, size_t bytes);
    uint64_t PrefixCacheChecksum(const std::vector<uint8_t> &bytes);

    bool CommitPersistentPrefixCacheGeneration(
        const std::filesystem::path &root,
        const PersistedPrefixCacheGeneration &source,
        std::vector<PersistentPayloadRef> &refs,
        std::string *error);
    bool LoadPersistentPrefixCacheGeneration(
        const std::filesystem::path &root,
        PersistedPrefixCacheGeneration &generation,
        std::vector<PersistentPayloadRef> &refs,
        std::string *error);
    bool ReadPersistentPrefixCachePayload(
        const std::filesystem::path &root,
        uint64_t generation,
        const PersistentPayloadRef &ref,
        std::vector<uint8_t> &bytes,
        std::string *error);
    std::string PersistentPrefixCacheKeyDirectoryName(
        const std::string &cacheKey);
    void SetPersistentPrefixCacheCommitFailpointForTest(
        PersistentPrefixCommitFailpoint failpoint);

    struct PersistentPrefixCacheOptions {
        bool enabled = false;
        std::filesystem::path diskDirectory;
        std::string cacheKey;
    };

    struct PersistentPrefixCacheStatus {
        bool enabled = false;
        uint64_t loadedGeneration = 0;
        uint64_t checkpointCount = 0;
        uint64_t restoreHitCount = 0;
        uint64_t payloadBytes = 0;
        double lastDurationMs = 0.0;
        std::string lastError;
    };

    struct PersistentPrefixCheckpointStats {
        uint64_t generation = 0;
        uint64_t pages = 0;
        uint64_t bytes = 0;
        double durationMs = 0.0;
    };

    bool PreparePersistentPrefixCache(
        const PersistentPrefixCacheOptions &options,
        basellm *model,
        PersistentPrefixCacheStatus &status,
        std::string *error);
    bool PreparePersistentPrefixCacheFromEnv(
        basellm *model,
        PersistentPrefixCacheStatus &status,
        std::string *error);
    bool CheckpointPersistentPrefixCache(
        basellm *model,
        PersistentPrefixCheckpointStats &stats,
        std::string *error);
    PersistentPrefixCacheStatus GetPersistentPrefixCacheStatus();
    void ResetPersistentPrefixCacheForTest();

    namespace persistent_prefix_cache_internal {
        void AttachPreparedManager(
            int managerId, PagedCacheManager *manager);
        void ObserveRestoreHit();
    }

    bool PersistentPayloadFromData(
        PersistentPayloadKind kind,
        const std::string &name,
        const Data &data,
        PersistentPayloadRecord &record,
        std::string *error);
    bool RestoreDataFromPersistentPayload(
        const PersistentPayloadRef &ref,
        const std::vector<uint8_t> &bytes,
        Data &data,
        std::string *error);
}
