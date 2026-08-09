#include "prefixcache_persistence.h"
#include "fastllm.h"
#include "models/basellm.h"

#include <algorithm>
#include <cctype>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>
#include <mutex>

#ifdef FASTLLM_USE_ZSTD
#include <zstd.h>
#endif

#ifdef _WIN32
#include <io.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fastllm {
namespace {
    constexpr uint8_t MANIFEST_MAGIC[8] = {
        'F', 'L', 'P', 'C', 'G', 'E', 'N', '1'
    };
    constexpr uint32_t MANIFEST_VERSION = 1;
    constexpr uint64_t MAX_MANIFEST_BYTES = UINT64_C(64) << 20;
    constexpr uint32_t MAX_STRING_BYTES = UINT32_C(1) << 20;
    constexpr uint32_t MAX_DIMENSIONS = 64;
    constexpr uint32_t MAX_PAYLOADS = UINT32_C(1) << 20;
    constexpr uint64_t MAX_SINGLE_PAYLOAD_BYTES = UINT64_C(64) << 30;
    constexpr uint32_t FLAG_ZSTD = 1;

    std::atomic<PersistentPrefixCommitFailpoint> commitFailpoint(
        PersistentPrefixCommitFailpoint::NONE);

    void SetError(std::string *error, const std::string &message) {
        if (error != nullptr) {
            *error = message;
        }
    }

    uint64_t Checksum(const uint8_t *data, size_t bytes) {
        uint64_t value = UINT64_C(1469598103934665603);
        for (size_t i = 0; i < bytes; i++) {
            value ^= data[i];
            value *= UINT64_C(1099511628211);
        }
        return value;
    }

    uint64_t Checksum(const std::vector<uint8_t> &data) {
        return Checksum(data.data(), data.size());
    }

    class BufferWriter {
    public:
        void U32(uint32_t value) {
            for (int i = 0; i < 4; i++) {
                bytes.push_back((uint8_t)(value >> (8 * i)));
            }
        }

        void U64(uint64_t value) {
            for (int i = 0; i < 8; i++) {
                bytes.push_back((uint8_t)(value >> (8 * i)));
            }
        }

        bool String(const std::string &value, std::string *error) {
            if (value.size() > MAX_STRING_BYTES) {
                SetError(error, "persistent prefix manifest string exceeds limit");
                return false;
            }
            U32((uint32_t)value.size());
            bytes.insert(bytes.end(), value.begin(), value.end());
            return true;
        }

        void Raw(const uint8_t *source, size_t count) {
            bytes.insert(bytes.end(), source, source + count);
        }

        std::vector<uint8_t> bytes;
    };

    class BufferReader {
    public:
        BufferReader(const uint8_t *data, size_t size)
            : data(data), size(size) {}

        bool U32(uint32_t &value) {
            if (Remaining() < 4) {
                return false;
            }
            value = 0;
            for (int i = 0; i < 4; i++) {
                value |= (uint32_t)data[offset + i] << (8 * i);
            }
            offset += 4;
            return true;
        }

        bool U64(uint64_t &value) {
            if (Remaining() < 8) {
                return false;
            }
            value = 0;
            for (int i = 0; i < 8; i++) {
                value |= (uint64_t)data[offset + i] << (8 * i);
            }
            offset += 8;
            return true;
        }

        bool String(std::string &value) {
            uint32_t length = 0;
            if (!U32(length) || length > MAX_STRING_BYTES ||
                Remaining() < length) {
                return false;
            }
            value.assign(reinterpret_cast<const char *>(data + offset), length);
            offset += length;
            return true;
        }

        bool Raw(const uint8_t *&value, size_t count) {
            if (Remaining() < count) {
                return false;
            }
            value = data + offset;
            offset += count;
            return true;
        }

        size_t Remaining() const {
            return size - offset;
        }

        size_t Offset() const {
            return offset;
        }

    private:
        const uint8_t *data = nullptr;
        size_t size = 0;
        size_t offset = 0;
    };

    bool ParseEnvBytes(const char *name, uint64_t &value,
                       bool &present, std::string *error) {
        const char *text = std::getenv(name);
        present = text != nullptr && text[0] != '\0';
        if (!present) {
            value = 0;
            return true;
        }
        errno = 0;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0') {
            SetError(error, std::string("invalid ") + name);
            return false;
        }
        value = (uint64_t)parsed;
        return true;
    }

    uint64_t DirectoryBytes(const std::filesystem::path &root) {
        uint64_t total = 0;
        std::error_code error;
        if (!std::filesystem::exists(root, error)) {
            return 0;
        }
        std::filesystem::recursive_directory_iterator it(
            root, std::filesystem::directory_options::skip_permission_denied,
            error);
        const std::filesystem::recursive_directory_iterator end;
        while (!error && it != end) {
            if (it->is_regular_file(error)) {
                uint64_t size = it->file_size(error);
                if (!error && size <= std::numeric_limits<uint64_t>::max() - total) {
                    total += size;
                }
            }
            it.increment(error);
        }
        return total;
    }

    uint64_t ProcessId() {
#ifdef _WIN32
        return (uint64_t)_getpid();
#else
        return (uint64_t)getpid();
#endif
    }

    uint64_t Nonce() {
        return (uint64_t)std::chrono::steady_clock::now()
            .time_since_epoch().count();
    }

#ifndef _WIN32
    class ScopedFileLock {
    public:
        ScopedFileLock(const std::filesystem::path &path, bool exclusive,
                       std::string *error) {
            fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (fd < 0) {
                SetError(error, "failed to open persistent prefix cache lock: " +
                                std::string(std::strerror(errno)));
                return;
            }
            if (flock(fd, exclusive ? LOCK_EX : LOCK_SH) != 0) {
                SetError(error, "failed to lock persistent prefix cache: " +
                                std::string(std::strerror(errno)));
                close(fd);
                fd = -1;
            }
        }

        ~ScopedFileLock() {
            if (fd >= 0) {
                flock(fd, LOCK_UN);
                close(fd);
            }
        }

        bool Valid() const {
            return fd >= 0;
        }

    private:
        int fd = -1;
    };

    bool WriteSpan(int fd, const uint8_t *data, size_t bytes,
                   std::string *error) {
        size_t offset = 0;
        while (offset < bytes) {
            ssize_t written = write(fd, data + offset, bytes - offset);
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                SetError(error, "failed to write persistent prefix cache: " +
                                std::string(std::strerror(errno)));
                return false;
            }
            offset += (size_t)written;
        }
        return true;
    }

    bool WriteFileAndSync(const std::filesystem::path &path,
                          const uint8_t *data, size_t bytes,
                          std::string *error) {
        int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                      0600);
        if (fd < 0) {
            SetError(error, "failed to open persistent prefix cache file: " +
                            std::string(std::strerror(errno)));
            return false;
        }
        bool ok = WriteSpan(fd, data, bytes, error);
        if (ok && fsync(fd) != 0) {
            SetError(error, "failed to sync persistent prefix cache file: " +
                            std::string(std::strerror(errno)));
            ok = false;
        }
        if (close(fd) != 0 && ok) {
            SetError(error, "failed to close persistent prefix cache file: " +
                            std::string(std::strerror(errno)));
            ok = false;
        }
        return ok;
    }

    bool SyncDirectory(const std::filesystem::path &path,
                       std::string *error) {
        int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0) {
            SetError(error, "failed to open persistent prefix cache directory: " +
                            std::string(std::strerror(errno)));
            return false;
        }
        bool ok = fsync(fd) == 0;
        if (!ok) {
            SetError(error, "failed to sync persistent prefix cache directory: " +
                            std::string(std::strerror(errno)));
        }
        close(fd);
        return ok;
    }
#else
    class ScopedFileLock {
    public:
        ScopedFileLock(const std::filesystem::path &, bool, std::string *) {}
        bool Valid() const { return true; }
    };

    bool WriteFileAndSync(const std::filesystem::path &path,
                          const uint8_t *data, size_t bytes,
                          std::string *error) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) {
            SetError(error, "failed to open persistent prefix cache file");
            return false;
        }
        file.write(reinterpret_cast<const char *>(data), bytes);
        file.flush();
        if (!file) {
            SetError(error, "failed to write persistent prefix cache file");
            return false;
        }
        return true;
    }

    bool SyncDirectory(const std::filesystem::path &, std::string *) {
        return true;
    }
#endif

    bool ReadBoundedFile(const std::filesystem::path &path, uint64_t maxBytes,
                         std::vector<uint8_t> &bytes, std::string *error) {
        std::error_code filesystemError;
        uint64_t size = std::filesystem::file_size(path, filesystemError);
        if (filesystemError || size > maxBytes ||
            size > (uint64_t)std::numeric_limits<size_t>::max()) {
            SetError(error, filesystemError ?
                "failed to stat persistent prefix cache file: " +
                    filesystemError.message() :
                "persistent prefix cache file exceeds limit");
            return false;
        }
        bytes.resize((size_t)size);
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            SetError(error, "failed to open persistent prefix cache file");
            return false;
        }
        if (size > 0) {
            file.read(reinterpret_cast<char *>(bytes.data()), (std::streamsize)size);
        }
        if (!file || file.peek() != std::ifstream::traits_type::eof()) {
            SetError(error, "persistent prefix cache file changed while reading");
            return false;
        }
        return true;
    }

    bool ValidateKind(PersistentPayloadKind kind) {
        return kind == PersistentPayloadKind::PAGED_CACHE_TRIE ||
               kind == PersistentPayloadKind::PAGED_CACHE_PAGE ||
               kind == PersistentPayloadKind::MODEL_EXTRA;
    }

    bool ValidateSource(const PersistedPrefixCacheGeneration &source,
                        uint64_t &payloadBytes, std::string *error) {
        if (source.generation == 0) {
            SetError(error, "persistent prefix cache generation must be positive");
            return false;
        }
        if (source.cacheKey.empty() || source.cacheKey.size() > MAX_STRING_BYTES) {
            SetError(error, "persistent prefix cache key is empty or too large");
            return false;
        }
        if (source.payloads.size() > MAX_PAYLOADS) {
            SetError(error, "persistent prefix cache has too many payloads");
            return false;
        }
        payloadBytes = 0;
        std::set<std::pair<uint32_t, std::string> > names;
        for (const PersistentPayloadRecord &payload : source.payloads) {
            if (!ValidateKind(payload.kind) || payload.name.empty() ||
                payload.name.size() > MAX_STRING_BYTES ||
                payload.dimensions.size() > MAX_DIMENSIONS ||
                payload.bytes.size() > MAX_SINGLE_PAYLOAD_BYTES) {
                SetError(error, "persistent prefix cache payload metadata is invalid");
                return false;
            }
            if (!names.insert({(uint32_t)payload.kind, payload.name}).second) {
                SetError(error, "persistent prefix cache payload name is duplicated");
                return false;
            }
            if (payload.bytes.size() >
                std::numeric_limits<uint64_t>::max() - payloadBytes) {
                SetError(error, "persistent prefix cache payload size overflow");
                return false;
            }
            payloadBytes += payload.bytes.size();
        }
        return true;
    }

    bool EncodePayload(const std::vector<uint8_t> &source,
                       std::vector<uint8_t> &stored, bool &compressed,
                       std::string *error) {
        compressed = false;
        stored.clear();
#ifdef FASTLLM_USE_ZSTD
        if (source.size() >= 256) {
            size_t bound = ZSTD_compressBound(source.size());
            if (ZSTD_isError(bound) || bound > MAX_SINGLE_PAYLOAD_BYTES ||
                bound > std::numeric_limits<size_t>::max()) {
                SetError(error, "persistent prefix cache zstd bound is invalid");
                return false;
            }
            stored.resize(bound);
            size_t result = ZSTD_compress(
                stored.data(), stored.size(), source.data(), source.size(), 1);
            if (ZSTD_isError(result)) {
                SetError(error, std::string("persistent prefix cache zstd error: ") +
                                ZSTD_getErrorName(result));
                return false;
            }
            if (result < source.size()) {
                stored.resize(result);
                compressed = true;
                return true;
            }
            stored.clear();
        }
#else
        (void)error;
#endif
        return true;
    }

    bool BuildManifest(const PersistedPrefixCacheGeneration &source,
                       const std::vector<PersistentPayloadRef> &refs,
                       std::vector<uint8_t> &manifest, std::string *error) {
        BufferWriter writer;
        writer.Raw(MANIFEST_MAGIC, sizeof(MANIFEST_MAGIC));
        writer.U32(MANIFEST_VERSION);
        writer.U64(source.generation);
        if (!writer.String(source.cacheKey, error)) {
            return false;
        }
        writer.U32((uint32_t)refs.size());
        for (const PersistentPayloadRef &ref : refs) {
            writer.U32((uint32_t)ref.kind);
            if (!writer.String(ref.name, error)) {
                return false;
            }
            writer.U32(ref.dataType);
            writer.U32((uint32_t)ref.dimensions.size());
            for (int64_t dimension : ref.dimensions) {
                writer.U64((uint64_t)dimension);
            }
            writer.U64(ref.fileOffset);
            writer.U64(ref.storedBytes);
            writer.U64(ref.uncompressedBytes);
            writer.U64(ref.checksum);
            writer.U32(ref.zstdCompressed ? FLAG_ZSTD : 0);
            if (writer.bytes.size() > MAX_MANIFEST_BYTES - 8) {
                SetError(error, "persistent prefix cache manifest exceeds limit");
                return false;
            }
        }
        writer.U64(Checksum(writer.bytes));
        manifest = std::move(writer.bytes);
        return true;
    }

    bool DecodeManifest(const std::vector<uint8_t> &manifest,
                        PersistedPrefixCacheGeneration &generation,
                        std::vector<PersistentPayloadRef> &refs,
                        std::string *error) {
        if (manifest.size() < sizeof(MANIFEST_MAGIC) + 4 + 8 + 4 + 4 + 8) {
            SetError(error, "persistent prefix cache manifest is truncated");
            return false;
        }
        uint64_t expectedChecksum = 0;
        BufferReader checksumReader(
            manifest.data() + manifest.size() - 8, 8);
        if (!checksumReader.U64(expectedChecksum) ||
            expectedChecksum != Checksum(manifest.data(), manifest.size() - 8)) {
            SetError(error, "persistent prefix cache manifest checksum mismatch");
            return false;
        }

        BufferReader reader(manifest.data(), manifest.size() - 8);
        const uint8_t *magic = nullptr;
        uint32_t version = 0;
        if (!reader.Raw(magic, sizeof(MANIFEST_MAGIC)) ||
            std::memcmp(magic, MANIFEST_MAGIC, sizeof(MANIFEST_MAGIC)) != 0 ||
            !reader.U32(version) || version != MANIFEST_VERSION ||
            !reader.U64(generation.generation) ||
            generation.generation == 0 ||
            !reader.String(generation.cacheKey) || generation.cacheKey.empty()) {
            SetError(error, "persistent prefix cache manifest header is invalid");
            return false;
        }

        uint32_t payloadCount = 0;
        if (!reader.U32(payloadCount) || payloadCount > MAX_PAYLOADS) {
            SetError(error, "persistent prefix cache payload count is invalid");
            return false;
        }
        refs.clear();
        refs.reserve(payloadCount);
        std::set<std::pair<uint32_t, std::string> > names;
        uint64_t expectedOffset = 0;
        for (uint32_t i = 0; i < payloadCount; i++) {
            PersistentPayloadRef ref;
            uint32_t kind = 0;
            uint32_t dimensions = 0;
            uint32_t flags = 0;
            if (!reader.U32(kind) ||
                !reader.String(ref.name) || ref.name.empty() ||
                !reader.U32(ref.dataType) ||
                !reader.U32(dimensions) || dimensions > MAX_DIMENSIONS) {
                SetError(error, "persistent prefix cache payload metadata is invalid");
                return false;
            }
            ref.kind = (PersistentPayloadKind)kind;
            if (!ValidateKind(ref.kind) ||
                !names.insert({kind, ref.name}).second) {
                SetError(error, "persistent prefix cache payload identity is invalid");
                return false;
            }
            ref.dimensions.resize(dimensions);
            for (uint32_t d = 0; d < dimensions; d++) {
                uint64_t value = 0;
                if (!reader.U64(value)) {
                    SetError(error, "persistent prefix cache dimensions are truncated");
                    return false;
                }
                ref.dimensions[d] = (int64_t)value;
            }
            if (!reader.U64(ref.fileOffset) ||
                !reader.U64(ref.storedBytes) ||
                !reader.U64(ref.uncompressedBytes) ||
                !reader.U64(ref.checksum) || !reader.U32(flags) ||
                (flags & ~FLAG_ZSTD) != 0 ||
                ref.fileOffset != expectedOffset ||
                ref.storedBytes > MAX_SINGLE_PAYLOAD_BYTES ||
                ref.uncompressedBytes > MAX_SINGLE_PAYLOAD_BYTES ||
                ref.storedBytes > std::numeric_limits<uint64_t>::max() -
                    expectedOffset) {
                SetError(error, "persistent prefix cache payload bounds are invalid");
                return false;
            }
            ref.zstdCompressed = (flags & FLAG_ZSTD) != 0;
            if ((!ref.zstdCompressed &&
                 ref.storedBytes != ref.uncompressedBytes) ||
                (ref.zstdCompressed && ref.uncompressedBytes == 0)) {
                SetError(error, "persistent prefix cache compression metadata is invalid");
                return false;
            }
            expectedOffset += ref.storedBytes;
            refs.push_back(std::move(ref));
        }
        if (reader.Remaining() != 0) {
            SetError(error, "persistent prefix cache manifest has trailing bytes");
            return false;
        }
        generation.payloads.clear();
        return true;
    }

    bool ReadCurrentGeneration(const std::filesystem::path &root,
                               uint64_t &generation, std::string *error) {
        std::vector<uint8_t> current;
        if (!ReadBoundedFile(root / "CURRENT", 64, current, error)) {
            return false;
        }
        if (current.empty() || current.size() > 21) {
            SetError(error, "persistent prefix cache CURRENT is invalid");
            return false;
        }
        if (current.back() == '\n') {
            current.pop_back();
        }
        if (current.empty()) {
            SetError(error, "persistent prefix cache CURRENT is empty");
            return false;
        }
        generation = 0;
        for (uint8_t byte : current) {
            if (byte < '0' || byte > '9' ||
                generation > (std::numeric_limits<uint64_t>::max() -
                              (byte - '0')) / 10) {
                SetError(error, "persistent prefix cache CURRENT is invalid");
                return false;
            }
            generation = generation * 10 + (byte - '0');
        }
        if (generation == 0) {
            SetError(error, "persistent prefix cache CURRENT is invalid");
            return false;
        }
        return true;
    }

    std::filesystem::path GenerationPath(
        const std::filesystem::path &root, uint64_t generation) {
        return root / ("gen-" + std::to_string(generation));
    }

    bool ParseGenerationDirectoryName(
            const std::string &name, uint64_t &generation) {
        if (name.rfind("gen-", 0) != 0 || name.size() <= 4) {
            return false;
        }
        generation = 0;
        for (size_t index = 4; index < name.size(); index++) {
            const char digit = name[index];
            if (digit < '0' || digit > '9' ||
                generation >
                    (std::numeric_limits<uint64_t>::max() -
                     (digit - '0')) / 10) {
                return false;
            }
            generation =
                generation * 10 + static_cast<uint64_t>(digit - '0');
        }
        return generation != 0;
    }

    bool PrunePersistentPrefixCacheGenerations(
            const std::filesystem::path &root,
            uint64_t current,
            size_t keep,
            std::string *error) {
        std::vector<std::pair<uint64_t, std::filesystem::path>>
            generations;
        std::error_code filesystemError;
        for (std::filesystem::directory_iterator it(
                 root,
                 std::filesystem::directory_options::
                     skip_permission_denied,
                 filesystemError), end;
             !filesystemError && it != end;
             it.increment(filesystemError)) {
            uint64_t generation = 0;
            if (it->is_directory(filesystemError) &&
                !filesystemError &&
                ParseGenerationDirectoryName(
                    it->path().filename().string(), generation)) {
                generations.emplace_back(generation, it->path());
            }
        }
        if (filesystemError) {
            SetError(error,
                     "failed to inspect persistent prefix cache generations: " +
                     filesystemError.message());
            return false;
        }
        std::sort(
            generations.begin(), generations.end(),
            [](const auto &left, const auto &right) {
                return left.first > right.first;
            });
        size_t retainedOther = 0;
        for (const auto &generation : generations) {
            if (generation.first == current) {
                continue;
            }
            if (retainedOther + 1 < keep) {
                retainedOther++;
                continue;
            }
            std::filesystem::remove_all(
                generation.second, filesystemError);
            if (filesystemError) {
                SetError(error,
                         "failed to prune persistent prefix cache generation: " +
                         filesystemError.message());
                return false;
            }
        }
        return SyncDirectory(root, error);
    }

    bool RemoveStaleStaging(const std::filesystem::path &root,
                            std::string *error) {
        std::error_code filesystemError;
        for (std::filesystem::directory_iterator it(root, filesystemError), end;
             !filesystemError && it != end; it.increment(filesystemError)) {
            std::string name = it->path().filename().string();
            if (name.rfind(".staging-", 0) == 0 ||
                name.rfind(".CURRENT-", 0) == 0) {
                std::filesystem::remove_all(it->path(), filesystemError);
                if (filesystemError) {
                    break;
                }
            }
        }
        if (filesystemError) {
            SetError(error, "failed to clean stale persistent prefix cache files: " +
                            filesystemError.message());
            return false;
        }
        return true;
    }
}

uint64_t PrefixCacheChecksum(const uint8_t *data, size_t bytes) {
    return Checksum(data, bytes);
}

uint64_t PrefixCacheChecksum(const std::vector<uint8_t> &bytes) {
    return Checksum(bytes);
}

std::string PersistentPrefixCacheKeyDirectoryName(
    const std::string &cacheKey) {
    std::string safe;
    safe.reserve(std::min<size_t>(cacheKey.size(), 96) + 17);
    for (char value : cacheKey) {
        if (safe.size() == 96) {
            break;
        }
        bool accepted = (value >= 'a' && value <= 'z') ||
                        (value >= 'A' && value <= 'Z') ||
                        (value >= '0' && value <= '9') ||
                        value == '.' || value == '_' || value == '-';
        safe.push_back(accepted ? value : '_');
    }
    if (safe.empty()) {
        safe = "cache";
    }
    std::ostringstream suffix;
    suffix << '-' << std::hex << std::setw(16) << std::setfill('0')
           << Checksum(reinterpret_cast<const uint8_t *>(cacheKey.data()),
                       cacheKey.size());
    safe += suffix.str();
    return safe;
}

void SetPersistentPrefixCacheCommitFailpointForTest(
    PersistentPrefixCommitFailpoint failpoint) {
    commitFailpoint.store(failpoint, std::memory_order_release);
}

bool CommitPersistentPrefixCacheGeneration(
    const std::filesystem::path &root,
    const PersistedPrefixCacheGeneration &source,
    std::vector<PersistentPayloadRef> &refs,
    std::string *error) {
    refs.clear();
    if (error != nullptr) {
        error->clear();
    }
    try {
        uint64_t payloadBytes = 0;
        if (root.empty() || !ValidateSource(source, payloadBytes, error)) {
            if (root.empty()) {
                SetError(error, "persistent prefix cache root is empty");
            }
            return false;
        }

        std::error_code filesystemError;
        std::filesystem::create_directories(root, filesystemError);
        if (filesystemError) {
            SetError(error, "failed to create persistent prefix cache root: " +
                            filesystemError.message());
            return false;
        }
        ScopedFileLock lock(root / "LOCK", true, error);
        if (!lock.Valid() || !RemoveStaleStaging(root, error)) {
            return false;
        }

        uint64_t maxBytes = 0;
        uint64_t minFreeBytes = 0;
        bool maxPresent = false;
        bool minFreePresent = false;
        if (!ParseEnvBytes("FASTLLM_PREFIX_CACHE_DISK_MAX_BYTES",
                           maxBytes, maxPresent, error) ||
            !ParseEnvBytes("FASTLLM_PREFIX_CACHE_DISK_MIN_FREE_BYTES",
                           minFreeBytes, minFreePresent, error)) {
            return false;
        }
        if (maxPresent) {
            uint64_t existingBytes = DirectoryBytes(root);
            if (payloadBytes > maxBytes || existingBytes > maxBytes - payloadBytes) {
                SetError(error, "persistent prefix cache disk byte limit exceeded");
                return false;
            }
        }
        if (minFreePresent) {
            std::filesystem::space_info space =
                std::filesystem::space(root, filesystemError);
            if (filesystemError || space.available < minFreeBytes ||
                payloadBytes > space.available - minFreeBytes) {
                SetError(error, "persistent prefix cache free-space floor reached");
                return false;
            }
        }

        const std::filesystem::path finalDirectory =
            GenerationPath(root, source.generation);
        if (std::filesystem::exists(finalDirectory, filesystemError)) {
            SetError(error, "persistent prefix cache generation already exists");
            return false;
        }
        const std::filesystem::path stagingDirectory = root /
            (".staging-" + std::to_string(ProcessId()) + '-' +
             std::to_string(Nonce()));
        if (!std::filesystem::create_directory(stagingDirectory,
                                               filesystemError)) {
            SetError(error, "failed to create persistent prefix cache staging directory: " +
                            filesystemError.message());
            return false;
        }

        bool committed = false;
        auto cleanup = [&]() {
            if (!committed) {
                std::error_code ignored;
                std::filesystem::remove_all(stagingDirectory, ignored);
            }
        };

#ifndef _WIN32
        int pagesFd = open((stagingDirectory / "pages.bin").c_str(),
                           O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
        if (pagesFd < 0) {
            SetError(error, "failed to create persistent prefix cache pages: " +
                            std::string(std::strerror(errno)));
            cleanup();
            return false;
        }
#else
        std::ofstream pagesFile(stagingDirectory / "pages.bin",
                                std::ios::binary | std::ios::trunc);
        if (!pagesFile) {
            SetError(error, "failed to create persistent prefix cache pages");
            cleanup();
            return false;
        }
#endif

        uint64_t offset = 0;
        std::vector<uint8_t> encoded;
        for (const PersistentPayloadRecord &payload : source.payloads) {
            bool compressed = false;
            if (!EncodePayload(payload.bytes, encoded, compressed, error)) {
#ifndef _WIN32
                close(pagesFd);
#endif
                cleanup();
                return false;
            }
            const uint8_t *storedData = compressed ? encoded.data() :
                payload.bytes.data();
            const size_t storedSize = compressed ? encoded.size() :
                payload.bytes.size();
#ifndef _WIN32
            if (!WriteSpan(pagesFd, storedData, storedSize, error)) {
                close(pagesFd);
                cleanup();
                return false;
            }
#else
            pagesFile.write(reinterpret_cast<const char *>(storedData),
                            (std::streamsize)storedSize);
            if (!pagesFile) {
                SetError(error, "failed to write persistent prefix cache pages");
                cleanup();
                return false;
            }
#endif
            PersistentPayloadRef ref;
            ref.kind = payload.kind;
            ref.name = payload.name;
            ref.dataType = payload.dataType;
            ref.dimensions = payload.dimensions;
            ref.fileOffset = offset;
            ref.storedBytes = storedSize;
            ref.uncompressedBytes = payload.bytes.size();
            ref.checksum = Checksum(payload.bytes);
            ref.zstdCompressed = compressed;
            refs.push_back(std::move(ref));
            offset += storedSize;
        }
#ifndef _WIN32
        const bool pagesSynced = fsync(pagesFd) == 0;
        const bool pagesClosed = close(pagesFd) == 0;
        if (!pagesSynced || !pagesClosed) {
            SetError(error, "failed to sync persistent prefix cache pages");
            cleanup();
            return false;
        }
#else
        pagesFile.flush();
        if (!pagesFile) {
            SetError(error, "failed to sync persistent prefix cache pages");
            cleanup();
            return false;
        }
        pagesFile.close();
#endif

        std::vector<uint8_t> manifest;
        if (!BuildManifest(source, refs, manifest, error) ||
            !WriteFileAndSync(stagingDirectory / "manifest.bin",
                              manifest.data(), manifest.size(), error) ||
            !SyncDirectory(stagingDirectory, error)) {
            cleanup();
            return false;
        }

        std::filesystem::rename(stagingDirectory, finalDirectory,
                                filesystemError);
        if (filesystemError) {
            SetError(error,
                     "failed to publish persistent prefix cache generation: " +
                     filesystemError.message());
            cleanup();
            return false;
        }
        if (!SyncDirectory(root, error)) {
            cleanup();
            return false;
        }
        committed = true;

        std::string currentText = std::to_string(source.generation) + "\n";
        const std::filesystem::path currentTemporary = root /
            (".CURRENT-" + std::to_string(ProcessId()) + '-' +
             std::to_string(Nonce()));
        if (!WriteFileAndSync(
                currentTemporary,
                reinterpret_cast<const uint8_t *>(currentText.data()),
                currentText.size(), error)) {
            return false;
        }
        if (commitFailpoint.load(std::memory_order_acquire) ==
            PersistentPrefixCommitFailpoint::BEFORE_CURRENT_RENAME) {
            std::filesystem::remove(currentTemporary, filesystemError);
            SetError(error, "injected failure before CURRENT rename");
            return false;
        }
        std::filesystem::rename(currentTemporary, root / "CURRENT",
                                filesystemError);
        if (filesystemError) {
            SetError(error,
                     "failed to publish persistent prefix cache CURRENT: " +
                     filesystemError.message());
            return false;
        }
        if (!SyncDirectory(root, error) ||
            !PrunePersistentPrefixCacheGenerations(
                root, source.generation, 2, error)) {
            return false;
        }
        return true;
    } catch (const std::exception &exception) {
        SetError(error, std::string("persistent prefix cache commit failed: ") +
                        exception.what());
        return false;
    }
}

bool LoadPersistentPrefixCacheGeneration(
    const std::filesystem::path &root,
    PersistedPrefixCacheGeneration &generation,
    std::vector<PersistentPayloadRef> &refs,
    std::string *error) {
    generation = {};
    refs.clear();
    if (error != nullptr) {
        error->clear();
    }
    try {
        if (root.empty()) {
            SetError(error, "persistent prefix cache root is empty");
            return false;
        }
        ScopedFileLock lock(root / "LOCK", false, error);
        if (!lock.Valid()) {
            return false;
        }
        uint64_t current = 0;
        if (!ReadCurrentGeneration(root, current, error)) {
            return false;
        }
        std::vector<uint8_t> manifest;
        if (!ReadBoundedFile(GenerationPath(root, current) / "manifest.bin",
                             MAX_MANIFEST_BYTES, manifest, error) ||
            !DecodeManifest(manifest, generation, refs, error) ||
            generation.generation != current) {
            if (generation.generation != 0 && generation.generation != current) {
                SetError(error, "persistent prefix cache generation mismatch");
            }
            generation = {};
            refs.clear();
            return false;
        }
        std::error_code filesystemError;
        uint64_t pagesBytes = std::filesystem::file_size(
            GenerationPath(root, current) / "pages.bin", filesystemError);
        uint64_t expectedBytes = refs.empty() ? 0 :
            refs.back().fileOffset + refs.back().storedBytes;
        if (filesystemError || pagesBytes != expectedBytes) {
            SetError(error, "persistent prefix cache pages size mismatch");
            generation = {};
            refs.clear();
            return false;
        }
        return true;
    } catch (const std::exception &exception) {
        SetError(error, std::string("persistent prefix cache load failed: ") +
                        exception.what());
        generation = {};
        refs.clear();
        return false;
    }
}

bool ReadPersistentPrefixCachePayload(
    const std::filesystem::path &root,
    uint64_t generation,
    const PersistentPayloadRef &ref,
    std::vector<uint8_t> &bytes,
    std::string *error) {
    bytes.clear();
    if (error != nullptr) {
        error->clear();
    }
    try {
        if (root.empty() || generation == 0 || !ValidateKind(ref.kind) ||
            ref.name.empty() || ref.name.size() > MAX_STRING_BYTES ||
            ref.dimensions.size() > MAX_DIMENSIONS ||
            ref.storedBytes > MAX_SINGLE_PAYLOAD_BYTES ||
            ref.uncompressedBytes > MAX_SINGLE_PAYLOAD_BYTES ||
            ref.fileOffset > std::numeric_limits<uint64_t>::max() -
                ref.storedBytes ||
            ref.storedBytes > (uint64_t)std::numeric_limits<size_t>::max() ||
            ref.uncompressedBytes >
                (uint64_t)std::numeric_limits<size_t>::max()) {
            SetError(error, "persistent prefix cache payload reference is invalid");
            return false;
        }
        ScopedFileLock lock(root / "LOCK", false, error);
        if (!lock.Valid()) {
            return false;
        }
        const std::filesystem::path pagesPath =
            GenerationPath(root, generation) / "pages.bin";
        std::error_code filesystemError;
        uint64_t pagesBytes = std::filesystem::file_size(
            pagesPath, filesystemError);
        if (filesystemError || ref.fileOffset + ref.storedBytes > pagesBytes) {
            SetError(error, "persistent prefix cache payload exceeds pages file");
            return false;
        }
        std::vector<uint8_t> stored((size_t)ref.storedBytes);
        std::ifstream file(pagesPath, std::ios::binary);
        if (!file) {
            SetError(error, "failed to open persistent prefix cache pages");
            return false;
        }
        file.seekg((std::streamoff)ref.fileOffset);
        if (ref.storedBytes > 0) {
            file.read(reinterpret_cast<char *>(stored.data()),
                      (std::streamsize)ref.storedBytes);
        }
        if (!file) {
            SetError(error, "persistent prefix cache payload is truncated");
            return false;
        }
        if (ref.zstdCompressed) {
#ifdef FASTLLM_USE_ZSTD
            bytes.resize((size_t)ref.uncompressedBytes);
            size_t result = ZSTD_decompress(
                bytes.data(), bytes.size(), stored.data(), stored.size());
            if (ZSTD_isError(result) || result != bytes.size()) {
                SetError(error, "persistent prefix cache payload decompression failed");
                bytes.clear();
                return false;
            }
#else
            SetError(error, "persistent prefix cache requires unavailable zstd");
            return false;
#endif
        } else {
            if (ref.storedBytes != ref.uncompressedBytes) {
                SetError(error, "persistent prefix cache payload size mismatch");
                return false;
            }
            bytes = std::move(stored);
        }
        if (Checksum(bytes) != ref.checksum) {
            SetError(error, "persistent prefix cache payload checksum mismatch");
            bytes.clear();
            return false;
        }
        return true;
    } catch (const std::exception &exception) {
        SetError(error, std::string("persistent prefix cache payload read failed: ") +
                        exception.what());
        bytes.clear();
        return false;
    }
}

namespace {
    struct PersistentPrefixCacheRuntime {
        std::mutex locker;
        PersistentPrefixCacheOptions options;
        std::filesystem::path root;
        uint64_t generation = 0;
        std::vector<PersistentPayloadRef> refs;
        PersistentPrefixCacheStatus status;
    };

    PersistentPrefixCacheRuntime &GetPersistentPrefixCacheRuntime() {
        static PersistentPrefixCacheRuntime runtime;
        return runtime;
    }

    bool PersistentPrefixCacheEnvEnabled(const char *name) {
        const char *value = std::getenv(name);
        if (value == nullptr) {
            return false;
        }
        std::string normalized(value);
        std::transform(normalized.begin(), normalized.end(),
                       normalized.begin(), [](unsigned char character) {
            return (char)std::tolower(character);
        });
        return normalized == "1" || normalized == "true" ||
               normalized == "yes" || normalized == "on";
    }

    uint64_t NextPersistentPrefixCacheGeneration(
            const std::filesystem::path &root) {
        uint64_t next = 1;
        std::error_code error;
        for (std::filesystem::directory_iterator it(
                 root,
                 std::filesystem::directory_options::skip_permission_denied,
                 error), end;
             !error && it != end; it.increment(error)) {
            const std::string name =
                it->path().filename().string();
            uint64_t value = 0;
            if (ParseGenerationDirectoryName(name, value) &&
                value >= next &&
                value < std::numeric_limits<uint64_t>::max()) {
                next = value + 1;
            }
        }
        return next;
    }
}

bool PreparePersistentPrefixCache(
    const PersistentPrefixCacheOptions &options,
    basellm *model,
    PersistentPrefixCacheStatus &status,
    std::string *error) {
    (void)model;
    if (error != nullptr) {
        error->clear();
    }
    PersistentPrefixCacheRuntime &runtime =
        GetPersistentPrefixCacheRuntime();
    {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.options = options;
        runtime.root.clear();
        runtime.generation = 0;
        runtime.refs.clear();
        runtime.status = {};
        runtime.status.enabled = options.enabled;
    }
    if (!options.enabled) {
        status = GetPersistentPrefixCacheStatus();
        return true;
    }
    if (options.diskDirectory.empty() || options.cacheKey.empty()) {
        SetError(error,
                 "persistent prefix cache requires a disk directory and key");
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.lastError =
            error == nullptr ? "invalid persistent prefix cache options" :
                               *error;
        status = runtime.status;
        return false;
    }

    const std::filesystem::path root =
        options.diskDirectory /
        PersistentPrefixCacheKeyDirectoryName(options.cacheKey);
    std::error_code filesystemError;
    std::filesystem::create_directories(root, filesystemError);
    if (filesystemError) {
        SetError(error, "failed to create persistent prefix cache root: " +
                        filesystemError.message());
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.lastError =
            error == nullptr ? "failed to create cache root" : *error;
        status = runtime.status;
        return false;
    }
    if (!std::filesystem::exists(root / "CURRENT", filesystemError)) {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.root = root;
        status = runtime.status;
        return true;
    }
    if (filesystemError) {
        SetError(error, "failed to inspect persistent prefix cache CURRENT: " +
                        filesystemError.message());
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.lastError =
            error == nullptr ? "failed to inspect cache generation" : *error;
        status = runtime.status;
        return false;
    }

    PersistedPrefixCacheGeneration loaded;
    std::vector<PersistentPayloadRef> refs;
    if (!LoadPersistentPrefixCacheGeneration(
            root, loaded, refs, error) ||
        loaded.cacheKey != options.cacheKey) {
        if (loaded.cacheKey != options.cacheKey) {
            SetError(error, "persistent prefix cache key mismatch");
        }
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.root = root;
        runtime.status.lastError =
            error == nullptr ? "failed to load cache generation" : *error;
        status = runtime.status;
        return false;
    }

    if (model != nullptr) {
        std::vector<PersistentPayloadRecord> extras;
        for (const PersistentPayloadRef &ref : refs) {
            if (ref.kind != PersistentPayloadKind::MODEL_EXTRA) {
                continue;
            }
            PersistentPayloadRecord record;
            record.kind = ref.kind;
            record.name = ref.name;
            record.dataType = ref.dataType;
            record.dimensions = ref.dimensions;
            if (!ReadPersistentPrefixCachePayload(
                    root,
                    loaded.generation,
                    ref,
                    record.bytes,
                    error)) {
                std::lock_guard<std::mutex> guard(runtime.locker);
                runtime.root = root;
                runtime.status.lastError =
                    error == nullptr ?
                        "failed to load model prefix cache extras" :
                        *error;
                status = runtime.status;
                return false;
            }
            extras.push_back(std::move(record));
        }
        if (!model->ImportPersistentPrefixCacheExtras(
                extras, error)) {
            std::lock_guard<std::mutex> guard(runtime.locker);
            runtime.root = root;
            runtime.status.lastError =
                error == nullptr ?
                    "model prefix cache extras are incompatible" :
                    *error;
            status = runtime.status;
            return false;
        }
    }

    uint64_t payloadBytes = 0;
    for (const PersistentPayloadRef &ref : refs) {
        if (ref.uncompressedBytes <=
            std::numeric_limits<uint64_t>::max() - payloadBytes) {
            payloadBytes += ref.uncompressedBytes;
        }
    }
    {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.root = root;
        runtime.generation = loaded.generation;
        runtime.refs = std::move(refs);
        runtime.status.loadedGeneration = loaded.generation;
        runtime.status.payloadBytes = payloadBytes;
    }
    AttachPreparedPersistentPrefixCacheManagers();
    status = GetPersistentPrefixCacheStatus();
    return true;
}

bool PreparePersistentPrefixCacheFromEnv(
    basellm *model,
    PersistentPrefixCacheStatus &status,
    std::string *error) {
    PersistentPrefixCacheOptions options;
    options.enabled =
        PersistentPrefixCacheEnvEnabled("FASTLLM_PREFIX_CACHE_PERSIST");
    if (const char *directory =
            std::getenv("FASTLLM_PREFIX_CACHE_DISK_DIR")) {
        options.diskDirectory = directory;
    }
    if (const char *key =
            std::getenv("FASTLLM_PREFIX_CACHE_PERSIST_KEY")) {
        options.cacheKey = key;
    }
    return PreparePersistentPrefixCache(
        options, model, status, error);
}

bool CheckpointPersistentPrefixCache(
    basellm *model,
    PersistentPrefixCheckpointStats &stats,
    std::string *error) {
    stats = {};
    if (error != nullptr) {
        error->clear();
    }
    PersistentPrefixCacheRuntime &runtime =
        GetPersistentPrefixCacheRuntime();
    PersistentPrefixCacheOptions options;
    std::filesystem::path root;
    {
        std::lock_guard<std::mutex> guard(runtime.locker);
        options = runtime.options;
        root = runtime.root;
    }
    if (!options.enabled || root.empty()) {
        SetError(error, "persistent prefix cache is disabled");
        return false;
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<PersistentPayloadRecord> records;
    uint64_t pages = 0;
    if (!ExportPersistentPagedCacheRecords(
            records, pages, error)) {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.lastError =
            error == nullptr ? "failed to export paged cache" : *error;
        return false;
    }
    if (model != nullptr &&
        !model->ExportPersistentPrefixCacheExtras(
            records, error)) {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.lastError =
            error == nullptr ?
                "failed to export model prefix cache extras" :
                *error;
        return false;
    }
    PersistedPrefixCacheGeneration source;
    source.generation =
        NextPersistentPrefixCacheGeneration(root);
    source.cacheKey = options.cacheKey;
    source.payloads = std::move(records);

    uint64_t payloadBytes = 0;
    for (const PersistentPayloadRecord &record : source.payloads) {
        if (record.bytes.size() <=
            std::numeric_limits<uint64_t>::max() - payloadBytes) {
            payloadBytes += record.bytes.size();
        }
    }
    std::vector<PersistentPayloadRef> refs;
    if (!CommitPersistentPrefixCacheGeneration(
            root, source, refs, error)) {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.lastError =
            error == nullptr ? "failed to commit cache generation" : *error;
        return false;
    }
    stats.generation = source.generation;
    stats.pages = pages;
    stats.bytes = payloadBytes;
    stats.durationMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    {
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.checkpointCount++;
        runtime.status.payloadBytes = payloadBytes;
        runtime.status.lastDurationMs = stats.durationMs;
        runtime.status.lastError.clear();
    }
    return true;
}

PersistentPrefixCacheStatus GetPersistentPrefixCacheStatus() {
    PersistentPrefixCacheRuntime &runtime =
        GetPersistentPrefixCacheRuntime();
    std::lock_guard<std::mutex> guard(runtime.locker);
    return runtime.status;
}

void ResetPersistentPrefixCacheForTest() {
    PersistentPrefixCacheRuntime &runtime =
        GetPersistentPrefixCacheRuntime();
    std::lock_guard<std::mutex> guard(runtime.locker);
    runtime.options = {};
    runtime.root.clear();
    runtime.generation = 0;
    runtime.refs.clear();
    runtime.status = {};
    commitFailpoint.store(
        PersistentPrefixCommitFailpoint::NONE,
        std::memory_order_release);
}

namespace persistent_prefix_cache_internal {
    void AttachPreparedManager(
            int managerId, PagedCacheManager *manager) {
        if (manager == nullptr || manager->persistentRestoreAttempted) {
            return;
        }
        PersistentPrefixCacheRuntime &runtime =
            GetPersistentPrefixCacheRuntime();
        std::filesystem::path root;
        uint64_t generation = 0;
        std::vector<PersistentPayloadRef> refs;
        {
            std::lock_guard<std::mutex> guard(runtime.locker);
            if (!runtime.status.enabled ||
                runtime.generation == 0) {
                return;
            }
            root = runtime.root;
            generation = runtime.generation;
            refs = runtime.refs;
        }
        manager->persistentRestoreAttempted = true;
        const std::string trieName =
            "manager/" + std::to_string(managerId) + "/trie";
        auto trie = std::find_if(
            refs.begin(), refs.end(),
            [&](const PersistentPayloadRef &ref) {
                return ref.kind ==
                           PersistentPayloadKind::PAGED_CACHE_TRIE &&
                       ref.name == trieName;
            });
        if (trie == refs.end()) {
            return;
        }
        std::vector<uint8_t> trieBytes;
        std::string restoreError;
        if (!ReadPersistentPrefixCachePayload(
                root, generation, *trie, trieBytes, &restoreError) ||
            !manager->ImportPersistentRecords(
                root, generation, trieBytes, refs, &restoreError)) {
            std::lock_guard<std::mutex> guard(runtime.locker);
            runtime.status.lastError = restoreError;
        }
    }

    void ObserveRestoreHit() {
        PersistentPrefixCacheRuntime &runtime =
            GetPersistentPrefixCacheRuntime();
        std::lock_guard<std::mutex> guard(runtime.locker);
        runtime.status.restoreHitCount++;
    }
}
}
