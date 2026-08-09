# Recoverable Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist FastLLM's paged K/V trie plus Qwen3.5/3.6 linear-attention and MTP prefix snapshots as atomic disk generations that Thinking Proxy can checkpoint before unloading and restore lazily after restart.

**Architecture:** A new core archive component owns bounded binary parsing, immutable generation payloads, filesystem locking, and atomic `CURRENT` commits. `PagedCacheManager` exports/imports generic trie records; `basellm` supplies an optional model-extra hook implemented by Qwen3.5. The API server exposes one per-spawn-token-protected checkpoint route, and the owned-backend proxy calls it only after all leases drain and before it signals its child.

**Tech Stack:** C++17, FastLLM `Data`/paged-cache APIs, POSIX `flock`/`fsync` with portable fail-open fallback, zstd when compiled, json11 HTTP responses, Python 3.12 asyncio/httpx, `unittest`.

---

## File map

- Create `include/prefixcache_persistence.h`: bounded archive types, checkpoint/restore results, core API.
- Create `src/prefixcache_persistence.cpp`: binary codec, generation lock/commit/load, generic `Data` records.
- Modify `CMakeLists.txt`: compile the new source.
- Modify `include/fastllm.h`, `src/fastllm.cpp`: stable manager id, trie export/import, immutable generation disk refs, metrics.
- Modify `include/models/basellm.h`: optional model-extra capture/restore hooks.
- Modify `include/models/qwen3_5.h`, `src/models/qwen3_5.cpp`: Qwen linear/GDN/MTP extra archive.
- Modify `example/apiserver/apiserver.cpp`: prepare restore, authenticated checkpoint endpoint, props metrics.
- Create `example/apiserver/checkpoint_control.h`, `test/apiserver/checkpointControlTest.cpp`: pure control-route authorization/quiescence policy and tests.
- Modify `test/ops/regressionOps.cpp`: archive, corruption, paged-trie, and Qwen extra regressions.
- Modify `../v100-perfs/thinking_proxy.py`: per-child token, checkpoint-before-signal, lifecycle metrics.
- Modify `../v100-perfs/test_thinking_proxy_lifecycle.py`: ordering, failure, and external-isolation tests.
- Modify `../v100-perfs/scripts/start_proxy.sh`, `../v100-perfs/README.md`: deployment knobs and exact persistence semantics.
- Add durable V100 evidence under `../v100-perfs/benchmarks/fastllm/results/` only after the end-to-end run passes.

### Task 1: Bounded binary archive and atomic generations

**Files:**
- Create: `include/prefixcache_persistence.h`
- Create: `src/prefixcache_persistence.cpp`
- Modify: `CMakeLists.txt:209-216`
- Test: `test/ops/regressionOps.cpp`

- [ ] **Step 1: Write failing archive corruption and atomic-commit regressions**

Add `RunPersistentPrefixArchiveRegression()` and a `FASTLLM_REGRESSION_ONLY=persistent_prefix_cache` dispatch. The fixture must use a unique directory under `std::filesystem::temp_directory_path()`, write generation 1, inject a failed generation-2 commit before `CURRENT` rename, and assert generation 1 still loads. Mutate each of magic, schema, payload offset, payload checksum, and manifest truncation and assert `LoadPersistentPrefixCacheGeneration` returns `false` with a non-empty error.

```cpp
void RunPersistentPrefixArchiveRegression() {
    ScopedTempDirectory root("fastllm-prefix-persist");
    fastllm::PersistentPrefixGeneration source;
    source.cacheKey = "qwen36-q8-turbo3-page128";
    source.generation = 1;
    source.payloads.push_back({{1, 2, 3, 4}, false, 4,
                               fastllm::PrefixCacheChecksum({1, 2, 3, 4})});
    std::string error;
    std::vector<fastllm::PersistentPayloadRef> refs;
    Expect(fastllm::CommitPersistentPrefixCacheGeneration(
               root.path(), source, refs, &error), error);

    fastllm::PersistentPrefixGeneration loaded;
    Expect(fastllm::LoadPersistentPrefixCacheGeneration(
               root.path(), source.cacheKey, loaded, &error), error);
    Expect(loaded.generation == 1 && loaded.payloads.size() == 1,
           "persistent generation did not round trip");

    fastllm::SetPersistentPrefixCacheCommitFailpointForTest(
        fastllm::PersistentCommitFailpoint::BEFORE_CURRENT_RENAME);
    source.generation = 2;
    refs.clear();
    Expect(!fastllm::CommitPersistentPrefixCacheGeneration(
               root.path(), source, refs, &error),
           "injected commit failure unexpectedly succeeded");
    fastllm::SetPersistentPrefixCacheCommitFailpointForTest(
        fastllm::PersistentCommitFailpoint::NONE);
    Expect(fastllm::LoadPersistentPrefixCacheGeneration(
               root.path(), source.cacheKey, loaded, &error) &&
           loaded.generation == 1,
           "failed commit replaced the last valid CURRENT");

    CorruptAndExpectPersistentLoadFailure(root.path(), "magic");
    CorruptAndExpectPersistentLoadFailure(root.path(), "schema");
    CorruptAndExpectPersistentLoadFailure(root.path(), "offset");
    CorruptAndExpectPersistentLoadFailure(root.path(), "checksum");
    CorruptAndExpectPersistentLoadFailure(root.path(), "truncate");
}
```

- [ ] **Step 2: Build and prove the selector fails before implementation**

Run:

```bash
cmake --build build --target regressionOps -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
```

Expected: compile failure for undefined persistent archive types/functions.

- [ ] **Step 3: Implement the bounded codec and generation transaction**

Define exact public types in `include/prefixcache_persistence.h`:

```cpp
namespace fastllm {
constexpr uint32_t PERSISTENT_PREFIX_SCHEMA = 1;
struct PersistentPayloadRef {
    uint64_t offset = 0, storedBytes = 0, uncompressedBytes = 0;
    bool zstdCompressed = false;
    uint64_t checksum = 0;
};
struct PersistentPayload {
    std::vector<uint8_t> bytes;
    bool zstdCompressed = false;
    uint64_t uncompressedBytes = 0;
    uint64_t checksum = 0;
};
struct PersistentPrefixGeneration {
    std::string cacheKey;
    uint64_t generation = 0;
    std::vector<PersistentPayload> payloads;
    std::vector<uint8_t> managerManifest;
    std::vector<uint8_t> modelExtra;
};
enum class PersistentCommitFailpoint { NONE, BEFORE_CURRENT_RENAME };
uint64_t PrefixCacheChecksum(const uint8_t *data, size_t bytes);
uint64_t PrefixCacheChecksum(const std::vector<uint8_t> &bytes);
bool CommitPersistentPrefixCacheGeneration(
    const std::string &root, const PersistentPrefixGeneration &generation,
    std::vector<PersistentPayloadRef> &refs, std::string *error);
bool LoadPersistentPrefixCacheGeneration(
    const std::string &root, const std::string &cacheKey,
    PersistentPrefixGeneration &generation, std::string *error);
void SetPersistentPrefixCacheCommitFailpointForTest(
    PersistentCommitFailpoint failpoint);
}
```

Implement little-endian `ArchiveWriter`/`ArchiveReader` with `WriteU8/U32/U64/I32/I64/String/Bytes` and matching reads. Every read must check `requested <= remaining` before pointer arithmetic; cap strings/vectors at the manifest's remaining bytes and reject trailing bytes. Escape the key to `[A-Za-z0-9._-]`, append a checksum suffix, and never accept a path from the archive.

On POSIX: open `<key>/LOCK`, `flock(LOCK_EX|LOCK_NB)`, clean only `.staging-*`, write and `fsync` `pages.bin`, write/checksum/`fsync` `manifest.bin`, rename staging to `gen-N`, write/`fsync` `CURRENT.new`, rename to `CURRENT`, `fsync` the key directory, then remove generations not named by `CURRENT`. On Windows return a clear unsupported error while persistence remains disabled by default.

- [ ] **Step 4: Compile the component and run the focused regression**

Add `src/prefixcache_persistence.cpp` to `FASTLLM_CXX_SOURCES`, then run the selector above. Expected: `persistent prefix archive regression: PASS`.

- [ ] **Step 5: Commit the archive layer**

```bash
git add CMakeLists.txt include/prefixcache_persistence.h src/prefixcache_persistence.cpp test/ops/regressionOps.cpp
git commit -m "feat: add atomic prefix cache generations"
```

### Task 2: Export and lazily restore paged K/V trie nodes

**Files:**
- Modify: `include/fastllm.h:649-737`
- Modify: `src/fastllm.cpp:4938-5075,6284-6639,6829-7625`
- Modify: `include/prefixcache_persistence.h`
- Modify: `src/prefixcache_persistence.cpp`
- Test: `test/ops/regressionOps.cpp:1004-1327`

- [ ] **Step 1: Write a failing two-manager restart regression**

Build two CPU `PagedCacheManager`s through `AllocatePagedCacheManager`, fill deterministic page bytes, record the same two token pages, checkpoint, call `ClearAllPagedCacheManagers`, re-create managers with the same geometry, prepare the generation, and query. Assert restored page ids are materialized, bytes match exactly, `GetPagedPrefixCachePersistentRestoreHits()` increases, and an altered page length rejects restore without affecting a normal miss.

```cpp
fastllm::PersistentPrefixCacheOptions options {
    true, root.path(), key
};
fastllm::PersistentPrefixCacheStatus status;
fastllm::PersistentPrefixCheckpointStats stats;
Expect(fastllm::PreparePersistentPrefixCache(
           options, nullptr, status, &error), error);
auto *k = MakeCpuPagedManager(20, 2, {0x11, 0x12, 0x21, 0x22});
auto *v = MakeCpuPagedManager(21, 2, {0x31, 0x32, 0x41, 0x42});
RecordTwoPagePrefix(*k, {10, 11, 12, 13});
RecordTwoPagePrefix(*v, {10, 11, 12, 13});
Expect(fastllm::CheckpointPersistentPrefixCache(nullptr, stats, &error), error);
fastllm::ClearAllPagedCacheManagers();
fastllm::ResetPersistentPrefixCacheForTest();
Expect(fastllm::PreparePersistentPrefixCache(
           options, nullptr, status, &error), error);
k = MakeCpuPagedManager(20, 2, {});
v = MakeCpuPagedManager(21, 2, {});
std::vector<int> kPages, vPages;
k->Query({10, 11, 12, 13}, kPages);
v->Query({10, 11, 12, 13}, vPages);
Expect(kPages.size() == 2 && vPages.size() == 2,
       "restored paged prefix did not materialize both K/V pages");
Expect(ReadLogicalPages(*k, kPages) ==
           std::vector<uint8_t>({0x11, 0x12, 0x21, 0x22}),
       "restored K bytes changed");
```

- [ ] **Step 2: Run the selector and observe the missing paged-manager APIs**

Run the focused regression. Expected: compile failure for `PreparePersistentPrefixCache`, `CheckpointPersistentPrefixCache`, persistent metrics, or stable manager ids.

Define the high-level API once and use it from tests and the API server:

```cpp
struct PersistentPrefixCacheOptions {
    bool enabled = false;
    std::string root;
    std::string cacheKey;
};
struct PersistentPrefixCacheStatus {
    bool enabled = false;
    uint64_t loadedGeneration = 0;
    uint64_t restoreHits = 0;
    uint64_t restoreBytes = 0;
    std::string lastError;
};
struct PersistentPrefixCheckpointStats {
    uint64_t generation = 0;
    uint64_t managers = 0;
    uint64_t nodes = 0;
    uint64_t payloadBytes = 0;
    double durationMilliseconds = 0.0;
};
bool PreparePersistentPrefixCache(
    const PersistentPrefixCacheOptions &options, basellm *model,
    PersistentPrefixCacheStatus &status, std::string *error);
bool PreparePersistentPrefixCacheFromEnv(
    basellm *model, PersistentPrefixCacheStatus &status,
    std::string *error);
bool CheckpointPersistentPrefixCache(
    basellm *model, PersistentPrefixCheckpointStats &stats,
    std::string *error);
```
- [ ] **Step 3: Add stable ids and immutable generation refs**

Add `int persistentId = -1` to `PagedCacheManager`; set it from `layerIndex` before `SetMaxPages`. Extend `PagedPrefixCacheTierDiskRef` with an optional `std::shared_ptr<PersistentGenerationFile>` plus `payloadIndex`. Existing session refs keep that pointer null. `ReleasePagedPrefixCacheDiskReference` must call session `Release()` only for session refs; immutable generation refs only reset shared ownership.

`PagedPrefixCacheDiskStore::Read` must delegate generation refs to a bounded pread/read function that verifies offset, stored length, and checksum. It must never make a generation file writable or truncate it.

- [ ] **Step 4: Serialize manager geometry and trie records**

Use this stable record contract:

```cpp
struct PersistentTrieNodeRecord {
    uint32_t parent = UINT32_MAX;
    uint64_t edgeHash = 0;
    std::vector<int> edgeTokens;
    uint64_t accessCount = 0;
    int64_t lastAccessOrder = 0;
    int32_t depthPages = 0;
    int32_t maxPrefixDepthPages = 0;
    uint32_t payloadIndex = UINT32_MAX;
};
struct PersistentManagerRecord {
    int32_t id = -1;
    int32_t type = 0;
    int32_t pageLen = 0;
    int32_t dataType = 0;
    uint64_t pageBytes = 0;
    std::vector<PersistentTrieNodeRecord> nodes;
};
```

At checkpoint, lock global manager-map mutex, sort by `persistentId`, then lock each `pageIndexLocker` in that order. Export only complete nodes with valid full `edgeTokens`. Capture payload from resident CPU/CUDA bytes, CPU tier payload, or existing disk ref; normalize to raw-or-zstd bytes and append to the new generation. Do not apply the runtime min-hit/min-token storage policy during an explicit checkpoint.

At prepare, parse the manager records once. During `AllocatePagedCacheManager`, validate id/type/pageLen/dtype/pageBytes; rebuild nodes with `pageId=-1` and generation refs. Preserve access ordering, but never restore physical page ids/refcounts/free lists.

- [ ] **Step 5: Run focused tests and commit paged persistence**

```bash
cmake --build build --target regressionOps -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
git add include/fastllm.h include/prefixcache_persistence.h src/fastllm.cpp src/prefixcache_persistence.cpp test/ops/regressionOps.cpp
git commit -m "feat: restore paged prefix tries across processes"
```

Expected: archive and two-manager restart tests pass; existing `FASTLLM_REGRESSION_ONLY=paged_prefix_cache_policy` also passes.

### Task 3: Persist Qwen3.5/3.6 GDN and MTP prefix extras

**Files:**
- Modify: `include/models/basellm.h:245-275`
- Modify: `include/models/qwen3_5.h`
- Modify: `src/models/qwen3_5.cpp:4147-4519,7957-8165`
- Modify: `include/prefixcache_persistence.h`
- Modify: `src/prefixcache_persistence.cpp`
- Test: `test/ops/regressionOps.cpp`

- [ ] **Step 1: Add a failing model-extra round-trip regression**

Expose only a deterministic test fixture builder on `Qwen3_5Model` under the existing regression friend pattern. Create one snapshot with tokens, one single-device linear layer, one two-local TP linear layer, and MTP key/value. Capture to bytes, destroy the source model, restore into a new model, then capture again and require byte equality. Corrupt one tensor checksum and require the whole snapshot to be rejected.

```cpp
Qwen3_5Model source;
source.InstallPersistentPrefixFixtureForTest(
    {101, 102, 103, 104}, MakeLinearFixture(), MakeMtpFixture());
std::vector<uint8_t> encoded;
std::string error;
Expect(source.CapturePersistentPrefixCacheExtra(encoded, &error), error);
Qwen3_5Model restored;
Expect(restored.RestorePersistentPrefixCacheExtra(encoded, &error), error);
std::vector<uint8_t> roundTrip;
Expect(restored.CapturePersistentPrefixCacheExtra(roundTrip, &error), error);
Expect(encoded == roundTrip, "Qwen linear/MTP prefix extra changed");
encoded.back() ^= 0x80;
Expect(!restored.RestorePersistentPrefixCacheExtra(encoded, &error),
       "corrupt Qwen prefix extra was accepted");
```

- [ ] **Step 2: Add default model hooks and generic CPU tensor records**

In `basellm` add:

```cpp
virtual bool CapturePersistentPrefixCacheExtra(
    std::vector<uint8_t> &bytes, std::string *error) const {
    bytes.clear();
    return true;
}
virtual bool RestorePersistentPrefixCacheExtra(
    const std::vector<uint8_t> &bytes, std::string *error) {
    return bytes.empty();
}
```

Add archive helpers that serialize a CPU `Data` record: dtype, dims, KV/linear/transposed flags, TP layout/axis/global dims/ranges, and exact `GetBytes()` bytes plus checksum. Reject CUDA-only pointers; Qwen capture already copies snapshots to CPU.

- [ ] **Step 3: Implement Qwen extra capture and all-or-nothing restore**

Under `Qwen35LinearPrefixSnapshotsMutex`, serialize records in deterministic timestamp/token order. For each snapshot write cached length, full tokens, request/timestamp order, layer count, each linear layer's first/second single or per-device local tensors and TP metadata, then optional MTP tokens/key/value. Restore into temporary `unique_ptr<Qwen35LinearPrefixSnapshot>` objects, validate every tensor and every linear layer, and only then swap them into `Qwen35LinearPrefixSnapshots()[this]`. A partial snapshot must never enter the map.

The top-level checkpoint calls `model->CapturePersistentPrefixCacheExtra`; prepare calls restore only after the generation and generic manager manifest validate. If Qwen extra is missing/corrupt, do not install paged manager records for a Qwen model with linear layers, because `QueryPagedPrefixCacheExtra` would otherwise correctly reduce the hit to zero.

- [ ] **Step 4: Run CPU regressions and commit model extras**

```bash
cmake --build build --target regressionOps -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
FASTLLM_REGRESSION_ONLY=paged_prefix_cache_policy ./build/regressionOps
git add include/models/basellm.h include/models/qwen3_5.h include/prefixcache_persistence.h src/models/qwen3_5.cpp src/prefixcache_persistence.cpp test/ops/regressionOps.cpp
git commit -m "feat: persist Qwen linear and MTP prefix state"
```

Expected: both selectors report PASS; corrupt Qwen extra restores an empty cache without throwing.

### Task 4: Add the protected FastLLM checkpoint control plane

**Files:**
- Modify: `example/apiserver/apiserver.cpp:186-205,358-533,1058-1135`
- Create: `example/apiserver/checkpoint_control.h`
- Create: `test/apiserver/checkpointControlTest.cpp`
- Modify: `CMakeLists.txt:627-633`
- Modify: `include/prefixcache_persistence.h`
- Modify: `src/prefixcache_persistence.cpp`

- [ ] **Step 1: Write failing checkpoint-control policy tests**

Extract a pure decision helper and test these inputs: GET → `METHOD_NOT_ALLOWED`; POST without/wrong token → `FORBIDDEN`; disabled persistence → `DISABLED`; POST with active or queued generation work → `BUSY`; correct token while enabled and idle → `ALLOW`. Include unequal-length and one-byte-different tokens, and assert no diagnostic string contains the expected token.

```cpp
using fastllm::apiserver::CheckpointControlDecision;
ExpectDecision(\"GET\", \"Bearer secret\", \"secret\", true, 0, 0,
               CheckpointControlDecision::METHOD_NOT_ALLOWED);
ExpectDecision(\"POST\", \"\", \"secret\", true, 0, 0,
               CheckpointControlDecision::FORBIDDEN);
ExpectDecision(\"POST\", \"Bearer secreu\", \"secret\", true, 0, 0,
               CheckpointControlDecision::FORBIDDEN);
ExpectDecision(\"POST\", \"Bearer secret\", \"secret\", false, 0, 0,
               CheckpointControlDecision::DISABLED);
ExpectDecision(\"POST\", \"Bearer secret\", \"secret\", true, 1, 0,
               CheckpointControlDecision::BUSY);
ExpectDecision(\"POST\", \"Bearer secret\", \"secret\", true, 0, 1,
               CheckpointControlDecision::BUSY);
ExpectDecision(\"POST\", \"Bearer secret\", \"secret\", true, 0, 0,
               CheckpointControlDecision::ALLOW);
```

- [ ] **Step 2: Prepare persistence after model load**

After model construction and before accepting sockets, call:

```cpp
std::string persistenceError;
fastllm::PersistentPrefixCacheStatus persistenceStatus;
if (!fastllm::PreparePersistentPrefixCacheFromEnv(
        model.get(), persistenceStatus, &persistenceError)) {
    fprintf(stderr, "[prefix-persist] restore skipped: %s\n",
            persistenceError.c_str());
}
```

The prepare helper reads the three required env vars; absent/false means a clean disabled status, not an error.

- [ ] **Step 3: Implement the authenticated route**

Normalize header key case and trim OWS around values. Compare `Authorization: Bearer <token>` in constant time against `FASTLLM_PREFIX_CACHE_CONTROL_TOKEN`. The endpoint accepts no body path/key. Under the WorkQueue lock calculate active work excluding the control request; return 409 when active or queued generation work exists. Serialize checkpoint calls with a dedicated mutex.

```cpp
enum class CheckpointControlDecision {
    ALLOW, METHOD_NOT_ALLOWED, FORBIDDEN, DISABLED, BUSY
};
CheckpointControlDecision EvaluateCheckpointControl(
    const std::string &method, const std::string &authorization,
    const std::string &expectedToken, bool persistenceEnabled,
    int activeGenerationRequests, int queuedGenerationRequests);
```

Add the focused target:

```cmake
add_executable(testCheckpointControl
    test/apiserver/checkpointControlTest.cpp)
target_include_directories(testCheckpointControl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
```

Return 503 on disabled persistence, 500 on checkpoint failure, and 200 with exact stats on success. Extend `/props` with enabled/loaded generation, checkpoint/restore counts, payload bytes, duration, and last error.

- [ ] **Step 4: Build, run socket/core selectors, and commit**

```bash
cmake --build build --target apiserver regressionOps testCheckpointControl -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
./build/testCheckpointControl
git add CMakeLists.txt example/apiserver/apiserver.cpp example/apiserver/checkpoint_control.h include/prefixcache_persistence.h src/prefixcache_persistence.cpp test/apiserver/checkpointControlTest.cpp
git commit -m \"feat: expose protected prefix checkpoint endpoint\"
```

### Task 5: Checkpoint owned backends before Thinking Proxy signals them

**Files:**
- Modify: `../v100-perfs/thinking_proxy.py:63-140,327-645,1643-1683`
- Modify: `../v100-perfs/test_thinking_proxy_lifecycle.py`
- Modify: `../v100-perfs/scripts/start_proxy.sh:1-140`

- [ ] **Step 1: Write failing lifecycle-order tests**

Add three observable contracts:

```python
async def test_owned_stop_drains_checkpoints_then_signals(self):
    order = []

    async def checkpoint_backend(child):
        order.append("checkpoint")
        return {"generation": child.generation, "duration_ms": 12.5}

    async def stop_backend(child):
        order.append("signal")
        child.returncode = 0

    manager, _ = self.make_manager(
        checkpoint_backend=checkpoint_backend,
        stop_backend=stop_backend,
    )
    lease = await manager.acquire()
    stop = asyncio.create_task(manager.drain_and_stop("idle", timeout=1.0))
    await asyncio.sleep(0)
    self.assertEqual(order, [])
    await lease.release()
    self.assertTrue(await stop)
    self.assertEqual(order, ["checkpoint", "signal"])

async def test_checkpoint_failure_still_signals_owned_child(self):
    order = []

    async def checkpoint_backend(child):
        order.append("checkpoint")
        raise RuntimeError("disk full")

    async def stop_backend(child):
        order.append("signal")
        child.returncode = 0

    manager, _ = self.make_manager(
        checkpoint_backend=checkpoint_backend,
        stop_backend=stop_backend,
    )
    lease = await manager.acquire()
    await lease.release()
    self.assertTrue(await manager.stop("memory_pressure"))
    self.assertEqual(order, ["checkpoint", "signal"])
    snapshot = manager.snapshot()
    self.assertEqual(snapshot["checkpoint_failures"], 1)
    self.assertIn("disk full", snapshot["last_checkpoint_error"])

async def test_external_backend_never_checkpoints_or_signals(self):
    calls = {"checkpoint": 0, "stop": 0}

    async def checkpoint_backend(child):
        calls["checkpoint"] += 1

    async def stop_backend(child):
        calls["stop"] += 1

    manager, _ = self.make_manager(
        owned=False,
        checkpoint_backend=checkpoint_backend,
        stop_backend=stop_backend,
    )
    lease = await manager.acquire()
    await lease.release()
    self.assertFalse(await manager.stop("shutdown"))
    self.assertEqual(calls, {"checkpoint": 0, "stop": 0})
```

Also assert each activation receives a different non-empty control token and that the token never appears in lifecycle snapshots/log messages.

- [ ] **Step 2: Run the focused lifecycle suite and observe failure**

```bash
../.venv-1cat/bin/python -m unittest -v test_thinking_proxy_lifecycle
```

Expected: failures for missing checkpoint callback/status/token propagation.

- [ ] **Step 3: Implement one shared checkpoint-and-stop path**

Generate `secrets.token_urlsafe(32)` inside `_spawn_owned_fastllm`, copy `os.environ`, set `FASTLLM_PREFIX_CACHE_CONTROL_TOKEN`, and pass that env to `create_subprocess_exec`. Store the token in a private generation record, not the public lifecycle snapshot.

Add `_checkpoint_owned_fastllm(child, token)` using `httpx.AsyncClient` and `FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT`. Call it in `BackendLifecycleManager.stop()` only after `_drained.wait()` and before `_stop_owned_fastllm`. Catch checkpoint errors, update counters/last error, then always continue the existing owned process-group stop. External/adopted mode must bypass both operations.

Snapshot fields:

```python
{
    "checkpoint_successes": self.checkpoint_successes,
    "checkpoint_failures": self.checkpoint_failures,
    "last_checkpoint_generation": self.last_checkpoint_generation,
    "last_checkpoint_duration_ms": self.last_checkpoint_duration_ms,
    "last_checkpoint_error": self.last_checkpoint_error,
}
```

- [ ] **Step 4: Pass through documented deployment knobs**

In `start_proxy.sh`, document and export `FASTLLM_PREFIX_CACHE_PERSIST`, `FASTLLM_PREFIX_CACHE_PERSIST_KEY`, `FASTLLM_PREFIX_CACHE_DISK_DIR`, and `FASTLLM_PREFIX_CACHE_CHECKPOINT_TIMEOUT`. Do not set persistence on by default and do not require these vars in external mode.

- [ ] **Step 5: Run proxy regressions and commit in v100-perfs**

```bash
../.venv-1cat/bin/python -m unittest -v test_thinking_proxy_fastllm test_thinking_proxy_lifecycle
bash -n scripts/start_proxy.sh
git add thinking_proxy.py test_thinking_proxy_lifecycle.py scripts/start_proxy.sh
git commit -m "feat: checkpoint owned FastLLM before unload"
```

Expected: all existing proxy tests plus the new order/failure/isolation tests pass.

### Task 6: CPU restart smoke and corruption fail-open

**Files:**
- Modify: `test/ops/regressionOps.cpp`
- Modify: `README.md` or existing FastLLM deployment doc only after behavior passes

- [ ] **Step 1: Add a subprocess restart fixture**

Use the focused regression binary in two child modes selected by `FASTLLM_PREFIX_PERSIST_CHILD=record|restore`. `record` creates CPU managers and commits deterministic pages; `restore` starts in a fresh process, queries the same tokens, and exits nonzero unless bytes and persistent-hit metrics match. Parent then truncates `manifest.bin`, runs `restore` again, and requires a normal cache miss rather than crash/output corruption.

- [ ] **Step 2: Run the true cross-process smoke**

```bash
cmake --build build --target regressionOps -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
```

Expected output includes:

```text
persistent prefix archive regression: PASS
persistent paged prefix process-restart regression: PASS
persistent corruption fail-open regression: PASS
```

- [ ] **Step 3: Run existing adjacent regressions**

```bash
FASTLLM_REGRESSION_ONLY=paged_prefix_cache_policy ./build/regressionOps
FASTLLM_REGRESSION_ONLY=qwen35_long_prefill_state ./build/regressionOps
```

Expected: both PASS.

- [ ] **Step 4: Update docs to remove the false process-scoped limitation**

Document that actual GPU/CPU/NVMe prefix state is recoverable only when the three opt-in persistence vars are set and a clean checkpoint succeeds; otherwise behavior remains process-scoped. State the double-generation disk-space requirement and last-valid-generation recovery.

- [ ] **Step 5: Commit cross-process evidence and docs**

```bash
git add test/ops/regressionOps.cpp README.md docs/qwen35_v100_local_stack.md
git commit -m "test: verify prefix cache process recovery"
```

### Task 7: V100 owned-backend end-to-end acceptance

**Files:**
- Create only after observed run: `../v100-perfs/benchmarks/fastllm/results/fastllm_persistent_prefix_epoch_restart.json`
- Modify only after observed run: `../v100-perfs/docs/fastllm_benchmark.md`

- [ ] **Step 1: Build the merged SM70 binary**

```bash
cmake --build build --target apiserver regressionOps -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
```

Expected: build succeeds and focused regression passes on the exact binary to deploy.

- [ ] **Step 2: Launch Thinking Proxy in isolated owned mode**

Use non-production ports, an absolute child command, Turbo3 gate, and a stable key:

```bash
FASTLLM_OWNED=1 \
FASTLLM_BACKEND_URL=http://127.0.0.1:18082 \
FASTLLM_BACKEND_COMMAND='/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/fastllm/build/apiserver --path /run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models/ThinkingCap-Qwen3.6-27B-GGUF/ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf --threads 2 --atype float16 --kv_cache_dtype turbo3 --batch 5 --tokens 65536 --model_name qwen3.6-fastllm --port 18082 --device cuda' \
FASTLLM_QWEN35_TURBO3_KV=1 \
FASTLLM_PREFIX_CACHE_PERSIST=1 \
FASTLLM_PREFIX_CACHE_PERSIST_KEY=thinkingcap-q4km-turbo3-page128-sm70 \
FASTLLM_PREFIX_CACHE_DISK_DIR=/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/cache/fastllm-prefix \
FASTLLM_IDLE_TIMEOUT=10 \
PROXY_HOST=127.0.0.1 PROXY_PORT=18080 AUTH_TOKEN=smoke-token \
./scripts/start_proxy.sh
```

- [ ] **Step 3: Exercise epoch 1 and observe atomic unload**

Send one deterministic page-aligned long-prefix greedy request. Save response content hash, TTFT, usage, backend `/props`, proxy `/health`, generation directory sizes, and GPU free memory before/after idle. Wait for lifecycle `COLD`; require `checkpoint_successes=1`, a nonzero generation, no child listener on 18082, and cold free memory restored.

- [ ] **Step 4: Exercise epoch 2 and prove a real restored hit**

Send the identical request. Require a new child generation, `loaded_generation` equal to epoch-1 checkpoint, increased `prefix_cache_persistent_restore_hits` and read bytes, lower cold-prefill work/TTFT than an empty-cache cold run, and identical greedy output hash. Let it unload again and require a second valid generation.

- [ ] **Step 5: Corrupt latest generation and prove fail-open**

With the child cold, truncate the latest manifest. Send the same request. Require FastLLM to report restore error, serve a normal cache miss, and return the same greedy output hash. No crash, OOM, stuck `DRAINING`, or orphan child is allowed.

- [ ] **Step 6: Record only observed evidence and commit v100-perfs**

The JSON artifact must include exact commit, model/hash, env, request hash, response hashes, epoch timings, persistence metrics, generation sizes, VRAM samples, corruption result, and command lines. Update the Chinese benchmark doc with both positive and negative results, then:

```bash
git add benchmarks/fastllm/results/fastllm_persistent_prefix_epoch_restart.json docs/fastllm_benchmark.md README.md
git commit -m "bench: verify recoverable FastLLM prefix cache"
```

## Final verification

Run only after every task passes independently:

```bash
cmake --build build --target apiserver regressionOps testCheckpointControl -j4
FASTLLM_REGRESSION_ONLY=persistent_prefix_cache ./build/regressionOps
FASTLLM_REGRESSION_ONLY=paged_prefix_cache_policy ./build/regressionOps
FASTLLM_REGRESSION_ONLY=qwen35_long_prefill_state ./build/regressionOps
cd ../v100-perfs
../.venv-1cat/bin/python -m unittest -v test_thinking_proxy_fastllm test_thinking_proxy_lifecycle
bash -n scripts/start_proxy.sh
```

Acceptance requires all focused checks plus the V100 two-epoch/corruption smoke. Do not enable persistence in production profiles until the observed artifact satisfies every gate.
