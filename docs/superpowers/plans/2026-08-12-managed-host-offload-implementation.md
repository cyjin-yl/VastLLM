# Managed Host Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the IQ6 Qwen3.6 256K Turbo3-KV MTP2 model topology alive across memory-tier suspend, release GPU memory to the cold baseline, and resume from a model-weight-plus-prefix-KV shared 12 GiB host budget with fail-open disk recovery and stable vision output.

**Architecture:** Extend the existing `VRAM → RAM → disk` cache hierarchy rather than adding a parallel offload path. `HostOffloadManager` is the RAM-tier implementation: it moves selected final `Data` tensors GPU→CPU, records deterministic reload recipes for uncached tensors, releases all remaining CUDA state, restores cached tensors CPU→GPU, materializes misses from disk-tier GGUF recipes, and rebuilds only non-serializable Qwen runtime objects. One process-wide `HostCacheBudget` covers both model tensors and prefix-cache CPU payloads; SSD generations remain the authoritative disk tier.

**Tech Stack:** C++17, FastLLM `Data`/`WeightMap`, GGUF loader, CUDA backend, zstd prefix-cache tier, FastLLM API server, CMake unit/regression executables, Python proxy smoke scripts.

---

## File map

- Create `fastllm/include/host_offload.h`: budget, source identity, materialization recipe, metrics, and manager public interfaces.
- Create `fastllm/src/host_offload.cpp`: dynamic `MemAvailable` accounting, candidate ordering, tensor migration, checksums, rollback, and host/disk fallback decisions.
- Modify `fastllm/include/fastllm.h` and `fastllm/src/fastllm.cpp`: narrowly scoped `Data` storage migration helpers and shared prefix-tier budget hooks.
- Modify `fastllm/include/models/basellm.h` and `fastllm/src/models/basellm.cpp`: model-level offload hooks and materialization-plan ownership.
- Modify `fastllm/src/model.cpp` and `fastllm/third_party/gguf/gguf.{h,cpp}`: record source identity/direct-transform recipes during initial load and reload a dependency-closed subset into existing final `Data` slots.
- Modify `fastllm/include/models/qwen3_5.h` and `fastllm/src/models/qwen3_5.cpp`: destroy/rebuild Qwen3.5/3.6 CUDA graph, linear-slot, MTP, and paged-cache runtime state without destroying model topology.
- Modify `fastllm/src/prefixcache_persistence.cpp`: keep SSD generation identity intact while CPU payloads participate in the shared host budget.
- Modify `fastllm/example/apiserver/apiserver.cpp`: explicit lifecycle states, host-first memory suspend, disk fallback, metrics, and response fields.
- Modify `fastllm/CMakeLists.txt`: compile the new source and tests.
- Create `fastllm/test/host_offload/hostCacheBudgetTest.cpp`, `materializationPlanTest.cpp`, and `hostOffloadLifecycleTest.cpp`.
- Modify `v100-perfs/runtime/fastllm-native-profiles/q6-ablit-262k-mtp2.env`: opt in to the final shared-budget feature after all gates pass.
- Modify `v100-perfs/docs/EXPERIENCE.md`: record only observed IQ6 256K vision and suspend/resume results.

### Task 1: Shared host-cache budget

**Files:**
- Create: `include/host_offload.h`
- Create: `src/host_offload.cpp`
- Create: `test/host_offload/hostCacheBudgetTest.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing budget tests**

Create a standalone test executable that exercises real budget logic without CUDA. Cover these contracts:

```cpp
constexpr uint64_t GiB = UINT64_C(1024) * 1024 * 1024;
HostCacheBudget budget(/*hardMax=*/12 * GiB, /*minFree=*/12 * GiB,
    [] { return 20 * GiB; });
auto weight = budget.TryReserve(6 * GiB, HostCacheClass::MODEL_WEIGHT);
auto prefix = budget.TryReserve(3 * GiB, HostCacheClass::PREFIX_KV);
Expect(weight && prefix, "reservations within dynamic budget succeed");
Expect(budget.ResidentBytes() == 9 * GiB, "resident bytes are exact");
Expect(!budget.TryReserve(1, HostCacheClass::PREFIX_KV),
       "MemAvailable reserve is a hard ceiling");
weight.Reset();
Expect(budget.ResidentBytes() == 3 * GiB, "RAII release is exact");
```

Also test: hard max 0; `MemAvailable <= minFree`; overflow-safe subtraction; model reservation evicts registered prefix reservations first; a failed allocation callback leaves accounting unchanged; peak bytes never decrease.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cmake -S . -B build -DUNIT_TEST=ON
cmake --build build --target testHostCacheBudget -j8
./build/testHostCacheBudget
```

Expected: compile failure because `host_offload.h` and `HostCacheBudget` do not exist.

- [ ] **Step 3: Implement the minimal budget core**

Define:

```cpp
enum class HostCacheClass { DERIVED_WEIGHT, MODEL_WEIGHT, PREFIX_KV };
class HostCacheReservation {
public:
    HostCacheReservation() = default;
    HostCacheReservation(HostCacheReservation&&) noexcept;
    HostCacheReservation &operator=(HostCacheReservation&&) noexcept;
    ~HostCacheReservation();
    explicit operator bool() const;
    uint64_t Bytes() const;
    void Reset();
};
class HostCacheBudget {
public:
    using AvailableBytesReader = std::function<uint64_t()>;
    HostCacheBudget(uint64_t hardMax, uint64_t minFree,
                    AvailableBytesReader reader = ReadLinuxMemAvailableBytes);
    HostCacheReservation TryReserve(uint64_t bytes, HostCacheClass kind);
    void RegisterPrefixEvictor(std::function<uint64_t(uint64_t)> evictor);
    uint64_t CurrentLimitBytes() const;
    uint64_t ResidentBytes() const;
    uint64_t PeakResidentBytes() const;
    uint64_t Bytes(HostCacheClass kind) const;
};
```

`TryReserve` must read `MemAvailable` on every call, compute `min(hardMax, available - minFree)` with saturating subtraction, evict prefix bytes before denying a weight reservation, and update accounting only after capacity is available. Do not allocate a 12 GiB arena.

- [ ] **Step 4: Run GREEN and the CPU baseline tests**

Run:

```bash
cmake --build build --target testHostCacheBudget -j8
./build/testHostCacheBudget
./build/testCheckpointControl
```

Expected: both exit 0 and the new test prints `host cache budget: PASS`.

### Task 2: Exact `Data` storage migration

**Files:**
- Modify: `include/fastllm.h`
- Modify: `src/fastllm.cpp`
- Create: `test/host_offload/dataStorageMigrationTest.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing migration tests**

Add a CUDA-gated regression executable. Allocate deterministic bytes in a `Data`, move it CPU→CUDA, call the new offload operation, assert the CUDA pointer is null and the host checksum matches, restore it, copy back, and assert byte equality. Add fault-injection cases for D2H and H2D; after each injected failure assert the source copy remains authoritative and no double-free occurs.

Required API:

```cpp
struct DataOffloadRecord {
    DataDevice originalDevice;
    int originalDeviceId;
    uint64_t bytes;
    uint64_t checksum;
};
bool Data::MoveCudaStorageToHost(DataOffloadRecord &record, std::string *error);
bool Data::RestoreCudaStorageFromHost(const DataOffloadRecord &record,
                                      std::string *error);
bool Data::ReleaseCudaStorageWithoutHostCopy(std::string *error);
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build --target testDataStorageMigration -j8
./build/testDataStorageMigration
```

Expected: compile failure on the missing methods.

- [ ] **Step 3: Implement transactional migration**

Use existing `Data::ToDevice` allocation/copy paths, but do not expose a half-transition. Compute a stable 64-bit checksum over exactly `expansionBytes`; only free CUDA after D2H plus checksum succeeds; only release non-mmap host storage after H2D plus verification succeeds. `ReleaseCudaStorageWithoutHostCopy` must preserve dtype, dims, strides, quant metadata, expansion size, and tensor name while setting storage pointers to null. It must reject borrowed CUDA memory and tensor-parallel layouts rather than guessing ownership.

- [ ] **Step 4: Run GREEN under compute-sanitizer-compatible behavior**

Run:

```bash
cmake --build build --target testDataStorageMigration -j8
./build/testDataStorageMigration
```

Expected: exit 0, exact round-trip hash, and no CUDA allocation remaining after object destruction.

### Task 3: Persist GGUF materialization recipes

**Files:**
- Modify: `include/host_offload.h`
- Modify: `include/models/basellm.h`
- Modify: `src/models/basellm.cpp`
- Modify: `src/model.cpp`
- Modify: `third_party/gguf/gguf.h`
- Modify: `third_party/gguf/gguf.cpp`
- Create: `test/host_offload/materializationPlanTest.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing source-identity and dependency tests**

Build a tiny temporary GGUF fixture with two direct tensors and one merge output. Assert that initial load records canonical path, size, nanosecond mtime, tensor-table fingerprint, source offset/type/transform parameters, ordered merge inputs, and final metadata. Assert `BuildReloadClosure({"merged.weight"})` returns the two sources before the merge output. Mutating file size, mtime, or tensor-table bytes must invalidate the whole plan.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build --target testMaterializationPlan -j8
./build/testMaterializationPlan
```

Expected: compile failure because materialization-plan types and reload closure are absent.

- [ ] **Step 3: Add immutable recipe types**

Define immutable records:

```cpp
enum class WeightRecipeKind { GGUF_DIRECT, GGUF_TRANSFORM, MERGE, MODEL_GENERATED };
struct WeightSourceIdentity {
    std::string canonicalPath;
    uint64_t size;
    int64_t mtimeNanoseconds;
    uint64_t tensorTableFingerprint;
};
struct WeightMaterializationRecipe {
    uint64_t id;
    WeightRecipeKind kind;
    std::string outputName;
    DataType outputType;
    std::vector<int> outputDims;
    std::vector<uint64_t> inputIds;
    std::string sourcePath;
    uint64_t sourceOffset;
    uint64_t sourceBytes;
    int ggmlType;
    GGUFWeightReplaceRule::GGUFWeightReplaceType replaceType;
    int untileNumKHeads;
    int untileNumVHeads;
    int untileVRowStart;
    bool untileComposeNegLog;
    std::string mergeType;
};
```

Store recipes and source identities in `basellm`. Record them while `ReadGGUFTask` and `weightMergeRules` are already fully resolved; never infer recipes later from names.

- [ ] **Step 4: Implement targeted materialization**

Add:

```cpp
bool ReloadGGUFWeightSubset(basellm *model,
                            const std::set<std::string> &finalWeightNames,
                            MaterializationMetrics *metrics,
                            std::string *error);
```

Validate every source identity before reading any payload. Build a dependency closure, read sources in plan order with `WeightImportGGUFTensor`, apply the existing merge primitive, verify final dtype/dims/bytes/checksum against the first-load manifest, move each final tensor to its configured CUDA device, and release temporary source tensors immediately. Never create a second model and never hold the full GGUF in anonymous RAM.

- [ ] **Step 5: Run GREEN and hash-equivalence tests**

Run:

```bash
cmake --build build --target testMaterializationPlan -j8
./build/testMaterializationPlan
```

Expected: direct, transform, merge, and source-identity cases pass; peak temporary bytes stay below the fixture's largest dependency group.

### Task 4: Per-model host offload manager

**Files:**
- Modify: `include/host_offload.h`
- Modify: `src/host_offload.cpp`
- Modify: `include/models/basellm.h`
- Modify: `src/models/basellm.cpp`
- Create: `test/host_offload/hostOffloadLifecycleTest.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing lifecycle tests**

Use a fake model with direct, derived, and uncached final tensors. Cover 12GiB-equivalent full fit, partial fit, below-must-cache, and zero-budget paths using small byte counts. Assert candidate order `must-cache > derived > ordinary > prefix`; each `SUSPENDED_HOST` tensor has exactly one authoritative host or source copy; each `READY` tensor has a GPU copy and manager-owned host weight bytes are zero; checksum/allocation/materialization failures request a full disk fallback.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build --target testHostOffloadLifecycle -j8
./build/testHostOffloadLifecycle
```

Expected: compile failure because `HostOffloadManager::Suspend` and `Resume` are absent.

- [ ] **Step 3: Implement candidate classification and suspend transaction**

Add `basellm::ClassifyHostOffloadWeight(name)` with a boring default: merge/transform outputs are derived, direct GGUF outputs are ordinary, `MODEL_GENERATED` without an independent recipe is must-cache, non-weight runtime storage is excluded. `HostOffloadManager::Suspend` must preflight every final tensor before freeing anything, reject TP/multi-CUDA, evict prefix RAM first, reserve per tensor, D2H chosen candidates, release source-rebuildable misses without copying, and rollback cached tensors to GPU if a mid-suspend failure occurs. If rollback fails, return `DISK_FALLBACK_REQUIRED`.

- [ ] **Step 4: Implement host resume and disk fallback result**

Validate generation, source identity, metadata, and checksums before H2D. Restore cached tensors in dependency order and release each reservation after successful H2D. Reload misses through `ReloadGGUFWeightSubset`. Return a structured result with actual tier, timings, cached/rebuilt bytes, hit ratio, and fallback reason. Do not silently leave a partial model READY.

- [ ] **Step 5: Run GREEN**

Run:

```bash
cmake --build build --target testHostOffloadLifecycle -j8
./build/testHostOffloadLifecycle
```

Expected: all budget paths and rollback/fallback cases pass.

### Task 5: Qwen3.5/3.6 runtime teardown and rebuild

**Files:**
- Modify: `include/models/basellm.h`
- Modify: `include/models/qwen3_5.h`
- Modify: `src/models/qwen3_5.cpp`
- Extend: `test/host_offload/hostOffloadLifecycleTest.cpp`

- [ ] **Step 1: Write failing hook-order tests**

Add a probe subclass that records `OnHostSuspendBegin`, `OnHostSuspendComplete`, `OnHostResumeWeightsReady`, and `OnHostResumeComplete`. Assert the scheduler is stopped before CUDA runtime destruction, weights restore before runtime rebuild, and requests cannot execute between these transitions.

- [ ] **Step 2: Run and verify RED**

Run `./build/testHostOffloadLifecycle` and expect failure because the hooks are missing.

- [ ] **Step 3: Add default model hooks and Qwen implementation**

Add virtual no-op hooks to `basellm`. In Qwen3.5/3.6 suspend: stop TP workers, erase CUDA graph decode states, linear slot pools, temporary MTP draft lm-head replicas, linear prefix snapshots, paged managers, and model-owned CUDA scratch; do not destroy `weight`, tokenizer, source recipes, or model configuration. In resume: re-create device-specific MTP state from restored weights, run only required warmup/runtime preparation, and reset graph/pool flags so existing preparation functions execute once.

- [ ] **Step 4: Run GREEN plus a short direct model inference test**

Run the lifecycle test, then start the test apiserver profile with a small token limit and exercise one text request before and after host suspend. Expected: identical greedy bytes and no stale CUDA pointer diagnostics.

### Task 6: Shared prefix-KV RAM budget

**Files:**
- Modify: `include/fastllm.h`
- Modify: `src/fastllm.cpp`
- Modify: `src/prefixcache_persistence.cpp`
- Extend: `test/host_offload/hostCacheBudgetTest.cpp`
- Extend: `test/host_offload/hostOffloadLifecycleTest.cpp`

- [ ] **Step 1: Write failing shared-accounting tests**

Create real `PagedPrefixCacheTierPayload` objects against the test budget. Assert payload creation reserves `PREFIX_KV`, destruction releases it, a later model-weight reservation evicts prefix payloads until it fits, and the corresponding SSD reference/generation remains materializable. Assert the legacy `FASTLLM_PREFIX_CACHE_CPU_MAX_BYTES` cannot raise total host residency above the shared cap when shared budgeting is enabled.

- [ ] **Step 2: Run and verify RED**

Run both host-cache tests and expect the prefix accounting assertions to fail.

- [ ] **Step 3: Integrate the budget at payload ownership boundaries**

Attach a movable `HostCacheReservation` to each CPU-tier payload. Register a prefix evictor that drops least valuable RAM payloads while preserving `tierDisk`. When shared budgeting is disabled, retain existing CPU max-byte behavior. When enabled, the effective prefix limit is the unreserved shared remainder, never `shared + legacy`.

- [ ] **Step 4: Run GREEN and persistent-prefix regression**

Run:

```bash
./build/testHostCacheBudget
./build/testHostOffloadLifecycle
```

Then run the existing persistent-prefix focused executable or its narrow regression command. Expected: prefix RAM eviction followed by SSD materialization restores identical page checksums.

### Task 7: API server lifecycle state machine and observability

**Files:**
- Modify: `example/apiserver/apiserver.cpp`
- Create: `example/apiserver/host_offload_control.h`
- Create: `test/apiserver/hostOffloadControlTest.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing control-state tests**

Extract a pure transition evaluator following the established `checkpoint_control.h` pattern. Test `READY → SUSPENDING → SUSPENDED_HOST|SUSPENDED_DISK`, `SUSPENDED_* → RESUMING → READY|ERROR`, checkpoint/suspend/resume mutual exclusion, idle requirement, duplicate suspend/resume 409s, and request 503s carrying `backend_reloading` during resume.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build --target testHostOffloadControl -j8
./build/testHostOffloadControl
```

Expected: compile failure because the transition evaluator does not exist.

- [ ] **Step 3: Wire host-first suspend and fallback**

Extend the existing tier lifecycle instead of adding a second suspend path. For `tier=memory`, checkpoint prefix SSD state, call the Qwen suspend hook, call the RAM-tier `HostOffloadManager::Suspend`, clear non-semantic paged/idle CUDA pools, and keep `model` alive on `SUSPENDED_HOST`. On zero budget, unsupported TP, preflight failure, or rollback failure, descend to the disk tier by destroying the model and entering `SUSPENDED_DISK`. `tier=disk` bypasses RAM caching and uses that same disk-tier transition directly.

- [ ] **Step 4: Wire host resume and full disk recovery**

For `SUSPENDED_HOST`, call manager resume and Qwen runtime rebuild. On any validation/copy/materialization/warmup failure, destroy the partial model, clear host reservations, recreate through `CreateLLMModelFromFile`, apply token/KV settings, and restore persistent prefix state. Only publish `READY` after the final model is complete.

- [ ] **Step 5: Add bounded metrics**

Add `/props` and suspend/resume JSON fields: `actual_tier`, dynamic budget, resident/peak bytes, derived/ordinary/prefix bytes, source-evicted bytes, D2H/H2D/materialize/warmup milliseconds, hit ratio, fallback count/reason, source mismatch count, checksum failure count. Emit one summary line per transition; never print one line per tensor unless the explicit top-N debug flag is set.

- [ ] **Step 6: Run GREEN and API regressions**

Run:

```bash
cmake --build build --target testHostOffloadControl testCheckpointControl apiserver -j8
./build/testHostOffloadControl
./build/testCheckpointControl
```

Expected: both tests exit 0 and `apiserver` links.

### Task 8: Direct-process suspend/resume verification

**Files:**
- No production source changes unless a reproduced defect requires another red-green cycle.

- [ ] **Step 1: Establish disk baseline**

Launch directly with `setsid` using the IQ6 profile but a separate port/log/cache key. Do not use hub or tmux. Poll `/health` in a short detached process. Record bounded metrics: start-to-READY, warmup stages, GPU used, host RSS, prefix generation, and deterministic greedy output hash.

- [ ] **Step 2: Verify 12GiB host suspend**

POST `/admin/suspend` with `{"tier":"memory"}`. Verify actual tier, GPU used near the observed cold baseline, manager resident bytes `<= min(12GiB, MemAvailable-12GiB)`, total model-plus-prefix shared accounting equals observed manager accounting, and no model GGUF appears under `/dev/shm`.

- [ ] **Step 3: Verify host resume**

POST `/admin/resume`, poll READY, and run the same deterministic text request. Verify output hash, source identity, prefix generation, host weight bytes 0 at READY, and bounded logs contain no CUDA OOM, stale pointer, packed KV copy failure, or fallback.

- [ ] **Step 4: Verify partial and disk paths**

Repeat with 8GiB, 1GiB, and 0GiB caps. Verify smaller budgets cache fewer bytes, all outputs match, and 0GiB reports `SUSPENDED_DISK`/full disk reload without OOM. Inject one checksum mismatch and one source-identity mismatch in the isolated test instance; verify automatic disk recovery.

### Task 9: IQ6 256K vision stability and production cutover

**Files:**
- Modify: `v100-perfs/runtime/fastllm-native-profiles/q6-ablit-262k-mtp2.env`
- Modify: `v100-perfs/docs/EXPERIENCE.md`

- [ ] **Step 1: Verify one image request at 256K configuration**

Use `v100-perfs/benchmarks/fastllm/requests/thinkingcap_vision_red_pixel.json` against the isolated direct-process server. Record HTTP status, prompt/completion tokens, wall time, GPU peak, output hash/text, and bounded error patterns.

- [ ] **Step 2: Verify repeated images across suspend/resume**

Run at least: image A twice, memory suspend/resume, image A again, image B or a changed payload, then a short text request. Require all requests 200, READY preserved, no monotonic GPU leak beyond allocator reuse, no invalid image placeholder/tokenization result, and no CUDA allocation/copy/paged-cache error.

- [ ] **Step 3: Enable production profile**

Set:

```text
FASTLLM_HOST_SUSPEND_CACHE=1
FASTLLM_HOST_SUSPEND_CACHE_MAX_BYTES=12884901888
FASTLLM_HOST_SUSPEND_MIN_FREE_BYTES=12884901888
FASTLLM_HOST_SUSPEND_STAGING_BYTES=268435456
FASTLLM_HOST_SUSPEND_PREFIX_MODE=opportunistic
```

Keep `--tokens 262144`, Turbo3 KV, MTP2, current IQ6 abliterated model, and existing persistent-prefix directories unchanged.

- [ ] **Step 4: Direct-process production replacement**

Stop only verified existing production PIDs, launch the profile directly with `setsid`, wait for READY through bounded polling, run one text and one image request, and confirm proxy health. Do not use hub/tmux and do not read the full live log.

- [ ] **Step 5: Record observed results**

Append exact build/profile hashes, READY time, suspend/resume time, GPU cold/ready/vision peak, host resident peak, prefix generation, image sequence, and test outputs to `EXPERIENCE.md`. Report only the exercised IQ6 256K configuration.
