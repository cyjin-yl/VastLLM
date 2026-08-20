# SM70 Shared-KV Packed Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make packed paged attention faster than production BATCH_GQA for qLen=512 and qLen=2048 on V100, then reuse the proven packed loader in XQA with Turbo3/Turbo4.

**Architecture:** A single launch covers all fixed query tiles. Each CTA owns `(kvHead, queryTile, groupChunk, split)`, loads each Q8/Turbo packed KV fragment once, and reuses it across at most six query/head sets with fused online softmax. Full score matrices are forbidden; split partials retain the existing stable `headDim+2` scratch representation.

**Tech Stack:** C++17, CUDA SM70, FastLLM paged cache, Q8_0_KV, Turbo3/Turbo4 KV, CUDA events, CPU/fp64 reference tests.

**Spec:** `docs/superpowers/specs/2026-08-21-sm70-shared-kv-packed-prefill-design.md`

## Global Constraints

- Target GPU is V100 SM70; no SM75+ instructions or `cp.async`.
- qLen=512 and qLen=2048 each enable only after independent CPU/fp64 comparison and a warmed CUDA-event speedup over BATCH_GQA.
- Preserve qLen=1–4 XQA and qLen=5–32 packed-prefill behavior until their replacements independently pass.
- Never materialize `query × key × head` scores.
- No runtime-sized register arrays; each CTA owns at most six online-softmax states.
- Physical page order, partial final pages, causal tails, Turbo3, and Turbo4 are required cases.
- Every retained/reverted experiment is synchronized to VastLLM issue #1 with exact shape and timing.

---

### Task 1: Long-Query Reference and Baselines

**Files:**
- Modify: `test/ops/turboPagedAttentionTest.cpp:589-767`

**Interfaces:**
- Consumes: existing `BuildPackedFixture`, `DequantQ8RowHost`, `DequantTurbo3RowHost`, `DequantTurbo4RowHost`, `TimeMs`.
- Produces: `ReferenceAttentionRows(...)`, `RunLongCorrectness(...)`, and `--bench-long` baseline output used by every kernel task.

- [ ] **Step 1: Add selected-row fp64 reference coverage**

Add a row selector so qLen=2048 does not perform an impractical full CPU reference:

```cpp
struct ReferenceRow { int token; int head; };

std::vector<double> ReferenceAttentionRows(
    const Fixture &f, const std::vector<float> &qHost, int qoLen,
    const std::vector<ReferenceRow> &rows);
```

For each requested row, independently decode packed K/V bytes, use
`visibleEnd = f.kvLen - qoLen + token`, compute fp64 QK/softmax/PV, and return rows in input order. Required rows are tokens `{0, 1, qoLen / 2, qoLen - 2, qoLen - 1}` for heads `{0, 5, 6, 23}`.

- [ ] **Step 2: Add long-query route assertions that fail before implementation**

Add `--shared-long` correctness cases for Turbo3 and Turbo4 at:

```cpp
{qoLen=64, kvLen=321},
{qoLen=512, kvLen=769},
{qoLen=2048, kvLen=2305}
```

Use non-monotonic physical pages for the small fixture and a partial final page. Assert the new route returns true and selected output rows satisfy the existing `MaxRel() < 5e-3` tolerance without loosening it.

- [ ] **Step 3: Build and verify the new route test fails**

Run:

```bash
cmake --build build-rw --target testTurboPagedAttention -j 8
./build-rw/testTurboPagedAttention --shared-long
```

Expected: FAIL because the shared-long route/counter is not implemented; BATCH_GQA reference output still succeeds.

- [ ] **Step 4: Capture production baselines**

Add `--bench-long [kvLen ...]` with qLen `{512, 2048}`, Turbo3/Turbo4, three warmups, and 20 CUDA-event iterations. It must print median-equivalent per-iteration time and never silently route through an existing fused path. Capture initial BATCH_GQA numbers for kvLen `{2048, 8192}` before kernel work.

- [ ] **Step 5: Commit the fixture**

```bash
git add test/ops/turboPagedAttentionTest.cpp
git commit -m "测试 SM70 shared-KV 长 query 基线"
```

Update issue #1 with baseline table and the expected failing route test.

---

### Task 2: Cooperative Packed Fragment Loader

**Files:**
- Create: `src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-loader.cuh`
- Modify: `test/ops/turboPagedAttentionTest.cpp`

**Interfaces:**
- Produces:

```cpp
template <bool TURBO4>
struct Sm70PackedKvFragment {
    float k[8];
    float v[8];
};

template <bool TURBO4>
__device__ __forceinline__ Sm70PackedKvFragment<TURBO4>
LoadSm70PackedKvFragment(const uint8_t *pagedK, const uint8_t *pagedV,
                         size_t row, int lane,
                         const float *turbo3Lut,
                         const float *turbo4Lut);
```

- Consumes: row-byte helpers and packed block definitions from `fastllm-turboquant-kv.cuh`.

- [ ] **Step 1: Add a failing loader parity kernel test**

Create a tiny test kernel in `turboPagedAttentionTest.cpp` that calls the loader for all 32 lanes, writes 256 decoded K/V values, and compares them with `DequantQ8RowHost` plus the pre-inverse-WHT Turbo3/Turbo4 host representation. Include rows on misaligned 100-byte and 132-byte strides.

- [ ] **Step 2: Run the loader test and verify compile failure**

Run the targeted test binary build. Expected: FAIL because the loader header/API does not exist.

- [ ] **Step 3: Implement aligned-safe packed loads**

The implementation must:

```cpp
// Q8: lane owns 8 dimensions, one scale load plus 8 int8 values.
// Turbo3: uint16 low2 only when naturally aligned; high1 is byte-loaded.
// Turbo4: four explicit byte loads; never reinterpret an unaligned uint32_t*.
```

Return decoded fragments in registers. Do not stage an already single-consumer fragment through shared memory. The executor will share the returned fragment across all fixed sets handled by that warp.

- [ ] **Step 4: Run Turbo3/Turbo4 loader parity and compute-sanitizer**

Run:

```bash
cmake --build build-rw --target testTurboPagedAttention -j 8
./build-rw/testTurboPagedAttention --loader
compute-sanitizer --tool memcheck ./build-rw/testTurboPagedAttention --loader
```

Expected: all numerical checks pass; zero misaligned/illegal accesses.

- [ ] **Step 5: Commit the loader**

```bash
git add src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-loader.cuh test/ops/turboPagedAttentionTest.cpp
git commit -m "加入 SM70 packed KV 协作加载器"
```

Update issue #1 with loader parity and sanitizer evidence.

---

### Task 3: Single-Launch Shared-KV Long Prefill

**Files:**
- Create: `src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-shared.cu`
- Modify: `CMakeLists.txt:331-341`
- Modify: `include/devices/cuda/attention/fastllm-paged-attention-turbo-xqa.cuh`
- Modify: `src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-xqa.cu:775-848`
- Modify: `include/fastllm-kernel-route.h:37-56`
- Modify: `src/fastllm-kernel-route.cpp:57-69`
- Test: `test/ops/turboPagedAttentionTest.cpp`

**Interfaces:**
- Produces:

```cpp
bool FastllmCudaTrySm70PagedTurboSharedPrefill(
    void *qData, fastllm::DataType qType,
    int H, int qoLen, int qDim, int qHeadStride, int qTokenStride,
    const int32_t *pageIndicesGpu, int numPages, int lastPageLen,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int pageLen, int numKvHeads, int headDim,
    void *outData, fastllm::DataType outType,
    int outHeadStride, int outTokenStride, int group, float scale);
```

- Adds route `KERNEL_ROUTE_ATTN_SM70_TURBO_SHARED_PREFILL` at the end of the enum, named `attn.sm70_turbo_shared_prefill`.
- Consumes `LoadSm70PackedKvFragment<TURBO4>()` and existing stable scratch/combine semantics.

- [ ] **Step 1: Add the route declaration and keep the test red**

Declare the public trial entry and route enum/name, but do not add a stub implementation. Build the targeted test. Expected: link failure for the new function, proving the fixture exercises the intended API.

- [ ] **Step 2: Implement only the `Q_TILE=2, GROUP_CHUNK=3` kernel**

Use compile-time constants:

```cpp
template <bool TURBO4, int Q_TILE, int GROUP_CHUNK>
__global__ void Sm70PagedTurboSharedSplit(..., int qoLen, ...);
```

Grid:

```cpp
const int groupChunks = group / GROUP_CHUNK;
dim3 grid(numKvHeads * groupChunks,
          (qoLen + Q_TILE - 1) / Q_TILE,
          splits);
```

Each warp loads one logical KV row, calls the packed fragment loader once, and reuses `k[8]/v[8]` across `Q_TILE * GROUP_CHUNK` sets. Query indices are global; invalid tail sets remain neutral. Scratch uses `(head * qoLen + globalToken) * splits + split`.

- [ ] **Step 3: Implement split combine and trial eligibility**

Reuse the existing delayed inverse-WHT rule. Eligibility requires SM70, FP16 Q/output, Q8 K, Turbo3/Turbo4 V, headDim 256, group 6, power-of-two page length, and qLen at least 64. On any failure return false without writing output. Add `FASTLLM_CUDA_SM70_TURBO_SHARED_PREFILL=0` as an independent kill switch.

- [ ] **Step 4: Make correctness green before benchmarking**

Run:

```bash
cmake --build build-rw --target testTurboPagedAttention -j 8
./build-rw/testTurboPagedAttention --shared-long
```

Expected: Turbo3/Turbo4 qLen 64/512/2048 selected-row comparisons pass, including tail and causal boundaries.

- [ ] **Step 5: Benchmark the first candidate**

Run `--bench-long 2048 8192`. Record BATCH_GQA and `2×3` times for both packed types and qLen values. If either target is slower, inspect register/local-memory and launch evidence before adding another candidate; do not route it yet.

- [ ] **Step 6: Commit the first correct candidate**

```bash
git add CMakeLists.txt include/devices/cuda/attention/fastllm-paged-attention-turbo-xqa.cuh include/fastllm-kernel-route.h src/fastllm-kernel-route.cpp src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-shared.cu src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-xqa.cu test/ops/turboPagedAttentionTest.cpp
git commit -m "实现 SM70 single-launch shared-KV prefill"
```

Update issue #1 with correctness and timings, including a retained/rejected decision.

---

### Task 4: Tile Selection and Conservative Routing

**Files:**
- Modify: `src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-shared.cu`
- Modify: `src/devices/cuda/attention/paged/fastllm-paged-attention-native.cu`
- Test: `test/ops/turboPagedAttentionTest.cpp`

**Interfaces:**
- Adds compile-time candidates `<1,6>`, `<3,2>`, and `<4,1>`.
- Adds experiment override `FASTLLM_CUDA_SM70_TURBO_SHARED_TILE=1x6|2x3|3x2|4x1`; unset uses measured per-shape winner.

- [ ] **Step 1: Add failing dispatch/counter tests**

For qLen=512 and 2048, assert default Data-layer dispatch output is bitwise equal to direct shared route output only for shapes selected as wins. Assert `/props` route census increments `attn.sm70_turbo_shared_prefill`. A disabled kill switch must leave the shared counter unchanged and hit BATCH_GQA fallback.

- [ ] **Step 2: Add one candidate at a time**

For each of `1×6`, `3×2`, `4×1`: instantiate, run independent correctness, then benchmark. Revert an instantiation if it is both slower than `2×3` and offers no occupancy/local-memory explanation useful to another target.

- [ ] **Step 3: Select qLen=512 and qLen=2048 winners**

Encode only measured choices. If a target has no candidate above 1.0× BATCH_GQA, leave that target on fallback and continue kernel optimization; do not weaken the gate.

- [ ] **Step 4: Run dispatch and route tests**

Run targeted correctness plus `--bench-long`. Expected: selected shapes hit the shared route; unselected/disabled shapes preserve BATCH_GQA output and counters.

- [ ] **Step 5: Commit routing**

```bash
git add src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-shared.cu src/devices/cuda/attention/paged/fastllm-paged-attention-native.cu test/ops/turboPagedAttentionTest.cpp
git commit -m "按实测路由 shared-KV 长 query tile"
```

Update issue #1 with all four candidate tables.

---

### Task 5: XQA Turbo3/Turbo4 Loader Integration

**Files:**
- Modify: `src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-xqa.cu`
- Modify: `src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-loader.cuh`
- Test: `test/ops/turboPagedAttentionTest.cpp`

**Interfaces:**
- Existing public `FastllmCudaTrySm70PagedTurboXqa(...)` remains unchanged.
- XQA accepts both `TURBO3_KV` and `TURBO4_KV` and invokes the common packed fragment loader.

- [ ] **Step 1: Add failing Turbo4 XQA tests**

Extend unordered/partial-page fixtures so qLen `{1,2,3,4}` with Turbo4 must return true, match independent fp64, and hit `attn.sm70_turbo_xqa`.

- [ ] **Step 2: Verify red**

Run the targeted binary. Expected: Turbo4 XQA returns false at the existing dtype gate.

- [ ] **Step 3: Template XQA value decoding through the loader**

Replace duplicated Q8/Turbo3 fragment decode with `LoadSm70PackedKvFragment<TURBO4>()`. Keep the existing `1×6`, `2×3`, `3×2`, `4×1` set layouts and delayed inverse-WHT. Do not add shared-memory traffic where the warp already loads a KV row once and all its sets consume registers.

- [ ] **Step 4: Verify XQA correctness and performance**

Run qLen 1–4 Turbo3/Turbo4 fp64 tests and benchmarks. Existing Turbo3 must not regress materially; Turbo4 only routes when faster than its prior fallback.

- [ ] **Step 5: Commit XQA integration**

```bash
git add src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-loader.cuh src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-xqa.cu test/ops/turboPagedAttentionTest.cpp
git commit -m "复用 packed loader 并加入 Turbo4 XQA"
```

Update issue #1 with qLen 1–4 parity and timings.

---

### Task 6: Final Verification and Production Smoke

**Files:**
- Modify: `v100-perfs/docs/EXPERIENCE.md`
- Modify only if route exposure requires it: `example/apiserver/apiserver.cpp`

**Interfaces:**
- Produces final `/props` route evidence and issue #1 completion table.

- [ ] **Step 1: Run targeted correctness suites**

```bash
cmake --build build-rw --target testTurboPagedAttention apiserver regressionOps -j 8
./build-rw/testTurboPagedAttention
./build-rw/testTurboPagedAttention --shared-long
FASTLLM_REGRESSION_ONLY=none ./build-rw/regressionOps
```

Expected: all applicable tests pass; skips remain hardware-feature skips only.

- [ ] **Step 2: Run memory safety**

```bash
compute-sanitizer --tool memcheck ./build-rw/testTurboPagedAttention --shared-long
```

Expected: zero CUDA memory errors for Turbo3/Turbo4, unordered pages, tails, qLen512/2048.

- [ ] **Step 3: Run exclusive performance proof**

With production idle, run warmed `--bench-long` at the accepted kvLen/qLen matrix. Both qLen=512 and qLen=2048 must show >1.0× over BATCH_GQA for the route to be called complete. Continue candidate/kernel optimization if either misses.

- [ ] **Step 4: Deploy and smoke actual production routing**

Restart `thinking-proxy-prod-iq4xs`, wait for READY, send bounded requests that exercise selected long-query chunks, and inspect `/props`. Verify shared route counts increase, short XQA remains active, output succeeds, and fallback handles unselected shapes.

- [ ] **Step 5: Document and synchronize evidence**

Append exact correctness, CUDA-event timing, route counts, and retained/reverted candidates to `v100-perfs/docs/EXPERIENCE.md` and VastLLM issue #1. Do not describe qLen512/2048 as optimized unless both performance gates passed.

- [ ] **Step 6: Commit final evidence**

```bash
git add example/apiserver/apiserver.cpp
git commit -m "验证并部署 shared-KV packed attention"
```

Omit unchanged paths from `git add`; the documentation may live outside the FastLLM subrepository and is recorded separately.
