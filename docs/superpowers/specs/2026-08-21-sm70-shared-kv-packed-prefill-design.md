# SM70 shared-KV packed paged-attention design

## Objective

Accelerate Qwen3.5/Qwen3.8 paged attention on V100 (SM70) for `qoLen=512` and `qoLen=2048` while retaining packed `Q8_0_KV` keys and `TURBO3_KV`/`TURBO4_KV` values.

A new route is acceptable only when it:

1. passes an independent CPU/fp64 numerical reference;
2. handles the existing paged-cache layouts and causal boundary exactly;
3. is faster than the production BATCH_GQA fallback at the same shape;
4. does not regress the existing qLen=1–4 XQA or qLen=5–32 packed paths.

A shape that does not beat BATCH_GQA remains on BATCH_GQA. Correctness alone does not enable routing.

## Existing limitation

The current long-prefill grid is `(queryHead, queryToken, split)`. Every CTA owns one query head and one query token, so the six GQA query heads and neighboring query tokens repeatedly fetch the same paged K/V row.

The rejected 8-query experiment expanded the fixed-Q kernel to 48 resident sets and launched 64 query tiles from the host. It measured 40.03 ms at `kvLen=1024, qoLen=512`, versus 1.63 ms for BATCH_GQA. The failure came from register pressure/spilling plus repeated launches, not from the shared-KV objective.

## Chosen architecture

Use a fixed, small set count and cover all query tiles in one kernel launch.

The logical grid is:

```text
(kvHead, queryTile, groupChunk, split)
```

CUDA's three grid dimensions will flatten `(kvHead, groupChunk)` into one dimension. `queryTile` and `split` occupy the remaining dimensions. No host loop launches one kernel per query tile.

Benchmark these compile-time tile candidates:

| Query tile | GQA head chunk | Sets per CTA |
|---:|---:|---:|
| 1 | 6 | 6 |
| 2 | 3 | 6 |
| 3 | 2 | 6 |
| 4 | 1 | 4 |

No candidate may allocate an array sized by runtime `qoLen`. Each CTA keeps at most six online-softmax states. The router may select different winning tiles for qLen=512 and qLen=2048.

## Packed loader

A dedicated device-side packed-row loader handles physical page lookup and byte layout without changing cache representation.

For each logical key position and KV head:

1. resolve `pageIndices[logicalPage]` and the in-page offset;
2. cooperatively load the 272-byte Q8 key row with aligned/coalesced transactions;
3. cooperatively load the 100-byte Turbo3 or 132-byte Turbo4 value row without misaligned vector dereferences;
4. expose decoded key fragments and packed/dequantized value fragments to every set in the CTA;
5. preserve Turbo3/Turbo4 template specialization so only value decoding differs.

The first implementation may stage packed bytes, decoded fragments, or both. Selection is empirical: retain the variant with lower runtime and no bank-conflict or occupancy regression. Global K/V loads must not be repeated per set.

## Shared-KV executor

Each CTA processes one fixed query/head tile across one KV split:

1. load and scale the fixed query tile;
2. iterate visible KV rows in blocks;
3. load each packed K/V row once through the packed loader;
4. reuse that row across all sets in the tile;
5. calculate QK and update fused online-softmax `(m, l, accumulator)` state;
6. write one partial state per `(queryHead, queryToken, split)` to the existing stable scratch allocation;
7. run the split-combine kernel and perform the single delayed inverse-WHT before writing FP16 output.

Causality uses the global query index:

```text
visibleEnd = kvLen - qoLen + globalQuery + 1
```

A tail query tile masks missing query rows. Empty split ranges emit the existing neutral partial state and must not read K/V.

The long packed-prefill executor is implemented first because qLen=512/2048 is the acceptance target. After it passes its gates, the existing qLen=1–4 XQA route adopts the same packed loader and route instrumentation. XQA keeps warp-register sharing when a KV row is consumed by one warp and all of its query/head sets; forcing an extra shared-memory round trip where there is no duplicate global load is prohibited. Long query tiles use CTA shared memory because multiple warps consume the same row. Both are one shared-KV architecture: one global packed-row load per consuming tile, with the narrowest on-chip sharing scope that reaches every consumer. Turbo4 support is added to XQA during this integration and must pass the same independent reference and beat its prior fallback.

## Scratch and memory constraints

Do not materialize a full `query × key × head` score matrix. At 262K context this would erase the memory benefit and can exceed device capacity.

Continue using split partials of `headDim + 2` floats. Scratch addressing uses the global query index and stays stable for CUDA graph compatibility. Growth retains old allocations under the current allocator rule; implementation must compute capacity for qLen=2048 without integer overflow.

Shared memory and registers are compile-time bounded by the selected tile. Build output and profiler evidence must show no order-of-magnitude spill regression. A candidate whose occupancy or local-memory traffic explains a slowdown is rejected rather than special-cased.

## Routing

The production route remains conservative:

- qLen=1–4: existing `attn.sm70_turbo_xqa`;
- qLen=5–32: existing `attn.sm70_turbo_prefill`;
- qLen=64/512/2048: new shared-KV route only for independently validated, benchmarked shapes;
- every unsupported or slower shape: BATCH_GQA.

The route name must be separately observable in `/props`; counters distinguish Turbo3, Turbo4, qLen=512, qLen=2048, and fallback. An environment switch disables the new route without disabling the proven short paths.

## Independent numerical reference

The test reference must not call either GPU attention path. It independently:

1. decodes Q8 key blocks;
2. decodes Turbo3 and Turbo4 value blocks;
3. applies paged logical-to-physical mapping;
4. computes causal scaled attention in fp64;
5. performs inverse-WHT independently;
6. converts only the final comparison output.

Required cases include:

- Turbo3 and Turbo4;
- qLen 64, 512, and 2048;
- full and partial final pages;
- multiple pages and non-monotonic physical page indices;
- kvLen equal to qLen and kvLen greater than qLen;
- first, middle, and last query tiles;
- tail tiles for candidates where `qoLen % Q_TILE != 0`;
- split counts of one and greater than one.

Report `maxAbs`, RMS error, reference maximum, and `maxAbs/refMax`. Compare against both fp64 and BATCH_GQA. Tolerances follow the existing Turbo paged-attention fixture and may not be loosened solely to admit the new kernel.

## Performance verification

Measure warmed CUDA event time, not process wall time. For each packed type and qLen 512/2048:

1. benchmark BATCH_GQA;
2. benchmark each valid tile candidate;
3. repeat across representative kvLen values, including the production chunked-prefill shape;
4. record median and dispersion after warm-up;
5. enable a route only where its selected candidate is consistently faster than BATCH_GQA.

The first optimization target is eliminating redundant packed K/V global loads. Kernel launch count, register use, local-memory traffic, shared-memory throughput, and occupancy are supporting evidence, not substitutes for elapsed-time improvement.

## Issue synchronization

VastLLM issue #1 is the progress ledger. Update it after:

1. design approval and baseline capture;
2. each tile candidate's correctness result;
3. each material performance result, including rejected experiments;
4. route selection;
5. production smoke verification.

Each update includes commit, exact shape, packed type, baseline, candidate time, speedup, correctness result, and whether code was retained or reverted.

## Delivery sequence

1. Extend the independent fixture to qLen 64/512/2048 and capture BATCH_GQA baselines.
2. Extract a safe cooperative packed-row loader without changing existing routes.
3. Implement the smallest single-launch shared-KV candidate.
4. Numerically compare before benchmarking.
5. Add remaining fixed tile candidates only when the prior result identifies reuse, occupancy, or causality as the limiting factor.
6. Select per-shape winners and add conservative routing/counters.
7. Integrate the proven packed loader with qLen=1–4 XQA, including Turbo4, and retain only measured wins.
8. Run full targeted correctness, performance, and production smoke verification.
9. Synchronize all evidence to issue #1.
