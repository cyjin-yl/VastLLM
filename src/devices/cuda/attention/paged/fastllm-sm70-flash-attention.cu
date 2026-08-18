// Copyright (c) 2024, flash-attention-v100 contributors.
// Copyright (c) 2025, FastLLM contributors.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
//
// The tiled causal/paged online-softmax structure is adapted from the local
// flash-attention-v100 paged forward kernel. This specialization targets the
// short-query FP8-KV validation shape used by Qwen3.5/3.6 MTP on Volta.

#include "devices/cuda/attention/fastllm-paged-attention-native.cuh"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace nvcuda::wmma;

constexpr int kHeadDim = 256;
constexpr int kTile = 16;
constexpr int kThreads = 512;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kGroup = 6;
constexpr int kPageLen = 128;
constexpr int kMaxBatch = 5;
constexpr int kMaxQLen = 10;
constexpr int kDefaultMaxKv = 512;

struct alignas(128) Sm70FlashPrefillSmem {
    half q[kTile * kHeadDim];
    half k[kTile * kHeadDim];
    half v[kTile * kHeadDim];
    float scores[kTile * kTile];
    half probabilities[kTile * kTile];
    float output[kTile * kHeadDim];
    float rowMax[kTile];
    float rowSum[kTile];
    float rescale[kTile];
};

__device__ __forceinline__ half LoadFp8Half(const __nv_fp8_e4m3 *base,
                                             int64_t offset) {
    __nv_fp8_e4m3 value;
    value.__x = __ldg(reinterpret_cast<const unsigned char *>(base) + offset);
    return __float2half_rn(static_cast<float>(value));
}

__global__ void Sm70FlashAttentionFp8PrefillKernel(
    const half *__restrict__ q,
    const __nv_fp8_e4m3 *__restrict__ kCache,
    const __nv_fp8_e4m3 *__restrict__ vCache,
    half *__restrict__ output,
    const int32_t *__restrict__ qSizes,
    const int32_t *__restrict__ pageSizes,
    const int32_t *__restrict__ pageIndices,
    const int32_t *__restrict__ lastPageLens,
    int qStrideHead,
    int qStrideToken,
    float softmaxScale,
    int totalQTokens,
    int totalPageSlots,
    int maxPages) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    extern __shared__ char rawSmem[];
    Sm70FlashPrefillSmem &smem =
        *reinterpret_cast<Sm70FlashPrefillSmem *>(rawSmem);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int request = blockIdx.x;
    const int qHead = blockIdx.y;
    const int kvHead = qHead / kGroup;
    // The launch decision is taken on the host against Data::cpuIntDatas, which is only a
    // *mirror* of these buffers: several decode paths (fillLastPageLensOnDevice, external
    // decode metadata, restored/swapped requests) update the device copy without refreshing
    // the mirror -- see the note in fastllm-paged-attention-native.cu.  The kernel therefore
    // must stay in bounds for *any* content of qSizes/pageSizes/lastPageLens: every value
    // read below is range-checked before it is used as an index.
    const int qBegin = qSizes[request];
    const int qEnd = qSizes[request + 1];
    const int pageBegin = pageSizes[request];
    const int pageEnd = pageSizes[request + 1];
    if (qBegin < 0 || qEnd <= qBegin || qEnd > totalQTokens
        || qEnd - qBegin > kTile
        || pageBegin < 0 || pageEnd <= pageBegin || pageEnd > totalPageSlots) {
        return;
    }
    const int qLen = qEnd - qBegin;
    const int pageCount = pageEnd - pageBegin;
    // Clamping lastPageLen to [0, kPageLen] bounds kvLen by pageCount * kPageLen, which is
    // exactly what makes logicalPage < pageCount hold for every tile below.
    const int lastPageLen = min(max(lastPageLens[request], 0), kPageLen);
    const int kvLen = (pageCount - 1) * kPageLen + lastPageLen;

    for (int i = tid; i < kTile * kHeadDim; i += kThreads) {
        const int row = i / kHeadDim;
        const int dim = i - row * kHeadDim;
        smem.q[i] = row < qLen
            ? q[(int64_t)qHead * qStrideHead
                + (int64_t)(qBegin + row) * qStrideToken + dim]
            : __float2half(0.0f);
        smem.output[i] = 0.0f;
    }
    if (tid < kTile) {
        smem.rowMax[tid] = -1.0e30f;
        smem.rowSum[tid] = 0.0f;
        smem.rescale[tid] = 0.0f;
        // Rows >= qLen are never written by the softmax below but are still fed to the P*V
        // WMMA tile, so zero them once here instead of reading stale shared memory.
        if (tid >= qLen) {
#pragma unroll
            for (int col = 0; col < kTile; ++col) {
                smem.probabilities[tid * kTile + col] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    const int causalOffset = kvLen - qLen;
    for (int tileStart = 0; tileStart < kvLen; tileStart += kTile) {
        const int tileRows = min(kTile, kvLen - tileStart);
        for (int i = tid; i < kTile * kHeadDim; i += kThreads) {
            const int tokenInTile = i / kHeadDim;
            const int dim = i - tokenInTile * kHeadDim;
            half kValue = __float2half(0.0f);
            half vValue = __float2half(0.0f);
            if (tokenInTile < tileRows) {
                const int logicalToken = tileStart + tokenInTile;
                const int logicalPage = logicalToken / kPageLen;
                const int pageOffset = logicalToken - logicalPage * kPageLen;
                // logicalToken < kvLen <= pageCount * kPageLen (see the clamp above), so
                // logicalPage is always inside this request's slice of the page table.
                // The physical page still comes from device memory, so clamp it into the
                // pool: branchless, and it cannot turn a bad page id into a wild address.
                const int rawPage = pageIndices[pageBegin + logicalPage];
                const int physicalPage = min(max(rawPage, 0), maxPages - 1);
                const int64_t cacheOffset =
                    (((int64_t)physicalPage * kPageLen + pageOffset) * kKvHeads
                        + kvHead) * kHeadDim + dim;
                kValue = LoadFp8Half(kCache, cacheOffset);
                vValue = LoadFp8Half(vCache, cacheOffset);
            }
            smem.k[i] = kValue;
            smem.v[i] = vValue;
        }
        __syncthreads();

        if (warp == 0) {
            fragment<matrix_a, kTile, kTile, kTile, half, row_major> qFrag;
            fragment<matrix_b, kTile, kTile, kTile, half, col_major> kFrag;
            fragment<accumulator, kTile, kTile, kTile, float> scoreFrag;
            fill_fragment(scoreFrag, 0.0f);
#pragma unroll
            for (int dim = 0; dim < kHeadDim; dim += kTile) {
                load_matrix_sync(qFrag, smem.q + dim, kHeadDim);
                load_matrix_sync(kFrag, smem.k + dim, kHeadDim);
                mma_sync(scoreFrag, qFrag, kFrag, scoreFrag);
            }
            store_matrix_sync(smem.scores, scoreFrag, kTile, mem_row_major);
        }
        __syncthreads();

        if (tid < qLen) {
            float tileMax = -1.0e30f;
#pragma unroll
            for (int col = 0; col < kTile; ++col) {
                const int keyPosition = tileStart + col;
                if (col < tileRows && keyPosition <= causalOffset + tid) {
                    tileMax = fmaxf(tileMax,
                                    smem.scores[tid * kTile + col]
                                        * softmaxScale);
                }
            }
            const float oldMax = smem.rowMax[tid];
            const float newMax = fmaxf(oldMax, tileMax);
            const float rescale = __expf(oldMax - newMax);
            float tileSum = 0.0f;
#pragma unroll
            for (int col = 0; col < kTile; ++col) {
                const int keyPosition = tileStart + col;
                float probability = 0.0f;
                if (col < tileRows && keyPosition <= causalOffset + tid) {
                    probability = __expf(
                        smem.scores[tid * kTile + col] * softmaxScale - newMax);
                }
                smem.probabilities[tid * kTile + col] =
                    __float2half_rn(probability);
                tileSum += probability;
            }
            smem.rowMax[tid] = newMax;
            smem.rowSum[tid] = smem.rowSum[tid] * rescale + tileSum;
            smem.rescale[tid] = rescale;
        }
        __syncthreads();

        for (int i = tid; i < qLen * kHeadDim; i += kThreads) {
            const int row = i / kHeadDim;
            smem.output[i] *= smem.rescale[row];
        }
        __syncthreads();

        if (warp < kHeadDim / kTile) {
            fragment<matrix_a, kTile, kTile, kTile, half, row_major> pFrag;
            fragment<matrix_b, kTile, kTile, kTile, half, row_major> vFrag;
            fragment<accumulator, kTile, kTile, kTile, float> outFrag;
            const int dim = warp * kTile;
            load_matrix_sync(pFrag, smem.probabilities, kTile);
            load_matrix_sync(vFrag, smem.v + dim, kHeadDim);
            load_matrix_sync(outFrag, smem.output + dim, kHeadDim, mem_row_major);
            mma_sync(outFrag, pFrag, vFrag, outFrag);
            store_matrix_sync(smem.output + dim, outFrag, kHeadDim, mem_row_major);
        }
        __syncthreads();
    }

    for (int i = tid; i < qLen * kHeadDim; i += kThreads) {
        const int row = i / kHeadDim;
        const int dim = i - row * kHeadDim;
        const float invSum = 1.0f / fmaxf(smem.rowSum[row], 1.0e-24f);
        output[((int64_t)(qBegin + row) * kQHeads + qHead) * kHeadDim
               + dim] = __float2half_rn(smem.output[i] * invSum);
    }
#endif
}

bool EnvEnabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}

int MaxKvTokens() {
    const char *value = std::getenv("FASTLLM_CUDA_SM70_FLASH_ATTN_MAX_KV");
    if (value == nullptr) {
        return 512;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= INT_MAX
        ? static_cast<int>(parsed)
        : 512;
}

bool StrictDeviceCheckEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_SM70_FLASH_ATTN_STRICT");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

// The values the kernel will actually index with.  Data::cpuIntDatas is only a host mirror of
// these buffers and is allowed to lag them (fillLastPageLensOnDevice / externally supplied
// decode metadata / restored requests), so the launch decision has to be taken against the
// device copy -- otherwise a shape the guard never approved can reach the kernel.
struct Sm70PagedMeta {
    std::vector<int32_t> qSizes;
    std::vector<int32_t> pageSizes;
    std::vector<int32_t> lastPageLens;
};

bool FetchDeviceMeta(const fastllm::Data &qSizes, const fastllm::Data &pageSizes,
                     const fastllm::Data &lastPageLens, int batch, Sm70PagedMeta &meta) {
    meta.qSizes.assign((size_t)batch + 1, 0);
    meta.pageSizes.assign((size_t)batch + 1, 0);
    meta.lastPageLens.assign((size_t)batch, 0);
    auto copy = [](void *dst, const void *src, size_t bytes) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                               cudaStreamPerThread) == cudaSuccess;
    };
    // Same stream as the appends that produced the metadata, so the copies see the
    // post-append values rather than whatever was there at enqueue time.
    if (!copy(meta.qSizes.data(), qSizes.cudaData,
              (size_t)(batch + 1) * sizeof(int32_t))
        || !copy(meta.pageSizes.data(), pageSizes.cudaData,
                 (size_t)(batch + 1) * sizeof(int32_t))
        || !copy(meta.lastPageLens.data(), lastPageLens.cudaData,
                 (size_t)batch * sizeof(int32_t))
        || cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

// Every constraint the kernel actually relies on, checked against one consistent snapshot.
bool MetaWithinKernelLimits(const std::vector<int32_t> &qSizesHost,
                            const std::vector<int32_t> &pageSizesHost,
                            const std::vector<int32_t> &lastPageLensHost,
                            int batch, int totalQTokens, int totalPageSlots,
                            int maxKvTokens) {
    if (qSizesHost.front() != 0 || pageSizesHost.front() != 0
        || qSizesHost.back() != totalQTokens
        || pageSizesHost.back() != totalPageSlots) {
        return false;
    }
    for (int request = 0; request < batch; ++request) {
        const int qBegin = qSizesHost[request];
        const int qLen = qSizesHost[request + 1] - qBegin;
        const int pageBegin = pageSizesHost[request];
        const int pages = pageSizesHost[request + 1] - pageBegin;
        const int lastPageLen = lastPageLensHost[request];
        const int kvLen = pages > 0 ? (pages - 1) * kPageLen + lastPageLen : 0;
        if (qBegin < 0 || pageBegin < 0
            || qLen < 2 || qLen > kMaxQLen || qLen > kTile
            || pages <= 0 || lastPageLen <= 0 || lastPageLen > kPageLen
            || kvLen < qLen || kvLen > maxKvTokens) {
            return false;
        }
    }
    return true;
}

bool IsCudaInt32Vector(const fastllm::Data &data) {
    return data.dataType == fastllm::DataType::INT32
        && data.dataDevice == fastllm::DataDevice::CUDA
        && data.dims.size() == 1 && data.cudaData != nullptr;
}

} // namespace

bool FastllmCudaTrySm70FlashAttentionPrefill(
    fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches,
    fastllm::Data &qSizes, fastllm::Data &pageSizes,
    fastllm::Data &pageIndexs, fastllm::Data &lastPageLens,
    fastllm::Data &output, int group, float scale, int attentionType) {
    if (!EnvEnabled("FASTLLM_CUDA_SM70_FLASH_ATTN")
        || attentionType != 1 || group != kGroup
        || q.dataType != fastllm::DataType::FLOAT16
        || output.dataType != fastllm::DataType::FLOAT16
        || q.dataDevice != fastllm::DataDevice::CUDA
        || output.dataDevice != fastllm::DataDevice::CUDA
        || q.cudaData == nullptr || output.cudaData == nullptr
        || q.dims.size() != 3 || output.dims != q.dims
        || q.dims[0] != kQHeads || q.dims[1] < 2 || q.dims[1] > 50
        || q.dims[2] != kHeadDim
        || q.strides.size() != 3 || output.strides.size() != 3
        || q.strides[2] != 1 || output.strides != q.strides
        || qSizes.dims.size() != 1 || qSizes.dims[0] < 2
        || qSizes.dims[0] > 6
        || pageSizes.dims.size() != 1
        || pageSizes.dims[0] != qSizes.dims[0]
        || lastPageLens.dims.size() != 1
        || lastPageLens.dims[0] != qSizes.dims[0] - 1
        || !IsCudaInt32Vector(qSizes) || !IsCudaInt32Vector(pageSizes)
        || !IsCudaInt32Vector(pageIndexs) || !IsCudaInt32Vector(lastPageLens)) {
        return false;
    }
    fastllm::PagedCacheManager *pagedK = kCaches.pagedKVCacheData;
    fastllm::PagedCacheManager *pagedV = vCaches.pagedKVCacheData;
    if (!kCaches.isPagedKVCache || !vCaches.isPagedKVCache
        || pagedK == nullptr || pagedV == nullptr
        || pagedK->dataType != fastllm::DataType::FP8_E4M3
        || pagedV->dataType != fastllm::DataType::FP8_E4M3
        || pagedK->dataDevice != fastllm::DataDevice::CUDA
        || pagedV->dataDevice != fastllm::DataDevice::CUDA
        || pagedK->cudaData == nullptr || pagedV->cudaData == nullptr
        || pagedK->dims.size() != 4 || pagedV->dims != pagedK->dims
        || pagedK->dims[1] != kPageLen || pagedK->dims[2] != kKvHeads
        || pagedK->dims[3] != kHeadDim) {
        return false;
    }

    const int batch = qSizes.dims[0] - 1;
    const int maxKvTokens = MaxKvTokens();
    if (!std::isfinite(scale) || scale <= 0.0f
        || (int)qSizes.cpuIntDatas.size() != batch + 1
        || (int)pageSizes.cpuIntDatas.size() != batch + 1
        || (int)pageIndexs.cpuIntDatas.size() < pageIndexs.dims[0]
        || (int)lastPageLens.cpuIntDatas.size() != batch
        || qSizes.cpuIntDatas.front() != 0
        || pageSizes.cpuIntDatas.front() != 0
        || qSizes.cpuIntDatas.back() != q.dims[1]
        || pageSizes.cpuIntDatas.back() != pageIndexs.dims[0]) {
        return false;
    }
    for (int request = 0; request < batch; ++request) {
        const int qLen = qSizes.cpuIntDatas[request + 1]
            - qSizes.cpuIntDatas[request];
        const int pages = pageSizes.cpuIntDatas[request + 1]
            - pageSizes.cpuIntDatas[request];
        const int lastPageLen = lastPageLens.cpuIntDatas[request];
        const int kvLen = pages > 0 ? (pages - 1) * kPageLen + lastPageLen : 0;
        if (qLen < 2 || qLen > kMaxQLen || pages <= 0
            || lastPageLen <= 0 || lastPageLen > kPageLen
            || kvLen < qLen || kvLen > maxKvTokens) {
            return false;
        }
        for (int page = pageSizes.cpuIntDatas[request];
             page < pageSizes.cpuIntDatas[request + 1]; ++page) {
            const int physicalPage = pageIndexs.cpuIntDatas[page];
            if (physicalPage < 0 || physicalPage >= pagedK->dims[0]) {
                return false;
            }
        }
    }

    int device = -1;
    int major = 0;
    int minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                  device) != cudaSuccess
        || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                  device) != cudaSuccess
        || major != 7 || minor != 0) {
        cudaGetLastError();
        return false;
    }
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus) != cudaSuccess
        || captureStatus != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return false;
    }

    // Re-run the shape contract against the buffers the kernel dereferences.  The checks
    // above only prove that the *mirror* is in range; this is what stops a long-KV / long-qLen
    // shape from reaching a kernel that the mirror said was short.
    if (StrictDeviceCheckEnabled()) {
        Sm70PagedMeta deviceMeta;
        if (!FetchDeviceMeta(qSizes, pageSizes, lastPageLens, batch, deviceMeta)
            || !MetaWithinKernelLimits(deviceMeta.qSizes, deviceMeta.pageSizes,
                                       deviceMeta.lastPageLens, batch, (int)q.dims[1],
                                       (int)pageIndexs.dims[0], maxKvTokens)) {
            return false;
        }
    }

    const size_t smemBytes = sizeof(Sm70FlashPrefillSmem);
    if (cudaFuncSetAttribute(Sm70FlashAttentionFp8PrefillKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smemBytes)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    dim3 grid(batch, kQHeads, 1);
    Sm70FlashAttentionFp8PrefillKernel<<<grid, kThreads, smemBytes,
                                         cudaStreamPerThread>>>(
        reinterpret_cast<const half *>(q.cudaData),
        reinterpret_cast<const __nv_fp8_e4m3 *>(pagedK->cudaData),
        reinterpret_cast<const __nv_fp8_e4m3 *>(pagedV->cudaData),
        reinterpret_cast<half *>(output.cudaData),
        reinterpret_cast<const int32_t *>(qSizes.cudaData),
        reinterpret_cast<const int32_t *>(pageSizes.cudaData),
        reinterpret_cast<const int32_t *>(pageIndexs.cudaData),
        reinterpret_cast<const int32_t *>(lastPageLens.cudaData),
        static_cast<int>(q.strides[0]), static_cast<int>(q.strides[1]), scale,
        static_cast<int>(q.dims[1]), static_cast<int>(pageIndexs.dims[0]),
        static_cast<int>(pagedK->dims[0]));
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    output.Resize({q.dims[1], kQHeads, kHeadDim});
    static thread_local bool logged = false;
    if (!logged) {
        std::printf("[FastLLM] SM70 FlashAttention FP8 paged prefill enabled "
                    "(Volta WMMA, page128, batch1..%d, qLen2..%d, "
                    "KV<=%d, Q24/KV4 D256 GQA6, device-meta check %s).\n",
                    kMaxBatch, kMaxQLen, maxKvTokens,
                    StrictDeviceCheckEnabled() ? "on" : "OFF");
        logged = true;
    }
    return true;
}
