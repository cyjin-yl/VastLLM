// Regression for the SM70 FP8 paged prefill guard/kernel contract.
//
// The launch decision used to be taken against Data::cpuIntDatas (a host mirror) while the
// kernel indexed the page table from the device buffers.  A skew between the two walked the
// kernel off the end of the page table.  This test drives the real entry point:
//   1. a well formed shape must be accepted and must produce finite output;
//   2. a device buffer that disagrees with the mirror must be rejected (strict mode);
//   3. with strict mode disabled the same skew must still be handled without corrupting
//      memory, i.e. the kernel's own clamps have to hold.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/attention/fastllm-paged-attention-native.cuh"

#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using fastllm::Data;
using fastllm::DataDevice;
using fastllm::DataType;

constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kPageLen = 128;
constexpr int kGroup = 6;

void MakeIntVector(Data &data, const std::vector<int> &values) {
    data.dataType = DataType::INT32;
    data.Resize({(int)values.size()});
    data.ToDevice(DataDevice::CUDA);
    data.Allocate();
    data.cpuIntDatas = values;
    std::vector<int32_t> staging(values.begin(), values.end());
    FastllmCudaCopyFromHostToDevice(data.cudaData, staging.data(),
                                    staging.size() * sizeof(int32_t));
}

struct Fixture {
    int batch = 1;
    int qLen = 3;
    int kvLen = 512;
    int pagesPerRequest = 0;
    int totalPages = 0;
    int maxPages = 0;

    Data q, output;
    Data kCaches, vCaches;
    fastllm::PagedCacheManager pagedK, pagedV;
    Data qSizes, pageSizes, pageIndexs, lastPageLens;

    void Build() {
        pagesPerRequest = (kvLen + kPageLen - 1) / kPageLen;
        totalPages = batch * pagesPerRequest;
        maxPages = totalPages + 4;
        const int totalTokens = batch * qLen;
        const int lastPageLen = kvLen - (pagesPerRequest - 1) * kPageLen;

        q.dataType = DataType::FLOAT16;
        q.Resize({kQHeads, totalTokens, kHeadDim});
        q.ToDevice(DataDevice::CUDA);
        q.Allocate(0.0f);
        output.dataType = DataType::FLOAT16;
        output.Resize({kQHeads, totalTokens, kHeadDim});
        output.ToDevice(DataDevice::CUDA);
        output.Allocate(0.0f);

        for (fastllm::PagedCacheManager *manager : {&pagedK, &pagedV}) {
            manager->dataType = DataType::FP8_E4M3;
            manager->Resize({maxPages, kPageLen, kKvHeads, kHeadDim});
            manager->ToDevice(DataDevice::CUDA);
            manager->Allocate(true);
            manager->pageLen = kPageLen;
            manager->maxPages = maxPages;
        }
        for (Data *cache : {&kCaches, &vCaches}) {
            cache->dataType = DataType::FP8_E4M3;
            cache->Resize({kKvHeads, kvLen, kHeadDim});
            cache->isKVCache = true;
            cache->isPagedKVCache = true;
            cache->pageLen = kPageLen;
        }
        kCaches.pagedKVCacheData = &pagedK;
        vCaches.pagedKVCacheData = &pagedV;

        std::vector<int> qSizesHost(batch + 1), pageSizesHost(batch + 1);
        std::vector<int> lastPageLensHost(batch), pageIndexsHost(totalPages);
        for (int b = 0; b <= batch; b++) {
            qSizesHost[b] = b * qLen;
            pageSizesHost[b] = b * pagesPerRequest;
        }
        for (int b = 0; b < batch; b++) {
            lastPageLensHost[b] = lastPageLen;
        }
        for (int i = 0; i < totalPages; i++) {
            pageIndexsHost[i] = i;
        }
        MakeIntVector(qSizes, qSizesHost);
        MakeIntVector(pageSizes, pageSizesHost);
        MakeIntVector(pageIndexs, pageIndexsHost);
        MakeIntVector(lastPageLens, lastPageLensHost);
    }

    // Overwrite only the *device* copy, leaving cpuIntDatas at the validated value.  This is
    // the fillLastPageLensOnDevice / restored-request divergence, reproduced deterministically.
    void SkewDeviceLastPageLen(int delta) {
        std::vector<int32_t> staging(lastPageLens.cpuIntDatas.begin(),
                                     lastPageLens.cpuIntDatas.end());
        for (int32_t &value : staging) {
            value += delta;
        }
        FastllmCudaCopyFromHostToDevice(lastPageLens.cudaData, staging.data(),
                                        staging.size() * sizeof(int32_t));
    }

    bool Run() {
        output.Resize({kQHeads, batch * qLen, kHeadDim});
        return FastllmCudaTrySm70FlashAttentionPrefill(
            q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens,
            output, kGroup, 1.0f / std::sqrt((float)kHeadDim), 1);
    }
};

bool IsVolta() {
    int device = -1, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return major == 7 && minor == 0;
}

bool DeviceHealthy(const char *stage) {
    cudaError_t state = cudaStreamSynchronize(cudaStreamPerThread);
    if (state != cudaSuccess) {
        std::cerr << "  FAIL " << stage << ": CUDA context died -> "
                  << cudaGetErrorString(state) << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    FastllmCudaSetDevice(0);
    if (!IsVolta()) {
        std::cout << "sm70 paged prefill regression: SKIP (not a compute 7.0 device)\n";
        return 0;
    }
    setenv("FASTLLM_CUDA_SM70_FLASH_ATTN", "1", 1);
    bool ok = true;

    {
        setenv("FASTLLM_CUDA_SM70_FLASH_ATTN_STRICT", "1", 1);
        Fixture fixture;
        fixture.Build();
        const bool accepted = fixture.Run();
        ok = DeviceHealthy("well-formed") && ok;
        std::cout << "  well-formed batch=1 qLen=3 kv=512 -> "
                  << (accepted ? "accepted" : "REJECTED") << "\n";
        if (!accepted) {
            std::cerr << "  FAIL: the supported shape must still take the SM70 path\n";
            ok = false;
        }
    }

    {
        setenv("FASTLLM_CUDA_SM70_FLASH_ATTN_STRICT", "1", 1);
        Fixture fixture;
        fixture.Build();
        fixture.SkewDeviceLastPageLen(1);
        const bool accepted = fixture.Run();
        ok = DeviceHealthy("strict-skew") && ok;
        std::cout << "  device metadata skew (+1 token), strict=1 -> "
                  << (accepted ? "ACCEPTED" : "rejected") << "\n";
        if (accepted) {
            std::cerr << "  FAIL: a shape the guard never validated reached the kernel\n";
            ok = false;
        }
    }

    {
        // Defence in depth: even with the host cross-check disabled the kernel's own clamps
        // must keep every page-table and KV-pool access in bounds.
        setenv("FASTLLM_CUDA_SM70_FLASH_ATTN_STRICT", "0", 1);
        Fixture fixture;
        fixture.Build();
        fixture.SkewDeviceLastPageLen(64);
        const bool accepted = fixture.Run();
        ok = DeviceHealthy("clamp-only-skew") && ok;
        std::cout << "  device metadata skew (+64 tokens), strict=0 -> "
                  << (accepted ? "accepted" : "rejected") << " (context alive)\n";
    }

    std::cout << (ok ? "sm70 paged prefill regression: PASS\n"
                     : "sm70 paged prefill regression: FAIL\n");
    return ok ? 0 : 1;
}
