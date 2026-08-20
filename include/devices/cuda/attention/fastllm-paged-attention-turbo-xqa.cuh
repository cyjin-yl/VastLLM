//
// 融合式打包分页注意力 decode/短 query kernel 的对外入口。
// 实现见 src/devices/cuda/attention/paged/fastllm-paged-attention-turbo-xqa.cu。
//
#pragma once

#include "fastllm.h"

// 试探路径(trial path)：任何条件不满足一律返回 false 且**不写 outData**,
// 调用方原有的 gather + chunked cuBLAS 路径完全不受影响。
// 参数与 fastllm-paged-attention-native.cu 里
// FastllmCudaPagedAttentionNativeChunkedCublasRaw 的同名参数一一对应。
bool FastllmCudaTrySm70PagedTurboXqa(
    void *qData, fastllm::DataType qType,
    int H, int qoLen, int qDim, int qHeadStride, int qTokenStride,
    const int32_t *pageIndicesGpu, int numPages, int lastPageLen,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int pageLen, int numKvHeads, int headDim,
    void *outData, fastllm::DataType outType, int outHeadStride, int outTokenStride,
    int group, float scale);

bool FastllmCudaTrySm70PagedTurboPrefill(
    void *qData, fastllm::DataType qType,
    int H, int qoLen, int qDim, int qHeadStride, int qTokenStride,
    const int32_t *pageIndicesGpu, int numPages, int lastPageLen,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int pageLen, int numKvHeads, int headDim,
    void *outData, fastllm::DataType outType,
    int outHeadStride, int outTokenStride, int group, float scale);
