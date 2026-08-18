// Offline microbenchmark for the fastllm GGUF prefill "dequantise-then-cuBLAS"
// path -- the fastllm analogue of 1Cat-vLLM 1.3.0 "long-prefill exact-dense".
//
// Shapes are the real Qwen3.8-27B-Uncensored-Cyber (arch qwen35) projections.
// Baselines are copied verbatim from src/devices/cuda/fastllm-ggml-cuda.cu.
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 -Wno-deprecated-gpu-targets \
//        -ccbin <conda>/bin/x86_64-conda-linux-gnu-g++ bench.cu -o bench -lcublas
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} } while(0)
#define CHKB(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s_);exit(1);} } while(0)

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

typedef __half  ggml_half;
typedef __half2 ggml_half2;

typedef struct { union { struct { ggml_half d; ggml_half dmin; } data; ggml_half2 dm; };
                 uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;   // 176 B
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;                                          // 34 B
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K; // 210 B

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4); }
}
static __device__ __forceinline__ unsigned pack2(__half lo, __half hi) {
    return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
static __device__ __forceinline__ void store16_cs(__half * dst, unsigned a, unsigned b, unsigned c, unsigned d) {
    __stcs(reinterpret_cast<uint4 *>(dst), make_uint4(a, b, c, d));
}

// ===================== baselines (verbatim from fastllm) ====================
// fastllm-ggml-cuda.cu:1615  dequantize_block_q5_K   <<<k/256, 64>>>
__global__ void base_q5_K(const void * __restrict__ vx, __half * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;
    const int64_t i = blockIdx.x, tid = threadIdx.x;
    const int64_t il = tid/16, ir = tid%16, is = 2*il;
    __half * y = yy + i*QK_K + 64*il + 2*ir;
    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);
    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    uint8_t hm = 1 << (2*il);
    y[ 0] = __float2half_rn(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    y[ 1] = __float2half_rn(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = __float2half_rn(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
    y[33] = __float2half_rn(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
}
// fastllm-ggml-cuda.cu:1423 dequantize_block<QK8_0,QR8_0,dequantize_q8_0>
__global__ void base_q8_0(const void * __restrict__ vx, __half * __restrict__ y, int64_t k) {
    const int64_t i = (int64_t)2*(blockDim.x*blockIdx.x + threadIdx.x);
    if (i >= k) return;
    const int64_t ib = i/QK8_0;
    const int iqs = (int)(i%QK8_0);
    const int64_t iybs = i - i%QK8_0;
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const __half d = x[ib].d;
    __half2 v = __halves2half2(__int2half_rn(x[ib].qs[iqs+0]), __int2half_rn(x[ib].qs[iqs+1]));
    v = __hmul2(v, __halves2half2(d,d));
    y[iybs + iqs + 0] = __low2half(v);
    y[iybs + iqs + 1] = __high2half(v);
}
// fastllm-ggml-cuda.cu:1649 dequantize_block_q6_K  <<<k/256, 64>>>
__global__ void base_q6_K(const void * __restrict__ vx, __half * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;
    const int64_t i = blockIdx.x, tid = threadIdx.x;
    const int64_t ip = tid/32, il = tid - 32*ip, is = 8*ip + il/16;
    __half * y = yy + i*QK_K + 128*ip + il;
    const float d = x[i].d;
    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t qh = x[i].qh[32*ip + il];
    const int8_t * sc = x[i].scales + is;
    y[ 0] = __float2half_rn(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = __float2half_rn(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = __float2half_rn(d * sc[4] * ((int8_t)((ql[ 0] >>  4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = __float2half_rn(d * sc[6] * ((int8_t)((ql[32] >>  4) | (((qh >> 6) & 3) << 4)) - 32));
}

// ============ candidates: 8 consecutive outputs / lane, 16B store ===========
// STREAM=0 -> ordinary st.global.v4 ; STREAM=1 -> st.global.cs.v4 (evict-first)
template<int STREAM>
__global__ void cand_q5_K(const void * __restrict__ vx, __half * __restrict__ yy, int64_t nblocks) {
    const block_q5_K * __restrict__ x = (const block_q5_K *) vx;
    const int64_t gt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ib = gt >> 5;
    if (ib >= nblocks) return;
    const int lane = (int)(gt & 31);
    const int il = lane >> 3;          // 0..3   64-element group
    const int o  = (lane & 7) << 3;    // 0,8,..,56
    const int hi = o >> 5;             // 0 low nibble / 1 high nibble
    const int j  = o & 31;             // 0,8,16,24
    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);
    uint8_t sc, mn;
    get_scale_min_k4(2*il + hi, x[ib].scales, sc, mn);
    const float dd = dall * sc, mm = dmin * mn;
    const uint8_t hbit = (uint8_t)(1u << (2*il + hi));
    const uint2 qsv = *(const uint2 *)(x[ib].qs + 32*il + j);   // qs off 48, blk 176 -> 8B aligned
    const uint2 qhv = *(const uint2 *)(x[ib].qh + j);           // qh off 16          -> 8B aligned
    const uint8_t * qs8 = (const uint8_t *)&qsv;
    const uint8_t * qh8 = (const uint8_t *)&qhv;
    __half v[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int q = hi ? (qs8[t] >> 4) : (qs8[t] & 0xF);
        v[t] = __float2half_rn(dd * (q + ((qh8[t] & hbit) ? 16 : 0)) - mm);
    }
    const unsigned a=pack2(v[0],v[1]), b=pack2(v[2],v[3]), c=pack2(v[4],v[5]), e=pack2(v[6],v[7]);
    __half * dst = yy + ib*QK_K + 64*il + 32*hi + j;
    if (STREAM) store16_cs(dst,a,b,c,e); else *(uint4 *)dst = make_uint4(a,b,c,e);
}

template<int STREAM>
__global__ void cand_q8_0(const void * __restrict__ vx, __half * __restrict__ yy, int64_t nblocks) {
    const block_q8_0 * __restrict__ x = (const block_q8_0 *) vx;
    const int64_t gt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ib = gt >> 2;
    if (ib >= nblocks) return;
    const int j = (int)(gt & 3) << 3;
    const __half d = x[ib].d;
    const int8_t * q = x[ib].qs + j;      // 34B stride -> only 2B aligned, byte loads
    __half v[8];
#pragma unroll
    for (int t = 0; t < 8; t += 2) {
        __half2 h = __halves2half2(__int2half_rn(q[t]), __int2half_rn(q[t+1]));
        h = __hmul2(h, __halves2half2(d, d));
        v[t]   = __low2half(h);
        v[t+1] = __high2half(h);
    }
    const unsigned a=pack2(v[0],v[1]), b=pack2(v[2],v[3]), c=pack2(v[4],v[5]), e=pack2(v[6],v[7]);
    __half * dst = yy + ib*QK8_0 + j;
    if (STREAM) store16_cs(dst,a,b,c,e); else *(uint4 *)dst = make_uint4(a,b,c,e);
}

template<int STREAM>
__global__ void cand_q6_K(const void * __restrict__ vx, __half * __restrict__ yy, int64_t nblocks) {
    const block_q6_K * __restrict__ x = (const block_q6_K *) vx;
    const int64_t gt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ib = gt >> 5;
    if (ib >= nblocks) return;
    const int lane = (int)(gt & 31);
    const int n0 = lane << 3;            // first output, multiple of 8
    const int ip = n0 >> 7;              // 0/1
    const int r  = n0 & 127;
    const int g  = r >> 5;               // 0..3
    const int il = r & 31;               // 0,8,16,24
    const float d = __half2float(x[ib].d);
    const int8_t sc = x[ib].scales[8*ip + (il >> 4) + 2*g];
    const uint8_t * ql = x[ib].ql + 64*ip + 32*(g & 1) + il;   // 210B stride -> byte loads
    const uint8_t * qh = x[ib].qh + 32*ip + il;
    const int shift = 2*g;
    __half v[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int q = (g < 2) ? (ql[t] & 0xF) : (ql[t] >> 4);
        const int h = (qh[t] >> shift) & 3;
        v[t] = __float2half_rn(d * sc * ((int8_t)(q | (h << 4)) - 32));
    }
    const unsigned a=pack2(v[0],v[1]), b=pack2(v[2],v[3]), c=pack2(v[4],v[5]), e=pack2(v[6],v[7]);
    __half * dst = yy + ib*QK_K + n0;
    if (STREAM) store16_cs(dst,a,b,c,e); else *(uint4 *)dst = make_uint4(a,b,c,e);
}

// roofline: identical memory pattern to cand_q5_K, trivial arithmetic
__global__ void roof_q5_K(const void * __restrict__ vx, __half * __restrict__ yy, int64_t nblocks) {
    const block_q5_K * __restrict__ x = (const block_q5_K *) vx;
    const int64_t gt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ib = gt >> 5;
    if (ib >= nblocks) return;
    const int lane = (int)(gt & 31);
    const int il = lane >> 3, o = (lane & 7) << 3, hi = o >> 5, j = o & 31;
    const uint2 qsv = *(const uint2 *)(x[ib].qs + 32*il + j);
    const uint8_t * q8 = (const uint8_t *)&qsv;
    const unsigned a=pack2(__int2half_rn(q8[0]),__int2half_rn(q8[1]));
    const unsigned b=pack2(__int2half_rn(q8[2]),__int2half_rn(q8[3]));
    const unsigned c=pack2(__int2half_rn(q8[4]),__int2half_rn(q8[5]));
    const unsigned e=pack2(__int2half_rn(q8[6]),__int2half_rn(q8[7]));
    *(uint4 *)(yy + ib*QK_K + 64*il + 32*hi + j) = make_uint4(a,b,c,e);
}
__global__ void roof_q5_K_cs(const void * __restrict__ vx, __half * __restrict__ yy, int64_t nblocks) {
    const block_q5_K * __restrict__ x = (const block_q5_K *) vx;
    const int64_t gt = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ib = gt >> 5;
    if (ib >= nblocks) return;
    const int lane = (int)(gt & 31);
    const int il = lane >> 3, o = (lane & 7) << 3, hi = o >> 5, j = o & 31;
    const uint2 qsv = *(const uint2 *)(x[ib].qs + 32*il + j);
    const uint8_t * q8 = (const uint8_t *)&qsv;
    store16_cs(yy + ib*QK_K + 64*il + 32*hi + j,
               pack2(__int2half_rn(q8[0]),__int2half_rn(q8[1])),
               pack2(__int2half_rn(q8[2]),__int2half_rn(q8[3])),
               pack2(__int2half_rn(q8[4]),__int2half_rn(q8[5])),
               pack2(__int2half_rn(q8[6]),__int2half_rn(q8[7])));
}

// ================================= driver ==================================
struct Shape { const char * name; int rows; int cols; int qtype; int count; };
static Shape kShapes[] = {
  {"ffn_gate",       17408,  5120, 5, 64},
  {"ffn_up",         17408,  5120, 5, 64},
  {"ffn_down",        5120, 17408, 5, 64},
  {"in_proj_qkv",    10240,  5120, 8, 48},
  {"in_proj_z",       6144,  5120, 8, 48},
  {"ssm_out_proj",    5120,  6144, 8, 48},
  {"attn_q",         12288,  5120, 8, 16},
  {"attn_k",          1024,  5120, 8, 16},
  {"attn_v",          1024,  5120, 8, 16},
  {"attn_o_proj",     5120,  6144, 6, 16},
};
static const int kNShapes = sizeof(kShapes)/sizeof(kShapes[0]);
static const int kM[] = {256, 512, 1024, 2048};
static const int kNM = sizeof(kM)/sizeof(kM[0]);

static size_t qbytes(int qtype, size_t n) {
    if (qtype == 5) return n/QK_K * sizeof(block_q5_K);
    if (qtype == 6) return n/QK_K * sizeof(block_q6_K);
    return n/QK8_0 * sizeof(block_q8_0);
}
struct Timer {
    cudaEvent_t a, b;
    Timer(){ CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b)); }
    void start(){ CHK(cudaEventRecord(a)); }
    float stop(){ CHK(cudaEventRecord(b)); CHK(cudaEventSynchronize(b));
                  float ms; CHK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};

int main(int argc, char ** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 40;
    cudaDeviceProp prop; CHK(cudaGetDeviceProperties(&prop, 0));
    size_t fm=0, tm=0; CHK(cudaMemGetInfo(&fm,&tm));
    printf("# %s sm_%d%d %dSM peakBW=%.0f GB/s | free=%.0f MiB | iters=%d (min-of-N)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           2.0*prop.memoryClockRate*1e3*(prop.memoryBusWidth/8.0)/1e9,
           fm/1048576.0, iters);

    cublasHandle_t hb; CHKB(cublasCreate(&hb));
    size_t maxN=0, maxCols=0, maxRows=0;
    for (int i=0;i<kNShapes;i++){ maxN=std::max(maxN,(size_t)kShapes[i].rows*kShapes[i].cols);
        maxCols=std::max(maxCols,(size_t)kShapes[i].cols); maxRows=std::max(maxRows,(size_t)kShapes[i].rows); }
    const int Mmax = kM[kNM-1];
    const size_t qbufBytes = std::max(std::max(qbytes(5,maxN),qbytes(6,maxN)),qbytes(8,maxN)) + 8192;

    void *dq=nullptr; __half *dW=nullptr,*dX=nullptr,*dY=nullptr;
    CHK(cudaMalloc(&dq, qbufBytes));
    CHK(cudaMalloc(&dW, maxN*sizeof(__half)));
    CHK(cudaMalloc(&dX, (size_t)Mmax*maxCols*sizeof(__half)));
    CHK(cudaMalloc(&dY, (size_t)Mmax*maxRows*sizeof(__half)));
    { std::vector<uint8_t> h(qbufBytes); std::mt19937 r(12345);
      for (size_t i=0;i<h.size();++i) h[i]=(uint8_t)(r()&0xFF);
      CHK(cudaMemcpy(dq,h.data(),h.size(),cudaMemcpyHostToDevice)); }
    { std::vector<__half> h((size_t)Mmax*maxCols); std::mt19937 r(999);
      for (size_t i=0;i<h.size();++i) h[i]=__float2half((((int)(r()%2001))-1000)/1000.0f*0.05f);
      CHK(cudaMemcpy(dX,h.data(),h.size()*sizeof(__half),cudaMemcpyHostToDevice)); }

    std::vector<__half> hA(maxN), hB(maxN);
    std::vector<double> tBase(kNShapes,0), tVec(kNShapes,0), tCs(kNShapes,0);

    printf("\n== 1. weight expansion (dequant -> fp16), per call ==\n");
    printf("%-14s %6s %6s %3s | %8s %8s %8s | %8s %8s %8s | %6s %s\n",
           "shape","rows","cols","q","base_ms","vec_ms","cs_ms","baseGB/s","vecGB/s","csGB/s","best","exactness");
    printf("%s\n", std::string(118, 0x2D).c_str());

    Timer T;
    for (int s=0;s<kNShapes;s++) {
        const Shape &sh = kShapes[s];
        const size_t n = (size_t)sh.rows*sh.cols;
        const double moved = (double)qbytes(sh.qtype,n) + (double)n*2;
        float mb=1e9f, mv=1e9f, mc=1e9f;
        const int blk = 256;
        if (sh.qtype==5) {
            const int64_t nb=n/QK_K, gr=(nb*32+blk-1)/blk;
            for(int i=0;i<iters;i++){T.start(); base_q5_K<<<nb,64>>>(dq,dW); float t=T.stop(); if(i&&t<mb)mb=t;}
            CHK(cudaMemcpy(hA.data(),dW,n*2,cudaMemcpyDeviceToHost));
            for(int i=0;i<iters;i++){T.start(); cand_q5_K<0><<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<mv)mv=t;}
            for(int i=0;i<iters;i++){T.start(); cand_q5_K<1><<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<mc)mc=t;}
            CHK(cudaMemcpy(hB.data(),dW,n*2,cudaMemcpyDeviceToHost));
        } else if (sh.qtype==8) {
            const int64_t nb=n/QK8_0, gr0=((int64_t)n+2*blk-1)/(2*blk), gr=(nb*4+blk-1)/blk;
            for(int i=0;i<iters;i++){T.start(); base_q8_0<<<gr0,blk>>>(dq,dW,(int64_t)n); float t=T.stop(); if(i&&t<mb)mb=t;}
            CHK(cudaMemcpy(hA.data(),dW,n*2,cudaMemcpyDeviceToHost));
            for(int i=0;i<iters;i++){T.start(); cand_q8_0<0><<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<mv)mv=t;}
            for(int i=0;i<iters;i++){T.start(); cand_q8_0<1><<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<mc)mc=t;}
            CHK(cudaMemcpy(hB.data(),dW,n*2,cudaMemcpyDeviceToHost));
        } else {
            const int64_t nb=n/QK_K, gr=(nb*32+blk-1)/blk;
            for(int i=0;i<iters;i++){T.start(); base_q6_K<<<nb,64>>>(dq,dW); float t=T.stop(); if(i&&t<mb)mb=t;}
            CHK(cudaMemcpy(hA.data(),dW,n*2,cudaMemcpyDeviceToHost));
            for(int i=0;i<iters;i++){T.start(); cand_q6_K<0><<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<mv)mv=t;}
            for(int i=0;i<iters;i++){T.start(); cand_q6_K<1><<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<mc)mc=t;}
            CHK(cudaMemcpy(hB.data(),dW,n*2,cudaMemcpyDeviceToHost));
        }
        CHK(cudaGetLastError());
        const char * ok = (memcmp(hA.data(),hB.data(),n*2)==0) ? "BITWISE-EQUAL" : "*** MISMATCH ***";
        const double best = std::min((double)mb, std::min((double)mv,(double)mc));
        const char * bn = (best==(double)mb) ? "base" : ((best==(double)mv) ? "vec" : "cs");
        tBase[s]=mb; tVec[s]=mv; tCs[s]=mc;
        printf("%-14s %6d %6d %3d | %8.4f %8.4f %8.4f | %8.1f %8.1f %8.1f | %6s %s\n",
               sh.name, sh.rows, sh.cols, sh.qtype, mb, mv, mc,
               moved/1e6/mb, moved/1e6/mv, moved/1e6/mc, bn, ok);
    }

    // roofline
    {
        const size_t n=(size_t)17408*5120; const int64_t nb=n/QK_K; const int blk=256;
        const int64_t gr=(nb*32+blk-1)/blk;
        const double moved=(double)qbytes(5,n)+(double)n*2;
        float r0=1e9f, r1=1e9f;
        for(int i=0;i<iters;i++){T.start(); roof_q5_K<<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<r0)r0=t;}
        for(int i=0;i<iters;i++){T.start(); roof_q5_K_cs<<<gr,blk>>>(dq,dW,nb); float t=T.stop(); if(i&&t<r1)r1=t;}
        printf("\n# roofline 17408x5120 (same traffic, no real math): plain-store %.4f ms (%.1f GB/s), "
               "cs-store %.4f ms (%.1f GB/s)\n", r0, moved/1e6/r0, r1, moved/1e6/r1);
    }

    // ------------------------------- GEMM ---------------------------------
    printf("\n== 2. cuBLAS fp16 GEMM, same call as fastllm-ggml-cuda.cu:2836 ==\n");
    printf("%-14s", "shape");
    for (int i=0;i<kNM;i++) printf("  M=%-4d ms TFLOP/s", kM[i]);
    printf("\n%s\n", std::string(14+kNM*20, 0x2D).c_str());
    std::vector<std::vector<double>> g(kNShapes, std::vector<double>(kNM,0.0));
    __half alpha=__float2half(1.0f), beta=__float2half(0.0f);
    for (int s=0;s<kNShapes;s++) {
        printf("%-14s", kShapes[s].name);
        for (int mi=0; mi<kNM; mi++) {
            const int M=kM[mi], k=kShapes[s].rows, m=kShapes[s].cols;
            float best=1e9f; const int it=std::max(8, iters/2);
            for (int i=0;i<it;i++) {
                T.start();
                CHKB(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, k, M, m,
                     &alpha, dW, CUDA_R_16F, m, dX, CUDA_R_16F, m, &beta,
                     dY, CUDA_R_16F, k, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
                float t=T.stop(); if(i&&t<best) best=t;
            }
            g[s][mi]=best;
            printf("  %9.4f %8.1f", best, 2.0*M*k*m/(best*1e-3)/1e12);
        }
        printf("\n");
    }

    // ------------------------------ roll-up --------------------------------
    double DB=0, DV=0, DC=0;
    for (int s=0;s<kNShapes;s++){ DB+=tBase[s]*kShapes[s].count; DV+=tVec[s]*kShapes[s].count;
        DC+=std::min(std::min(tBase[s],tVec[s]),tCs[s])*kShapes[s].count; }
    printf("\n== 3. ONE prefill chunk of M tokens, all 176 quantised projections ==\n");
    printf("%-8s %11s %11s %11s %11s %10s %11s %10s\n",
           "chunk M","deq_base","deq_best","gemm_ms","tot_base","deq_share","tot_best","gain");
    printf("%s\n", std::string(88, 0x2D).c_str());
    for (int mi=0; mi<kNM; mi++) {
        double G=0; for (int s=0;s<kNShapes;s++) G+=g[s][mi]*kShapes[s].count;
        const double tb=DB+G, tc=DC+G;
        printf("%-8d %11.2f %11.2f %11.2f %11.2f %9.1f%% %11.2f %9.2f%%\n",
               kM[mi], DB, DC, G, tb, 100.0*DB/tb, tc, 100.0*(tb-tc)/tb);
    }
    printf("\n== 4. chunk-size amortisation (the exact-dense lever) ==\n");
    printf("%-8s %14s %14s %14s %14s\n","chunk M","us/token base","us/token best","tok/s base","vs M=512");
    double ref=0;
    for (int mi=0; mi<kNM; mi++){ if(kM[mi]!=512) continue; double G=0;
        for(int s=0;s<kNShapes;s++) G+=g[s][mi]*kShapes[s].count; ref=(DB+G)*1000.0/512; }
    for (int mi=0; mi<kNM; mi++) {
        double G=0; for (int s=0;s<kNShapes;s++) G+=g[s][mi]*kShapes[s].count;
        const double pb=(DB+G)*1000.0/kM[mi], pc=(DC+G)*1000.0/kM[mi];
        printf("%-8d %14.2f %14.2f %14.1f %13.1f%%\n", kM[mi], pb, pc, 1e6/pb, 100.0*(ref-pb)/ref);
    }
    printf("\n# roll-up excludes attention, GDN, norms, activations, embedding and lm_head.\n");

    CHK(cudaFree(dq)); CHK(cudaFree(dW)); CHK(cudaFree(dX)); CHK(cudaFree(dY));
    cublasDestroy(hb);
    return 0;
}
