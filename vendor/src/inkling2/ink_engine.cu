#include <filesystem>
// src/inkling2/ink_engine.cu - the engine path for Inkling 2 (GGUF architecture inkling2), llama.cpp PR #32695's graph
#include "strata/artifact/gguf_reader.hpp"
#include "strata/inkling2/engine.hpp"
#include "cpu_attn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/qwen35/sampling.hpp"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace K = strata::kernels;
using strata::inkling2::CpuAttnJob;
using strata::inkling2::cpu_attn;
using strata::inkling2::CpuExperts;
using strata::inkling2::CpuMoeJob;
using strata::inkling2::cpu_experts_supported;
using strata::inkling2::cpu_moe;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-inkling2: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,    \
                         __LINE__);                                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kHD = 128;          // head width (the kernels' block)
constexpr int kChunkKeys = 256;   // attention split-K chunk
constexpr int kMaxDev = 8;
constexpr int kMaxStates = 8;     // depth-mix inputs (the blocks and the current state)
constexpr int kMaxG = 4;          // query heads per KV head the attention kernels take
constexpr int kGemmT = 8;
// GENI: a picture's rows take its run of <|unused_200054|> tokens (the chat template's image placeholder). In the engine
// each row's id is one past the vocabulary, made from the picture's key and the row, so every match by ids (lanes,
// checkpoints, parked conversations) tells pictures apart; a forward reads those rows' values from the request's.
constexpr int32_t kImagePad = 200054;
constexpr int32_t kImgId0 = 1 << 30;
inline bool is_img_id(int32_t t) { return t >= kImgId0; }
struct ImgRows {
    int n_embd = 0;
    std::vector<float> rows;                           // the request's pictures' rows, in prompt order
    std::unordered_map<int32_t, size_t> at;            // a row's id -> its row
};         // above this many rows a matrix product is a GEMM, else the native MMVQ
constexpr int kExpertGemmT = 32;  // from this many rows the experts are grouped GEMMs, else the native grouped kernels
constexpr int kAttTileT = 16;     // from this many rows attention is the tiled kernel, else split-K
constexpr size_t kSplitFloats = (size_t) 4 << 20;   // the f32 products' split-K partials
constexpr int kMaxLanes = 8;
constexpr int kStage = 16384;     // cells of a RAM part staged in VRAM at a time
constexpr int kHostKeys = 2048;   // keys a decode partial of a staged RAM block (the tensor-core kernel)
int g_cc_mib = 8192, g_cc_slots = 4, g_cc_min = 2048;   // --conversation-cache-mib / -slots / -min-tokens
int g_dspark_n = 3;              // --dspark-n: the most tokens DSpark drafts a step
uint64_t g_head_extra = 0;       // bytes beside the head on its card (the DSpark drafter: its file)
// Room on card d for `bytes` more (the pool moves its coldest experts to RAM; nothing without the pool): the drafter's
// weights and buffers come after the experts filled the cards.
std::function<void(int, size_t)> g_make_room;
void make_room(int d, size_t bytes) {
    if (g_make_room) g_make_room(d, bytes);
}
int g_dspark_window = 32768;     // --dspark-window: the drafter's full-attention layer reads this many positions back
int g_prefill_pipe = -1;      // --prefill-pipe 0|1: prompt chunks on two executors at once (-1: with 2+ cards)
int g_reserve_mib = -1;       // --vram-reserve-mib: VRAM the pool leaves free on each card
int g_experts_vram = -1;      // --experts-vram N: each layer keeps N experts in VRAM (-1: all), the rest in RAM
constexpr int kStageExp = 16;     // experts a prompt chunk copies into VRAM at a time (two such halves)
constexpr int kCpuSegs = 8;       // a decode's CPU share of a RAM part in this many segments a KV head
constexpr int kCpuMin = 16384;    // a RAM part shorter than this is read by the GPU alone

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
}

__global__ void k_rms2cat(const float* cur, const float* msg, float* out, int n, float eps) {
    __shared__ float sh[kThreads];
    const size_t t = blockIdx.x;
    for (int part = 0; part < 2; ++part) {
        const float* xr = (part ? msg : cur) + t * n;
        float s = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
        s = block_sum(s, sh);
        const float r = rsqrtf(s / (float) n + eps);
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[t * 2 * n + (size_t) part * n + i] = xr[i] * r;
    }
}

/// A decode's Q8_0 projections of up to 5 matrices in one launch (each its own q8_1 input and output; their rows
/// one after the other in the grid): a warp a row, 4 lanes a 32-value block, 8 blocks a step, so a small matrix
/// (k, v, rel) fills the card as well as a large one. Up to 8 rows (verify): the weights read once for all.
struct QSeg {
    const uint8_t* w;   // [n_out][n_in / 32] Q8_0 blocks (34 bytes)
    const uint8_t* x;   // [T][n_in / 32] q8_1 blocks (36 bytes)
    float* y;           // [T][n_out]
    int n_out, blk0;    // its first grid block
};
struct QSegs {
    QSeg s[5];
    int n;
};
__device__ __forceinline__ int ld_i32_b2(const uint8_t* p) {   // 4 bytes at a 2-byte aligned address
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (int) ((uint32_t) q[0] | ((uint32_t) q[1] << 16));
}
template <int T>
__global__ void __launch_bounds__(256) k_q8_seg(QSegs a, int n_in) {
    int k = 0;
    while (k + 1 < a.n && (int) blockIdx.x >= a.s[k + 1].blk0) ++k;
    const QSeg sg = a.s[k];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = ((int) blockIdx.x - sg.blk0) * 8 + warp;
    if (row >= sg.n_out) return;
    const int nb = n_in / 32, kq = lane % 4, b0 = lane / 4;
    const uint8_t* wr = sg.w + (size_t) row * nb * 34;
    float acc[T];
#pragma unroll
    for (int c = 0; c < T; ++c) acc[c] = 0.f;
    for (int kb = b0; kb < nb; kb += 8) {
        const uint8_t* wb = wr + (size_t) kb * 34;
        const float dw = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(wb)));
        const int w0 = ld_i32_b2(wb + 2 + kq * 8), w1 = ld_i32_b2(wb + 6 + kq * 8);
#pragma unroll
        for (int c = 0; c < T; ++c) {
            const uint8_t* xb = sg.x + ((size_t) c * nb + kb) * 36;
            const float dx = __low2float(*reinterpret_cast<const __half2*>(xb));
            const int* xi = reinterpret_cast<const int*>(xb + 4 + kq * 8);
            const int si = __dp4a(w1, xi[1], __dp4a(w0, xi[0], 0));
            acc[c] += dw * dx * (float) si;
        }
    }
#pragma unroll
    for (int c = 0; c < T; ++c) {
        float v = acc[c];
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) sg.y[(size_t) c * sg.n_out + row] = v;
    }
}

/// A looped layer's controls for a decode's rows, up to 3 at once (blockIdx.z): (1) the down projection of
/// [x, msg] split over K (kCtlSplits slices, half over x, half over msg) so a decode has blocks for every SM: each
/// (row group, slice) block a warp a row's unscaled partial, and the slice's sum of squares (the RMS scales apply
/// after: rx * sum(w x) = sum(w rx x)); (2) a warp two outputs of the gain and forget projections (R inputs) from the
/// summed, scaled partials, and the mix (1 - tanh f) x + tanh g msg.
constexpr int kCtlSplits = 8;
struct CtlArgs {
    const float *x[3], *msg[3], *wd[3], *wg[3], *wf[3];
    float* out[3];
};
__global__ void __launch_bounds__(256) k_ctl_down(CtlArgs a, float* h, int n, int R, float eps) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.y, j = blockIdx.z, T = gridDim.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int rg = blockIdx.x / kCtlSplits, sl = blockIdx.x % kCtlSplits;
    const int L = 2 * n / kCtlSplits, e0 = sl * L;   // this slice: [e0, e0 + L) of [x, msg] (inside one of them)
    const float* src = e0 < n ? a.x[j] + (size_t) t * n + e0 : a.msg[j] + (size_t) t * n + (e0 - n);
    float* part = h + ((size_t) j * T + t) * (kCtlSplits * R + kCtlSplits);   // [slice][R] partials, then [slice] squares
    if (rg == 0) {
        float sq = 0.f;
        for (int i = threadIdx.x; i < L; i += blockDim.x) sq += src[i] * src[i];
        sq = block_sum(sq, sh);
        if (threadIdx.x == 0) part[kCtlSplits * R + sl] = sq;
    }
    const int o = rg * 8 + warp;
    if (o >= R) return;
    const float4* w4 = reinterpret_cast<const float4*>(a.wd[j] + (size_t) o * 2 * n + e0);
    const float4* v4 = reinterpret_cast<const float4*>(src);
    float acc = 0.f;
    for (int i = lane; i < L / 4; i += 32) {
        const float4 w = w4[i], v = v4[i];
        acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) part[(size_t) sl * R + o] = acc;
}
__global__ void __launch_bounds__(256) k_ctl_up(CtlArgs a, const float* h, int n, int R, float eps) {
    const int t = blockIdx.y, j = blockIdx.z, T = gridDim.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int half = lane / 16, l16 = lane % 16;          // a half-warp a row: R / 4 float4 (R <= 64)
    const int i = (blockIdx.x * 8 + warp) * 2 + half;
    const float* part = h + ((size_t) j * T + t) * (kCtlSplits * R + kCtlSplits);
    float sx = 0.f, sm = 0.f;
#pragma unroll
    for (int k = 0; k < kCtlSplits / 2; ++k) {
        sx += part[kCtlSplits * R + k];
        sm += part[kCtlSplits * R + kCtlSplits / 2 + k];
    }
    const float rx = rsqrtf(sx / (float) n + eps), rm = rsqrtf(sm / (float) n + eps);
    float g = 0.f, f = 0.f;
    if (i < n && l16 < R / 4) {
        float4 hv = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < kCtlSplits; ++k) {
            const float4 p = reinterpret_cast<const float4*>(part + (size_t) k * R)[l16];
            const float r = k < kCtlSplits / 2 ? rx : rm;
            hv.x += p.x * r;
            hv.y += p.y * r;
            hv.z += p.z * r;
            hv.w += p.w * r;
        }
        const float4 wg = reinterpret_cast<const float4*>(a.wg[j] + (size_t) i * R)[l16];
        const float4 wf = reinterpret_cast<const float4*>(a.wf[j] + (size_t) i * R)[l16];
        g = wg.x * hv.x + wg.y * hv.y + wg.z * hv.z + wg.w * hv.w;
        f = wf.x * hv.x + wf.y * hv.y + wf.z * hv.z + wf.w * hv.w;
    }
    for (int m = 8; m > 0; m >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, m);
        f += __shfl_xor_sync(0xffffffffu, f, m);
    }
    if (i < n && l16 == 0) {
        const size_t o = (size_t) t * n + i;
        a.out[j][o] = (1.0f - tanhf(f)) * a.x[j][o] + tanhf(g) * a.msg[j][o];
    }
}

__global__ void k_ctrl(const float* cur, const float* msg, const float* g, const float* f, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.0f - tanhf(f[i])) * cur[i] + tanhf(g[i]) * msg[i];
}

__global__ void k_add(float* out, const float* a, const float* b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

struct States {
    const float* p[kMaxStates];
};

__global__ void k_mix(States st, int ns, const float* Q, int nq, float* out, int T, int n) {
    __shared__ float red[kMaxStates * 5];   // [0, ns): sum of squares; ns + q * ns + s: q . s (up to 4 queries)
    __shared__ float sc[4][kMaxStates];
    const size_t t = blockIdx.x;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    const int tasks = ns * (1 + nq);
    for (int k = warp; k < tasks; k += nw) {
        const int s = k < ns ? k : (k - ns) % ns;
        const float* xr = st.p[s] + t * n;
        const float* qr = k < ns ? xr : Q + (size_t) ((k - ns) / ns) * n;
        float a = 0.f;
        for (int i = lane * 4; i < n; i += 128) {
            const float4 x = *reinterpret_cast<const float4*>(xr + i), y = *reinterpret_cast<const float4*>(qr + i);
            a += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) red[k] = a;
    }
    __syncthreads();
    if (threadIdx.x < (unsigned) nq) {
        const int q = threadIdx.x;
        const float inv = rsqrtf((float) n);
        float m = -INFINITY;
        for (int s = 0; s < ns; ++s) {
            sc[q][s] = red[ns + q * ns + s] * rsqrtf(red[s] / (float) n + 1e-6f) * inv;
            m = fmaxf(m, sc[q][s]);
        }
        float sum = 0.f;
        for (int s = 0; s < ns; ++s) sum += (sc[q][s] = expf(sc[q][s] - m));
        for (int s = 0; s < ns; ++s) sc[q][s] = sc[q][s] / sum * rsqrtf(red[s] / (float) n + 1e-6f);
    }
    __syncthreads();
    // blockIdx.y: this block's slice of the output (each block works the scores out itself)
    const int sl = (n + gridDim.y - 1) / gridDim.y, i0 = blockIdx.y * sl, i1 = min(n, i0 + sl);
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x)
        for (int q = 0; q < nq; ++q) {
            float a = 0.f;
            for (int s = 0; s < ns; ++s) a += sc[q][s] * st.p[s][t * n + i];
            out[(size_t) q * T * n + t * n + i] = a;
        }
}

__global__ void k_conv(const float* x, const float* hist, const float* w, const float* dyn, int nh, float* out, int C,
                       int T, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) C * T) return;
    const int t = (int) (i / C), c = (int) (i % C);
    auto in = [&](int tt) { return tt >= 0 ? x[(size_t) tt * C + c] : hist[c * 3 + 3 + tt]; };
    const float x0 = x[(size_t) t * C + c], x1 = in(t - 1), x2 = in(t - 2), x3 = in(t - 3);
    float o = w[c * 4] * x3 + w[c * 4 + 1] * x2 + w[c * 4 + 2] * x1 + w[c * 4 + 3] * x0;
    if (dyn) {
        const float* tp = dyn + ((size_t) t * nh + c / (C / nh)) * 4;
        o += tp[0] * x0 + tp[1] * x1 + tp[2] * x2 + tp[3] * x3;
    }
    out[(size_t) t * C + c] = (x0 + o) * scale;
}
/// A decode's short convolutions in one launch (blockIdx.y: up to two of them), a thread a channel over the rows: the
/// input (or the experts' weighted sum, as k_combine), the conv (as k_conv), a residual added (as k_add), then the
/// history moved on (as k_conv_hist), or the input kept for a verify's commit.
struct ConvJob {
    const float *x = nullptr, *parts = nullptr, *wts = nullptr;
    int K = 0;
    float* hist = nullptr;
    const float *w = nullptr, *dyn = nullptr;
    int nh = 1;
    float* out = nullptr;
    const float* res = nullptr;
    float* save = nullptr;   // a verify: the inputs here, the history left as it is
    int C = 0;
    float scale = 1.f;
};
__global__ void k_conv_fused(ConvJob a, ConvJob b, int T) {
    const ConvJob& j = blockIdx.y ? b : a;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= j.C) return;
    const int C = j.C;
    float xv[kGemmT];
#pragma unroll
    for (int t = 0; t < kGemmT; ++t) {
        if (t >= T) break;
        float v;
        if (j.x) v = j.x[(size_t) t * C + c];
        else {
            v = 0.f;
            for (int k = 0; k < j.K; ++k) v += j.wts[t * j.K + k] * j.parts[((size_t) t * j.K + k) * C + c];
        }
        xv[t] = v;
    }
    const float h0 = j.hist[c * 3], h1 = j.hist[c * 3 + 1], h2 = j.hist[c * 3 + 2];
    auto in = [&](int tt) { return tt >= 0 ? xv[tt] : tt == -1 ? h2 : tt == -2 ? h1 : h0; };
    const float w0 = j.w[c * 4], w1 = j.w[c * 4 + 1], w2 = j.w[c * 4 + 2], w3 = j.w[c * 4 + 3];
#pragma unroll
    for (int t = 0; t < kGemmT; ++t) {
        if (t >= T) break;
        const float x0 = xv[t], x1 = in(t - 1), x2 = in(t - 2), x3 = in(t - 3);
        float o = w0 * x3 + w1 * x2 + w2 * x1 + w3 * x0;
        if (j.dyn) {
            const float* tp = j.dyn + ((size_t) t * j.nh + c / (C / j.nh)) * 4;
            o += tp[0] * x0 + tp[1] * x1 + tp[2] * x2 + tp[3] * x3;
        }
        float r = (x0 + o) * j.scale;
        if (j.res) r = r + j.res[(size_t) t * C + c];
        j.out[(size_t) t * C + c] = r;
    }
    if (j.save) {
        for (int t = 0; t < T; ++t) j.save[(size_t) t * C + c] = xv[t];
        return;
    }
    float h[3];
    for (int k = 0; k < 3; ++k) h[k] = in(T - 3 + k);
    for (int k = 0; k < 3; ++k) j.hist[c * 3 + k] = h[k];
}
__global__ void k_conv_hist(const float* x, float* hist, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3];
    for (int k = 0; k < 3; ++k) {
        const int tt = T - 3 + k;
        h[k] = tt >= 0 ? x[(size_t) tt * C + c] : hist[c * 3 + T + k];
    }
    for (int k = 0; k < 3; ++k) hist[c * 3 + k] = h[k];
}

struct Kv {
    void *k = nullptr, *v = nullptr;   // [kvh][cell][128]: FP16, or int8 with a scale per (kvh, cell)
    float *ks = nullptr, *vs = nullptr;
    int ring = 0, cells = 0;           // ring > 0: cell = position % ring; else cell = position
    int8_t *hk = nullptr, *hv = nullptr;   // tiered: every position in pinned host memory, [kvh][hcells][128]
    float *hks = nullptr, *hvs = nullptr;
    int hcells = 0;
    // split (two cards, tiered): the VRAM window's positions in 64-position blocks, every other one on the other card
    // (k2, v2, ks2, vs2 there), each half a ring of `cells` cells (kv_cell): the visit's attention reads both at once
    void *k2 = nullptr, *v2 = nullptr;
    float *ks2 = nullptr, *vs2 = nullptr;
    int split = 0, dev2 = -1;
};
constexpr int kSplitB = 64;   // positions a block of a split window
/// A split window's half for position p (0: the visit's card) and its index in that half's ring.
__host__ __device__ __forceinline__ int kv_part(int64_t p) { return (int) ((p / kSplitB) & 1); }
__host__ __device__ __forceinline__ int64_t kv_local(int64_t p) { return (p / (2 * kSplitB)) * kSplitB + p % kSplitB; }

template <bool Q8>
__global__ void k_prep(const float* Q, const float* Kx, const float* Vx, const float* R, const float* qclip,
                       const float* rclip, float* qo, float* ro, Kv c, int H, int HKV, int n_rel,
                       int pos0, int ring, int cells, float thr, float rthr, float eps, int global_, float alpha,
                       float floor_) {
    __shared__ float sh[kThreads];
    const int head = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int pos = pos0 + t;
    const float tau = global_ ? 1.0f + alpha * logf(fmaxf(1.0f, (float) (pos + 1) / floor_)) : 1.0f;
    if (head < H) {
        const size_t o = ((size_t) t * H + head) * kHD + i;
        const float x = Q[o] * qclip[head * kHD + i];
        const float nrm = sqrtf(block_sum(x * x, sh)) + 1e-6f;
        qo[o] = x * fminf(1.0f, fmaxf(0.0f, thr / nrm)) * tau;
        const float xr = i < n_rel ? R[((size_t) t * H + head) * n_rel + i] * rclip[head * n_rel + i] : 0.f;
        const float rn = sqrtf(block_sum(xr * xr, sh)) + 1e-6f;
        if (i < n_rel) ro[((size_t) t * H + head) * n_rel + i] = xr * fminf(1.0f, fmaxf(0.0f, rthr / rn)) * tau;
    } else {
        const int kh = head - H;
        const size_t o = ((size_t) t * HKV + kh) * kHD + i;
        const int part = c.split ? kv_part(pos) : 0;
        const int cell = c.split ? (int) (kv_local(pos) % cells) : ring > 0 ? pos % ring : pos;
        const size_t co = ((size_t) kh * cells + cell) * kHD + i;
        int8_t* kd8 = (int8_t*) (part ? c.k2 : c.k);
        int8_t* vd8 = (int8_t*) (part ? c.v2 : c.v);
        float* ksd = part ? c.ks2 : c.ks;
        float* vsd = part ? c.vs2 : c.vs;
        const float k = Kx[o];
        const float kn = sqrtf(block_sum(k * k, sh)) + 1e-6f;
        const float kq = k * fminf(1.0f, fmaxf(0.0f, thr / kn));
        const float v = Vx[o];
        const float vs = block_sum(v * v, sh);
        const float vq = v * rsqrtf(vs / (float) kHD + eps);
        if constexpr (Q8) {
            const size_t so = (size_t) kh * cells + cell;
            const float km = block_max(fabsf(kq), sh), vm = block_max(fabsf(vq), sh);
            const float ksc = km > 0.f ? km / 127.f : 1.f, vsc = vm > 0.f ? vm / 127.f : 1.f;
            const int8_t k8 = (int8_t) __float2int_rn(kq / ksc), v8 = (int8_t) __float2int_rn(vq / vsc);
            kd8[co] = k8;   // the other card's half: a store over PCIe (peer access)
            vd8[co] = v8;
            if (i == 0) {
                ksd[so] = ksc;
                vsd[so] = vsc;
            }
            if (c.hk) {
                const size_t ho = (size_t) kh * c.hcells + pos;
                c.hk[ho * kHD + i] = k8;
                c.hv[ho * kHD + i] = v8;
                if (i == 0) {
                    c.hks[ho] = ksc;
                    c.hvs[ho] = vsc;
                }
            }
        } else {
            ((__half*) c.k)[co] = __float2half(kq);
            ((__half*) c.v)[co] = __float2half(vq);
        }
    }
}

/// One store of a visit's cells for attention: the VRAM cache (ring or by position) or the host copy (by position).
struct KvSrc {
    const void *k, *v;
    const float *ks, *vs;
    int ring, cells, cbase;
    int split = 0, part = 0;   // a split window's half: its keys only (kv_part(key) == part)
};
__device__ __forceinline__ int src_cell(const KvSrc& s, int kp) {
    return s.split ? (int) (kv_local(kp) % s.cells) : s.ring > 0 ? kp % s.ring : kp - s.cbase;
}

/// Decode attention over keys [kmin, kmax] of one store, one pass (FlashDecoding): a block a (256-key chunk, KV head,
/// query row). A group of R lanes reads a key row (R x 16 bytes: int8 or FP16), K and V together; each lane holds its
/// 16-byte slice's query dims for the KV head's G query heads and a running softmax over the rows its group reads.
/// The groups merge at the end into the split-K partial of the chunk.
template <bool Q8, int G>
__global__ void __launch_bounds__(128) k_attn_fd(const float* q, const float* rr, KvSrc src, const float* rel_bias,
                                                 int n_rel, int extent, int pos0, int t0, int H, int win, int kmin,
                                                 int kmax, int base, int c_off, int nc_total, float scale, float* pacc,
                                                 float* pm, float* pl) {
    constexpr int R = 16;                    // lanes a key row
    constexpr int DL = 8;                    // dims a lane (8 bytes of int8, 16 of FP16)
    constexpr int LB = Q8 ? 8 : 16;          // a lane's bytes of a row
    constexpr int NG = 128 / R;              // groups (key rows read at once)
    __shared__ float sb[G][kChunkKeys];      // the bias of each key of the chunk (within the bias range)
    __shared__ float rs[G][16];
    __shared__ float gm[NG][G], gl[NG][G];
    __shared__ float gacc[NG * G * kHD];
    const int c = blockIdx.z, kvh = blockIdx.y, j = threadIdx.x;   // a chunk's rows next to each other: the rest from L2
    const int t = t0 + blockIdx.x, p = pos0 + t;
    const int grp = j / R, slot = j % R, d0 = slot * DL;
    const int lo = max(kmin, win > 0 ? max(0, p - win + 1) : 0), hi = min(p, kmax);
    const int k0 = base + c * kChunkKeys;
    const int ka = max(lo, k0), kb = min(hi, k0 + kChunkKeys - 1);
    float qv[G][DL];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int e = 0; e < DL; ++e) qv[g][e] = q[((size_t) t * H + kvh * G + g) * kHD + d0 + e] * scale;
    const bool biased = ka <= kb && p - kb < extent;   // keys of this chunk inside the bias range (its newest one is)
    if (biased) {
        if (j < G * 16) rs[j / 16][j % 16] = j % 16 < n_rel ? rr[((size_t) t * H + kvh * G + j / 16) * n_rel + j % 16] : 0.f;
        __syncthreads();
        for (int i = j; i < G * kChunkKeys; i += 128) {
            const int g = i / kChunkKeys, kk = i % kChunkKeys, d = p - (k0 + kk);
            float b = 0.f;
            if (d >= 0 && d < extent) {
                const float* br = rel_bias + (size_t) d * n_rel;
                for (int r = 0; r < n_rel; ++r) b += br[r] * rs[g][r];
            }
            sb[g][kk] = b;
        }
        __syncthreads();
    }
    float m[G], l[G], acc[G][DL];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int e = 0; e < DL; ++e) acc[g][e] = 0.f;
    }
    if (ka <= kb) {
#pragma unroll 4
        for (int kk = grp; kk < kChunkKeys; kk += NG) {
            const int kp = k0 + kk;
            const bool ok = kp >= ka && kp <= kb;
            const int kc = min(max(kp, ka), kb);   // a valid row to read either way
            const size_t ro = (size_t) kvh * src.cells + src_cell(src, kc);
            float kf[DL], vf[DL];
            if constexpr (Q8) {
                const uint2 kw = *reinterpret_cast<const uint2*>((const char*) src.k + ro * kHD + slot * LB);
                const uint2 vw = *reinterpret_cast<const uint2*>((const char*) src.v + ro * kHD + slot * LB);
                const float ksc = src.ks[ro], vsc = src.vs[ro];
                const uint32_t kx[2] = {kw.x, kw.y}, vx[2] = {vw.x, vw.y};
#pragma unroll
                for (int w = 0; w < 2; ++w)
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        kf[4 * w + b] = (float) (int8_t) (kx[w] >> (8 * b)) * ksc;
                        vf[4 * w + b] = (float) (int8_t) (vx[w] >> (8 * b)) * vsc;
                    }
            } else {
                const uint4 kw = *reinterpret_cast<const uint4*>((const char*) src.k + ro * 2 * kHD + slot * LB);
                const uint4 vw = *reinterpret_cast<const uint4*>((const char*) src.v + ro * 2 * kHD + slot * LB);
                const __half2* kh = reinterpret_cast<const __half2*>(&kw);
                const __half2* vh = reinterpret_cast<const __half2*>(&vw);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const float2 a = __half22float2(kh[w]), b = __half22float2(vh[w]);
                    kf[2 * w] = a.x; kf[2 * w + 1] = a.y;
                    vf[2 * w] = b.x; vf[2 * w + 1] = b.y;
                }
            }
            float sc[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float a = 0.f;
#pragma unroll
                for (int e = 0; e < DL; ++e) a += qv[g][e] * kf[e];
#pragma unroll
                for (int o = R / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                sc[g] = a + (biased ? sb[g][kk] : 0.f);
            }
            if (!ok) continue;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float mn = fmaxf(m[g], sc[g]);
                const float cr = expf(m[g] - mn), e = expf(sc[g] - mn);
                l[g] = l[g] * cr + e;
#pragma unroll
                for (int x = 0; x < DL; ++x) acc[g][x] = acc[g][x] * cr + e * vf[x];
                m[g] = mn;
            }
        }
    }
    // the groups' results into one: the max, then each group's share rescaled to it
    for (int g = 0; g < G; ++g) {
        if (slot == 0) {
            gm[grp][g] = m[g];
            gl[grp][g] = l[g];
        }
#pragma unroll
        for (int e = 0; e < DL; ++e) gacc[((size_t) grp * G + g) * kHD + d0 + e] = acc[g][e];
    }
    __syncthreads();
    for (int g = 0; g < G; ++g) {
        float M = -INFINITY;
        for (int i = 0; i < NG; ++i) M = fmaxf(M, gm[i][g]);
        float a = 0.f, L = 0.f;
        if (M != -INFINITY)
            for (int i = 0; i < NG; ++i) {
                const float w = gm[i][g] == -INFINITY ? 0.f : expf(gm[i][g] - M);
                a += w * gacc[((size_t) i * G + g) * kHD + j];
                L += w * gl[i][g];
            }
        const size_t o = ((size_t) blockIdx.x * H + kvh * G + g) * nc_total + c_off + c;
        pacc[o * kHD + j] = a;
        if (j == 0) {
            pm[o] = M;
            pl[o] = L;
        }
    }
}

__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, int nc, const float* sink,
                             const float* gate, int t0, int H, float* out) {
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const float sk = sink[(size_t) t * H + h];
    float M = sk;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * nc + c]);
    float num = 0.f, den = expf(sk - M);
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * nc + c;
        if (pm[o] == -INFINITY) continue;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * kHD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * kHD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

constexpr int kQT = 16, kKT = 32, kKs = kHD + 2;
size_t attn_tile_smem(int G, int n_rel) {
    return (size_t) kKT * kKs * 2 + (size_t) kKT * kHD * 2 + (size_t) kQT * G * kHD * 4 + (size_t) kQT * G * n_rel * 4 +
           (size_t) (kQT + kKT) * n_rel * 4;
}
__global__ void __launch_bounds__(256) k_attn_tile(const float* q, const float* rr, const __half* kc, const __half* vc,
                                                   const float* rel_bias, int n_rel, int pos0, int T, int H, int G,
                                                   int ring, int cells, int win, int extent, float scale,
                                                   const float* sink, const float* gate, float* out) {
    constexpr int RW = kQT * kMaxG / 8;   // rows per warp at most
    extern __shared__ __align__(16) unsigned char smem[];
    __half* Ks = reinterpret_cast<__half*>(smem);
    __half* Vs = Ks + kKT * kKs;
    float* qs = reinterpret_cast<float*>(Vs + kKT * kHD);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int t0 = blockIdx.x * kQT, kvh = blockIdx.y;
    const int nt = min(kQT, T - t0), rows = nt * G;
    float* rsm = qs + kQT * G * kHD;
    float* bs = rsm + kQT * G * n_rel;
    for (int i = threadIdx.x; i < rows * kHD; i += blockDim.x) {
        const int r = i / kHD, d = i % kHD;
        const int t = t0 + r / G, h = kvh * G + r % G;
        qs[r * kHD + d] = q[((size_t) t * H + h) * kHD + d];
    }
    for (int i = threadIdx.x; i < rows * n_rel; i += blockDim.x) {
        const int r = i / n_rel, j = i % n_rel;
        const int t = t0 + r / G, h = kvh * G + r % G;
        rsm[r * n_rel + j] = rr[((size_t) t * H + h) * n_rel + j];
    }
    const int p_lo = pos0 + t0, p_hi = pos0 + t0 + nt - 1;
    const int lo = win > 0 ? max(0, p_lo - win + 1) : 0;
    float acc[RW][4], m[RW], l[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }
    for (int kb = lo; kb <= p_hi; kb += kKT) {
        __syncthreads();
        for (int i = threadIdx.x; i < kKT * (kHD / 8); i += blockDim.x) {
            const int k = i / (kHD / 8), c8 = (i % (kHD / 8)) * 8;
            const int kp = kb + k;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (kp <= p_hi) {
                const int cell = ring > 0 ? kp % ring : kp;
                const size_t o = ((size_t) kvh * cells + cell) * kHD + c8;
                kv = *reinterpret_cast<const uint4*>(kc + o);
                vv = *reinterpret_cast<const uint4*>(vc + o);
            }
            const __half2* k2 = reinterpret_cast<const __half2*>(&kv);
            __half2* kd = reinterpret_cast<__half2*>(Ks + k * kKs + c8);
            kd[0] = k2[0]; kd[1] = k2[1]; kd[2] = k2[2]; kd[3] = k2[3];
            *reinterpret_cast<uint4*>(Vs + k * kHD + c8) = vv;
        }
        const int d_min = p_lo - kb - kKT + 1;
        for (int i = threadIdx.x; i < (kQT + kKT) * n_rel; i += blockDim.x) {
            const int j = i / n_rel, r = i % n_rel, d = d_min + j;
            bs[j * n_rel + r] = d >= 0 && d < extent ? rel_bias[(size_t) d * n_rel + r] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int ri = 0; ri < RW; ++ri) {
            const int r = warp + 8 * ri;
            if (r >= rows) break;
            const int p = pos0 + t0 + r / G;
            const int rlo = win > 0 ? max(0, p - win + 1) : 0;
            const int kp = kb + lane;
            float sc = -INFINITY;
            if (kp <= p && kp >= rlo) {
                const __half2* kr = reinterpret_cast<const __half2*>(Ks + lane * kKs);
                const float* qr = qs + r * kHD;
                float a = 0.f;
#pragma unroll 16
                for (int i = 0; i < kHD / 2; ++i) {
                    const float2 kf = __half22float2(kr[i]);
                    a += qr[2 * i] * kf.x + qr[2 * i + 1] * kf.y;
                }
                const int d = p - kp;
                float b = 0.f;
                if (d < extent) {
                    const float* br = bs + (d - d_min) * n_rel;
                    const float* rq = rsm + r * n_rel;
                    for (int j = 0; j < n_rel; ++j) b += br[j] * rq[j];
                }
                sc = a * scale + b;
            }
            float mt = sc;
            for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            const float mn = fmaxf(m[ri], mt);
            if (mn == -INFINITY) continue;
            const float corr = m[ri] == -INFINITY ? 0.f : expf(m[ri] - mn);
            const float e = sc == -INFINITY ? 0.f : expf(sc - mn);
            float es = e;
            for (int o = 16; o > 0; o >>= 1) es += __shfl_xor_sync(0xffffffffu, es, o);
            l[ri] = l[ri] * corr + es;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[ri][i] *= corr;
#pragma unroll 8
            for (int k = 0; k < kKT; ++k) {
                const float pk = __shfl_sync(0xffffffffu, e, k);
                const __half2* vr = reinterpret_cast<const __half2*>(Vs + k * kHD + lane * 4);
                const float2 v01 = __half22float2(vr[0]), v23 = __half22float2(vr[1]);
                acc[ri][0] += pk * v01.x;
                acc[ri][1] += pk * v01.y;
                acc[ri][2] += pk * v23.x;
                acc[ri][3] += pk * v23.y;
            }
            m[ri] = mn;
        }
    }
#pragma unroll
    for (int ri = 0; ri < RW; ++ri) {
        const int r = warp + 8 * ri;
        if (r >= rows) break;
        const int t = t0 + r / G, h = kvh * G + r % G;
        const float sk = sink[(size_t) t * H + h];
        const float M = fmaxf(m[ri], sk);
        const float w = m[ri] == -INFINITY ? 0.f : expf(m[ri] - M);
        const float den = l[ri] * w + expf(sk - M);
        const size_t o = ((size_t) t * H + h) * kHD + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) out[o + i] = acc[ri][i] * w / den * sigm(gate[o + i]);
    }
}

constexpr int kEM = 64, kEN = 128, kEK = 64, kELd = kEK + 8;   // rows x columns x K per step; the FP16 tiles' stride

__device__ __forceinline__ float ue4m3_f(uint32_t x) {
    const uint32_t e = (x >> 3) & 0xF, mn = x & 7;
    if (x == 0 || x == 0x7F) return 0.f;
    return e == 0 ? (float) mn * (1.0f / 1024.0f) : __uint_as_float(((e + 119) << 23) | (mn << 20));
}
/// NVFP4's K order in the prompt's expert GEMM: within each 16-value sub-block, value j and j + 8 (a byte's low and
/// high nibble) next to each other, so a weight byte is a half2 (a dot product's terms in any order: the same sums).
__host__ __device__ __forceinline__ int nv4_perm(int k) {
    const int w = k & 15;
    return (k & ~15) | (w < 8 ? 2 * w : 2 * (w - 8) + 1);
}
/// ue4m3_f as FP16, without branches: its bits shifted into FP16's are the value / 2^7, subnormals too (0x7F: 0).
__device__ __forceinline__ __half2 ue4m3_h2(uint32_t x) {
    const uint32_t b = x == 0x7Fu ? 0u : x << 7;
    return __hmul2(__halves2half2(__ushort_as_half((unsigned short) b), __ushort_as_half((unsigned short) b)), __float2half2_rn(128.f));
}
__device__ __forceinline__ uint32_t fp4_lut(uint32_t q) {
    const uint32_t idx = q & 0x07070707u, sp = idx | (idx >> 4);
    const uint32_t sel = (sp & 0xFFu) | ((sp >> 8) & 0xFF00u);
    const uint32_t lo = __byte_perm(0x03020100u, 0x0C080604u, sel), hi = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel);
    const uint32_t m8 = ((q & 0x08080808u) >> 3) * 0xFFu;
    return (hi & m8) | (lo & ~m8);
}
__device__ __forceinline__ __half2 i8x2_half2(uint32_t r, uint32_t sel, __half2 d) {
    uint32_t x;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(x) : "r"(r), "r"(sel));
    const uint32_t b = __vadd2(x, 0x66006600u);
    return __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b), __float2half2_rn(1536.f)), d);
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src),
                 "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

/// A 64-value K step's weight bytes a row: NVFP4 a block, Q8_0 two, Q4_K a quarter super-block (144 / 4: copied
/// whole), Q5_0 two blocks of 22.
template <int TY> __host__ __device__ constexpr int seg_bytes() { return TY == 40 || TY == 12 ? 36 : TY == 6 ? 44 : 68; }
/// Steps a raw weight copy of the warp-specialized GEMM: a Q4_K super-block (4), Q5_0 2 (its rows only 8-byte aligned,
/// and 4 steps of it would not fit beside the other buffers).
template <int TY> __host__ __device__ constexpr int ws_raw_steps() { return TY == 6 || TY == 8 ? 2 : 4; }
/// A row's bytes of such a copy: NVFP4 4 blocks, a Q4_K / Q5_K / Q6_K super-block, Q5_0 4 blocks, Q8_0 8 blocks.
template <int TY> __host__ __device__ constexpr int ws_raw_bytes() {
    return TY == 40 || TY == 12 ? 144 : TY == 13 ? 176 : TY == 14 ? 210 : TY == 6 ? 88 : 136;   // (Q8_0: 4 blocks)
}
template <int TY> constexpr size_t expert_smem() {
    return (size_t) 2 * kEM * kELd * 2 + (size_t) 2 * kEN * seg_bytes<TY>() + (size_t) kEN * kELd * 2;
}

template <int TY, bool GU>
__global__ void __launch_bounds__(256) k_expert_mma(const __half* X, const unsigned long long* grp_ptr,
                                                    const int32_t* grp_start, const int32_t* n_groups, size_t w_off,
                                                    size_t row_bytes, int K, int n_ff, size_t tail_off, float beta, float lb,
                                                    float* Y, __half* H, int ld_y) {
    using namespace nvcuda;
    constexpr int SEG = seg_bytes<TY>(), QK = TY == 40 ? 64 : 32, BB = TY == 40 ? 36 : 34;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int s0 = grp_start[g], n = grp_start[g + 1] - s0;
    const int c0 = blockIdx.x * (GU ? kEN / 2 : kEN), warp = threadIdx.x / 32, tid = threadIdx.x;
    const uint8_t* W = (const uint8_t*) grp_ptr[g] + w_off;
    const float* tail = tail_off ? (const float*) (grp_ptr[g] + tail_off) : nullptr;
    extern __shared__ __align__(128) unsigned char esm[];
    __half* As = reinterpret_cast<__half*>(esm);                               // [2][kEM * kELd]
    uint8_t* raw = esm + (size_t) 2 * kEM * kELd * 2;                          // [2][kEN * SEG]
    __half* Bs = reinterpret_cast<__half*>(raw + (size_t) 2 * kEN * SEG);      // [kEN * kELd]
    float* Cs = reinterpret_cast<float*>(esm);                                 // [kEM][kEN], after the K loop
    const int steps = K / kEK;
    for (int r0 = 0; r0 < n; r0 += kEM) {
        const int nr = min(kEM, n - r0);
        auto issue = [&](int st, int buf) {
            const int k0 = st * kEK;
            uint8_t* rd = raw + (size_t) buf * kEN * SEG;
            for (int i = tid; i < kEN * (SEG / 4); i += 256) {
                const int j = i / (SEG / 4), w = i % (SEG / 4);
                const int row = GU ? (j < kEN / 2 ? c0 + j : n_ff + c0 + j - kEN / 2) : c0 + j;
                cp_async4(rd + j * SEG + 4 * w, W + (size_t) row * row_bytes + (size_t) (k0 / QK) * BB + 4 * w);
            }
            __half* ad = As + (size_t) buf * kEM * kELd;
            for (int i = tid; i < kEM * (kEK / 8); i += 256) {
                const int r = i / (kEK / 8), c8 = (i % (kEK / 8)) * 8;
                const int rr = r < nr ? r : 0;
                cp_async16(ad + r * kELd + c8, X + (size_t) (s0 + r0 + rr) * K + k0 + c8, r < nr ? 16 : 0);
            }
            cp_commit();
        };
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kEM / 16];
#pragma unroll
        for (int i = 0; i < kEM / 16; ++i) wmma::fill_fragment(acc[i], 0.f);
        issue(0, 0);
        for (int st = 0; st < steps; ++st) {
            if (st + 1 < steps) {
                issue(st + 1, (st + 1) & 1);
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            const uint8_t* rs = raw + (size_t) (st & 1) * kEN * SEG;
#pragma unroll
            for (int it = 0; it < kEN * 8 / 256; ++it) {
                const int idx = it * 256 + tid;
                const int col = (idx % 8) + 8 * (idx / 64), ch = (idx / 8) % 8;
                const uint8_t* sg = rs + col * SEG;
                __half2 v[4];
                if constexpr (TY == 40) {
                    const int sb = ch / 2, sh = (ch & 1) * 4;
                    const __half2 d = __float2half2_rn(ue4m3_f(sg[sb]));
                    const uint32_t r0 = fp4_lut((*reinterpret_cast<const uint32_t*>(sg + 4 + sb * 8) >> sh) & 0x0F0F0F0Fu);
                    const uint32_t r1 = fp4_lut((*reinterpret_cast<const uint32_t*>(sg + 8 + sb * 8) >> sh) & 0x0F0F0F0Fu);
                    v[0] = i8x2_half2(r0, 0x9180, d);
                    v[1] = i8x2_half2(r0, 0xB3A2, d);
                    v[2] = i8x2_half2(r1, 0x9180, d);
                    v[3] = i8x2_half2(r1, 0xB3A2, d);
                } else {
                    const int bl = ch / 4, off = (ch % 4) * 8;
                    const uint8_t* bp = sg + bl * 34;
                    const float d = __half2float(__ushort_as_half((uint16_t) (bp[0] | (bp[1] << 8))));
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        v[q] = __floats2half2_rn(d * (float) (int8_t) bp[2 + off + 2 * q], d * (float) (int8_t) bp[3 + off + 2 * q]);
                }
                *reinterpret_cast<uint4*>(Bs + col * kELd + ch * 8) = *reinterpret_cast<const uint4*>(v);
            }
            __syncthreads();
            const __half* ac = As + (size_t) (st & 1) * kEM * kELd;
#pragma unroll
            for (int kk = 0; kk < kEK; kk += 16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, Bs + warp * 16 * kELd + kk, kELd);
#pragma unroll
                for (int i = 0; i < kEM / 16; ++i) {
                    if (i * 16 >= nr) break;
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
                    wmma::load_matrix_sync(af, ac + i * 16 * kELd + kk, kELd);
                    wmma::mma_sync(acc[i], af, bf, acc[i]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < kEM / 16; ++i)
            if (i * 16 < nr) wmma::store_matrix_sync(Cs + i * 16 * kEN + warp * 16, acc[i], kEN, wmma::mem_row_major);
        __syncthreads();
        if constexpr (GU) {
            const float sg = tail ? tail[0] : 1.f, su = tail ? tail[1] : 1.f;
            for (int i = tid; i < nr * (kEN / 2); i += 256) {
                const int r = i / (kEN / 2), j = i % (kEN / 2);
                const float gv = Cs[r * kEN + j] * sg, u0 = Cs[r * kEN + kEN / 2 + j] * su;
                const float u = lb > 0.f ? lb * tanhf(u0 / lb) : u0;
                H[(size_t) (s0 + r0 + r) * n_ff + c0 + j] = __float2half(beta * tanhf(gv / beta) / (1.0f + expf(-gv)) * u);
            }
        } else {
            const float sd = tail ? tail[2] : 1.f;
            for (int i = tid; i < nr * kEN; i += 256) {
                const int r = i / kEN, c = i % kEN;
                Y[(size_t) (s0 + r0 + r) * ld_y + c0 + c] = Cs[r * kEN + c] * sd;
            }
        }
        __syncthreads();
    }
}

constexpr int kTQ = 16, kTK = 32, kQLd = kHD + 8, kSLd = kTK + 8, kRLd = 24, kTLd = 52, kMRows = 48;
// ---- prompt attention, FlashAttention-2 style: one pass, the softmax rescaled in registers -------------------------
constexpr int kFaRows = 128, kFaKeys = 64, kFaLd = kHD + 8;   // a block's rows (query x head), a key tile, a smem row
constexpr int kFaWarps = kFaRows / 16;
// Q goes to registers before the first tile, so K and V take its place
constexpr size_t kFaQKV = (size_t) kFaRows * kFaLd * 2 > (size_t) 2 * kFaKeys * kFaLd * 2 ? (size_t) kFaRows * kFaLd * 2
                                                                                         : (size_t) 2 * kFaKeys * kFaLd * 2;
constexpr size_t kFaRaw = (size_t) 2 * kFaKeys * kHD + (size_t) 2 * kFaKeys * 4;   // an int8 tile as it is: K, V, scales
constexpr size_t kFaRawH = (size_t) 2 * kFaKeys * kFaLd * 2;   // an FP16 tile (K, V) in ldmatrix layout: its second buffer
                                                                  // (the first is Q's room, free once Q is in registers)
// the rows' relative queries (FP32, 16 a row), then each warp's bias table for half a key tile: 16 rows x 48 distances
constexpr int kFaBiasW = 48;
constexpr size_t kFaRel = (size_t) kFaRows * 16 * 4 > (size_t) kFaWarps * 16 * kFaBiasW * 4 ? (size_t) kFaRows * 16 * 4
                                                                                            : (size_t) kFaWarps * 16 * kFaBiasW * 4;
constexpr size_t kFaSmem = kFaQKV + kFaRel + (2 * kFaRaw > kFaRawH ? 2 * kFaRaw : kFaRawH);

__device__ __forceinline__ void cp_async(void* s, const void* g, int bytes, bool ok) {
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(s);
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(ok ? 16 : 0));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sa), "l"(g), "r"(ok ? 4 : 0));
}


__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
/// x as FP16 high and low parts (x = hi + lo to ~22 bits): three FP16 products give an FP32-accurate one
__device__ __forceinline__ void split_h2(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half ha = __float2half_rn(a), hb = __float2half_rn(b);
    hi = pack_h2(__half2float(ha), __half2float(hb));
    lo = pack_h2(a - __half2float(ha), b - __half2float(hb));
}

/// Attention of a prompt chunk's rows (row = token x the KV head's query heads) over the keys [kmin, kmax] of one store: the running (O, m, l) carried between calls (first: from the sink; last: the sink and the
/// gate applied, the result written). A block: 64 rows of one KV head, 4 warps of 16; K and V in 64-key tiles converted
/// to FP16 once a block; S = Q K^T, the online softmax and P V on the tensor cores, P never leaving the registers; the
/// relative bias only on tiles within its range.
template <bool Q8>
__global__ void __launch_bounds__(kFaWarps * 32) k_attn_fa2(const float* q, const float* rr, KvSrc src, const float* rel_bias, int n_rel,
                                                  int pos0, int T, int H, int G, int win, int extent, float scale,
                                                  const float* sink, const float* gate, float* out, int kmin, int kmax,
                                                  float* run_o, float* run_m, float* run_l, int first, int last) {
    extern __shared__ __align__(128) unsigned char fa_sm[];
    __half* Qs = reinterpret_cast<__half*>(fa_sm);
    __half* Ks = Qs;                     // after Q is in registers
    __half* Vs = Ks + kFaKeys * kFaLd;
    float* Rr = reinterpret_cast<float*>(fa_sm + kFaQKV);
    uint8_t* raw = fa_sm + kFaQKV + kFaRel;   // [2][K 64x128 | V 64x128 | K scales | V scales]
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, tq = lane % 4;
    const int kvh = blockIdx.y, R = T * G, r0 = blockIdx.x * kFaRows;
    for (int i = tid; i < kFaRows * (kHD / 4); i += kFaWarps * 32) {   // the rows' queries in FP16
        const int r = i / (kHD / 4), d = (i % (kHD / 4)) * 4, rg = r0 + r;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (rg < R) v = *reinterpret_cast<const float4*>(q + ((size_t) (rg / G) * H + kvh * G + rg % G) * kHD + d);
        __half2* o = reinterpret_cast<__half2*>(Qs + r * kFaLd + d);
        o[0] = __floats2half2_rn(v.x, v.y);
        o[1] = __floats2half2_rn(v.z, v.w);
    }
    for (int i = tid; i < kFaRows * 16; i += kFaWarps * 32) {
        const int r = i / 16, j = i % 16, rg = r0 + r;
        Rr[i] = rg < R && j < n_rel ? rr[((size_t) (rg / G) * H + kvh * G + rg % G) * n_rel + j] : 0.f;
    }
    __syncthreads();
    uint32_t qf[kHD / 16][4];   // this warp's 16 rows of Q as mma A fragments
#pragma unroll
    for (int ks = 0; ks < kHD / 16; ++ks)
        ldsm_x4(qf[ks], Qs + (warp * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * kFaLd + ks * 16 + (lane / 16) * 8);
    // the relative bias on the tensor cores: its rows' relative queries (16 x n_rel) as A fragments, high and low
    // parts; Rr's room becomes the warps' bias tables once every warp has them (the first tile's barrier)
    uint32_t rah[4], ral[4];
    {
        const float* rw = Rr + (size_t) warp * 16 * 16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = g + 8 * (i & 1), k = 2 * tq + 8 * (i >> 1);
            split_h2(rw[r * 16 + k], rw[r * 16 + k + 1], rah[i], ral[i]);
        }
    }
    float* Bw = Rr + (size_t) warp * 16 * kFaBiasW;
    const int rpw = pos0 + min(r0 + warp * 16, R - 1) / G;   // the warp's first row's position
    // the two rows this thread holds (g and g + 8 of its warp's 16)
    int rp[2], rlo[2], rhi[2];
    bool rok[2];
    size_t rrow[2];
    float m[2], l[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int rg = r0 + warp * 16 + g + 8 * h;
        rok[h] = rg < R;
        const int t = rok[h] ? rg / G : 0;
        rrow[h] = (size_t) t * H + kvh * G + (rok[h] ? rg % G : 0);
        rp[h] = pos0 + t;
        rlo[h] = max(kmin, win > 0 ? max(0, rp[h] - win + 1) : 0);
        rhi[h] = min(rp[h], kmax);
        m[h] = rok[h] ? (first ? sink[rrow[h]] : run_m[rrow[h]]) : 0.f;
        l[h] = rok[h] && !first && tq == 0 ? run_l[rrow[h]] : 0.f;
    }
    float o[kHD / 8][4];
#pragma unroll
    for (int dn = 0; dn < kHD / 8; ++dn)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = i / 2, d = dn * 8 + 2 * tq + (i % 2);
            o[dn][i] = !first && rok[h] ? run_o[rrow[h] * kHD + d] : 0.f;
        }
    // the block's key range
    const int rg_lo = r0, rg_hi = min(R, r0 + kFaRows) - 1;
    const int p_lo = pos0 + rg_lo / G, p_hi = pos0 + rg_hi / G;
    const int k_lo = max(kmin, win > 0 ? max(0, p_lo - win + 1) : 0), k_hi = min(p_hi, kmax);
    const float sl2 = scale * 1.4426950408889634f;   // exponentials in base 2
    constexpr int NT = kFaWarps * 32;
    auto fetch = [&](int kb, int buf) {   // an int8 tile's rows and scales into raw[buf], asynchronously
        uint8_t* rb = raw + (size_t) buf * kFaRaw;
        for (int i = tid; i < kFaKeys * (kHD / 16) * 2; i += NT) {
            const int which = i / (kFaKeys * (kHD / 16)), e = i % (kFaKeys * (kHD / 16));
            const int kk = e / (kHD / 16), c16 = (e % (kHD / 16)) * 16, kp = kb + kk;
            const bool ok = kp <= k_hi;
            const size_t ro = (size_t) kvh * src.cells + src_cell(src, ok ? kp : k_lo);
            cp_async(rb + (size_t) which * kFaKeys * kHD + (size_t) kk * kHD + c16,
                     (const int8_t*) (which ? src.v : src.k) + ro * kHD + c16, 16, ok);
        }
        for (int i = tid; i < 2 * kFaKeys; i += NT) {
            const int which = i / kFaKeys, kk = i % kFaKeys, kp = kb + kk;
            const bool ok = kp <= k_hi;
            const size_t ro = (size_t) kvh * src.cells + src_cell(src, ok ? kp : k_lo);
            cp_async(rb + (size_t) 2 * kFaKeys * kHD + (size_t) (which * kFaKeys + kk) * 4, (which ? src.vs : src.ks) + ro, 4, ok);
        }
    };
    auto hbuf = [&](int buf) { return buf ? reinterpret_cast<__half*>(raw) : reinterpret_cast<__half*>(fa_sm); };
    auto fetch_h = [&](int kb, int buf) {   // an FP16 tile straight into its buffer in ldmatrix layout, asynchronously
        __half* kd = hbuf(buf);
        __half* vd = kd + kFaKeys * kFaLd;
        for (int i = tid; i < kFaKeys * (kHD / 8) * 2; i += NT) {
            const int which = i / (kFaKeys * (kHD / 8)), e = i % (kFaKeys * (kHD / 8));
            const int kk = e / (kHD / 8), c8 = (e % (kHD / 8)) * 8, kp = kb + kk;
            const bool ok = kp <= k_hi;
            const size_t ro = (size_t) kvh * src.cells + src_cell(src, ok ? kp : k_lo);
            cp_async((which ? vd : kd) + kk * kFaLd + c8, (const __half*) (which ? src.v : src.k) + ro * kHD + c8, 16, ok);
        }
    };
    if constexpr (!Q8) __syncthreads();   // every warp has its Q fragments before the first tile takes Q's room
    // a split window's half: its own 64-key blocks only (tiles on block bounds; the keys below k_lo masked per row)
    const int kstep = src.split ? 2 * kFaKeys : kFaKeys;
    int kstart = k_lo;
    if (src.split) {
        kstart = k_lo / kFaKeys * kFaKeys;
        if (kv_part(kstart) != src.part) kstart += kFaKeys;
    }
    if (kstart <= k_hi) {
        if constexpr (Q8) fetch(kstart, 0);
        else fetch_h(kstart, 0);
    }
    cp_commit();
    int tile = 0;
    for (int kb = kstart; kb <= k_hi; kb += kstep, ++tile) {
        if (kb + kstep <= k_hi) {   // the next tile while this one runs
            if constexpr (Q8) fetch(kb + kstep, (tile + 1) & 1);
            else fetch_h(kb + kstep, (tile + 1) & 1);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if constexpr (Q8) {   // this tile from raw to FP16
            const uint8_t* rb = raw + (size_t) (tile & 1) * kFaRaw;
            const float* rsc = reinterpret_cast<const float*>(rb + (size_t) 2 * kFaKeys * kHD);
            for (int i = tid; i < kFaKeys * (kHD / 8) * 2; i += NT) {
                const int which = i / (kFaKeys * (kHD / 8)), e = i % (kFaKeys * (kHD / 8));
                const int kk = e / (kHD / 8), c8 = (e % (kHD / 8)) * 8;
                const uint2 b = *reinterpret_cast<const uint2*>(rb + (size_t) which * kFaKeys * kHD + (size_t) kk * kHD + c8);
                const float sc = kb + kk <= k_hi ? rsc[which * kFaKeys + kk] : 0.f;
                uint4 v;
                __half2* hh = reinterpret_cast<__half2*>(&v);
#pragma unroll
                for (int q2 = 0; q2 < 4; ++q2) {
                    const uint32_t w = q2 < 2 ? b.x : b.y;
                    const int sh = 16 * (q2 & 1);
                    hh[q2] = __floats2half2_rn((float) (int8_t) (w >> sh) * sc, (float) (int8_t) (w >> (sh + 8)) * sc);
                }
                *reinterpret_cast<uint4*>((which ? Vs : Ks) + kk * kFaLd + c8) = v;
            }
        }
        const __half* Kt = Ks;   // this tile's K and V: converted (int8) or as copied (FP16)
        const __half* Vt = Vs;
        if constexpr (!Q8) {
            Kt = hbuf(tile & 1);
            Vt = Kt + kFaKeys * kFaLd;
        }
        __syncthreads();
        float sc[kFaKeys / 8][4];
#pragma unroll
        for (int j = 0; j < kFaKeys / 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) sc[j][i] = 0.f;
#pragma unroll
        for (int ks = 0; ks < kHD / 16; ++ks)
#pragma unroll
            for (int j = 0; j < kFaKeys / 8; j += 2) {
                uint32_t b[4];
                ldsm_x4(b, Kt + (j * 8 + (lane % 8) + (lane / 16) * 8) * kFaLd + ks * 16 + ((lane / 8) % 2) * 8);
                mma16816(sc[j], qf[ks], b);
                mma16816(sc[j + 1], qf[ks], b + 2);
            }
        const bool near = p_lo - (kb + kFaKeys - 1) < extent;   // some key of the tile within the bias range
#pragma unroll
        for (int j = 0; j < kFaKeys / 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int h = i / 2, kp = kb + j * 8 + 2 * tq + (i % 2);
                sc[j][i] = rok[h] && kp >= rlo[h] && kp <= rhi[h] ? sc[j][i] * sl2 : -INFINITY;
            }
        if (near) {   // each half tile: B(16 rows x 48 distances) = R (16 x n_rel) . bias table^T, then by distance
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
                const int dbase = rpw - (kb + hb * 32 + 31);   // the table's column 0: the farthest key from the first row
#pragma unroll
                for (int nt = 0; nt < kFaBiasW / 8; ++nt) {
                    const int d = dbase + nt * 8 + g;
                    float2 b0 = make_float2(0.f, 0.f), b1 = make_float2(0.f, 0.f);
                    if (d >= 0 && d < extent) {
                        const float* br = rel_bias + (size_t) d * n_rel;
                        if (2 * tq + 1 < n_rel) b0 = *reinterpret_cast<const float2*>(br + 2 * tq);
                        if (2 * tq + 9 < n_rel) b1 = *reinterpret_cast<const float2*>(br + 2 * tq + 8);
                    }
                    uint32_t bh[2], bl[2];
                    split_h2(b0.x, b0.y, bh[0], bl[0]);
                    split_h2(b1.x, b1.y, bh[1], bl[1]);
                    float c[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(c, rah, bh);
                    mma16816(c, rah, bl);
                    mma16816(c, ral, bh);
                    Bw[g * kFaBiasW + nt * 8 + 2 * tq] = c[0];
                    Bw[g * kFaBiasW + nt * 8 + 2 * tq + 1] = c[1];
                    Bw[(g + 8) * kFaBiasW + nt * 8 + 2 * tq] = c[2];
                    Bw[(g + 8) * kFaBiasW + nt * 8 + 2 * tq + 1] = c[3];
                }
                __syncwarp();
#pragma unroll
                for (int j = hb * 4; j < hb * 4 + 4; ++j)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int h = i / 2, kp = kb + j * 8 + 2 * tq + (i % 2), d = rp[h] - kp;
                        if (sc[j][i] != -INFINITY && d < extent)
                            sc[j][i] += Bw[(g + 8 * h) * kFaBiasW + (d - dbase)] * 1.4426950408889634f;
                    }
                __syncwarp();
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < kFaKeys / 8; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) mx[i / 2] = fmaxf(mx[i / 2], sc[j][i]);
        float corr[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
            const float mo = m[h] * 1.4426950408889634f, mn = fmaxf(mo, mx[h]);   // m is kept in natural units
            corr[h] = exp2f(mo - mn);
            m[h] = mn * 0.6931471805599453f;
            l[h] *= corr[h];
        }
#pragma unroll
        for (int dn = 0; dn < kHD / 8; ++dn) {
            o[dn][0] *= corr[0];
            o[dn][1] *= corr[0];
            o[dn][2] *= corr[1];
            o[dn][3] *= corr[1];
        }
        uint32_t pa[kFaKeys / 16][4];
#pragma unroll
        for (int j = 0; j < kFaKeys / 8; ++j) {
            float e[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int h = i / 2;
                e[i] = sc[j][i] == -INFINITY ? 0.f : exp2f(sc[j][i] - m[h] * 1.4426950408889634f);
                l[h] += e[i];
            }
            pa[j / 2][(j % 2) * 2 + 0] = pack_h2(e[0], e[1]);
            pa[j / 2][(j % 2) * 2 + 1] = pack_h2(e[2], e[3]);
        }
#pragma unroll
        for (int kc = 0; kc < kFaKeys / 16; ++kc)
#pragma unroll
            for (int dn = 0; dn < kHD / 8; dn += 2) {
                uint32_t b[4];
                ldsm_x4_t(b, Vt + (kc * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * kFaLd + dn * 8 + (lane / 16) * 8);
                const uint32_t a[4] = {pa[kc][0], pa[kc][1], pa[kc][2], pa[kc][3]};
                mma16816(o[dn], a, b);
                mma16816(o[dn + 1], a, b + 2);
            }
        if constexpr (!Q8) __syncthreads();   // every warp is done with this raw buffer before the next copy into it
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
    }
#pragma unroll
    for (int dn = 0; dn < kHD / 8; ++dn)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = i / 2, d = dn * 8 + 2 * tq + (i % 2);
            if (!rok[h]) continue;
            const size_t oi = rrow[h] * kHD + d;
            if (last) out[oi] = o[dn][i] / (l[h] + expf(sink[rrow[h]] - m[h])) * sigm(gate[oi]);
            else run_o[oi] = o[dn][i];
        }
    if (!last && tq == 0)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            if (rok[h]) {
                run_m[rrow[h]] = m[h];
                run_l[rrow[h]] = l[h];
            }
}


/// Decode attention on the tensor cores, split-K, straight from global memory: a block takes up to 16 rows (query x
/// the KV head's query heads, so up to 5 tokens; more rows: more blocks in x) over `dk` keys from `base + c * dk`, its 4
/// warps a quarter each, 16 keys a step. A dot product's terms and P V's keys can go in any order, so the head dims
/// are ordered for S = Q K^T and the dims of O for P V such that every thread reads whole 16-byte pieces of its rows
/// into its mma fragments: no shared memory but the merge, many warps an SM. Int8 goes to FP16 exactly (byte
/// permutes), its per-key scales in FP32 (K's on the scores, V's on P). The warps' softmaxes merge into the block's
/// split-K partial (O, m, l) for k_attn_merge.
__device__ __forceinline__ uint32_t i8x2_h2(uint32_t t) {   // bytes 0 and 2 of t (offset by 128) as two halves
    const uint32_t h = (t & 0x00ff00ffu) | 0x64006400u;
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&h), __float2half2_rn(1152.f));
    return *reinterpret_cast<const uint32_t*>(&r);
}
template <bool Q8>
__global__ void __launch_bounds__(128) k_attn_dreg(const float* q, const float* rr, KvSrc src, const float* rel_bias,
                                                  int n_rel, int extent, int pos0, int t0, int nq, int H, int G, int win,
                                                  int kmin, int kmax, int base, int dk, int c_off, int nc_total, float scale,
                                                  float* pacc, float* pm, float* pl) {
    __shared__ float Rr[16 * 16];
    extern __shared__ __align__(16) float dr_sm[];   // the merge: [warp][16][kHD], then m, l
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, tq = lane % 4;
    const int kvh = blockIdx.y, c = blockIdx.z, R = nq * G, r0 = blockIdx.x * 16;
    for (int i = tid; i < 16 * 16; i += 128) {
        const int r = i / 16, j = i % 16, rg = r0 + r;
        Rr[i] = rg < R && j < n_rel ? rr[((size_t) (t0 + rg / G) * H + kvh * G + rg % G) * n_rel + j] : 0.f;
    }
    int rp[2], rlo[2], rhi[2];
    bool rok[2];
    size_t qrow[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int rg = r0 + g + 8 * h;
        rok[h] = rg < R;
        const int t = rok[h] ? rg / G : 0;
        qrow[h] = (size_t) (t0 + t) * H + kvh * G + (rok[h] ? rg % G : 0);
        rp[h] = pos0 + t0 + t;
        rlo[h] = max(kmin, win > 0 ? max(0, rp[h] - win + 1) : 0);
        rhi[h] = min(rp[h], kmax);
    }
    // Q's A fragments: k-step ks, thread tq: dims tq * 32 + ks * 4 + {0, 1} (a0, a1) and {2, 3} (a2, a3)
    uint32_t qf[kHD / 16][4];
#pragma unroll
    for (int ks = 0; ks < kHD / 16; ++ks) {
        float4 v[2] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
#pragma unroll
        for (int h = 0; h < 2; ++h)
            if (rok[h]) v[h] = *reinterpret_cast<const float4*>(q + qrow[h] * kHD + tq * 32 + ks * 4);
        qf[ks][0] = pack_h2(v[0].x, v[0].y);
        qf[ks][1] = pack_h2(v[1].x, v[1].y);
        qf[ks][2] = pack_h2(v[0].z, v[0].w);
        qf[ks][3] = pack_h2(v[1].z, v[1].w);
    }
    __syncthreads();
    const int rg_hi = min(R, r0 + 16) - 1;
    const int p_lo = pos0 + t0 + r0 / G, p_hi = pos0 + t0 + rg_hi / G;
    const int b_lo = max(kmin, win > 0 ? max(0, p_lo - win + 1) : 0), b_hi = min(p_hi, kmax);
    const int wk = dk / 4;
    const int k_lo = max(b_lo, base + c * dk + warp * wk), k_hi = min(b_hi, base + c * dk + (warp + 1) * wk - 1);
    const float sl2 = scale * 1.4426950408889634f;
    constexpr int RB = Q8 ? kHD : 2 * kHD;   // a row's bytes
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[kHD / 8][4];   // n-tile dn, column n: dim n * 16 + dn
#pragma unroll
    for (int dn = 0; dn < kHD / 8; ++dn)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[dn][i] = 0.f;
    auto row_of = [&](int kp) { return (size_t) kvh * src.cells + src_cell(src, min(max(kp, 0), k_hi)); };
    // 16-key steps from a multiple of 16 (a step inside one block of a split window); a split half skips the other's
    for (int kb = k_lo & ~15; kb <= k_hi; kb += 16) {
        if (src.split && kv_part(kb) != src.part) {
            kb = (kb / kSplitB + 1) * kSplitB - 16;
            continue;
        }
        // K: rows kb + j * 8 + g, this thread's dims tq * 32 .. + 31
        uint4 kw[2][Q8 ? 2 : 4];
        float ksc[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const char* kr = (const char*) src.k + row_of(kb + j * 8 + g) * RB + tq * (RB / 4);
#pragma unroll
            for (int x = 0; x < (Q8 ? 2 : 4); ++x) kw[j][x] = *reinterpret_cast<const uint4*>(kr + x * 16);
            if constexpr (Q8)
#pragma unroll
                for (int i = 0; i < 2; ++i) ksc[j][i] = src.ks[row_of(kb + j * 8 + tq * 2 + i)];
        }
        // V: rows kb + {tq*2, tq*2+1, tq*2+8, tq*2+9}, dims g * 16 .. + 15
        uint4 vw[4][Q8 ? 1 : 2];
        float vsc[4] = {1.f, 1.f, 1.f, 1.f};
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int kk = tq * 2 + (x & 1) + (x >> 1) * 8;
            const char* vr = (const char*) src.v + row_of(kb + kk) * RB + g * (RB / 8);
#pragma unroll
            for (int y = 0; y < (Q8 ? 1 : 2); ++y) vw[x][y] = *reinterpret_cast<const uint4*>(vr + y * 16);
            if constexpr (Q8) vsc[x] = src.vs[row_of(kb + kk)];
        }
        float sc[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) sc[j][i] = 0.f;
            const uint32_t* w = reinterpret_cast<const uint32_t*>(kw[j]);
#pragma unroll
            for (int ks = 0; ks < kHD / 16; ++ks) {
                uint32_t bb[2];
                if constexpr (Q8) {
                    const uint32_t x = w[ks] ^ 0x80808080u;
                    bb[0] = i8x2_h2(__byte_perm(x, 0, 0x7170));
                    bb[1] = i8x2_h2(__byte_perm(x, 0, 0x7372));
                } else {
                    bb[0] = w[2 * ks];
                    bb[1] = w[2 * ks + 1];
                }
                mma16816(sc[j], qf[ks], bb);
            }
        }
        const bool near = p_lo - (kb + 15) < extent;
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int h = i / 2, kp = kb + j * 8 + 2 * tq + (i % 2);
                float v = -INFINITY;
                if (rok[h] && kp >= rlo[h] && kp <= rhi[h] && kp >= k_lo && kp <= k_hi) {
                    v = sc[j][i] * (Q8 ? ksc[j][i % 2] : 1.f) * sl2;
                    if (near) {
                        const int d = rp[h] - kp;
                        if (d < extent) {
                            const float* br = rel_bias + (size_t) d * n_rel;
                            const float* qr = Rr + (g + 8 * h) * 16;
                            float bs = 0.f;
                            for (int x2 = 0; x2 < n_rel; ++x2) bs += br[x2] * qr[x2];
                            v += bs * 1.4426950408889634f;
                        }
                    }
                }
                sc[j][i] = v;
                mx[h] = fmaxf(mx[h], v);
            }
        float corr[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
            const float mo = m[h] * 1.4426950408889634f, mn = fmaxf(mo, mx[h]);
            corr[h] = mn == -INFINITY ? 1.f : exp2f(mo - mn);
            m[h] = mn * 0.6931471805599453f;
            l[h] *= corr[h];
        }
#pragma unroll
        for (int dn = 0; dn < kHD / 8; ++dn) {
            o[dn][0] *= corr[0];
            o[dn][1] *= corr[0];
            o[dn][2] *= corr[1];
            o[dn][3] *= corr[1];
        }
        // P (V's scales in it) as the A fragment: keys tq*2, +1 (a0, a1) and tq*2+8, +9 (a2, a3)
        uint32_t pa[4];
        {
            float e[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int h = i / 2;
                    e[j][i] = sc[j][i] == -INFINITY ? 0.f : exp2f(sc[j][i] - m[h] * 1.4426950408889634f);
                    l[h] += e[j][i];
                }
            pa[0] = pack_h2(e[0][0] * vsc[0], e[0][1] * vsc[1]);
            pa[1] = pack_h2(e[0][2] * vsc[0], e[0][3] * vsc[1]);
            pa[2] = pack_h2(e[1][0] * vsc[2], e[1][1] * vsc[3]);
            pa[3] = pack_h2(e[1][2] * vsc[2], e[1][3] * vsc[3]);
        }
        // O += P V: n-tile dn, this thread (column g) dim g * 16 + dn
        if constexpr (Q8) {
            uint32_t vx[4][4];
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(&vw[x][0]);
#pragma unroll
                for (int y = 0; y < 4; ++y) vx[x][y] = w[y] ^ 0x80808080u;
            }
#pragma unroll
            for (int dn = 0; dn < kHD / 8; ++dn) {
                const int wi = dn / 4, bi = dn % 4;
                const uint32_t sel = (uint32_t) (bi | (bi << 4) | ((4 + bi) << 8) | ((4 + bi) << 12));
                uint32_t bb[2];
                bb[0] = i8x2_h2(__byte_perm(vx[0][wi], vx[1][wi], sel));
                bb[1] = i8x2_h2(__byte_perm(vx[2][wi], vx[3][wi], sel));
                mma16816(o[dn], pa, bb);
            }
        } else {
#pragma unroll
            for (int dn = 0; dn < kHD / 8; ++dn) {
                const int wi = dn / 2, hi = dn % 2;
                const uint32_t sel = hi ? 0x7632u : 0x5410u;
                const uint32_t* w0 = reinterpret_cast<const uint32_t*>(vw[0]);
                const uint32_t* w1 = reinterpret_cast<const uint32_t*>(vw[1]);
                const uint32_t* w2 = reinterpret_cast<const uint32_t*>(vw[2]);
                const uint32_t* w3 = reinterpret_cast<const uint32_t*>(vw[3]);
                uint32_t bb[2];
                bb[0] = __byte_perm(w0[wi], w1[wi], sel);
                bb[1] = __byte_perm(w2[wi], w3[wi], sel);
                mma16816(o[dn], pa, bb);
            }
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
    }
    float* sm_o = dr_sm;
    float* sm_m = sm_o + (size_t) 4 * 16 * kHD;
    float* sm_l = sm_m + 4 * 16;
#pragma unroll
    for (int dn = 0; dn < kHD / 8; ++dn)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = i / 2, d = (2 * tq + (i % 2)) * 16 + dn;
            sm_o[((size_t) warp * 16 + g + 8 * h) * kHD + d] = o[dn][i];
        }
    if (tq == 0)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            sm_m[warp * 16 + g + 8 * h] = m[h];
            sm_l[warp * 16 + g + 8 * h] = l[h];
        }
    __syncthreads();
    for (int i = tid; i < 16 * kHD; i += 128) {
        const int r = i / kHD, d = i % kHD, rg = r0 + r;
        if (rg >= R) continue;
        float M = -INFINITY;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, sm_m[w * 16 + r]);
        float a = 0.f, L = 0.f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; ++w) {
                const float mw = sm_m[w * 16 + r];
                if (mw == -INFINITY) continue;
                const float s2 = expf(mw - M);
                a += s2 * sm_o[((size_t) w * 16 + r) * kHD + d];
                L += s2 * sm_l[w * 16 + r];
            }
        const size_t oi = ((size_t) (rg / G) * H + kvh * G + rg % G) * nc_total + c_off + c;
        pacc[oi * kHD + d] = a;
        if (d == 0) {
            pm[oi] = M;
            pl[oi] = L;
        }
    }
}
/// A split window's two running results (the visit's card's, the other card's copied here) into the rows' output:
/// the sink and the gate as the prompt kernel's last call applies them.
__global__ void k_attn_join(const float* o1, const float* m1, const float* l1, const float* o2, const float* m2, const float* l2,
                            const float* sink, const float* gate, float* out) {
    const int row = blockIdx.x, d = threadIdx.x;
    const float a = m1[row], b = m2[row], M = fmaxf(fmaxf(a, b), sink[row]);
    const float wa = expf(a - M), wb = expf(b - M);
    const size_t i = (size_t) row * kHD + d;
    out[i] = (o1[i] * wa + o2[i] * wb) / (l1[row] * wa + l2[row] * wb + expf(sink[row] - M)) * sigm(gate[i]);
}
/// The partials [0, nc) of each row merged into one (O, m, l), written to slot `slot` of nc_out a row in (po, pm2,
/// pl2): the other card's half of a split window, into the visit's card's partials.
__global__ void k_attn_fold(const float* pacc, const float* pm, const float* pl, int nc, float* po, float* pmo, float* plo,
                            int nc_out, int slot) {
    const int row = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * nc + c]);
    float a = 0.f, L = 0.f;
    if (M != -INFINITY)
        for (int c = 0; c < nc; ++c) {
            const size_t o = (size_t) row * nc + c;
            if (pm[o] == -INFINITY) continue;
            const float w = expf(pm[o] - M);
            a += w * pacc[o * kHD + d];
            L += w * pl[o];
        }
    const size_t oo = (size_t) row * nc_out + slot;
    po[oo * kHD + d] = a;
    if (d == 0) {
        pmo[oo] = M;
        plo[oo] = L;
    }
}
constexpr size_t kDrSmem = (size_t) 4 * 16 * kHD * 4 + 2 * 4 * 16 * 4;


// ---- DSpark (--dspark FILE): Inkling 2's own block drafter (Unsloth's dspark GGUF, arch dflash). Three layers read
// the target's hidden states at four visits (injected into the drafter's own cache as each token is committed) and
// draft a block from [last token, mask x 7] in one pass; a recurrent head turns the block's rows into tokens through
// the target's output head. Verification keeps the reply exactly as without it (see spec_step).
struct DsMix {
    float w[4][4];          // softmax of each fuse row: [layer 0..2, head context][feature]
    const float* norm[4];   // dflare_norm of each layer, dspark_norm
};
/// FP32 rows into FP16 rows `pitch` values apart (a feature slot of [row][4][N]).
__global__ void k_f2h_rows(const float* x, __half* y, int n, size_t pitch) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[(size_t) t * pitch + i] = __float2half(x[(size_t) t * n + i]);
}
/// out[t] = RMS-normed (sum_k w[j][k] feat[t][k]) * norm[j]: drafter layer j's context rows (j = 3: the head's).
__global__ void k_ds_mix(const __half* feat, int n, DsMix a, int j, float eps, float* out) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x;
    float* o = out + (size_t) t * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = 0.f;
        for (int k = 0; k < 4; ++k) x += a.w[j][k] * __half2float(feat[((size_t) t * 4 + k) * n + i]);
        o[i] = x;
        ss += x * x;
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = o[i] * r * a.norm[j][i];
}
/// A head's 128 values RMS-normed (times w), then NeoX-rotated at position pos (x[i], x[i + 64] a pair).
__device__ __forceinline__ float ds_norm_rope(const float* x, const float* w, int pos, float base, float eps, float* sh) {
    const int i = threadIdx.x;
    const float v = x[i];
    sh[i] = v * v;
    __syncthreads();
    for (int s2 = 64; s2 > 0; s2 >>= 1) {
        if (i < s2) sh[i] += sh[i + s2];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) kHD + eps);
    __syncthreads();
    sh[i] = v * r * w[i];
    __syncthreads();
    const int pi = i % 64;
    const float th = (float) pos * powf(base, -2.0f * (float) pi / (float) kHD);
    float sn, cs;
    sincosf(th, &sn, &cs);
    const float a = sh[pi], b = sh[pi + 64];
    const float o = i < 64 ? a * cs - b * sn : b * cs + a * sn;
    __syncthreads();
    return o;
}
/// Rows' K (normed, rotated) and V into a drafter cache ring (FP16, [head][cell][128], cell = position % ring).
__global__ void k_ds_kput(const float* K, const float* V, int pos0, const float* knorm, int ring, __half* kc, __half* vc,
                          float eps, float base, int HKV) {
    __shared__ float sh[kHD];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x, pos = pos0 + t;
    const size_t src = ((size_t) t * HKV + h) * kHD;
    const float k = ds_norm_rope(K + src, knorm, pos, base, eps, sh);
    const size_t cell = ((size_t) h * ring + (size_t) (pos % ring)) * kHD + i;
    kc[cell] = __float2half(k);
    vc[cell] = __float2half(V[src + i]);
}
/// The block's queries (normed, rotated) laid out for k_attn_dreg as one token's heads: row (kvh, r * G + j) is block
/// row r's head kvh * G + j, so all 8 rows x G heads of a KV head read its keys together.
__global__ void k_ds_qprep(const float* Q, int pos0, const float* qnorm, float* qd, float eps, float base, int G, int HKV) {
    __shared__ float sh[kHD];
    const int r = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const float q = ds_norm_rope(Q + ((size_t) r * HKV * G + h) * kHD, qnorm, pos0 + r, base, eps, sh);
    const int kvh = h / G, j = h % G;
    qd[((size_t) kvh * (gridDim.x * G) + r * G + j) * kHD + i] = q;
}
/// k_attn_dreg's partials of those rows merged (no sink, no gate) back into [row][head][128].
__global__ void k_ds_merge(const float* pacc, const float* pm, const float* pl, int nc, float* out, int nrows, int G) {
    const int row = blockIdx.x, d = threadIdx.x;   // row = kvh * (nrows * G) + r * G + j
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * nc + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * nc + c;
        if (pm[o] == -INFINITY) continue;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * kHD + d];
        den += w * pl[o];
    }
    const int per = nrows * G, kvh = row / per, r = (row % per) / G, j = row % G;
    const int HKV = gridDim.x / per;
    out[((size_t) r * HKV * G + kvh * G + j) * kHD + d] = den > 0.f ? num / den : 0.f;
}
__global__ void k_silu_mul(const float* g, const float* u, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}
/// The head's input: [state, markov_w1[prev], the block row, the committed context].
__global__ void k_ds_head_in(const uint16_t* markov, const int32_t* prev, const float* state, const float* row, const float* ctx,
                             float* x, int rank, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, tot = 2 * rank + 2 * n;
    if (i >= tot) return;
    float v;
    if (i < rank) v = state[i];
    else if (i < 2 * rank) v = __uint_as_float((uint32_t) markov[(size_t) prev[0] * rank + (i - rank)] << 16);
    else if (i < 2 * rank + n) v = row[i - 2 * rank];
    else v = ctx[i - 2 * rank - n];
    x[i] = v;
}
/// joint = joint_proj x + b: gate, candidate and output thirds; the state on, the latent out.
__global__ void k_ds_state(const float* j, const float* jb, float* state, float* latent, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    const float g = 1.0f / (1.0f + expf(-(j[i] + jb[i])));
    const float c = tanhf(j[rank + i] + jb[rank + i]);
    state[i] = c + g * (state[i] - c);
    latent[i] = tanhf(j[2 * rank + i] + jb[2 * rank + i]);
}
constexpr int kDsCand = 256;   // a draft row's candidates: the top of the target head's logits of the row
/// Each row's kDsCand largest values (index, value; in no order): a radix select on the float's ordered bits, 11, 11
/// and 10 bits a pass, each pass's histogram built by many blocks (k_tk_hist) and read by one (k_tk_pick), then the
/// ones above the threshold and enough of the ones equal to it gathered (k_tk_take). st: [row] {prefix, mask, need, gt, eq}.
__device__ __forceinline__ unsigned tk_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
constexpr int kTkBins = 2048, kTkBlocks = 48;
__global__ void k_tk_hist(const float* x, int n, int ld, const unsigned* st, unsigned* hist, int shift, int bits) {
    __shared__ unsigned h[kTkBins];
    const int row = blockIdx.y;
    for (int i = threadIdx.x; i < (1 << bits); i += blockDim.x) h[i] = 0;
    __syncthreads();
    const unsigned prefix = st[row * 5], mask = st[row * 5 + 1], dm = (1u << bits) - 1;
    const float* r = x + (size_t) row * ld;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned k = tk_key(r[i]);
        if ((k & mask) == prefix) atomicAdd(&h[(k >> shift) & dm], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (1 << bits); i += blockDim.x)
        if (h[i]) atomicAdd(&hist[(size_t) row * kTkBins + i], h[i]);
}
__global__ void k_tk_pick(unsigned* st, unsigned* hist, int shift, int bits) {
    const int row = blockIdx.x;
    unsigned* hr = hist + (size_t) row * kTkBins;
    if (threadIdx.x == 0) {
        unsigned need = st[row * 5 + 2], cum = 0;
        int b = (1 << bits) - 1;
        for (; b > 0; --b) {
            if (cum + hr[b] >= need) break;
            cum += hr[b];
        }
        st[row * 5] |= (unsigned) b << shift;
        st[row * 5 + 1] |= ((1u << bits) - 1) << shift;
        st[row * 5 + 2] = need - cum;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kTkBins; i += blockDim.x) hr[i] = 0;   // the next pass's
}
__global__ void k_tk_init(unsigned* st, int rows) {
    const int r = threadIdx.x;
    if (r < rows) {
        st[r * 5] = 0;
        st[r * 5 + 1] = 0;
        st[r * 5 + 2] = kDsCand;
        st[r * 5 + 3] = 0;
        st[r * 5 + 4] = 0;
    }
}
__global__ void k_tk_take(const float* x, int n, int ld, unsigned* st, int32_t* oi, float* ov) {
    const int row = blockIdx.y;
    const unsigned thr = st[row * 5], need_eq = st[row * 5 + 2];
    const unsigned n_gt = kDsCand - need_eq;
    const float* r = x + (size_t) row * ld;
    int32_t* o = oi + (size_t) row * kDsCand;
    float* v = ov + (size_t) row * kDsCand;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned k = tk_key(r[i]);
        if (k > thr) {
            const unsigned p = atomicAdd(&st[row * 5 + 3], 1u);
            if (p < n_gt) {
                o[p] = i;
                v[p] = r[i];
            }
        } else if (k == thr) {
            const unsigned p = atomicAdd(&st[row * 5 + 4], 1u);
            if (p < need_eq) {
                o[n_gt + p] = i;
                v[n_gt + p] = r[i];
            }
        }
    }
}
/// A draft step's candidates' logits: the row's head logit plus the head row (Q8_0) times the step's correction, softcapped.
__global__ void k_ds_cand(const int32_t* ci, const float* cv, const uint8_t* head, size_t row_bytes, const float* corr, int n,
                          float cap, float* out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, c = blockIdx.x * (blockDim.x / 32) + warp;
    if (c >= kDsCand) return;
    const uint8_t* row = head + (size_t) ci[c] * row_bytes;
    float s = 0.f;
    for (int b = lane; b < n / 32; b += 32) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
        const float* x = corr + (size_t) b * 32;
        float a = 0.f;
#pragma unroll 8
        for (int j = 0; j < 32; ++j) a += (float) q[j] * x[j];
        s += d * a;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) {
        const float l = cv[c] + s;
        out[c] = cap > 0.f ? cap * tanhf(l / cap) : l;
    }
}
/// The largest of the candidates' logits (the lowest token id of a tie), as a token.
__global__ void k_cand_argmax(const int32_t* ci, const float* l, int32_t* tok) {
    __shared__ float bv[kDsCand];
    __shared__ int bi[kDsCand];
    const int t = threadIdx.x;
    bv[t] = l[t];
    bi[t] = ci[t];
    __syncthreads();
    for (int s2 = kDsCand / 2; s2 > 0; s2 >>= 1) {
        if (t < s2 && (bv[t + s2] > bv[t] || (bv[t + s2] == bv[t] && bi[t + s2] < bi[t]))) {
            bv[t] = bv[t + s2];
            bi[t] = bi[t + s2];
        }
        __syncthreads();
    }
    if (t == 0) tok[0] = bi[0];
}
__global__ void k_ds_u(const float* row, const float* corr, const float* b, float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) u[i] = row[i] + corr[i] + b[i];   // the head of (row + correction): one read of it for both terms
}

/// The prompt's expert GEMM on mma.sync: a block an MR x 8192 / MR tile (rows = the group's routed entries, columns =
/// output channels; gate/up: half gate, half up), 8 warps of 32 x 32; A from the gathered FP16 rows, B the GGUF blocks
/// copied a step ahead and converted to FP16 in shared memory; SITU in the gate/up epilogue, s_down in the down one.
constexpr int kXLd = kEK + 8;
template <int MR> __host__ __device__ constexpr int x_cols() { return 8192 / MR; }   // a block's columns: MR rows x this, 8 warps of 32 x 32
template <int TY, int MR> constexpr size_t expert2_smem() {
    constexpr int NC = x_cols<MR>();
    constexpr size_t body = (size_t) 2 * MR * kEK * 2 + (size_t) 2 * NC * seg_bytes<TY>() + (size_t) NC * kXLd * 2;
    constexpr size_t cs = (size_t) MR * (NC + 4) * 4;
    return body > cs ? body : cs;
}
/// A row's 16-byte chunk c of a step's rows in shared memory: swizzled (no padding), ldmatrix without conflicts.
__device__ __forceinline__ int a_sw(int r, int c) { return r * kEK + ((c ^ (r & 7)) << 3); }
template <int TY, bool GU, int MR>
__global__ void __launch_bounds__(256) k_expert_mma2(const __half* X, const unsigned long long* grp_ptr,
                                                     const int32_t* grp_start, const int32_t* n_groups, size_t w_off,
                                                     size_t row_bytes, int K, int n_ff, size_t tail_off, float beta, float lb,
                                                     float* Y, __half* H, int ld_y, int h_perm, const int2* tiles,
                                                     const int32_t* n_tiles) {
    // NVFP4: a byte's two codes as a half2 (the rows' K in nv4_perm order); the table once a block
    __shared__ __half2 lut[TY == 40 ? 256 : 1];
    if constexpr (TY == 40) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const uint32_t c = fp4_lut((uint32_t) (i & 15) | ((uint32_t) (i >> 4) << 8));
            lut[i] = __halves2half2(__int2half_rn((int) (int8_t) (c & 0xFF)), __int2half_rn((int) (int8_t) ((c >> 8) & 0xFF)));
        }
        __syncthreads();
    }
    constexpr int SEG = seg_bytes<TY>(), QK = TY == 40 ? 64 : 32, BB = TY == 40 ? 36 : 34;
    if ((int) blockIdx.y >= *n_tiles) return;
    const int2 tl = tiles[blockIdx.y];   // (group, first row): a hot expert's rows over many blocks
    const int g = tl.x;
    const int s0 = grp_start[g], n = grp_start[g + 1] - s0;
    constexpr int NC = x_cols<MR>(), WM = MR / 32;
    const int c0 = blockIdx.x * (GU ? NC / 2 : NC), tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp % WM, wn = warp / WM, gq = lane / 4, tq = lane % 4;   // a warp: 32 rows x 32 columns
    const uint8_t* W = (const uint8_t*) grp_ptr[g] + w_off;
    const float* tail = tail_off ? (const float*) (grp_ptr[g] + tail_off) : nullptr;
    extern __shared__ __align__(128) unsigned char esm[];
    __half* As = reinterpret_cast<__half*>(esm);                               // [2][MR * kEK], swizzled
    uint8_t* raw = esm + (size_t) 2 * MR * kEK * 2;                            // [2][NC * SEG]
    __half* Bs = reinterpret_cast<__half*>(raw + (size_t) 2 * NC * SEG);       // [NC * kXLd], column major
    float* Cs = reinterpret_cast<float*>(esm);                                 // [MR][NC + 4], after the K loop
    const int steps = K / kEK;
    // this thread's copies, their addresses once: TPC threads a weight column (its words part, part + TPC, ..), and
    // A_PT 16-byte pieces of the rows
    constexpr int TPC = 256 / NC, WPC = SEG / 4, WPT = (WPC + TPC - 1) / TPC, A_PT = MR * (kEK / 8) / 256;
    constexpr int STEP_B = kEK / QK * BB;   // a step's bytes along a weight row
    const int wj = tid / TPC, wpart = tid % TPC;
    const int wrow = GU ? (wj < NC / 2 ? c0 + wj : n_ff + c0 + wj - NC / 2) : c0 + wj;
    const uint8_t* wsrc = W + (size_t) wrow * row_bytes + 4 * wpart;
    const int wdst = wj * SEG + 4 * wpart;
    for (int r0 = tl.y; r0 == tl.y; r0 += MR) {
        const int nr = min(MR, n - r0);
        const __half* asrc[A_PT];
        int adst[A_PT], asz[A_PT];
#pragma unroll
        for (int q = 0; q < A_PT; ++q) {
            const int i = tid + 256 * q, r = i / (kEK / 8), c = i % (kEK / 8);
            asrc[q] = X + (size_t) (s0 + r0 + (r < nr ? r : 0)) * K + c * 8;
            adst[q] = a_sw(r, c);
            asz[q] = r < nr ? 16 : 0;
        }
        auto issue = [&](int st, int buf) {
            uint8_t* rd = raw + (size_t) buf * NC * SEG + wdst;
            const uint8_t* ws = wsrc + (size_t) st * STEP_B;
#pragma unroll
            for (int q = 0; q < WPT; ++q)
                if (wpart + TPC * q < WPC) cp_async4(rd + 4 * TPC * q, ws + 4 * TPC * q);
            __half* ad = As + (size_t) buf * MR * kEK;
#pragma unroll
            for (int q = 0; q < A_PT; ++q) cp_async16(ad + adst[q], asrc[q] + st * kEK, asz[q]);
            cp_commit();
        };
        float acc[2][4][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nj = 0; nj < 4; ++nj)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[mi][nj][i] = 0.f;
        const bool live0 = wm * 32 < nr, live1 = wm * 32 + 16 < nr;   // this warp's row tiles with entries in them
        issue(0, 0);
        for (int st = 0; st < steps; ++st) {
            if (st + 1 < steps) {
                issue(st + 1, (st + 1) & 1);
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            const uint8_t* rs = raw + (size_t) (st & 1) * NC * SEG;
#pragma unroll
            for (int it = 0; it < NC * 8 / 256; ++it) {   // the step's weight blocks to FP16 (as k_expert_mma)
                const int idx = it * 256 + tid;
                const int col = (idx % 8) + 8 * (idx / 64), ch = (idx / 8) % 8;
                const uint8_t* sg = rs + col * SEG;
                __half2 v[4];
                if constexpr (TY == 40) {   // K positions ch * 8 .. + 7 in nv4_perm order: 4 bytes of sub-block ch / 2
                    const int sb = ch / 2;
                    const __half2 d = __float2half2_rn(ue4m3_f(sg[sb]));
                    const uint32_t w = *reinterpret_cast<const uint32_t*>(sg + 4 + sb * 8 + (ch & 1) * 4);
                    v[0] = __hmul2(lut[w & 0xFF], d);
                    v[1] = __hmul2(lut[(w >> 8) & 0xFF], d);
                    v[2] = __hmul2(lut[(w >> 16) & 0xFF], d);
                    v[3] = __hmul2(lut[w >> 24], d);
                } else {
                    const int bl = ch / 4, off = (ch % 4) * 8;
                    const uint8_t* bp = sg + bl * 34;
                    const float d = __half2float(__ushort_as_half((uint16_t) (bp[0] | (bp[1] << 8))));
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        v[q] = __floats2half2_rn(d * (float) (int8_t) bp[2 + off + 2 * q], d * (float) (int8_t) bp[3 + off + 2 * q]);
                }
                *reinterpret_cast<uint4*>(Bs + col * kXLd + ch * 8) = *reinterpret_cast<const uint4*>(v);
            }
            __syncthreads();
            const __half* ac = As + (size_t) (st & 1) * MR * kEK;
#pragma unroll
            for (int kk = 0; kk < kEK; kk += 16) {
                uint32_t af[2][4];
                if (live0) ldsm_x4(af[0], ac + a_sw(wm * 32 + (lane % 8) + ((lane / 8) % 2) * 8, kk / 8 + lane / 16));
                if (live1) ldsm_x4(af[1], ac + a_sw(wm * 32 + 16 + (lane % 8) + ((lane / 8) % 2) * 8, kk / 8 + lane / 16));
#pragma unroll
                for (int nj = 0; nj < 4; nj += 2) {
                    uint32_t bf[4];
                    ldsm_x4(bf, Bs + (wn * 32 + nj * 8 + (lane % 8) + (lane / 16) * 8) * kXLd + kk + ((lane / 8) % 2) * 8);
                    if (live0) {
                        mma16816(acc[0][nj], af[0], bf);
                        mma16816(acc[0][nj + 1], af[0], bf + 2);
                    }
                    if (live1) {
                        mma16816(acc[1][nj], af[1], bf);
                        mma16816(acc[1][nj + 1], af[1], bf + 2);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nj = 0; nj < 4; ++nj)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = wm * 32 + mi * 16 + gq + (i / 2) * 8, c = wn * 32 + nj * 8 + 2 * tq + (i % 2);
                    Cs[r * (NC + 4) + c] = acc[mi][nj][i];
                }
        __syncthreads();
        if constexpr (GU) {
            const float sg = tail ? tail[0] : 1.f, su = tail ? tail[1] : 1.f;
            for (int i = tid; i < nr * (NC / 2); i += 256) {
                const int r = i / (NC / 2), j = i % (NC / 2);
                const float gv = Cs[r * (NC + 4) + j] * sg, u0 = Cs[r * (NC + 4) + NC / 2 + j] * su;
                const float u = lb > 0.f ? lb * tanhf(u0 / lb) : u0;
                const int hc = h_perm ? nv4_perm(c0 + j) : c0 + j;   // (the down GEMM's K order)
                H[(size_t) (s0 + r0 + r) * n_ff + hc] = __float2half(beta * tanhf(gv / beta) / (1.0f + expf(-gv)) * u);
            }
        } else {
            const float sd = tail ? tail[2] : 1.f;
            for (int i = tid; i < nr * NC; i += 256) {
                const int r = i / NC, c = i % NC;
                Y[(size_t) (s0 + r0 + r) * ld_y + c0 + c] = Cs[r * (NC + 4) + c] * sd;
            }
        }
        __syncthreads();
    }
}

/// The prompt's expert GEMM, warp-specialized: half the warps copy the rows and convert a step's weights to FP16 while
/// the other half multiply the step before (a consumer warp 32 or 64 rows x 32 columns), handing the buffers over with
/// named barriers: the rows three steps deep, the converted weights two, the raw weights copied 4 steps at a time
/// (8- or 16-byte pieces: NVFP4's 36-byte blocks align only 4 at a time). Products as k_expert_mma2's.
constexpr int kWsRaw = 4;   // steps a raw weight copy
template <int MR, int NC, int WC> __host__ __device__ constexpr int ws_threads() { return 32 * (MR / (MR == 128 ? 64 : 32)) * (NC / WC) + 256; }
template <int TY, int MR, int NC> constexpr size_t expert_ws_smem() {
    constexpr size_t body = (size_t) 3 * MR * kEK * 2 + (size_t) 2 * NC * ws_raw_bytes<TY>() + (size_t) 2 * NC * kEK * 2;
    constexpr size_t cs = (size_t) MR * (NC + 4) * 4;
    return body > cs ? body : cs;
}
/// Bytes into numbers without conversion instructions (16 a clock an SM): byte k of w as 2^23 + it (a float's
/// mantissa), or bytes k, k + 1 as 1024 + them (a half2's).
__device__ __forceinline__ float byte_f(uint32_t w, int k) {
    return __int_as_float((int) __byte_perm(w, 0x4B000000u, 0x7650u | (uint32_t) k)) - 8388608.0f;
}
__device__ __forceinline__ __half2 bytes_h2(uint32_t w, uint32_t sel) {   // sel: 0x4140 (bytes 0, 1) or 0x4342
    const uint32_t r = __byte_perm(w, 0x64646464u, sel);
    return *reinterpret_cast<const __half2*>(&r);
}
__device__ __forceinline__ uint32_t ld_u32_a2(const uint8_t* p) {   // 4 bytes at a 2-byte aligned address
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) h[0] | ((uint32_t) h[1] << 16);
}
__device__ __forceinline__ uint32_t spread4(uint32_t b) {   // bits 0..3 to bit 0 of bytes 0..3
    return (b & 1u) | ((b & 2u) << 7) | ((b & 4u) << 14) | ((b & 8u) << 21);
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"((uint32_t) __cvta_generic_to_shared(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
template <int TY, bool GU, int MR, int NC, int WC>
__global__ void __launch_bounds__(ws_threads<MR, NC, WC>(), 1) k_expert_ws(const __half* X, const unsigned long long* grp_ptr,
                                                     const int32_t* grp_start, const int32_t* n_groups, size_t w_off,
                                                     size_t row_bytes, int K, int n_ff, size_t tail_off, float beta, float lb,
                                                     float* Y, __half* H, int ld_y, int h_perm, const int2* tiles,
                                                     const int32_t* n_tiles) {
    // a consumer warp: WR rows (MI tiles of 16) x WC columns (NJ tiles of 8); then 8 producer warps
    constexpr int WR = MR == 128 ? 64 : 32, MI = WR / 16, NJ = WC / 8, WMN = MR / WR, NT = ws_threads<MR, NC, WC>(), PT = 256;
    constexpr int CT = NT - PT;
    constexpr int kFull = 1, kEmpty = 3, kProd = 5;   // named barriers: full[2], empty[2], the producers' own
    __shared__ __half2 lut[TY == 40 ? 256 : 1];
    if constexpr (TY == 40) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const uint32_t c = fp4_lut((uint32_t) (i & 15) | ((uint32_t) (i >> 4) << 8));
            lut[i] = __halves2half2(__int2half_rn((int) (int8_t) (c & 0xFF)), __int2half_rn((int) (int8_t) ((c >> 8) & 0xFF)));
        }
    }
    __syncthreads();
    constexpr int RK = ws_raw_steps<TY>();
    constexpr bool KQ = TY == 12 || TY == 13 || TY == 14;   // a K-quant: a super-block a copy, converted by quarters
    if ((int) blockIdx.y >= *n_tiles) return;
    const int2 tl = tiles[blockIdx.y];   // (group, first row): a hot expert's rows over many blocks
    const int g = tl.x;
    const int s0 = grp_start[g], n = grp_start[g + 1] - s0;
    const int c0 = blockIdx.x * (GU ? NC / 2 : NC), tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const bool producer = tid >= CT;
    const uint8_t* W = (const uint8_t*) grp_ptr[g] + w_off;
    const float* tail = tail_off ? (const float*) (grp_ptr[g] + tail_off) : nullptr;
    extern __shared__ __align__(128) unsigned char esm[];
    __half* As = reinterpret_cast<__half*>(esm);                               // [3][MR * kEK], swizzled
    constexpr int RSEG = ws_raw_bytes<TY>();                                   // a column's raw bytes of a copy
    constexpr int SEG = RSEG / RK;                                             // (a step's, but for the K-quants)
    uint8_t* raw = esm + (size_t) 3 * MR * kEK * 2;                            // [2][NC * RSEG]
    __half* Bs = reinterpret_cast<__half*>(raw + (size_t) 2 * NC * RSEG);      // [2][NC * kEK], column major, swizzled
    float* Cs = reinterpret_cast<float*>(esm);                                 // [MR][NC + 4], after the K loop
    const int steps = K / kEK;
    // a producer's copies (TPC threads a weight column: pieces part, part + TPC, ..; A_PT pieces of the rows) and
    // conversions; 16-byte pieces where the rows allow (the down rows: 648 bytes, 8)
    constexpr int TPC = PT / NC, A_PT = MR * (kEK / 8) / PT;
    const bool p16 = ((row_bytes | (size_t) W | (size_t) RSEG) & 15) == 0;   // (every copy starts a multiple of RSEG on)
    const int pb = p16 ? 16 : 8, n_pc = RSEG / pb;
    const int pt = tid - CT;
    const int wj = (producer ? pt : 0) / TPC, wpart = (producer ? pt : 0) % TPC;
    const int wrow = GU ? (wj < NC / 2 ? c0 + wj : n_ff + c0 + wj - NC / 2) : c0 + wj;
    const uint8_t* wsrc = W + (size_t) wrow * row_bytes;
    uint8_t* const wdst = raw + wj * RSEG;
    const int wm = warp % WMN, wn = warp / WMN, gq = lane / 4, tq = lane % 4;   // a consumer: rows wm * WR, columns wn * WC
    for (int r0 = tl.y; r0 == tl.y; r0 += MR) {
        const int nr = min(MR, n - r0);
        float acc[MI][NJ][4];
        if (producer) {
            const __half* asrc[A_PT];
            int adst[A_PT], asz[A_PT];
#pragma unroll
            for (int q = 0; q < A_PT; ++q) {
                const int i = pt + PT * q, r = i / (kEK / 8), c = i % (kEK / 8);
                asrc[q] = X + (size_t) (s0 + r0 + (r < nr ? r : 0)) * K + c * 8;
                adst[q] = a_sw(r, c);
                asz[q] = r < nr ? 16 : 0;
            }
            auto issue = [&](int st) {
                const int buf = st % 3;
                if (st % RK == 0) {   // steps st .. st + 3's raw weights (the last copy: what is left of the row)
                    uint8_t* rd = wdst + (size_t) ((st / RK) & 1) * NC * RSEG;
                    const uint8_t* ws = wsrc + (size_t) (st / RK) * RSEG;
                    if constexpr (TY == 14) {   // Q6_K rows (1890 bytes) are 2-byte aligned only: plain 16-bit copies
                        for (int c = wpart; c < RSEG / 2; c += TPC)
                            reinterpret_cast<uint16_t*>(rd)[c] = reinterpret_cast<const uint16_t*>(ws)[c];
                    } else {
                        const int valid = KQ ? RSEG : min(RK, steps - st) * SEG;
                        for (int c = wpart; c < n_pc; c += TPC) {
                            const int bytes = max(0, min(pb, valid - c * pb));
                            if (p16) cp_async16(rd + c * 16, ws + c * 16, bytes);
                            else cp_async8(rd + c * 8, ws + c * 8, bytes);
                        }
                    }
                }
                __half* ad = As + (size_t) buf * MR * kEK;
#pragma unroll
                for (int q = 0; q < A_PT; ++q) cp_async16(ad + adst[q], asrc[q] + st * kEK, asz[q]);
                cp_commit();
            };
            // step st: its copies in, converted into Bs[st & 1] (free: the consumers' step st - 2 is done), handed
            // over; then step st + 2's copies into the stage the consumers' step st - 1 used, once it is done
            issue(0);
            if (steps > 1) issue(1);
            for (int st = 0; st < steps; ++st) {
                if (st + 1 < steps) cp_wait<1>();
                else cp_wait<0>();
                bar_sync(kProd, PT);   // every producer's copies of step st are in
                const uint8_t* rblk = raw + (size_t) ((st / RK) & 1) * NC * RSEG;   // this copy's (a Q4_K super-block)
                const uint8_t* rs = rblk + (st % RK) * SEG;
                __half* bd = Bs + (size_t) (st & 1) * NC * kEK;
#pragma unroll
                for (int it = 0; it < NC * 8 / PT; ++it) {
                    const int idx = it * PT + pt;
                    const int col = (idx % 8) + 8 * (idx / 64), ch = (idx / 8) % 8;
                    const uint8_t* sg = rs + col * RSEG;
                    __half2 v[4];
                    if constexpr (TY == 40) {
                        const int sb = ch / 2;
                        const __half2 d = ue4m3_h2(sg[sb]);
                        const uint32_t w = *reinterpret_cast<const uint32_t*>(sg + 4 + sb * 8 + (ch & 1) * 4);
                        v[0] = __hmul2(lut[w & 0xFF], d);
                        v[1] = __hmul2(lut[(w >> 8) & 0xFF], d);
                        v[2] = __hmul2(lut[(w >> 16) & 0xFF], d);
                        v[3] = __hmul2(lut[w >> 24], d);
                    } else if constexpr (TY == 12 || TY == 13) {   // Q4_K / Q5_K: d sc q - dmin m, 6-bit scales and mins a 32
                        const uint8_t* sb_ = rblk + col * RSEG;   // d, dmin, scales[12], (Q5_K: qh[32],) qs[128]
                        const int qt = st % RK, sb = 2 * qt + (ch >> 2), l0 = (ch & 3) * 8;
                        const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(sb_)));
                        const float dmin = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(sb_ + 2)));
                        const uint8_t* scl = sb_ + 4;
                        int sc, mn;
                        if (sb < 4) {
                            sc = scl[sb] & 63;
                            mn = scl[sb + 4] & 63;
                        } else {
                            sc = (scl[sb + 4] & 0xF) | ((scl[sb - 4] >> 6) << 4);
                            mn = (scl[sb + 4] >> 4) | ((scl[sb] >> 6) << 4);
                        }
                        const float ds = d * (float) sc, dm = dmin * (float) mn;
                        constexpr int QS = TY == 13 ? 48 : 16;
                        const uint2 qb = *reinterpret_cast<const uint2*>(sb_ + QS + 32 * qt + l0);
                        const int sh = (ch >> 2) * 4;   // the low nibbles: the first 32, the high: the second
                        uint32_t u[2] = {(qb.x >> sh) & 0x0F0F0F0Fu, (qb.y >> sh) & 0x0F0F0F0Fu};   // the values, a byte each
                        if constexpr (TY == 13) {   // Q5_K: the fifth bits, bit sb of qh[l]
                            const uint2 hb = *reinterpret_cast<const uint2*>(sb_ + 16 + l0);
                            u[0] |= ((hb.x >> sb) & 0x01010101u) << 4;
                            u[1] |= ((hb.y >> sb) & 0x01010101u) << 4;
                        }
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            v[q] = __floats2half2_rn(ds * byte_f(u[q / 2], 2 * (q & 1)) - dm, ds * byte_f(u[q / 2], 2 * (q & 1) + 1) - dm);
                    } else if constexpr (TY == 14) {   // Q6_K: d sc q, 6-bit q - 32, int8 scales a 16
                        const uint8_t* sb_ = rblk + col * RSEG;   // ql[128], qh[64], scales[16], d
                        const int qt = st % RK, hf = qt >> 1, k0 = (qt & 1) * 64 + (ch & 7) * 8;   // within the half: 0..127
                        const int grp = k0 >> 5, l0 = k0 & 31;   // q1..q4 of ggml's loop, and l
                        const uint8_t* ql = sb_ + 64 * hf + (grp & 1) * 32;
                        const uint8_t* qh = sb_ + 128 + 32 * hf;
                        const int8_t* scl = reinterpret_cast<const int8_t*>(sb_ + 192) + 8 * hf;
                        const float d = __half2float(__ushort_as_half((uint16_t) (sb_[208] | (sb_[209] << 8))));
                        const int lsh = (grp >> 1) * 4, hsh = grp * 2;
                        const float ds = d * (float) scl[l0 / 16 + 2 * grp];   // (l0 .. l0 + 7: one scale)
                        uint32_t u[2];
#pragma unroll
                        for (int h = 0; h < 2; ++h)
                            u[h] = ((ld_u32_a2(ql + l0 + 4 * h) >> lsh) & 0x0F0F0F0Fu) | (((ld_u32_a2(qh + l0 + 4 * h) >> hsh) & 0x03030303u) << 4);
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            v[q] = __floats2half2_rn(ds * (byte_f(u[q / 2], 2 * (q & 1)) - 32.0f), ds * (byte_f(u[q / 2], 2 * (q & 1) + 1) - 32.0f));
                    } else if constexpr (TY == 6) {   // Q5_0: (q - 16) d, the fifth bit from qh
                        const uint8_t* bp = sg + (ch >> 2) * 22;   // d, qh[4], qs[16] (2-byte aligned)
                        const __half dh = __ushort_as_half(*reinterpret_cast<const uint16_t*>(bp));
                        const uint32_t qh = ld_u32_a2(bp + 2);
                        const int j0 = (ch & 3) * 8, sh = j0 >= 16 ? 4 : 0;   // values j0 .. j0 + 7: qs[j0 % 16 ..]'s nibbles
                        const uint32_t hb = (qh >> j0) & 0xFFu;
                        const __half2 d2 = __halves2half2(dh, dh), o2 = __float2half2_rn(1040.0f);   // 1024 + 16
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint32_t u = ((ld_u32_a2(bp + 6 + (j0 & 15) + 4 * h) >> sh) & 0x0F0F0F0Fu) | (spread4(hb >> (4 * h)) << 4);
                            v[2 * h] = __hmul2(__hsub2(bytes_h2(u, 0x4140), o2), d2);   // (q - 16) d: exact, then one rounding
                            v[2 * h + 1] = __hmul2(__hsub2(bytes_h2(u, 0x4342), o2), d2);
                        }
                    } else {
                        const int bl = ch / 4, off = (ch % 4) * 8;
                        const uint8_t* bp = sg + bl * 34;   // (2-byte aligned)
                        const __half dh = __ushort_as_half(*reinterpret_cast<const uint16_t*>(bp));
                        const __half2 d2 = __halves2half2(dh, dh), o2 = __float2half2_rn(1152.0f);   // 1024 + 128
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint32_t u = ld_u32_a2(bp + 2 + off + 4 * h) ^ 0x80808080u;   // q + 128
                            v[2 * h] = __hmul2(__hsub2(bytes_h2(u, 0x4140), o2), d2);
                            v[2 * h + 1] = __hmul2(__hsub2(bytes_h2(u, 0x4342), o2), d2);
                        }
                    }
                    *reinterpret_cast<uint4*>(bd + a_sw(col, ch)) = *reinterpret_cast<const uint4*>(v);
                }
                bar_arrive(kFull + (st & 1), NT);
                if (st >= 1) bar_sync(kEmpty + ((st - 1) & 1), NT);
                if (st + 2 < steps) issue(st + 2);
                else cp_commit();   // (an empty group: the waits above count groups)
            }
            bar_sync(kEmpty + ((steps - 1) & 1), NT);
        } else {
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int nj = 0; nj < NJ; ++nj)
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[mi][nj][i] = 0.f;
            bool live[MI];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) live[mi] = wm * WR + mi * 16 < nr;
            for (int st = 0; st < steps; ++st) {
                bar_sync(kFull + (st & 1), NT);
                const __half* ac = As + (size_t) (st % 3) * MR * kEK;
                const __half* Bc = Bs + (size_t) (st & 1) * NC * kEK;
#pragma unroll
                for (int kk = 0; kk < kEK; kk += 16) {
                    uint32_t af[MI][4], bf[NJ / 2][4];
#pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        if (live[mi]) ldsm_x4(af[mi], ac + a_sw(wm * WR + mi * 16 + (lane % 8) + ((lane / 8) % 2) * 8, kk / 8 + lane / 16));
#pragma unroll
                    for (int np = 0; np < NJ / 2; ++np)
                        ldsm_x4(bf[np], Bc + a_sw(wn * WC + np * 16 + (lane % 8) + (lane / 16) * 8, kk / 8 + (lane / 8) % 2));
#pragma unroll
                    for (int mi = 0; mi < MI; ++mi) {
                        if (!live[mi]) continue;
#pragma unroll
                        for (int np = 0; np < NJ / 2; ++np) {
                            mma16816(acc[mi][2 * np], af[mi], bf[np]);
                            mma16816(acc[mi][2 * np + 1], af[mi], bf[np] + 2);
                        }
                    }
                }
                bar_arrive(kEmpty + (st & 1), NT);
            }
        }
        __syncthreads();   // (the K loop's buffers become the result tile)
        if (!producer) {
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int nj = 0; nj < NJ; ++nj)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int r = wm * WR + mi * 16 + gq + (i / 2) * 8, c = wn * WC + nj * 8 + 2 * tq + (i % 2);
                        Cs[r * (NC + 4) + c] = acc[mi][nj][i];
                    }
        }
        __syncthreads();
        if constexpr (GU) {
            const float sg = tail ? tail[0] : 1.f, su = tail ? tail[1] : 1.f;
            for (int i = tid; i < nr * (NC / 2); i += NT) {
                const int r = i / (NC / 2), j = i % (NC / 2);
                const float gv = Cs[r * (NC + 4) + j] * sg, u0 = Cs[r * (NC + 4) + NC / 2 + j] * su;
                const float u = lb > 0.f ? lb * tanhf(u0 / lb) : u0;
                const int hc = h_perm ? nv4_perm(c0 + j) : c0 + j;
                H[(size_t) (s0 + r0 + r) * n_ff + hc] = __float2half(beta * tanhf(gv / beta) / (1.0f + expf(-gv)) * u);
            }
        } else {
            const float sd = tail ? tail[2] : 1.f;
            for (int i = tid; i < nr * NC; i += NT) {
                const int r = i / NC, c = i % NC;
                Y[(size_t) (s0 + r0 + r) * ld_y + c0 + c] = Cs[r * (NC + 4) + c] * sd;
            }
        }
        __syncthreads();
    }
}

/// The (group, first row) tiles of MR rows of every group, in group order.
__global__ void k_tiles(const int32_t* grp_start, const int32_t* n_groups, int mr, int2* tiles, int32_t* n_tiles) {
    if (threadIdx.x != 0) return;
    int t = 0;
    for (int g = 0; g < *n_groups; ++g)
        for (int r = 0; r < grp_start[g + 1] - grp_start[g]; r += mr) tiles[t++] = make_int2(g, r);
    *n_tiles = t;
}
__global__ void k_gather_f16p(const float* x, const int32_t* tok, __half* Xg, int n_ent, int n) {
    const int p = blockIdx.x;
    if (p >= n_ent) return;
    const float* xr = x + (size_t) tok[p] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) Xg[(size_t) p * n + nv4_perm(i)] = __float2half(xr[i]);
}
__global__ void k_gather_f16(const float* x, const int32_t* tok, __half* Xg, int n_ent, int n) {
    const int p = blockIdx.x;
    if (p >= n_ent) return;
    const float* xr = x + (size_t) tok[p] * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) Xg[(size_t) p * n + i] = __float2half(xr[i]);
}
__global__ void k_combine_sorted(const float* dg, const int32_t* inv, const float* wts, int K_, float* out, int n) {
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float a = 0.f;
        for (int k = 0; k < K_; ++k) a += wts[t * K_ + k] * dg[(size_t) inv[t * K_ + k] * n + i];
        out[t * n + i] = a;
    }
}

__global__ void k_route(const float* logits, const float* bias, int n_expert, int k, float scale, int32_t* ids,
                        float* wts) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float p[8], sel[8];
    const int per = (n_expert + 31) / 32;
    for (int j = 0; j < 8; ++j) {
        const int e = lane + 32 * j;
        p[j] = 0.f;
        sel[j] = -INFINITY;
        if (j < per && e < n_expert) {
            p[j] = sigm(logits[(size_t) t * n_expert + e]);
            sel[j] = p[j] + (bias ? bias[e] : 0.f);
        }
    }
    float got[16];
    int gid[16];
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bj = -1;
        for (int j = 0; j < per; ++j) {
            const int e = lane + 32 * j;
            if (sel[j] > bv || (sel[j] == bv && e < bi)) { bv = sel[j]; bi = e; bj = j; }
        }
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, m);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float pw = 0.f;
        if (bj >= 0 && lane + 32 * bj == bi) {
            pw = p[bj];
            sel[bj] = -INFINITY;
        }
        for (int m = 16; m > 0; m >>= 1) pw += __shfl_xor_sync(0xffffffffu, pw, m);
        got[r] = pw;
        gid[r] = bi;
        sum += pw;
    }
    if (lane == 0)
        for (int r = 0; r < k; ++r) {
            ids[(size_t) t * k + r] = gid[r];
            wts[(size_t) t * k + r] = got[r] / (sum + 1e-20f) * scale;
        }
}

/// The routed entries grouped by expert, a group an expert in VRAM (eptr != 0) in `order` (none: by id); the entries of
/// an expert not in VRAM are left out (ent_inv -1): the CPU computes them. order's first n_keep experts take a group
/// even with no entries (an empty one), so their groups' numbers are known before the routing is.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const unsigned long long* eptr,
                        const int32_t* order, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                        int32_t* ent_dst, int32_t* ent_tok, int32_t* ent_inv, int n_keep = 0) {
    __shared__ int cnt[256], off[256];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int o = 0; o < n_expert; ++o) {
            const int x = order ? order[o] : o;
            off[x] = pos;
            if ((cnt[x] == 0 && o >= n_keep) || eptr[x] == 0) continue;
            grp_ptr[g] = eptr[x];
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) {   // each entry its slot in its group (any order: rows are independent)
        const int x = ids[e];
        if (eptr[x] == 0) {
            ent_inv[e] = -1;
            continue;
        }
        const int p = atomicAdd(&off[x], 1);
        ent_dst[p] = e;
        ent_tok[p] = e / K_;
        ent_inv[e] = p;
    }
}
__global__ void k_hist(const int32_t* ids, int n_ent, uint32_t* hist) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n_ent; e += gridDim.x * blockDim.x) atomicAdd(&hist[ids[e]], 1u);
}
/// The CPU's entries into the GPU's per-entry results.
/// A decode's entries of experts in RAM into the parts: an entry whose expert is in VRAM (the GPU computed it) is left
/// alone without waiting; for the others the block waits for the host function's flag (mapped host memory) and reads
/// the CPU's result from there. A visit with no entry in RAM never waits on the host.
__global__ void k_fill_cpu_wait(float* parts, const int32_t* ids, const unsigned long long* eptr, const float* cpu,
                                const volatile int* done, int seq, int n) {
    const int e = blockIdx.x;
    if (eptr[ids[e]] != 0) return;
    if (threadIdx.x == 0)
        while (*done - seq < 0) {
        }
    __syncthreads();
    __threadfence_system();
    const volatile float* c = cpu + (size_t) e * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = c[i];
}

__global__ void k_combine(const float* parts, const float* wts, int K_, float* out, int n) {
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float a = 0.f;
        for (int k = 0; k < K_; ++k) a += wts[t * K_ + k] * parts[(t * K_ + k) * n + i];
        out[t * n + i] = a;
    }
}

__global__ void k_softcap(float* x, int64_t n, int64_t ld, int rows, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * rows) return;
    const int64_t r = i / n, c = i % n;
    float& v = x[r * ld + c];
    v = cap * tanhf(v / cap);
}

constexpr int kArgBlocks = 128;
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}

__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}

__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int kz) {
    constexpr int TM = 4, BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);
    out += (size_t) blockIdx.z * T * n_out;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
        As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * n_out + n0 + tn + j] = acc[i][j];
}

__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

struct HP {
    int n_embd = 0, n_vis = 0, n_phys = 0, core0 = 0, core1 = 0, n_pass = 1, gate_rank = 0;
    int n_head = 0, n_kv = 0, hd = 0, n_rel = 0, rel_extent = 0, swa = 0, n_conv = 0, res_block = 0;
    int n_expert = 0, top_k = 0, n_ff = 0, vocab_pad = 0, vocab = 0, log_floor = 1;
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, situ_beta = 0, situ_lb = 0, w_scale = 1, softcap = 0, log_alpha = 0;
    std::vector<char> swa_vis;   // per visit
    int n_core() const { return core1 - core0; }
    int extra() const { return n_pass > 1 ? (n_pass - 1) * n_core() : 0; }
    bool looped() const { return n_pass > 1; }
    int phys(int il) const {
        if (!looped() || il < core0) return il;
        if (il < core0 + n_pass * n_core()) return core0 + (il - core0) % n_core();
        return il - extra();
    }
    int pass(int il) const { return looped() && il >= core0 && il < core0 + n_pass * n_core() ? (il - core0) / n_core() : 0; }
    bool in_core(int ip) const { return looped() && ip >= core0 && ip < core1; }
};

struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

constexpr size_t kMmqTail = 4096;   // zeroed bytes after an expert's blob (a block read past its end)

struct Layer {
    int dev = 0;
    bool swa = false;
    int extent = 0;               // the distance table's rows
    Mat wq, wk, wv, wo, wg, wrel;
    const float *q_clip = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    std::vector<const float*> rel_bias2;   // [card]
    const float *k_conv = nullptr, *v_conv = nullptr, *a_conv = nullptr, *f_conv = nullptr;
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;
    const float *kc_proj = nullptr, *vc_proj = nullptr, *sink_proj = nullptr;
    const float* attn_res_q = nullptr;   // [nq][N], none on layer 0
    int n_res_q = 0;
    const float *ffn_res_q = nullptr, *ffn_norm = nullptr, *router = nullptr, *probs_b = nullptr;
    const float *loop_down[4] = {}, *loop_gain[4] = {}, *loop_forget[4] = {};
    const float* msg_q = nullptr;            // the loop's message queries (core layers)
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;               // the VRAM slots, n_slots blobs `stride` apart
    size_t stride = 0;
    bool mma = false;   // the prompt's tensor-core expert GEMM reads these types
    uint8_t* hblobs = nullptr;              // every expert's blob in pinned RAM, `stride` apart
    unsigned long long* eptr = nullptr;     // device [n_expert]: the expert's VRAM blob, 0 when it is not in VRAM
    std::vector<unsigned long long> eptr_h; // host copy
    std::vector<uint8_t> on_gpu;            // eptr_h != 0
    int n_slots = 0, n_res = 0;             // VRAM slots, experts in them
    CpuExperts cx;                          // the CPU's view of hblobs
    uint32_t* hist = nullptr;               // device [n_expert]: the routings since the pool last read them
    bool all_on_gpu() const { return n_res == (int) on_gpu.size(); }
    std::vector<int> slab_of, slot_of;      // expert -> its slab and slot in VRAM (-1: in RAM)
    void upload_ptrs() { CK(cudaMemcpy(eptr, eptr_h.data(), eptr_h.size() * 8, cudaMemcpyHostToDevice)); }
};

/// kSlab VRAM slots of one expert size in one allocation (a lone expert would round up to the driver's 2 MiB pages).
constexpr int kSlab = 4;
struct Slab {
    int dev = 0;
    size_t stride = 0;
    uint8_t* p = nullptr;   // nullptr: freed
    int used = 0, user_l[kSlab] = {}, user_e[kSlab] = {};
};

struct Model {
    HP hp;
    std::vector<Slab> slabs;
    /// Expert e of layer l into a free slot of a slab of its size on the layer's card (a new slab when none has one;
    /// false when the card has no room). The device pointer table is the caller's to upload.
    bool load_expert(int l, int e) {
        Layer& L = layers[(size_t) l];
        if (L.on_gpu[(size_t) e]) return true;
        int si = -1;
        for (size_t i = 0; i < slabs.size() && si < 0; ++i)
            if (slabs[i].p && slabs[i].dev == L.dev && slabs[i].stride == L.stride && slabs[i].used < kSlab) si = (int) i;
        if (si < 0) {
            Slab sb;
            sb.dev = L.dev;
            sb.stride = L.stride;
            if (cudaMalloc(&sb.p, (size_t) kSlab * L.stride + kMmqTail) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            CK(cudaMemset(sb.p + (size_t) kSlab * L.stride, 0, kMmqTail));
            for (int k = 0; k < kSlab; ++k) sb.user_l[k] = -1;
            for (size_t i = 0; i < slabs.size() && si < 0; ++i)
                if (!slabs[i].p) si = (int) i;
            if (si < 0) {
                si = (int) slabs.size();
                slabs.push_back(sb);
            } else
                slabs[(size_t) si] = sb;
        }
        Slab& sb = slabs[(size_t) si];
        int k = 0;
        while (sb.user_l[k] >= 0) ++k;
        uint8_t* q = sb.p + (size_t) k * L.stride;
        CK(cudaMemcpy(q, L.hblobs + (size_t) e * L.stride, L.stride, cudaMemcpyHostToDevice));
        sb.user_l[k] = l;
        sb.user_e[k] = e;
        ++sb.used;
        L.slab_of[(size_t) e] = si;
        L.slot_of[(size_t) e] = k;
        L.eptr_h[(size_t) e] = (unsigned long long) q;
        L.on_gpu[(size_t) e] = 1;
        ++L.n_res;
        return true;
    }
    /// Expert e of layer l back to RAM only; its slab freed when it was the last one in it.
    void evict_expert(int l, int e) {
        Layer& L = layers[(size_t) l];
        if (!L.on_gpu[(size_t) e]) return;
        Slab& sb = slabs[(size_t) L.slab_of[(size_t) e]];
        sb.user_l[L.slot_of[(size_t) e]] = -1;
        if (--sb.used == 0) {
            CK(cudaFree(sb.p));
            sb.p = nullptr;
        }
        L.slab_of[(size_t) e] = L.slot_of[(size_t) e] = -1;
        L.eptr_h[(size_t) e] = 0;
        L.on_gpu[(size_t) e] = 0;
        --L.n_res;
    }
    std::shared_ptr<strata::GgufModel> file;
    std::vector<Layer> layers;   // physical
    int n_dev = 1;
    std::vector<int> dev_ids;    // CUDA ordinals
    const uint8_t* embd = nullptr;   // the mapped GGUF's blocks (the rows of a forward are copied to the card)
    int embd_type = -1;
    size_t embd_row = 0;
    const float* tok_norm = nullptr;          // on layer 0's card
    const float *out_norm = nullptr, *out_res_q = nullptr;
    Mat head;
    int head_dev = 0;
};

size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}
size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;
        } else {
            uint32_t m = mantissa, shift = 0;
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool host_f32(const strata::TensorInfo& t, const uint8_t* src, std::vector<float>& h, std::string& err) {
    h.resize((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        err = t.name + " is " + t.type_name() + ", expected F32, BF16, F16 or Q8_0";
        return false;
    }
    return true;
}

struct Loader {
    Model& m;
    const strata::GgufModel& gm;
    std::string err;
    uint64_t bytes[kMaxDev] = {};
    explicit Loader(Model& m_) : m(m_), gm(*m_.file) {}
    Loader(Model& m_, const strata::GgufModel& g) : m(m_), gm(g) {}
    /// A BF16 tensor as it is (2 bytes a value).
    const uint16_t* bf16(int dev, const std::string& name) {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        if (!t) return nullptr;
        if (t->type != 30) {
            err = name + " is " + t->type_name() + ", expected BF16";
            return nullptr;
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        CK(cudaSetDevice(m.dev_ids[(size_t) dev]));
        uint16_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes[dev] += nb;
        return d;
    }

    const strata::TensorInfo* find(const std::string& name, size_t& s, bool required = true) {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t && required && err.empty()) err = "missing tensor " + name;
        return t;
    }
    const float* f32(int dev, const std::string& name, bool required = true) {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s, required);
        if (!t) return nullptr;
        std::vector<float> h;
        if (!host_f32(*t, gm.shard(s).tensor_data(*t), h, err)) return nullptr;
        CK(cudaSetDevice(m.dev_ids[(size_t) dev]));
        float* d = nullptr;
        CK(cudaMalloc(&d, h.size() * 4));
        CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        bytes[dev] += h.size() * 4;
        return d;
    }
    Mat mat(int dev, const std::string& name) {
        size_t s = 0;
        Mat w;
        const strata::TensorInfo* t = find(name, s);
        if (!t) return w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type) && !K::iq_supported(w.type)) {
            err = name + " is " + t->type_name() + ", which the native matrix kernels do not take";
            return w;
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        CK(cudaSetDevice(m.dev_ids[(size_t) dev]));
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes[dev] += nb;
        w.d = d;
        return w;
    }
};

std::string Lname(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

bool read_hp(const strata::GgufFile& f, HP& hp, std::string& err) {
    const strata::MetaValue* arch = f.get("general.architecture");
    if (!arch || (arch->s != "inkling2" && arch->s != "inkling2_loop")) {
        err = "strata-inkling2 runs inkling2 and inkling2_loop files only (llama.cpp's Inkling 2 GGUFs)";
        return false;
    }
    const std::string p = arch->s + ".";
    auto num = [&](const char* k, bool req, double def = 0) -> double {
        const strata::MetaValue* v = f.get(p + k);
        if (!v) {
            if (req && err.empty()) err = "missing " + p + k;
            return def;
        }
        if (v->type == strata::MetaType::ARRAY) {
            if (v->items.empty()) return def;
            return v->items[0].num();
        }
        return v->num();
    };
    hp.n_embd = (int) num("embedding_length", true);
    hp.n_phys = (int) num("block_count", true);   // the physical layers; the loop's visits follow from loop.*
    hp.n_head = (int) num("attention.head_count", true);
    hp.n_kv = (int) num("attention.head_count_kv", true);
    hp.hd = (int) num("attention.key_length", true);
    hp.n_rel = (int) num("attention.relative_length", true);
    hp.rel_extent = (int) num("attention.relative_extent", true);
    hp.swa = (int) num("attention.sliding_window", true);
    hp.n_conv = (int) num("shortconv.l_cache", true);
    hp.res_block = (int) num("attn_res.block_size", true);
    hp.n_expert = (int) num("expert_count", true);
    hp.top_k = (int) num("expert_used_count", true);
    hp.n_ff = (int) num("expert_feed_forward_length", true);
    hp.vocab_pad = (int) num("vocab_size", true);
    hp.vocab = (int) num("vocab_size_unpadded", true);
    hp.log_floor = (int) num("attention.log_scale_floor", true);
    hp.eps = (float) num("attention.layer_norm_rms_epsilon", true);
    hp.qk_clip = (float) num("attention.qk_clip", true);
    hp.rel_clip = (float) num("attention.relative_clip", true);
    hp.situ_beta = (float) num("activation.situ_beta", true);
    hp.situ_lb = (float) num("activation.situ_linear_beta", true);
    hp.w_scale = (float) num("expert_weights_scale", false, 1.0);
    hp.softcap = (float) num("final_logit_softcapping", false, 0.0);
    hp.log_alpha = (float) num("attention.log_scale_alpha", true);
    if (!err.empty()) return false;
    if (arch->s == "inkling2_loop") {
        hp.core0 = (int) num("loop.start", true);
        hp.core1 = (int) num("loop.end", true);
        hp.n_pass = (int) num("loop.count", true);
        hp.gate_rank = (int) num("loop.gate_rank", true);
        if (!err.empty()) return false;
        const strata::MetaValue* varying = f.get(p + "loop.varying");
        if (varying && varying->u != 0) {
            err = "a loop with a varying pass count (loop.varying) is not supported";
            return false;
        }
        if (hp.n_pass < 2 || hp.gate_rank <= 0 || hp.core0 >= hp.core1 || hp.core1 > hp.n_phys) {
            err = "invalid loop layout";
            return false;
        }
        if (hp.core0 == 0 || hp.core0 % hp.res_block || (hp.core1 - hp.core0) % hp.res_block) {
            err = "the looped core must start after the first block and span whole blocks";
            return false;
        }
    }
    hp.n_vis = hp.n_phys + hp.extra();
    const strata::MetaValue* pat = f.get(p + "attention.sliding_window_pattern");
    if (!pat || pat->type != strata::MetaType::ARRAY || (int) pat->items.size() != hp.n_phys) {
        err = "the sliding window pattern must name every layer";
        return false;
    }
    hp.swa_vis.resize((size_t) hp.n_vis);
    for (int i = 0; i < hp.n_vis; ++i) hp.swa_vis[(size_t) i] = pat->items[(size_t) hp.phys(i)].u != 0;
    if (hp.hd != kHD || hp.n_head % hp.n_kv || hp.n_rel > 32 || hp.n_conv != 4 || hp.n_expert > 256 ||
        hp.top_k > 16 || hp.n_embd % 256 || hp.n_ff % 32 || hp.vocab > hp.vocab_pad || hp.vocab <= 0) {
        err = "a geometry this driver's kernels do not take (head 128, relative length <= 32, conv 4, <= 256 experts, "
              "top-k <= 16)";
        return false;
    }
    if (hp.n_phys / hp.res_block + 2 > kMaxStates) {
        err = "too many residual blocks for this driver";
        return false;
    }
    return true;
}

uint64_t layer_bytes(const strata::GgufModel& gm, const HP& hp, int l, size_t& stride, std::string& err) {
    uint64_t b = 0;
    const std::string pre = "blk." + std::to_string(l) + ".";
    for (size_t s = 0; s < gm.size(); ++s)
        for (const auto& t : gm.shard(s).tensors()) {
            if (t.name.rfind(pre, 0) != 0) continue;
            if (t.name.find("_exps.") != std::string::npos) continue;
            const bool dense = t.type != 0 && t.type != 1 && t.type != 30 && t.shape.size() == 2 && t.shape[0] >= 1024 &&
                               t.shape[1] >= 256;
            b += dense ? strata::tensor_payload_bytes(t) : t.elements() * 4;
        }
    size_t s1 = 0, s2 = 0;
    const strata::TensorInfo* gu = gm.find(pre + "ffn_gate_up_exps.weight", &s1);
    const strata::TensorInfo* dn = gm.find(pre + "ffn_down_exps.weight", &s2);
    if (!gu || !dn) {
        err = "missing " + pre + "ffn_gate_up_exps / ffn_down_exps";
        return 0;
    }
    const K::NativeExpertLayout L = K::native_expert_layout((int) gu->type, (int) dn->type, hp.n_embd, hp.n_ff);
    const size_t al = lcm_sz(lcm_sz(block_bytes(L.gu_type), block_bytes(L.d_type)), 16);
    stride = (L.bytes + al - 1) / al * al;
    const int res = g_experts_vram < 0 ? hp.n_expert : std::min(g_experts_vram, hp.n_expert);
    return b + stride * (size_t) res + kMmqTail;
}

/// A prompt's time a chunk on two cards with two executors (chunk c's visit waits for chunk c - 1's), each visit a unit:
/// the runs of visits on one card, scheduled earliest first.
double pipe_time(const HP& hp, const std::vector<int>& dev) {
    std::vector<std::pair<int, int>> seg;   // (card, visits)
    for (int il = 0; il < hp.n_vis; ++il) {
        const int d = dev[(size_t) hp.phys(il)];
        if (!seg.empty() && seg.back().first == d) ++seg.back().second;
        else seg.push_back({d, 1});
    }
    const int n_ch = 24, S = (int) seg.size();
    std::vector<double> done((size_t) n_ch * S, 1e30);   // (not yet: the next chunk waits)
    double free_at[2] = {0, 0}, at[2] = {0, 0};
    int c_of[2] = {0, 1}, s_of[2] = {0, 0};
    double end = 0;
    for (;;) {
        int w = -1;
        double st_w = 1e30;
        for (int e = 0; e < 2; ++e) {
            if (c_of[e] >= n_ch) continue;
            const int c = c_of[e], k = s_of[e];
            double st = std::max(at[e], free_at[seg[(size_t) k].first]);
            if (c > 0) st = std::max(st, done[(size_t) (c - 1) * S + k]);
            if (st < st_w) { w = e; st_w = st; }
        }
        if (w < 0) break;
        const int c = c_of[w], k = s_of[w], d = seg[(size_t) k].first;
        const double en = st_w + seg[(size_t) k].second;
        free_at[d] = at[w] = done[(size_t) c * S + k] = en;
        end = std::max(end, en);
        if (++s_of[w] == S) {
            s_of[w] = 0;
            c_of[w] += 2;
        }
    }
    return end / n_ch;
}
bool place(Model& m, const std::vector<uint64_t>& lb, const std::vector<uint64_t>& kv, uint64_t head_bytes,
           uint64_t reserve, std::string& err, bool all_cards = false) {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    n = std::min(n, kMaxDev);
    if (n < 1) {
        err = "no CUDA device";
        return false;
    }
    std::vector<uint64_t> room((size_t) n);
    for (int d = 0; d < n; ++d) {
        CK(cudaSetDevice(d));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        room[(size_t) d] = fr > reserve ? fr - reserve : 0;
    }
    const int L = (int) lb.size();
    uint64_t total = head_bytes;
    for (int l = 0; l < L; ++l) total += lb[(size_t) l] + kv[(size_t) l];
    int use = 0;
    uint64_t acc = 0;
    for (int d = 0; d < n; ++d) {
        acc += room[(size_t) d];
        ++use;
        if (acc >= total && !all_cards) break;   // the pool: every card, its VRAM shared by the experts and the caches
    }
    if (acc < total) {
        err = "the model needs " + std::to_string(total >> 20) + " MiB beside the engine's buffers and the " +
              std::to_string(n) + " card(s) have " + std::to_string(acc >> 20) +
              " MiB: use a smaller quant or a shorter --max-context";
        return false;
    }
    m.n_dev = use;
    m.dev_ids.clear();
    for (int d = 0; d < use; ++d) m.dev_ids.push_back(d);
    std::vector<uint64_t> cap((size_t) use);
    uint64_t sum_room = 0;
    for (int d = 0; d < use; ++d) sum_room += room[(size_t) d];
    int d = 0;
    uint64_t used = 0;
    for (int l = 0; l < L; ++l) {
        const uint64_t need = lb[(size_t) l] + kv[(size_t) l];
        const double share = (double) total * (double) room[(size_t) d] / (double) sum_room;
        const bool last = d == use - 1;
        const uint64_t lim = last ? room[(size_t) d] - head_bytes : room[(size_t) d];
        if (!last && used > 0 && ((double) (used + need) > share * 1.02 || used + need > lim)) {
            ++d;
            used = 0;
        }
        m.layers[(size_t) l].dev = d;
        used += need;
    }
    int head_card = -1;
    if (use == 2) {   // two cards: the split whose two-chunk prompt pipeline keeps both busiest, where both fit
        const HP& hp = m.hp;
        // card 0 takes [0, x) and [y, z): with a looped core, its halves on the two cards make every hand-off as long
        // (then, of those within 1% of the fastest, the one leaving both cards the closest free VRAM: the pool's experts;
        // 5% let through splits whose hand-offs leave a card waiting, 4-7% slower)
        struct Cand { int x, y, z, hd; double t, gap; };
        std::vector<Cand> cands;
        std::vector<int> dev((size_t) L);
        double best_t = 1e30;
        for (int x = 1; x < L; ++x)
            for (int y = x; y < L; ++y)
                for (int z = y; z <= L; ++z) {
                    if (z == L && y < L) continue;   // the last layer on card 1
                    uint64_t b[2] = {0, 0};
                    for (int l = 0; l < L; ++l) {
                        dev[(size_t) l] = l < x || (l >= y && l < z) ? 0 : 1;
                        b[dev[(size_t) l]] += lb[(size_t) l] + kv[(size_t) l];
                    }
                    const double t = pipe_time(hp, dev);
                    for (int hd = 0; hd < 2; ++hd) {   // the head (and a drafter) on either card: a row's hand-off
                        const uint64_t b0 = b[0] + (hd == 0 ? head_bytes : 0), b1 = b[1] + (hd == 1 ? head_bytes : 0);
                        if (b0 > room[0] || b1 > room[1]) continue;
                        cands.push_back({x, y, z, hd, t, std::fabs((double) room[0] - (double) b0 - ((double) room[1] - (double) b1))});
                        best_t = std::min(best_t, t);
                    }
                }
        int bx = -1, by = -1, bz = -1;
        double best_gap = 1e30;
        for (const Cand& c : cands)
            if (c.t <= best_t * 1.01 && c.gap < best_gap) { best_gap = c.gap; bx = c.x; by = c.y; bz = c.z; head_card = c.hd; }
        if (bx > 0)
            for (int l = 0; l < L; ++l) m.layers[(size_t) l].dev = l < bx || (l >= by && l < bz) ? 0 : 1;
    }
    m.head_dev = head_card >= 0 ? head_card : m.layers.back().dev;
    std::vector<uint64_t> on((size_t) use, 0);
    for (int l = 0; l < L; ++l) on[(size_t) m.layers[(size_t) l].dev] += lb[(size_t) l] + kv[(size_t) l];
    on[(size_t) m.head_dev] += head_bytes;
    for (int k = 0; k < use; ++k)
        if (on[(size_t) k] > room[(size_t) k]) {
            err = "the layers do not split over the cards (card " + std::to_string(k) + " would need " +
                  std::to_string(on[(size_t) k] >> 20) + " MiB of " + std::to_string(room[(size_t) k] >> 20) + ")";
            return false;
        }
    return true;
}

/// The VRAM cells of a full-attention cache: all of ctx, a fixed window (--kv-resident N), or the pool's smallest one.
uint64_t window_cells(int resident, int ctx, int tmax, bool q8) {
    if (!q8 || resident == 0) return (uint64_t) ctx;
    const int w = resident < 0 ? std::min(ctx, std::max(std::max(tmax, kGemmT), 4096) + 1024) : resident;
    return (uint64_t) std::min(w, ctx);
}
uint64_t row_bytes_est(const HP& hp) {
    const uint64_t N = hp.n_embd, QD = (uint64_t) hp.n_head * kHD, KD = (uint64_t) hp.n_kv * kHD, Kp = hp.top_k;
    const uint64_t floats = 30 * N + 4 * QD + 4 * KD + 2 * (uint64_t) hp.n_head * hp.n_rel + 64 + hp.n_expert + Kp;
    return floats * 4 + Kp * N * (2 + 4) + Kp * hp.n_ff * 2 + 2 * QD * 2 + 4 * N + Kp * 16;
}
uint64_t cache_bytes(const Model& m, int d, int ctx, int tmax, bool q8, int resident) {
    const HP& hp = m.hp;
    uint64_t b = 0;
    for (int il = 0; il < hp.n_vis; ++il) {
        if (m.layers[(size_t) hp.phys(il)].dev != d) continue;
        const bool swa = hp.swa_vis[(size_t) il];
        const uint64_t full = window_cells(resident, ctx, tmax, q8);
        const uint64_t cells = swa ? (uint64_t) (hp.swa + tmax) : full;
        const uint64_t per_cell = swa || !q8 ? (uint64_t) hp.n_kv * kHD * 2 * 2 : (uint64_t) hp.n_kv * (kHD + 4) * 2;
        b += cells * per_cell + (uint64_t) (2 * hp.n_kv * kHD + 2 * hp.n_embd) * 3 * 4;
    }
    return b;
}

/// The DSpark drafter's weights (on the target's head card) and its hyperparameters.
struct DsLayer {
    Mat wq, wk, wv, wo, kc, vc, gate, up, down;   // kc / vc: the injected context's K and V
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *kc_norm = nullptr, *ctx_norm = nullptr,
                *ffn_norm = nullptr;
    bool swa = false;
};
struct DsModel {
    std::shared_ptr<strata::GgufModel> file;
    int dev = 0, n_layer = 0, block = 8, rank = 0, n_ff = 0, swa = 0, window = 32768, n_head = 0, n_kv = 0;
    float eps = 1e-6f, base = 1e6f, softcap = 0;
    std::vector<int> taps;   // the target visits whose input is a feature (n_vis: the final normed hidden)
    std::vector<DsLayer> L;
    DsMix mix{};
    const float *out_norm = nullptr, *mask = nullptr, *joint_b = nullptr, *hidden_b = nullptr, *joint = nullptr, *hidden = nullptr;
    const uint16_t* markov = nullptr;
    int ring(const DsLayer& l) const { return l.swa ? 2048 : window; }
    int tap_slot(int il) const {   // the feature slot of the input of visit il, -1 for none
        for (size_t k = 0; k < taps.size(); ++k)
            if (taps[k] == il) return (int) k;
        return -1;
    }
};
/// A conversation's drafter cache: each layer's K and V rings (FP16), the head's context rows of the last positions,
/// and the first position since which every committed one was injected (the drafter forgets what came before).
struct DsState {
    std::vector<__half*> k, v;
    static constexpr int kCtx = 16;
    float* ctx = nullptr;          // [kCtx][N]
    int64_t ctx_pos[kCtx];
    int64_t valid_from = 0;
    int64_t drafted = 0, accepted = 0, steps = 0;   // the drafter's record (logged a reply)
    // decayed outcomes by draft position (draft i tried when 1 .. i - 1 stood) and the draft's time by its length
    double tried[8] = {}, stood[8] = {}, t_draft[8] = {};
    int64_t idle = 0;
    double rate(int i) const { return (stood[i] + 2.0) / (tried[i] + 3.0); }   // a prior of 2 in 3
    void outcome(int k, int n_stood) {
        for (int i = 1; i <= k; ++i) {
            tried[i] = tried[i] * 0.98 + (i <= n_stood + 1 ? 1.0 : 0.0);
            stood[i] = stood[i] * 0.98 + (i <= n_stood ? 1.0 : 0.0);
        }
    }
    void forget(int64_t pos) {
        valid_from = pos;
        for (auto& p : ctx_pos) p = -1;
    }
};


/// The DSpark drafter (--dspark FILE) onto the target's head card. False with err set when the file is not an Inkling 2
/// DSpark drafter matching the target.
bool load_dspark(const std::string& path, Model& m, DsModel& ds, std::string& err) {
    ds.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *ds.file;
    const strata::GgufFile& f = gm.meta();
    const strata::MetaValue* arch = f.get("general.architecture");
    if (!arch || arch->s != "dflash" || !gm.find("dspark_hidden_out.weight", nullptr)) {
        err = path + " is not an Inkling 2 DSpark drafter (general.architecture dflash with dspark_hidden_out)";
        return false;
    }
    auto num = [&](const char* k, double def = -1) -> double {
        const strata::MetaValue* v = f.get(std::string("dflash.") + k);
        if (!v) return def;
        if (v->type == strata::MetaType::ARRAY) return v->items.empty() ? def : v->items[0].num();
        return v->num();
    };
    const HP& hp = m.hp;
    ds.n_layer = (int) num("block_count");
    ds.block = (int) num("block_size", 8);
    ds.n_ff = (int) num("feed_forward_length");
    ds.n_head = (int) num("attention.head_count");
    ds.n_kv = (int) num("attention.head_count_kv");
    ds.swa = (int) num("attention.sliding_window") + 1;   // the anchor and the window before it
    ds.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
    ds.base = (float) num("rope.freq_base", 1e6);
    ds.softcap = (float) num("final_logit_softcapping", 0);
    const strata::MetaValue* tl = f.get("dflash.target_layers");
    const strata::MetaValue* pat = f.get("dflash.attention.sliding_window_pattern");
    if (tl && tl->type == strata::MetaType::ARRAY)
        for (const auto& it : tl->items) ds.taps.push_back((int) it.num());
    if ((int) num("embedding_length") != hp.n_embd || ds.n_layer != 3 || ds.block != 8 || ds.n_head != hp.n_head ||
        ds.n_kv != hp.n_kv || ds.taps.size() != 4 || ds.taps.back() != hp.n_vis || !pat || (int) pat->items.size() != ds.n_layer ||
        ds.n_ff % 32 || ds.swa > 2048 - ds.block) {
        err = path + ": not a drafter for this model (its width, heads, target layers or layout differ)";
        return false;
    }
    ds.dev = m.head_dev;
    {
        size_t need = 0;
        for (const auto& t : gm.meta().tensors()) need += strata::tensor_payload_bytes(t);
        make_room(ds.dev, need + (64ull << 20));
    }
    Loader ld(m, gm);
    const int d = ds.dev;
    ds.L.resize((size_t) ds.n_layer);
    for (int l = 0; l < ds.n_layer; ++l) {
        DsLayer& y = ds.L[(size_t) l];
        auto nm = [&](const char* x) { return Lname(l, x); };
        y.swa = pat->items[(size_t) l].u != 0;
        y.wq = ld.mat(d, nm("attn_q.weight"));
        y.wk = ld.mat(d, nm("attn_k.weight"));
        y.wv = ld.mat(d, nm("attn_v.weight"));
        y.wo = ld.mat(d, nm("attn_output.weight"));
        y.kc = ld.mat(d, nm("attn_k_ctx.weight"));
        y.vc = ld.mat(d, nm("attn_v_ctx.weight"));
        y.gate = ld.mat(d, nm("ffn_gate.weight"));
        y.up = ld.mat(d, nm("ffn_up.weight"));
        y.down = ld.mat(d, nm("ffn_down.weight"));
        y.attn_norm = ld.f32(d, nm("attn_norm.weight"));
        y.q_norm = ld.f32(d, nm("attn_q_norm.weight"));
        y.k_norm = ld.f32(d, nm("attn_k_norm.weight"));
        y.kc_norm = ld.f32(d, nm("attn_k_ctx_norm.weight"));
        y.ctx_norm = ld.f32(d, nm("dflare_norm.weight"));
        y.ffn_norm = ld.f32(d, nm("ffn_norm.weight"));
    }
    ds.out_norm = ld.f32(d, "output_norm.weight");
    ds.mask = ld.f32(d, "mask_embd.weight");
    ds.joint = ld.f32(d, "joint_proj.weight");
    ds.joint_b = ld.f32(d, "joint_proj.bias");
    ds.hidden = ld.f32(d, "dspark_hidden_out.weight");
    ds.hidden_b = ld.f32(d, "dspark_hidden_out.bias");
    ds.markov = ld.bf16(d, "markov_w1.weight");
    const float* dnorm = ld.f32(d, "dspark_norm.weight");
    {   // the fuse rows' softmax, once
        size_t sh = 0;
        const strata::TensorInfo* t = gm.find("dspark_fuse.weight", &sh);
        std::vector<float> h;
        if (!t || !host_f32(*t, gm.shard(sh).tensor_data(*t), h, ld.err) || h.size() != 16) {
            if (ld.err.empty()) ld.err = "dspark_fuse.weight: expected 4 x 4";
        } else
            for (int r = 0; r < 4; ++r) {
                float mx = -INFINITY, sum = 0.f;
                for (int k = 0; k < 4; ++k) mx = std::max(mx, h[(size_t) r * 4 + k]);
                for (int k = 0; k < 4; ++k) sum += std::exp(h[(size_t) r * 4 + k] - mx);
                for (int k = 0; k < 4; ++k) ds.mix.w[r][k] = std::exp(h[(size_t) r * 4 + k] - mx) / sum;
            }
    }
    for (int l = 0; l < 3; ++l) ds.mix.norm[l] = ds.L[(size_t) l].ctx_norm;
    ds.mix.norm[3] = dnorm;
    {
        size_t sh = 0;
        const strata::TensorInfo* t = gm.find("markov_w1.weight", &sh);
        ds.rank = t ? (int) t->shape[0] : 0;
    }
    if (!ld.err.empty()) {
        err = path + ": " + ld.err;
        return false;
    }
    if (ds.rank <= 0 || ds.rank > 256 || ds.rank % 4) {
        err = path + ": unexpected head rank";
        return false;
    }
    std::fprintf(stderr, "strata-inkling2: DSpark drafter %s: %d layers, block %d, rank %d, features of visits %d, %d, %d and "
                 "the last, %.0f MiB on GPU %d\n", path.c_str(), ds.n_layer, ds.block, ds.rank, ds.taps[0], ds.taps[1], ds.taps[2],
                 (double) ld.bytes[d] / (1 << 20), d);
    return true;
}

bool load(const std::string& path, Model& m, int ctx, int tmax, bool q8, int n_sess, int resident, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    HP& hp = m.hp;
    if (!read_hp(gm.meta(), hp, err)) return false;
    const int N = hp.n_embd;
    m.layers.resize((size_t) hp.n_phys);
    std::vector<uint64_t> lb((size_t) hp.n_phys), kv((size_t) hp.n_phys, 0);
    std::vector<size_t> strides((size_t) hp.n_phys);
    for (int l = 0; l < hp.n_phys; ++l) {
        lb[(size_t) l] = layer_bytes(gm, hp, l, strides[(size_t) l], err);
        if (!err.empty()) return false;
    }
    const uint64_t conv_bytes = (uint64_t) (2 * hp.n_kv * hp.hd + 2 * N) * 3 * 4;
    for (int il = 0; il < hp.n_vis; ++il) {
        const int ip = hp.phys(il);
        const bool swa = hp.swa_vis[(size_t) il];
        const uint64_t full = window_cells(resident, ctx, tmax, q8);
        const uint64_t cells = swa ? (uint64_t) (hp.swa + tmax) : full;
        const uint64_t per_cell = swa || !q8 ? (uint64_t) hp.n_kv * hp.hd * 2 * 2 : (uint64_t) hp.n_kv * (hp.hd + 4) * 2;
        kv[(size_t) ip] += (cells * per_cell + conv_bytes) * (uint64_t) n_sess;
    }
    size_t hs = 0;
    const strata::TensorInfo* th = gm.find("output.weight", &hs);
    if (!th) {
        err = "missing output.weight";
        return false;
    }
    const uint64_t head_bytes = strata::tensor_payload_bytes(*th);
    const uint64_t reserve = ((uint64_t) 1 << 30) + (uint64_t) tmax * row_bytes_est(hp);
    const bool pool = resident < 0;
    if (pool) {   // the experts that fit beside the rest start in VRAM, the same number in every layer
        uint64_t room = 0, total = head_bytes, experts = 0;
        int n = 0;
        CK(cudaGetDeviceCount(&n));
        for (int d = 0; d < std::min(n, kMaxDev); ++d) {
            CK(cudaSetDevice(d));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            room += fr > reserve ? fr - reserve : 0;
        }
        for (int l = 0; l < hp.n_phys; ++l) {
            total += lb[(size_t) l] + kv[(size_t) l];
            experts += strides[(size_t) l] * (uint64_t) hp.n_expert;
        }
        if (total > room && g_experts_vram < 0) {
            const double fit = (double) (room > total - experts ? room - (total - experts) : 0) / (double) experts;
            g_experts_vram = std::max(0, std::min(hp.n_expert, (int) (fit * hp.n_expert)));
            for (int l = 0; l < hp.n_phys; ++l) lb[(size_t) l] = layer_bytes(gm, hp, l, strides[(size_t) l], err);
        }
    }
    if (!place(m, lb, kv, head_bytes + g_head_extra, reserve, err, pool)) return false;
    // the pool: each card's layers start with the experts that fit beside its own weights and caches (the split is by
    // work, so one card may hold more)
    std::vector<int> slots_on((size_t) m.n_dev, g_experts_vram < 0 ? hp.n_expert : std::min(g_experts_vram, hp.n_expert));
    if (pool && g_experts_vram < 0)
        for (int d = 0; d < m.n_dev; ++d) {
            CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const uint64_t margin = reserve + (1792ull << 20);   // (the engine's buffers and the CUDA context: ~2 GiB a card measured)
            const uint64_t room = fr > margin ? fr - margin : 0;
            uint64_t dense = d == m.head_dev ? head_bytes + g_head_extra : 0, experts = 0;
            for (int l = 0; l < hp.n_phys; ++l)
                if (m.layers[(size_t) l].dev == d) {
                    const uint64_t ex = strides[(size_t) l] * (uint64_t) hp.n_expert;
                    experts += ex;
                    dense += lb[(size_t) l] - std::min(lb[(size_t) l], ex) + kv[(size_t) l];
                }
            if (experts > 0 && dense + experts > room) {
                const double fit = room > dense ? (double) (room - dense) / (double) experts : 0.0;
                slots_on[(size_t) d] = std::max(0, std::min(hp.n_expert, (int) (fit * hp.n_expert)));
            }
        }
    Loader ld(m);
    for (int l = 0; l < hp.n_phys; ++l) {
        Layer& y = m.layers[(size_t) l];
        const int dv = y.dev;
        auto nm = [&](const char* s) { return Lname(l, s); };
        y.swa = hp.swa_vis[(size_t) l];
        y.wq = ld.mat(dv, nm("attn_q.weight"));
        y.wk = ld.mat(dv, nm("attn_k.weight"));
        y.wv = ld.mat(dv, nm("attn_v.weight"));
        y.wo = ld.mat(dv, nm("attn_output.weight"));
        y.wg = ld.mat(dv, nm("attn_gate.weight"));
        y.wrel = ld.mat(dv, nm("attn_rel.weight"));
        y.q_clip = ld.f32(dv, nm("attn_q_clip.weight"));
        y.rel_bias = ld.f32(dv, nm("attn_rel_bias.weight"));
        y.rel_bias2.assign((size_t) m.n_dev, nullptr);   // on every card (the other half of a split window reads it there)
        for (int od = 0; od < m.n_dev; ++od) y.rel_bias2[(size_t) od] = od == dv ? y.rel_bias : ld.f32(od, nm("attn_rel_bias.weight"));
        y.rel_clip = ld.f32(dv, nm("attn_rel_clip.weight"));
        y.sink_proj = ld.f32(dv, nm("attn_sink_proj.weight"));
        y.k_conv = ld.f32(dv, nm("attn_k_conv.weight"));
        y.v_conv = ld.f32(dv, nm("attn_v_conv.weight"));
        y.a_conv = ld.f32(dv, nm("attn_post_conv.weight"));
        y.f_conv = ld.f32(dv, nm("ffn_post_conv.weight"));
        {
            size_t s = 0;
            const strata::TensorInfo* t = gm.find(nm("attn_rel_bias.weight"), &s);
            if (t) y.extent = (int) t->shape[1];
        }
        if (y.swa) {
            y.attn_norm = ld.f32(dv, nm("attn_norm.weight"));
        } else {
            y.q_norm = ld.f32(dv, nm("attn_q_inp_norm.weight"));
            y.k_norm = ld.f32(dv, nm("attn_k_inp_norm.weight"));
            y.v_norm = ld.f32(dv, nm("attn_v_inp_norm.weight"));
            y.kc_proj = ld.f32(dv, nm("attn_k_conv_proj.weight"));
            y.vc_proj = ld.f32(dv, nm("attn_v_conv_proj.weight"));
        }
        if (l > 0) {
            y.attn_res_q = ld.f32(dv, nm("attn_res_query.weight"));
            y.n_res_q = y.swa ? 1 : 3;
        }
        y.ffn_res_q = ld.f32(dv, nm("ffn_res_query.weight"));
        y.ffn_norm = ld.f32(dv, nm("ffn_norm.weight"));
        y.router = ld.f32(dv, nm("ffn_gate_inp.weight"));
        y.probs_b = ld.f32(dv, nm("exp_probs_b.bias"), false);
        if (hp.in_core(l)) {
            static const char* const mod[4] = {"q", "k", "v", "ffn"};   // the controllers of Q, K, V and the FFN input
            for (int k = 0; k < 4; ++k) {
                y.loop_down[k] = ld.f32(dv, Lname(l, ("loop_" + std::string(mod[k]) + "_ctrl_down.weight").c_str()));
                y.loop_gain[k] = ld.f32(dv, Lname(l, ("loop_" + std::string(mod[k]) + "_ctrl_gain.weight").c_str()));
                y.loop_forget[k] = ld.f32(dv, Lname(l, ("loop_" + std::string(mod[k]) + "_ctrl_forget.weight").c_str()));
            }
            y.msg_q = ld.f32(dv, Lname(l, "loop_msg_query.weight"));   // [n_embd, 4]: the Q, K, V and FFN messages
        }
        if (!ld.err.empty()) {
            err = ld.err;
            return false;
        }
        if (y.extent <= 0) {
            err = nm("attn_rel_bias.weight") + ": no distance rows";
            return false;
        }
        size_t sgu = 0, sd = 0;
        const strata::TensorInfo* tgu = gm.find(nm("ffn_gate_up_exps.weight"), &sgu);
        const strata::TensorInfo* td = gm.find(nm("ffn_down_exps.weight"), &sd);
        y.mma = (tgu->type == 40 || tgu->type == 8 || tgu->type == 12 || tgu->type == 13 || tgu->type == 14) &&
                (td->type == 40 || td->type == 8 || td->type == 6) && N % kEN == 0 &&
                hp.n_ff % (kEN / 2) == 0 && hp.n_ff % kEK == 0 && N % kEK == 0;
        if (!K::native_expert_supported((int) tgu->type, (int) td->type, N, hp.n_ff)) {
            err = nm("experts are ") + tgu->type_name() + "/" + td->type_name() +
                  ", which the native expert kernels do not take at this width";
            return false;
        }
        y.L = K::native_expert_layout((int) tgu->type, (int) td->type, N, hp.n_ff);
        y.L.situ_beta = hp.situ_beta;
        y.L.situ_lb = hp.situ_lb;
        y.stride = strides[(size_t) l];
        std::vector<float> s_gu((size_t) hp.n_expert, 1.0f), s_d((size_t) hp.n_expert, 1.0f);
        for (int r = 0; r < 2; ++r) {
            size_t ss = 0;
            const strata::TensorInfo* ts = gm.find(nm(r ? "ffn_down_exps.scale" : "ffn_gate_up_exps.scale"), &ss);
            if (!ts) continue;
            if ((r ? td : tgu)->type != 40) {
                err = ts->name + ": a per-expert scale on experts that are not NVFP4 (this driver folds it into the NVFP4 tail)";
                return false;
            }
            std::vector<float> h;
            if (!host_f32(*ts, gm.shard(ss).tensor_data(*ts), h, err)) return false;
            if ((int) h.size() != hp.n_expert) {
                err = ts->name + ": not one scale per expert";
                return false;
            }
            (r ? s_d : s_gu) = h;
        }
        CK(cudaSetDevice(m.dev_ids[(size_t) dv]));
        // every expert in pinned RAM (the CPU's experts, and where a VRAM slot is filled from)
        const size_t gub = (size_t) 2 * hp.n_ff * y.L.gu_row, db = (size_t) N * y.L.d_row;
        const uint8_t* src_gu = gm.shard(sgu).tensor_data(*tgu);
        const uint8_t* src_d = gm.shard(sd).tensor_data(*td);
        CK(cudaHostAlloc((void**) &y.hblobs, y.stride * (size_t) hp.n_expert, cudaHostAllocPortable));
        for (int e = 0; e < hp.n_expert; ++e) {
            uint8_t* dst = y.hblobs + (size_t) e * y.stride;
            std::memset(dst, 0, y.stride);
            std::memcpy(dst, src_gu + (size_t) e * gub, gub);
            std::memcpy(dst + y.L.down_off, src_d + (size_t) e * db, db);
            if (y.L.tail_off) {
                const float tail[4] = {s_gu[(size_t) e], s_gu[(size_t) e], s_d[(size_t) e], 0.f};
                std::memcpy(dst + y.L.tail_off, tail, sizeof tail);
            }
        }
        y.n_slots = slots_on[(size_t) y.dev];
        y.n_res = 0;
        y.eptr_h.assign((size_t) hp.n_expert, 0);
        y.on_gpu.assign((size_t) hp.n_expert, 0);
        y.slab_of.assign((size_t) hp.n_expert, -1);
        y.slot_of.assign((size_t) hp.n_expert, -1);
        CK(cudaMalloc(&y.eptr, (size_t) hp.n_expert * 8));
        for (int e = 0; e < y.n_slots; ++e)   // the first n_slots experts start in VRAM
            if (!m.load_expert(l, e)) {
                err = "no VRAM for " + nm("experts") + " (--experts-vram)";
                return false;
            }
        y.upload_ptrs();
        CK(cudaMalloc(&y.hist, (size_t) hp.n_expert * 4));
        CK(cudaMemset(y.hist, 0, (size_t) hp.n_expert * 4));
        y.cx = CpuExperts{y.hblobs, y.stride, y.L.gu_row, y.L.d_row, y.L.up_off, y.L.down_off, y.L.tail_off,
                          N, hp.n_ff, y.L.gu_type, y.L.d_type, hp.situ_beta, hp.situ_lb};
        if (!y.all_on_gpu() && !cpu_experts_supported(y.L.gu_type, y.L.d_type)) {
            err = nm("experts") + ": only NVFP4, Q8_0, Q4_K, Q5_K, Q6_K and Q5_0 experts can run from RAM (--experts-vram)";
            return false;
        }
        ld.bytes[dv] += y.stride * (size_t) y.n_slots;
    }
    const int d0 = m.layers.front().dev;
    m.tok_norm = ld.f32(d0, "token_embd_norm.weight");
    m.out_norm = ld.f32(m.head_dev, "output_norm.weight");
    m.out_res_q = ld.f32(m.head_dev, "output_res_query.weight");
    m.head = ld.mat(m.head_dev, "output.weight");
    if (!ld.err.empty()) {
        err = ld.err;
        return false;
    }
    if (m.head.n_out != hp.vocab_pad || m.head.n_in != N) {
        err = "output.weight is not [n_embd, vocab_size]";
        return false;
    }
    {
        size_t s = 0;
        const strata::TensorInfo* t = gm.find("token_embd.weight", &s);
        if (!t || !K::embed_type_supported((int) t->type)) {
            err = "token_embd.weight is missing or of a type the embedding kernel does not take";
            return false;
        }
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        m.embd = gm.shard(s).tensor_data(*t);
    }
    std::string where;
    for (int dv = 0; dv < m.n_dev; ++dv) {
        std::string runs;   // its runs of layers, "0-5, 12-17"
        for (int l = 0; l < hp.n_phys; ++l) {
            if (m.layers[(size_t) l].dev != dv || (l > 0 && m.layers[(size_t) l - 1].dev == dv)) continue;
            int e = l;
            while (e + 1 < hp.n_phys && m.layers[(size_t) e + 1].dev == dv) ++e;
            runs += (runs.empty() ? "" : ", ") + std::to_string(l) + "-" + std::to_string(e);
        }
        char b[200];
        std::snprintf(b, sizeof b, "%sGPU %d: layers %s%s, %.2f GiB", dv ? "; " : "", m.dev_ids[(size_t) dv], runs.c_str(),
                      m.head_dev == dv ? " and the head" : "", (double) ld.bytes[dv] / (1 << 30));
        where += b;
    }
    std::fprintf(stderr, "strata-inkling2: %s: %d layers (%d visits: core %d-%d x %d), %s, the embedding read from the file, "
                 "loaded in %.1f s\n", path.c_str(), hp.n_phys, hp.n_vis, hp.core0, hp.core1 - 1, hp.n_pass, where.c_str(),
                 ms_since(t0) / 1000.0);
    return true;
}

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

struct Engine;

/// A conversation's state: per visit its K / V cache and convolution state, and how far it has read.
struct DsState;
struct Session {
    DsState* ds = nullptr;   // the DSpark drafter's cache for this conversation (--dspark)
    std::vector<Kv> kv;
    std::vector<char> kv_q8;
    std::vector<float*> conv;
    int64_t pos = 0;
    int win = 0;   // the cells of each tiered full-attention cache in VRAM (the pool grows and shrinks it)
    int split = 0;   // its tiered windows split over the two cards now (decode), or whole on each visit's card
};

/// A conversation's state at `pos` besides its tiered caches' cells (which a lane's RAM copy or a parked conversation
/// holds): each sliding-window visit's last swa cells (K then V, [HKV][swa][kHD] FP16, oldest first) and each visit's
/// conv history, in pinned RAM. With the cells [0, pos) it is the whole state, so a request that starts with `ids`
/// continues from it. Immutable once taken; lanes and parked conversations share them.
struct Ckpt {
    int64_t pos = 0;
    std::vector<int32_t> ids;     // the tokens [0, pos)
    std::vector<uint8_t*> swa;    // [visit], nullptr for a full-attention one
    std::vector<float*> conv;     // [visit]
    size_t swa_bytes = 0, conv_bytes = 0;
    size_t bytes() const {
        size_t b = ids.size() * 4;
        for (auto* p : swa) b += p ? swa_bytes : 0;
        return b + conv.size() * conv_bytes;
    }
    ~Ckpt() {
        for (auto* p : swa) if (p) cudaFreeHost(p);
        for (auto* p : conv) if (p) cudaFreeHost(p);
    }
};
using CkptP = std::shared_ptr<Ckpt>;

struct Value {
    Engine* E = nullptr;
    int slot[kMaxDev];
    Value() { std::fill(slot, slot + kMaxDev, -1); }
    ~Value();
};
using V = std::shared_ptr<Value>;

struct Dev {
    float* cv_save = nullptr;   // a verify forward's conv inputs, kept until commit() knows the rows kept
    cudaEvent_t evmx = nullptr;   // experts in RAM: a decode's ids and q8_1 rows copied out for the CPU
    // ... and a prompt chunk's: experts copied into two halves of kStageExp VRAM slots, the groups' order and pointers
    uint8_t* stage = nullptr;
    cudaEvent_t evsr[2] = {}, evsf[2] = {};
    struct StagePlan {   // a layer's: its experts in VRAM, then those in RAM (U) into the slots; built for `key`
        std::vector<unsigned long long> key;   // the layer's eptr_h it was built for
        std::vector<int> U, rng;               // rng: the groups of the VRAM experts, then of each batch of U
        int32_t *order_d = nullptr, *rng_d = nullptr;
        unsigned long long* eptr_d = nullptr;
    };
    std::vector<StagePlan> plans;   // by physical layer
    size_t cv_cap = 0, cv_used = 0;
    int id = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    strata::prefill::Gemm gemm;
    std::vector<float*> slots;
    std::vector<int> free_slots;
    float* hctl = nullptr;   // the controls' split down partials (k_ctl_down)
    float *xn[3] = {}, *msg[4] = {}, *mixo = nullptr, *ctl = nullptr, *cat2 = nullptr, *h64 = nullptr, *gb = nullptr,
          *fb = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *kc = nullptr, *vc = nullptr, *r = nullptr, *sink = nullptr,
          *dk = nullptr, *dv = nullptr, *g = nullptr, *att = nullptr, *o = nullptr, *qb = nullptr, *rb = nullptr;
    float *moe = nullptr, *eparts = nullptr, *ewts = nullptr, *rlog = nullptr, *sk_part = nullptr;
    float *pacc = nullptr, *pm = nullptr, *pl = nullptr;
    float *logits = nullptr, *amv = nullptr;
    void *q8 = nullptr, *xq = nullptr, *escr = nullptr;
    void *q8b = nullptr, *q8c = nullptr;
    __half* x16 = nullptr;
    int2* tiles = nullptr;
    int32_t* n_tiles = nullptr;
    int ent_cap = 0;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr,
            *tok = nullptr, *pick = nullptr, *ami = nullptr;
    unsigned long long* grp_ptr = nullptr;
    uint8_t* erows = nullptr;              // a forward's embedding rows (GGUF blocks)
    int32_t* iota = nullptr;
    __half *xg = nullptr, *hg = nullptr;   // the prompt experts' FP16 rows
    float* dg = nullptr;
    float *run_o = nullptr, *run_m = nullptr, *run_l = nullptr;   // the prompt attention's running result (tiered)
    struct Stage {
        int8_t *k = nullptr, *v = nullptr;
        float *ks = nullptr, *vs = nullptr;
    } stg[2];                                                     // blocks of a RAM part staged in VRAM, double-buffered
    cudaStream_t cs = nullptr;                                    // their copies
    cudaEvent_t evc[2] = {}, evu[2] = {};
    cudaStream_t hs = nullptr;                                    // the CPU share of a decode's RAM part
    cudaEvent_t evq = nullptr, evcpu = nullptr;
    float *qh = nullptr, *cacc = nullptr, *cm = nullptr, *cl = nullptr;   // pinned: its query rows and partials
    int32_t* ent_inv = nullptr;
    // the other card's half of a split window, read here for a visit there: its own stream, the visit's query rows,
    // this half's partials (decode) or running result (prompt)
    cudaStream_t ps = nullptr;
    cudaEvent_t evh = nullptr, evp = nullptr;   // evh: the visit's rows are ready (recorded there); evp: this half done
    float *hq = nullptr, *hrb = nullptr, *hpacc = nullptr, *hpm = nullptr, *hpl = nullptr, *hro = nullptr, *hrm = nullptr,
          *hrl = nullptr;
    int hnc = 0;   // partials a row in hpacc
};

struct Engine {
    const Model* m = nullptr;
    int ctx = 32768, tmax = 512, ring = 0, att_q = 64, n_chunk_max = 0;
    Dev dev[kMaxDev];
    int nd = 0;
    bool q8_kv = true;
    Ckpt* ck_out = nullptr;   // a prompt chunk's forward copies each visit's state into it, as the visit ends
    bool kv_split = false;
    // DSpark: a forward's features ([row][4][N] on the drafter's card), the injection's context rows, the draft's buffers
    const DsModel* dsm = nullptr;
    int ds_mode = 0, ds_from = 0;   // 1: features from row ds_from on injected at the end; 2: features only (a verify)
    __half* ds_feat = nullptr;   // FP16: a draft's input only (the verify keeps the reply exact)
    float *ds_ctxb = nullptr, *dsx = nullptr, *dsh = nullptr, *ds_g = nullptr, *ds_up = nullptr,
          *ds_hid = nullptr, *ds_state = nullptr, *ds_lat = nullptr, *ds_j = nullptr, *ds_x = nullptr, *ds_corr = nullptr,
          *ds_u = nullptr;
    void* ds_q8 = nullptr;
    int32_t *ds_tok = nullptr, *ds_tok_h = nullptr;
    float *ds_base = nullptr, *ds_cv = nullptr, *ds_cl = nullptr;   // the rows' head logits, their candidates' values, a step's
    int32_t* ds_ci = nullptr;                                       // the rows' candidates
    unsigned *ds_tk = nullptr, *ds_tkh = nullptr;                   // their select's state and histograms
    float* ds_cl_h = nullptr;                                       // pinned: a step's candidates' logits (a sampled draft)
    int32_t* ds_ci_h = nullptr;
    int kv_resident = 0;              // > 0: a full-attention visit keeps its last kv_resident cells in VRAM, all in RAM;
                                      // < 0: the pool sizes each session's window (--kv-resident auto)
    /// A decode's read of a RAM part on CPU threads, one record a (visit, row group): the job the host function runs,
    /// and the times of both shares, which set the next split.
    struct CpuRec {
        CpuAttnJob job;
        double cpu_ms = 0;
        int cpu_keys = 0, gpu_keys = 0;
        cudaEvent_t g0 = nullptr, g1 = nullptr;
        bool pending = false;
    };
    std::vector<CpuRec> cpu_rec;
    bool experts_in_ram = false;   // some layer keeps experts in RAM: the CPU's and the staged paths
    bool decode_first = false;     // a lane's executor: its streams at the highest priority
    bool count_routes = false;     // the pool reads each layer's routings
    struct MoeRec {   // a visit's: its own pinned buffers (the GPU goes on past a visit whose job is still running)
        CpuMoeJob job;
        int32_t* ids_h = nullptr;
        uint8_t *xq_h = nullptr, *mask_h = nullptr;
        float* out_h = nullptr;
        volatile int* done = nullptr;   // mapped: the job's seq once its results are in out_h
        int seq = 0;
    };
    std::vector<MoeRec> moe_rec;   // a visit's CPU experts (the job the host function runs)
    static void CUDART_CB moe_run(void* p) {
        MoeRec* r = static_cast<MoeRec*>(p);
        cpu_moe(r->job);
        std::atomic_thread_fence(std::memory_order_release);
        *r->done = r->seq;
    }
    float cpu_frac = 0.4f;            // the CPU's share of a RAM part
    static void CUDART_CB cpu_run(void* p) {
        CpuRec& r = *static_cast<CpuRec*>(p);
        const auto t = std::chrono::steady_clock::now();
        cpu_attn(r.job);
        r.cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    }
    Session* S = nullptr;             // the session a forward runs
    int conv_c = 0;
    int last_dev = 0;                 // the card holding the logits
    int T_now = 0;                    // rows of the forward in flight (the values' copies)
    int32_t* hpin = nullptr;
    uint8_t* hemb = nullptr;   // pinned: a forward's embedding rows
    const ImgRows* img = nullptr;   // GENI: the request's pictures (a forward's rows with their ids get their values)
    float* himg = nullptr;          // pinned: a forward's picture rows
    int himg_rows = 0;

    V newv(int d) {
        Dev& D = dev[(size_t) d];
        if (D.free_slots.empty()) {
            std::fprintf(stderr, "strata-inkling2: out of residual slots on GPU %d\n", D.id);
            std::exit(1);
        }
        V v = std::make_shared<Value>();
        v->E = this;
        v->slot[d] = D.free_slots.back();
        D.free_slots.pop_back();
        return v;
    }
    void release(Value& v) {
        for (int d = 0; d < nd; ++d)
            if (v.slot[d] >= 0) dev[(size_t) d].free_slots.push_back(v.slot[d]);
    }
    float* at(const V& v, int d) {
        Dev& D = dev[(size_t) d];
        if (v->slot[d] >= 0) return D.slots[(size_t) v->slot[d]];
        int s = -1;
        for (int k = 0; k < nd; ++k)
            if (v->slot[k] >= 0) { s = k; break; }
        if (D.free_slots.empty()) {
            std::fprintf(stderr, "strata-inkling2: out of residual slots on GPU %d\n", D.id);
            std::exit(1);
        }
        v->slot[d] = D.free_slots.back();
        D.free_slots.pop_back();
        Dev& S = dev[(size_t) s];
        CK(cudaSetDevice(D.id));
        CK(cudaEventRecord(D.ev, D.st));
        CK(cudaSetDevice(S.id));
        CK(cudaStreamWaitEvent(S.st, D.ev, 0));
        CK(cudaMemcpyPeerAsync(D.slots[(size_t) v->slot[d]], D.id, S.slots[(size_t) v->slot[s]], S.id,
                               (size_t) T_now * m->hp.n_embd * 4, S.st));
        CK(cudaEventRecord(S.ev, S.st));
        CK(cudaSetDevice(D.id));
        CK(cudaStreamWaitEvent(D.st, S.ev, 0));
        return D.slots[(size_t) v->slot[d]];
    }

    void init(const Model& model, int ctx_, int tmax_, int ring_) {
        m = &model;
        static std::once_flag peers;   // the cards' copies to each other straight over PCIe, not through host memory
        std::call_once(peers, [&] {
            for (int a = 0; a < m->n_dev; ++a)
                for (int b = 0; b < m->n_dev; ++b) {
                    int ok = 0;
                    if (a == b || cudaDeviceCanAccessPeer(&ok, m->dev_ids[(size_t) a], m->dev_ids[(size_t) b]) != cudaSuccess || !ok) continue;
                    CK(cudaSetDevice(m->dev_ids[(size_t) a]));
                    const cudaError_t e = cudaDeviceEnablePeerAccess(m->dev_ids[(size_t) b], 0);
                    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) std::fprintf(stderr, "strata-inkling2: no peer access %d -> %d (%s)\n", a, b, cudaGetErrorString(e));
                    cudaGetLastError();
                }
        });
        ctx = ctx_;
        tmax = std::max(tmax_, kGemmT);
        ring = ring_;
        const bool prompt = tmax > kGemmT;
        const HP& hp = m->hp;
        const size_t T = (size_t) tmax, N = (size_t) hp.n_embd, Kp = (size_t) hp.top_k, FF = (size_t) hp.n_ff;
        const size_t H = (size_t) hp.n_head, HKV = (size_t) hp.n_kv, QD = H * kHD, KD = HKV * kHD;
        // a decode's partials a row: the VRAM part's blocks (65 at most), the other card's, the RAM part's staged blocks
        // (kHostKeys keys a partial) and the CPU's segments; the old one-pass kernel (more query heads a KV head than
        // the tensor-core one takes) reads kChunkKeys a partial
        const bool dm_ok = hp.n_head / hp.n_kv <= 3 && hp.n_rel <= 16;
        n_chunk_max = dm_ok ? 66 + 1 + (ctx + kHostKeys - 1) / kHostKeys + 1 + (kv_resident != 0 ? kCpuSegs + 2 : 0)
                            : (ctx + kChunkKeys - 1) / kChunkKeys + 1 + (kv_resident != 0 ? kCpuSegs + 2 : 0);
        att_q = (int) std::max<size_t>(1, std::min<size_t>(64, ((size_t) 32 << 20) / (H * (size_t) n_chunk_max * kHD)));
        conv_c = (int) (2 * KD + 2 * N);
        experts_in_ram = kv_resident < 0;   // the pool may move experts to RAM at any time
        // two cards: each full-attention window half on each, read by both at once (needs the tiered int8 caches)
        kv_split = m->n_dev == 2 && kv_resident != 0 && q8_kv && hp.n_head / hp.n_kv <= 3 && hp.n_rel <= 16;
        for (const Layer& L : m->layers) experts_in_ram |= !L.all_on_gpu();
        count_routes = kv_resident < 0;
        nd = m->n_dev;
        int64_t big = 0;
        for (const Layer& L : m->layers)
            for (const Mat* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.wg, &L.wrel})
                big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
        bool all_mma = true;
        for (const Layer& L : m->layers) all_mma = all_mma && L.mma;
        const size_t TG = all_mma ? std::min(T, (size_t) kExpertGemmT) : T;
        for (int d = 0; d < m->n_dev; ++d) {
            Dev& D = dev[(size_t) d];
            D.id = m->dev_ids[(size_t) d];
            CK(cudaSetDevice(D.id));
            int lo_pr = 0, hi_pr = 0;   // decode (a lane) before prompt chunks on the same card
            CK(cudaDeviceGetStreamPriorityRange(&lo_pr, &hi_pr));
            const int pr = decode_first ? hi_pr : lo_pr;
            CK(cudaStreamCreateWithPriority(&D.st, cudaStreamNonBlocking, pr));
            {
                size_t vis = 0;
                for (int il = 0; il < hp.n_vis; ++il) vis += m->layers[(size_t) hp.phys(il)].dev == d;
                D.cv_cap = vis * (size_t) conv_c * kGemmT;
                D.cv_save = dalloc(D.cv_cap);
            }
            CK(cudaEventCreateWithFlags(&D.ev, cudaEventDisableTiming));
            const int n_slots = 12;
            for (int s = 0; s < n_slots; ++s) {
                D.slots.push_back(dalloc(T * N));
                D.free_slots.push_back(n_slots - 1 - s);
            }
            for (auto& p : D.xn) p = dalloc(T * N);
            D.msg[0] = dalloc(4 * T * N);   // the four messages of a looped layer, one after the other
            D.mixo = dalloc(3 * T * N);
            D.ctl = dalloc(T * N);
            D.cat2 = dalloc(2 * T * N);
            D.h64 = dalloc(3 * T * (size_t) std::max(64, hp.gate_rank));   // up to 3 controls at once (decode)
            D.hctl = dalloc(3 * (size_t) std::min(T, (size_t) kGemmT) * (kCtlSplits * (size_t) std::max(64, hp.gate_rank) + kCtlSplits));
            D.gb = dalloc(T * N);
            D.fb = dalloc(T * N);
            D.q = dalloc(T * QD);
            D.k = dalloc(T * KD);
            D.v = dalloc(T * KD);
            D.kc = dalloc(T * KD);
            D.vc = dalloc(T * KD);
            D.r = dalloc(T * H * hp.n_rel);
            D.rb = dalloc(T * H * hp.n_rel);
            D.qb = dalloc(T * QD);
            D.sink = dalloc(T * H);
            D.dk = dalloc(T * HKV * 4);
            D.dv = dalloc(T * HKV * 4);
            D.g = dalloc(T * QD);
            D.att = dalloc(T * QD);
            D.o = dalloc(T * N);
            D.moe = dalloc(T * N);
            D.eparts = dalloc(TG * Kp * N);
            D.ewts = dalloc(T * Kp);
            D.rlog = dalloc(T * hp.n_expert);
            if (prompt) D.sk_part = dalloc(kSplitFloats);
            const size_t rows = (size_t) std::min({att_q, tmax, kAttTileT - 1}) * H;   // (more rows: the prompt kernel)
            D.pacc = dalloc(rows * n_chunk_max * kHD);
            D.pm = dalloc(rows * n_chunk_max);
            D.pl = dalloc(rows * n_chunk_max);
            D.logits = dalloc((size_t) kGemmT * hp.vocab_pad);
            D.amv = dalloc((size_t) kArgBlocks);
            D.ami = dalloc_t<int32_t>((size_t) kArgBlocks);
            D.pick = dalloc_t<int32_t>(1);
            const int max_in = (int) std::max<size_t>(QD, 2 * N);
            CK(cudaMalloc(&D.q8, K::native_q8_1_bytes(max_in, 1) * T));
            for (void** p : {&D.q8b, &D.q8c}) CK(cudaMalloc(p, K::native_q8_1_bytes((int) N, 1) * std::min(T, (size_t) 8)));   // (k_q8_seg's k, v inputs)
            CK(cudaMalloc(&D.xq, K::native_q8_1_bytes((int) N, 1) * T));
            D.x16 = dalloc_t<__half>(T * (size_t) max_in);
            CK(cudaMalloc(&D.escr, K::native_expert_scratch_bytes((int64_t) (TG * Kp), (int64_t) FF)));
            D.ids = dalloc_t<int32_t>(T * Kp);
            D.grp_ptr = dalloc_t<unsigned long long>((size_t) hp.n_expert + 1);
            D.grp_start = dalloc_t<int32_t>((size_t) hp.n_expert + 2);
            D.tiles = dalloc_t<int2>(T * Kp / 64 + (size_t) hp.n_expert + 1);   // the expert GEMM's row tiles
            D.n_tiles = dalloc_t<int32_t>(1);
            D.ent_cap = (int) (T * Kp);
            D.n_groups = dalloc_t<int32_t>(1);
            D.ent_dst = dalloc_t<int32_t>(T * Kp);
            D.ent_tok = dalloc_t<int32_t>(T * Kp);
            D.tok = dalloc_t<int32_t>(T);
            D.erows = dalloc_t<uint8_t>(T * m->embd_row);
            {
                std::vector<int32_t> io(T);
                for (size_t i = 0; i < T; ++i) io[i] = (int32_t) i;
                D.iota = dalloc_t<int32_t>(T);
                CK(cudaMemcpy(D.iota, io.data(), T * 4, cudaMemcpyHostToDevice));
            }
            D.ent_inv = dalloc_t<int32_t>(T * Kp);
            if (prompt && kv_resident != 0) {
                D.run_o = dalloc(T * QD);
                D.run_m = dalloc(T * H);
                D.run_l = dalloc(T * H);
            }
            CK(cudaStreamCreateWithPriority(&D.cs, cudaStreamNonBlocking, pr));
            CK(cudaStreamCreateWithPriority(&D.hs, cudaStreamNonBlocking, pr));
            CK(cudaEventCreateWithFlags(&D.evq, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&D.evcpu, cudaEventDisableTiming));
            if (dsm && d == dsm->dev) ds_alloc();
            if (kv_split) {
                CK(cudaStreamCreateWithPriority(&D.ps, cudaStreamNonBlocking, hi_pr));   // ahead of this card's own work: a visit there waits on it
                CK(cudaEventCreateWithFlags(&D.evh, cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&D.evp, cudaEventDisableTiming));
                D.hq = dalloc(T * QD);
                D.hrb = dalloc(T * H * hp.n_rel);
                D.hnc = std::min(ctx / 2048, 64) + 18;   // a decode's partials (fewer than kAttTileT rows take that path)
                const size_t hrows = (size_t) std::min({att_q, tmax, kAttTileT - 1}) * H;
                D.hpacc = dalloc(hrows * (size_t) D.hnc * kHD);
                D.hpm = dalloc(hrows * (size_t) D.hnc);
                D.hpl = dalloc(hrows * (size_t) D.hnc);
                if (prompt) {
                    D.hro = dalloc(T * QD);
                    D.hrm = dalloc(T * H);
                    D.hrl = dalloc(T * H);
                }
            }
            if (experts_in_ram) {
                CK(cudaEventCreateWithFlags(&D.evmx, cudaEventDisableTiming));   // (the jobs' buffers: a visit's MoeRec)
                if (T >= (size_t) kExpertGemmT) {
                    size_t stride = 0;
                    for (const Layer& L : m->layers)
                        if (L.dev == d) stride = std::max(stride, L.stride);
                    CK(cudaMalloc(&D.stage, 2 * (size_t) kStageExp * stride + kMmqTail));
                    for (int i = 0; i < 2; ++i) {
                        CK(cudaEventCreateWithFlags(&D.evsr[i], cudaEventDisableTiming));
                        CK(cudaEventCreateWithFlags(&D.evsf[i], cudaEventDisableTiming));
                    }
                    D.plans.resize(m->layers.size());
                }
            }
            if (kv_resident != 0) {
                for (auto& b : D.stg) {
                    b.k = dalloc_t<int8_t>(HKV * kStage * kHD);
                    b.v = dalloc_t<int8_t>(HKV * kStage * kHD);
                    b.ks = dalloc(HKV * kStage);
                    b.vs = dalloc(HKV * kStage);
                }
                for (int i = 0; i < 2; ++i) {
                    CK(cudaEventCreateWithFlags(&D.evc[i], cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&D.evu[i], cudaEventDisableTiming));
                }
                const size_t rq = (size_t) kAttTileT * H;
                CK(cudaHostAlloc((void**) &D.qh, rq * kHD * 4, cudaHostAllocPortable));
                CK(cudaHostAlloc((void**) &D.cacc, rq * kCpuSegs * kHD * 4, cudaHostAllocPortable));
                CK(cudaHostAlloc((void**) &D.cm, rq * kCpuSegs * 4, cudaHostAllocPortable));
                CK(cudaHostAlloc((void**) &D.cl, rq * kCpuSegs * 4, cudaHostAllocPortable));
            }
            if (T >= (size_t) kExpertGemmT) {
                D.xg = dalloc_t<__half>(T * Kp * N);
                D.hg = dalloc_t<__half>(T * Kp * FF);
                D.dg = dalloc(T * Kp * N);
            }
            CK(cudaFuncSetAttribute(k_expert_mma<40, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_smem<40>()));
            CK(cudaFuncSetAttribute(k_expert_mma<40, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_smem<40>()));
            CK(cudaFuncSetAttribute(k_expert_mma<8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_smem<8>()));
            CK(cudaFuncSetAttribute(k_expert_mma<8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_smem<8>()));
            CK(cudaFuncSetAttribute(k_expert_ws<40, true, 64, 128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<40, 64, 128>()));
            CK(cudaFuncSetAttribute(k_expert_ws<40, false, 64, 128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<40, 64, 128>()));
            CK(cudaFuncSetAttribute(k_expert_ws<12, true, 64, 128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<12, 64, 128>()));
            CK(cudaFuncSetAttribute(k_expert_ws<6, false, 64, 128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<6, 64, 128>()));
            CK(cudaFuncSetAttribute(k_expert_ws<8, true, 64, 128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<8, 64, 128>()));
            CK(cudaFuncSetAttribute(k_expert_ws<8, false, 64, 128, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<8, 64, 128>()));
            CK(cudaFuncSetAttribute(k_expert_ws<13, true, 64, 64, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<13, 64, 64>()));
            CK(cudaFuncSetAttribute(k_expert_ws<14, true, 64, 64, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert_ws_smem<14, 64, 64>()));
            CK(cudaFuncSetAttribute(k_expert_mma2<8, true, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert2_smem<8, 64>()));
            CK(cudaFuncSetAttribute(k_expert_mma2<8, false, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) expert2_smem<8, 64>()));
            CK(cudaFuncSetAttribute(k_attn_fa2<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kFaSmem));
            CK(cudaFuncSetAttribute(k_attn_fa2<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kFaSmem));
            CK(cudaFuncSetAttribute(k_attn_dreg<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kDrSmem));
            CK(cudaFuncSetAttribute(k_attn_dreg<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kDrSmem));
            CK(cudaFuncSetAttribute(k_attn_tile, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int) attn_tile_smem(kMaxG, std::max(hp.n_rel, 1))));
            std::string gerr;
            if (prompt && !D.gemm.init(D.st, big, gerr)) {
                std::fprintf(stderr, "strata-inkling2: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        last_dev = m->head_dev;
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &hemb, T * m->embd_row, cudaHostAllocDefault));
        for (int di = 0; di < nd; ++di) {
            Dev& D = dev[di];
            CK(cudaSetDevice(D.id));
            CK(cudaDeviceSynchronize());
        }
        std::string fr;
        for (int di = 0; di < nd; ++di) {
            Dev& D = dev[di];
            CK(cudaSetDevice(D.id));
            size_t f = 0, t = 0;
            CK(cudaMemGetInfo(&f, &t));
            char b[64];
            std::snprintf(b, sizeof b, "%sGPU %d %.2f GiB", fr.empty() ? "" : ", ", D.id, (double) f / (1 << 30));
            fr += b;
        }
        if (prompt)
            std::fprintf(stderr, "strata-inkling2: context %d cells, prompt chunk %d, free VRAM after the caches: %s\n", ctx,
                         tmax, fr.c_str());
    }

    /// A session's caches and states (on the cards of their layers).
    void alloc_session(Session& ss) const {
        const HP& hp = m->hp;
        const size_t H = (size_t) hp.n_head, HKV = (size_t) hp.n_kv;
        for (int il = 0; il < hp.n_vis; ++il) {
            const Layer& L = m->layers[(size_t) hp.phys(il)];
            CK(cudaSetDevice(m->dev_ids[(size_t) L.dev]));
            Kv c;
            const bool q8 = q8_kv && !L.swa && H / HKV <= 3 && hp.n_rel <= 16;
            const int w0 = kv_resident < 0 ? min_window() : kv_resident;
            const bool tier = q8 && kv_resident != 0 && w0 < ctx;
            c.ring = L.swa ? ring : tier ? w0 : 0;
            c.cells = L.swa ? ring : tier ? w0 : ctx;
            if (tier) ss.win = w0;
            const size_t cells = (size_t) c.cells;
            if (tier) {
                c.hcells = ctx;
                const size_t hb = HKV * (size_t) ctx;
                CK(cudaHostAlloc((void**) &c.hk, hb * kHD, cudaHostAllocPortable | cudaHostAllocMapped));
                CK(cudaHostAlloc((void**) &c.hv, hb * kHD, cudaHostAllocPortable | cudaHostAllocMapped));
                CK(cudaHostAlloc((void**) &c.hks, hb * 4, cudaHostAllocPortable | cudaHostAllocMapped));
                CK(cudaHostAlloc((void**) &c.hvs, hb * 4, cudaHostAllocPortable | cudaHostAllocMapped));
            }
            if (tier && kv_split) {   // half here, half on the other card
                ss.split = 1;
                c.split = 1;
                c.dev2 = 1 - L.dev;
                c.cells = split_cells(w0);
                alloc_halves(c, L.dev);
            } else if (q8) {
                c.k = dalloc_t<int8_t>(HKV * cells * kHD);
                c.v = dalloc_t<int8_t>(HKV * cells * kHD);
                c.ks = dalloc(HKV * cells);
                c.vs = dalloc(HKV * cells);
            } else {
                c.k = dalloc_t<__half>(HKV * cells * kHD);
                c.v = dalloc_t<__half>(HKV * cells * kHD);
            }
            ss.kv.push_back(c);
            ss.kv_q8.push_back(q8 ? 1 : 0);
            ss.conv.push_back(dalloc((size_t) conv_c * 3));
        }
        if (dsm) {   // the drafter's cache: each layer's ring (the sliding ones 2048 cells, the full one its window)
            size_t need = (size_t) DsState::kCtx * hp.n_embd * 4;
            for (const DsLayer& l : dsm->L) need += 2 * HKV * (size_t) dsm->ring(l) * kHD * 2;
            make_room(dsm->dev, need);
            CK(cudaSetDevice(m->dev_ids[(size_t) dsm->dev]));
            auto* st = new DsState;
            for (const DsLayer& l : dsm->L) {
                const size_t n = HKV * (size_t) dsm->ring(l) * kHD;
                st->k.push_back(dalloc_t<__half>(n));
                st->v.push_back(dalloc_t<__half>(n));
            }
            st->ctx = dalloc((size_t) DsState::kCtx * hp.n_embd);
            st->forget(0);
            ss.ds = st;
        }
    }

    /// The smallest VRAM window of a tiered cache: a prompt chunk and the bias range.
    int min_window() const { return std::min(ctx, std::max(tmax, 4096) + 1024); }
    /// A split window of w positions: each half's ring (its blocks of the window, a block more for a window that does
    /// not start on a block pair).
    static int split_cells(int w) { return (w + 2 * kSplitB - 1) / (2 * kSplitB) * kSplitB + kSplitB; }
    /// The DSpark buffers on the drafter's card (room made for them first).
    void ds_alloc() {
        const size_t T = (size_t) tmax, N = (size_t) m->hp.n_embd, FF = (size_t) dsm->n_ff, B = (size_t) dsm->block,
                     R = (size_t) dsm->rank;
        make_room(dsm->dev, 3 * T * N * 4 + 3 * B * FF * 4 + (8ull << 20));
        CK(cudaSetDevice(m->dev_ids[(size_t) dsm->dev]));
        ds_feat = dalloc_t<__half>(T * 4 * N);
        ds_ctxb = dalloc(T * N);   // a layer's context rows at a time
        dsx = dalloc(B * N);
        dsh = dalloc(B * N);
        ds_g = dalloc(B * FF);
        ds_up = dalloc(B * FF);
        ds_hid = dalloc(B * FF);
        CK(cudaMalloc(&ds_q8, K::native_q8_1_bytes((int) FF, 1) * B));
        ds_state = dalloc(R);
        ds_lat = dalloc(R);
        ds_j = dalloc(3 * R);
        ds_x = dalloc(2 * R + 2 * N);
        ds_corr = dalloc(N);
        ds_u = dalloc(N);
        ds_tok = dalloc_t<int32_t>(B + 1);
        ds_base = dalloc(B * (size_t) m->hp.vocab_pad);
        ds_cv = dalloc(B * kDsCand);
        ds_cl = dalloc(kDsCand);
        ds_ci = dalloc_t<int32_t>(B * kDsCand);
        ds_tk = dalloc_t<unsigned>(B * 5);
        ds_tkh = dalloc_t<unsigned>(B * kTkBins);
        CK(cudaMemset(ds_tkh, 0, B * kTkBins * 4));
        CK(cudaHostAlloc((void**) &ds_cl_h, kDsCand * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &ds_ci_h, B * kDsCand * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &ds_tok_h, (B + 1) * 4, cudaHostAllocPortable));
    }
    void alloc_halves(Kv& c, int d) const {
        const size_t HKV = (size_t) m->hp.n_kv, n = HKV * (size_t) c.cells;
        CK(cudaSetDevice(m->dev_ids[(size_t) d]));
        c.k = dalloc_t<int8_t>(n * kHD);
        c.v = dalloc_t<int8_t>(n * kHD);
        c.ks = dalloc(n);
        c.vs = dalloc(n);
        CK(cudaSetDevice(m->dev_ids[(size_t) c.dev2]));
        c.k2 = dalloc_t<int8_t>(n * kHD);
        c.v2 = dalloc_t<int8_t>(n * kHD);
        c.ks2 = dalloc(n);
        c.vs2 = dalloc(n);
        CK(cudaSetDevice(m->dev_ids[(size_t) d]));
    }
    /// A split window's cells for positions [p0, pos) from the RAM copy: each half's 64-position blocks, every other
    /// block of the copy, as strided copies (runs between the ring's wrap).
    void refill_halves(Kv& c, int d, int64_t p0, int64_t pos) const {
        const size_t HKV = (size_t) m->hp.n_kv;
        for (int part = 0; part < 2; ++part) {
            CK(cudaSetDevice(m->dev_ids[(size_t) (part ? c.dev2 : d)]));
            int8_t* dk = (int8_t*) (part ? c.k2 : c.k);
            int8_t* dv = (int8_t*) (part ? c.v2 : c.v);
            float* dks = part ? c.ks2 : c.ks;
            float* dvs = part ? c.vs2 : c.vs;
            auto copy = [&](int64_t a, int64_t nb, int64_t len) {   // nb blocks from position a (len positions each)
                const int64_t cell = kv_local(a) % c.cells;
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t ho = kh * (size_t) c.hcells + (size_t) a, ro = kh * (size_t) c.cells + (size_t) cell;
                    CK(cudaMemcpy2D(dk + ro * kHD, kSplitB * kHD, c.hk + ho * kHD, 2 * kSplitB * kHD, (size_t) len * kHD, (size_t) nb,
                                    cudaMemcpyHostToDevice));
                    CK(cudaMemcpy2D(dv + ro * kHD, kSplitB * kHD, c.hv + ho * kHD, 2 * kSplitB * kHD, (size_t) len * kHD, (size_t) nb,
                                    cudaMemcpyHostToDevice));
                    CK(cudaMemcpy2D(dks + ro, kSplitB * 4, c.hks + ho, 2 * kSplitB * 4, (size_t) len * 4, (size_t) nb, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy2D(dvs + ro, kSplitB * 4, c.hvs + ho, 2 * kSplitB * 4, (size_t) len * 4, (size_t) nb, cudaMemcpyHostToDevice));
                }
            };
            int64_t b = p0 / kSplitB;
            if ((b & 1) != part) ++b;
            for (; b * kSplitB < pos;) {
                const int64_t a = std::max<int64_t>(p0, b * kSplitB), e = std::min<int64_t>(pos, (b + 1) * kSplitB);
                if (a != b * kSplitB || e != (b + 1) * kSplitB) {   // a partial block: on its own
                    copy(a, 1, e - a);
                    b += 2;
                    continue;
                }
                // whole blocks b, b + 2, ... while inside [p0, pos) and before the ring wraps
                const int64_t cell0 = kv_local(a) % c.cells;
                int64_t nb = 0;
                while ((b + 2 * nb + 1) * kSplitB <= pos && cell0 + (nb + 1) * kSplitB <= c.cells) ++nb;
                if (nb == 0) nb = 1;
                copy(a, nb, kSplitB);
                b += 2 * nb;
            }
        }
        CK(cudaSetDevice(m->dev_ids[(size_t) d]));
    }
    /// Each tiered cache's VRAM ring rebuilt at w cells from the RAM copy (the old ring freed first: the copy has every
    /// cell). The session's executors are idle (the pool holds its lock).
    void set_window(Session& ss, int w, bool refill = false, int layout = -1) const {
        const HP& hp = m->hp;
        const size_t HKV = (size_t) hp.n_kv;
        w = std::max(min_window(), std::min(w, ctx));
        if (layout < 0) layout = ss.split;
        if (layout != ss.split) refill = true;   // the other layout: every window rebuilt from the RAM copy
        if ((w == ss.win && !refill) || ss.win == 0) return;
        const int64_t pos = ss.pos, p0 = std::max<int64_t>(0, pos - w);
        for (int d = 0; d < m->n_dev; ++d) {
            CK(cudaSetDevice(m->dev_ids[(size_t) d]));
            CK(cudaDeviceSynchronize());
        }
        for (int il = 0; il < hp.n_vis; ++il) {   // every old window freed first (a card takes others' halves after)
            Kv& c = ss.kv[(size_t) il];
            if (!c.hk) continue;
            for (void* x : {c.k, c.v, (void*) c.ks, (void*) c.vs, c.k2, c.v2, (void*) c.ks2, (void*) c.vs2})
                if (x) CK(cudaFree(x));
            c.k = c.v = c.k2 = c.v2 = nullptr;
            c.ks = c.vs = c.ks2 = c.vs2 = nullptr;
        }
        for (int il = 0; il < hp.n_vis; ++il) {
            Kv& c = ss.kv[(size_t) il];
            if (!c.hk) continue;
            const int d = m->layers[(size_t) hp.phys(il)].dev;
            c.ring = w;
            if (layout) {
                c.split = 1;
                c.dev2 = 1 - d;
                c.cells = split_cells(w);
                alloc_halves(c, d);
                refill_halves(c, d, p0, pos);
                continue;
            }
            c.split = 0;
            c.cells = w;
            CK(cudaSetDevice(m->dev_ids[(size_t) d]));
            c.k = dalloc_t<int8_t>(HKV * (size_t) w * kHD);
            c.v = dalloc_t<int8_t>(HKV * (size_t) w * kHD);
            c.ks = dalloc(HKV * (size_t) w);
            c.vs = dalloc(HKV * (size_t) w);
            for (size_t kh = 0; kh < HKV; ++kh)
                for (int64_t a = p0; a < pos;) {   // cells a .. b - 1 are contiguous in the new ring
                    const int64_t b = std::min<int64_t>(pos, (a / w + 1) * w), n = b - a;
                    const size_t ho = kh * (size_t) c.hcells + (size_t) a, ro = kh * (size_t) w + (size_t) (a % w);
                    CK(cudaMemcpy((int8_t*) c.k + ro * kHD, c.hk + ho * kHD, (size_t) n * kHD, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy((int8_t*) c.v + ro * kHD, c.hv + ho * kHD, (size_t) n * kHD, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(c.ks + ro, c.hks + ho, (size_t) n * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(c.vs + ro, c.hvs + ho, (size_t) n * 4, cudaMemcpyHostToDevice));
                    a = b;
                }
        }
        ss.win = w;
        ss.split = layout;
    }
    /// The VRAM bytes a cell of a tiered window takes on card d (every tiered visit there).
    uint64_t window_cell_bytes(const Session& ss, int d) const {
        uint64_t b = 0;
        for (int il = 0; il < m->hp.n_vis; ++il)
            if (ss.kv[(size_t) il].hk && ss.kv[(size_t) il].split) b += (uint64_t) m->hp.n_kv * (kHD + 4);   // half of it here
            else if (ss.kv[(size_t) il].hk && m->layers[(size_t) m->hp.phys(il)].dev == d) b += (uint64_t) m->hp.n_kv * (kHD + 4) * 2;
        return b;
    }

    void reset() {
        for (size_t il = 0; il < S->conv.size(); ++il) {
            const Layer& L = m->layers[(size_t) m->hp.phys((int) il)];
            Dev& D = dev[(size_t) L.dev];
            CK(cudaSetDevice(D.id));
            CK(cudaMemsetAsync(S->conv[il], 0, (size_t) conv_c * 3 * 4, D.st));
        }
        sync_all();
        S->pos = 0;
    }
    void sync_all() {
        for (int di = 0; di < nd; ++di) {
            Dev& D = dev[di];
            CK(cudaSetDevice(D.id));
            CK(cudaStreamSynchronize(D.st));
        }
    }

    void prep(Dev& D, const float* x, int n, int T) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, D.st>>>(x, D.x16, (int64_t) n * T);
        else K::native_quantize_q8_1(x, D.q8, n * T, 1, D.st);
    }
    /// Q8_0 projections of one input each (q8_1), one launch (k_q8_seg); T <= 8.
    void apply_q8_seg(Dev& D, const Mat* const* w, const void* const* x, float* const* out, int n, int T) {
        QSegs a{};
        int blk = 0;
        for (int i = 0; i < n; ++i) {
            a.s[i] = QSeg{(const uint8_t*) w[i]->d, (const uint8_t*) x[i], out[i], w[i]->n_out, blk};
            blk += (w[i]->n_out + 7) / 8;
        }
        a.n = n;
        const int n_in = w[0]->n_in;
        switch (T) {
#define STRATA_INK_Q8SEG(TT) case TT: k_q8_seg<TT><<<(unsigned) blk, 256, 0, D.st>>>(a, n_in); break;
            STRATA_INK_Q8SEG(1) STRATA_INK_Q8SEG(2) STRATA_INK_Q8SEG(3) STRATA_INK_Q8SEG(4)
            STRATA_INK_Q8SEG(5) STRATA_INK_Q8SEG(6) STRATA_INK_Q8SEG(7) STRATA_INK_Q8SEG(8)
#undef STRATA_INK_Q8SEG
        }
    }
    void apply(Dev& D, const Mat& w, float* out, int T) {
        if (T > kGemmT) {
            D.gemm.native((const uint16_t*) D.x16, w.type, w.d, out, T, w.n_out, w.n_in);
            return;
        }
        if (K::native_mmvq_supported(w.type)) K::native_mmvq(w.type, w.d, D.q8, out, w.n_in, w.n_out, T, D.st);
        else K::iq_mmvq(w.type, w.d, D.q8, out, w.n_in, w.n_out, T, D.st);
    }
    void f32mm(Dev& D, const float* W, const float* x, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3((unsigned) ((n_out + 7) / 8), (unsigned) T), kThreads, 0, D.st>>>(W, x, out, n_in, n_out);
            return;
        }
        const int gx = (n_out + 63) / 64, gy = (T + 63) / 64;
        const int want = std::max(1, std::min(16, 128 / std::max(1, gx * gy)));
        const int fit = (int) std::max<size_t>(1, kSplitFloats / ((size_t) T * n_out));
        const int kz = (n_in / std::min(want, fit) + 15) / 16 * 16;
        const int splits = (n_in + kz - 1) / kz;
        float* dst = splits > 1 ? D.sk_part : out;
        k_sgemm_nt<<<dim3((unsigned) gx, (unsigned) gy, (unsigned) splits), 256, 0, D.st>>>(W, x, dst, n_in, n_out, T, kz);
        if (splits > 1) k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, D.st>>>(D.sk_part, out, (int64_t) T * n_out, splits);
    }

    void mix(Dev& D, int d, const std::vector<V>& blocks, const V& cur, const float* Q, int nq, float* out, int T) {
        States st{};
        int ns = 0;
        for (const V& b : blocks) st.p[ns++] = at(b, d);
        st.p[ns++] = at(cur, d);
        // a decode's few rows: their output over 4 blocks of 512 threads (one block a row was one SM's work)
        if (T <= kGemmT) k_mix<<<dim3((unsigned) T, 4), 512, 0, D.st>>>(st, ns, Q, nq, out, T, m->hp.n_embd);
        else k_mix<<<T, kThreads, 0, D.st>>>(st, ns, Q, nq, out, T, m->hp.n_embd);
    }
    /// Controls k0 .. k0 + nk - 1 of a looped layer for a decode's rows, in two launches (see k_ctl_down).
    void controls(Dev& D, const Layer& L, int k0, int nk, const float* const* x, const float* const* msg, float* const* out, int T) {
        const int N = m->hp.n_embd, R = m->hp.gate_rank;
        CtlArgs a{};
        for (int j = 0; j < nk; ++j) {
            a.x[j] = x[j];
            a.msg[j] = msg[j];
            a.wd[j] = L.loop_down[k0 + j];
            a.wg[j] = L.loop_gain[k0 + j];
            a.wf[j] = L.loop_forget[k0 + j];
            a.out[j] = out[j];
        }
        k_ctl_down<<<dim3((unsigned) ((R + 7) / 8 * kCtlSplits), (unsigned) T, (unsigned) nk), 256, 0, D.st>>>(a, D.hctl, N, R, 1e-6f);
        k_ctl_up<<<dim3((unsigned) ((N + 15) / 16), (unsigned) T, (unsigned) nk), 256, 0, D.st>>>(a, D.hctl, N, R, 1e-6f);
    }
    void control(Dev& D, const Layer& L, int k, const float* x, const float* msg, float* out, int T) {
        const int N = m->hp.n_embd, R = m->hp.gate_rank;
        k_rms2cat<<<T, kThreads, 0, D.st>>>(x, msg, D.cat2, N, 1e-6f);
        f32mm(D, L.loop_down[k], D.cat2, D.h64, 2 * N, R, T);
        f32mm(D, L.loop_gain[k], D.h64, D.gb, R, N, T);
        f32mm(D, L.loop_forget[k], D.h64, D.fb, R, N, T);
        k_ctrl<<<grid1((int64_t) T * N), kThreads, 0, D.st>>>(x, msg, D.gb, D.fb, out, (int64_t) T * N);
    }

    template <int TY, bool GU>
    void expert_mma(Dev& D, const Layer& L, const __half* X, size_t w_off, size_t row_bytes, int n_cols, int K, int g0 = 0,
                    const int32_t* ng = nullptr, int grid_g = 0) {
        const HP& hp = m->hp;
        // NVFP4: the warp-specialized kernel (64 x 128 tiles); Q8_0 (its raw stages do not fit there): 64 x 128 as well
        constexpr int MR = 64;
        const unsigned gy = (unsigned) (D.ent_cap / MR + (grid_g > 0 ? grid_g : hp.n_expert) + 1);   // the most tiles
        const int hpm = L.L.d_type == 40 ? 1 : 0;
        const int32_t* ngp = ng ? ng : D.n_groups;
        k_tiles<<<1, 32, 0, D.st>>>(D.grp_start + g0, ngp, MR, D.tiles, D.n_tiles);
        if constexpr (TY == 13 || TY == 14)   // (their super-blocks: 64 columns a block, so the buffers fit)
            k_expert_ws<TY, GU, MR, 64, 32><<<dim3((unsigned) (n_cols / (GU ? 32 : 64)), gy), ws_threads<MR, 64, 32>(), expert_ws_smem<TY, MR, 64>(), D.st>>>(
                X, D.grp_ptr + g0, D.grp_start + g0, ngp, w_off, row_bytes, K, hp.n_ff, L.L.tail_off, hp.situ_beta, hp.situ_lb, D.dg, D.hg, hp.n_embd, hpm, D.tiles, D.n_tiles);
        else if constexpr (TY == 40 || TY == 12 || TY == 6 || TY == 8)
            k_expert_ws<TY, GU, MR, 128, 32><<<dim3((unsigned) (n_cols / (GU ? 64 : 128)), gy), ws_threads<MR, 128, 32>(), expert_ws_smem<TY, MR, 128>(), D.st>>>(
                X, D.grp_ptr + g0, D.grp_start + g0, ngp, w_off, row_bytes, K, hp.n_ff, L.L.tail_off, hp.situ_beta, hp.situ_lb, D.dg, D.hg, hp.n_embd, hpm, D.tiles, D.n_tiles);
        else
            k_expert_mma2<TY, GU, MR><<<dim3((unsigned) (n_cols / (GU ? x_cols<MR>() / 2 : x_cols<MR>())), gy), 256, expert2_smem<TY, MR>(), D.st>>>(
                X, D.grp_ptr + g0, D.grp_start + g0, ngp, w_off, row_bytes, K, hp.n_ff, L.L.tail_off, hp.situ_beta, hp.situ_lb, D.dg, D.hg, hp.n_embd, hpm, D.tiles, D.n_tiles);
    }
    void experts_gemm(Dev& D, const Layer& L, const float* xn, int n_ent, int g0 = 0, const int32_t* ng = nullptr,
                      int grid_g = 0, bool gather = true) {
        const HP& hp = m->hp;
        const int N = hp.n_embd, FF = hp.n_ff;
        if (gather && L.L.gu_type == 40) k_gather_f16p<<<n_ent, kThreads, 0, D.st>>>(xn, D.ent_tok, D.xg, n_ent, N);
        else if (gather) k_gather_f16<<<n_ent, kThreads, 0, D.st>>>(xn, D.ent_tok, D.xg, n_ent, N);
        if (L.L.gu_type == 40) expert_mma<40, true>(D, L, D.xg, 0, L.L.gu_row, FF, N, g0, ng, grid_g);
        else if (L.L.gu_type == 12) expert_mma<12, true>(D, L, D.xg, 0, L.L.gu_row, FF, N, g0, ng, grid_g);
        else if (L.L.gu_type == 13) expert_mma<13, true>(D, L, D.xg, 0, L.L.gu_row, FF, N, g0, ng, grid_g);
        else if (L.L.gu_type == 14) expert_mma<14, true>(D, L, D.xg, 0, L.L.gu_row, FF, N, g0, ng, grid_g);
        else expert_mma<8, true>(D, L, D.xg, 0, L.L.gu_row, FF, N, g0, ng, grid_g);
        if (L.L.d_type == 40) expert_mma<40, false>(D, L, D.hg, L.L.down_off, L.L.d_row, N, FF, g0, ng, grid_g);
        else if (L.L.d_type == 6) expert_mma<6, false>(D, L, D.hg, L.L.down_off, L.L.d_row, N, FF, g0, ng, grid_g);
        else expert_mma<8, false>(D, L, D.hg, L.L.down_off, L.L.d_row, N, FF, g0, ng, grid_g);
    }
    /// A prompt chunk's experts when some are in RAM: the groups ordered (the experts in VRAM, then the layer's others
    /// kStageExp at a time, every one: a chunk's routings reach all of them, so no read back of the routing holds the
    /// host thread), each batch copied into one half of the staging slots while the GPU runs the one before. The order,
    /// pointers and batches are the layer's, built again when its experts in VRAM change. The same products as with
    /// every expert in VRAM.
    void experts_staged(Dev& D, const Layer& L, const float* xn, int T, int n_ent) {
        const HP& hp = m->hp;
        const int E = hp.n_expert, N = hp.n_embd;
        Dev::StagePlan& P = D.plans[(size_t) (&L - m->layers.data())];
        if (P.key != L.eptr_h) {
            std::vector<int32_t> order;
            P.U.clear();
            for (int x = 0; x < E; ++x)
                if (L.on_gpu[(size_t) x]) order.push_back(x);
                else P.U.push_back(x);
            const int nR = (int) order.size(), nb = (int) ((P.U.size() + kStageExp - 1) / kStageExp);
            order.insert(order.end(), P.U.begin(), P.U.end());
            std::vector<unsigned long long> ep(L.eptr_h);
            for (size_t j = 0; j < P.U.size(); ++j)
                ep[(size_t) P.U[j]] = (unsigned long long) (D.stage + ((j / kStageExp) % 2 * kStageExp + j % kStageExp) * L.stride);
            P.rng.assign(1, nR);
            for (int b = 0; b < nb; ++b) P.rng.push_back((int) std::min<size_t>(kStageExp, P.U.size() - (size_t) b * kStageExp));
            if (!P.order_d) {
                P.order_d = dalloc_t<int32_t>((size_t) E);
                P.rng_d = dalloc_t<int32_t>((size_t) E + 2);
                P.eptr_d = dalloc_t<unsigned long long>((size_t) E);
            }
            CK(cudaStreamSynchronize(D.st));   // (an earlier chunk's visit may still read the old ones)
            CK(cudaMemcpy(P.order_d, order.data(), (size_t) E * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(P.eptr_d, ep.data(), (size_t) E * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(P.rng_d, P.rng.data(), P.rng.size() * 4, cudaMemcpyHostToDevice));
            P.key = L.eptr_h;
        }
        k_group<<<1, kThreads, 0, D.st>>>(D.ids, n_ent, hp.top_k, E, P.eptr_d, P.order_d, D.grp_ptr, D.grp_start,
                                           D.n_groups, D.ent_dst, D.ent_tok, D.ent_inv, E);
        const bool mma = L.mma;
        auto run = [&](int g0, int k) {   // the groups [g0, g0 + rng[k])
            const int n = P.rng[(size_t) k];
            if (n <= 0) return;
            if (mma)
                experts_gemm(D, L, xn, n_ent, g0, P.rng_d + k, n, false);
            else
                K::native_expert_grouped(L.L, D.grp_ptr + g0, D.grp_start + g0, P.rng_d + k, D.ent_dst, D.ent_tok, n, n_ent,
                                         D.xq, D.escr, D.eparts, D.st);
        };
        if (mma && L.L.gu_type == 40) k_gather_f16p<<<n_ent, kThreads, 0, D.st>>>(xn, D.ent_tok, D.xg, n_ent, N);
        else if (mma) k_gather_f16<<<n_ent, kThreads, 0, D.st>>>(xn, D.ent_tok, D.xg, n_ent, N);
        run(0, 0);
        const int nR = P.rng[0];
        for (int b = 0; b + 1 < (int) P.rng.size(); ++b) {
            const int half = b % 2;
            CK(cudaStreamWaitEvent(D.cs, D.evsf[half], 0));
            for (int i = 0; i < P.rng[(size_t) (1 + b)]; ++i) {
                const int x = P.U[(size_t) b * kStageExp + (size_t) i];
                uint8_t* dst = D.stage + (size_t) (half * kStageExp + i) * L.stride;
                CK(cudaMemcpyAsync(dst, L.hblobs + (size_t) x * L.stride, L.stride, cudaMemcpyHostToDevice, D.cs));
            }
            CK(cudaEventRecord(D.evsr[half], D.cs));
            CK(cudaStreamWaitEvent(D.st, D.evsr[half], 0));
            run(nR + b * kStageExp, 1 + b);
            CK(cudaEventRecord(D.evsf[half], D.st));
        }
        (void) T;
    }
    /// The cells [0, end) of a tiered cache's RAM part, kStage at a time copied into VRAM on the copy stream (the next
    /// block's copy runs while `f` reads this one), f(store, first cell, last cell) on the compute stream.
    template <class F> void each_host_block(Dev& D, const Kv& cv, int end, F&& f) {
        const size_t HKV = (size_t) m->hp.n_kv;
        for (int b0 = 0, b = 0; b0 < end; b0 += kStage, ++b) {
            const int i = b & 1, n = std::min(kStage, end - b0);
            const Dev::Stage& st = D.stg[i];
            CK(cudaStreamWaitEvent(D.cs, D.evu[i], 0));
            CK(cudaMemcpy2DAsync(st.k, (size_t) kStage * kHD, cv.hk + (size_t) b0 * kHD, (size_t) cv.hcells * kHD,
                                 (size_t) n * kHD, HKV, cudaMemcpyHostToDevice, D.cs));
            CK(cudaMemcpy2DAsync(st.v, (size_t) kStage * kHD, cv.hv + (size_t) b0 * kHD, (size_t) cv.hcells * kHD,
                                 (size_t) n * kHD, HKV, cudaMemcpyHostToDevice, D.cs));
            CK(cudaMemcpy2DAsync(st.ks, (size_t) kStage * 4, cv.hks + b0, (size_t) cv.hcells * 4, (size_t) n * 4, HKV,
                                 cudaMemcpyHostToDevice, D.cs));
            CK(cudaMemcpy2DAsync(st.vs, (size_t) kStage * 4, cv.hvs + b0, (size_t) cv.hcells * 4, (size_t) n * 4, HKV,
                                 cudaMemcpyHostToDevice, D.cs));
            CK(cudaEventRecord(D.evc[i], D.cs));
            CK(cudaStreamWaitEvent(D.st, D.evc[i], 0));
            f(KvSrc{st.k, st.v, st.ks, st.vs, 0, kStage, b0}, b0, b0 + n - 1);
            CK(cudaEventRecord(D.evu[i], D.st));
        }
    }

    /// Rows r0 .. r0 + n - 1 of the last forward's features (positions pos ..) into the drafter's cache: each layer's
    /// context row (the fused features, RMS-normed) to K and V, and the head's context rows of the last positions.
    void ds_inject(Dev& D, DsState& st, int r0, int n, int64_t pos) {
        const int N = m->hp.n_embd, HKV = m->hp.n_kv;
        const __half* f = ds_feat + (size_t) r0 * 4 * N;
        for (int l = 0; l < dsm->n_layer; ++l) {
            const DsLayer& y = dsm->L[(size_t) l];
            k_ds_mix<<<(unsigned) n, kThreads, 0, D.st>>>(f, N, dsm->mix, l, dsm->eps, ds_ctxb);
            prep(D, ds_ctxb, N, n);
            apply(D, y.kc, D.k, n);
            apply(D, y.vc, D.v, n);
            k_ds_kput<<<dim3((unsigned) n, (unsigned) HKV), kHD, 0, D.st>>>(D.k, D.v, (int) pos, y.kc_norm, dsm->ring(y), st.k[(size_t) l],
                                                                            st.v[(size_t) l], dsm->eps, dsm->base, HKV);
        }
        const int t0 = std::max(0, n - DsState::kCtx);   // the head's context rows: the last positions only
        k_ds_mix<<<(unsigned) (n - t0), kThreads, 0, D.st>>>(f + (size_t) t0 * 4 * N, N, dsm->mix, 3, dsm->eps, ds_ctxb);
        for (int t = t0; t < n; ++t) {
            const int64_t p = pos + t;
            CK(cudaMemcpyAsync(st.ctx + (size_t) (p % DsState::kCtx) * N, ds_ctxb + (size_t) (t - t0) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, D.st));
            st.ctx_pos[p % DsState::kCtx] = p;
        }
    }
    /// Up to kmax tokens to follow `anchor` (the last token, at position S->pos, not yet in the state): the block
    /// [anchor, mask x 7] through the drafter's layers, then its head a token at a time (greedy). Empty when the drafter
    /// has no context for this position (a state it did not see being made).
    /// `pick` (a sampled request): each step's logits to the host, the token drawn there (the caller keeps its
    /// distribution for the verify); else the argmax on the card.
    using DsPick = std::function<int32_t(const int32_t*, const float*, int, int)>;   // (tokens, logits, n, step) -> token
    static int32_t pick_cand(const DsPick& pick, const int32_t* ids, const float* lg, int step) { return pick(ids, lg, kDsCand, step); }
    std::vector<float> ds_lgv;
    std::vector<int32_t> ds_draft(Session& ss, int32_t anchor, int kmax, const DsPick& pick = nullptr) {
        std::vector<int32_t> out;
        DsState* st = ss.ds;
        const int64_t pos0 = ss.pos;
        kmax = std::min(kmax, dsm ? dsm->block - 1 : 0);
        if (!st || kmax <= 0 || pos0 < 1 || st->ctx_pos[(pos0 - 1) % DsState::kCtx] != pos0 - 1 || st->valid_from > pos0 - 1) return out;
        const HP& hp = m->hp;
        const int N = hp.n_embd, HKV = hp.n_kv, G = hp.n_head / hp.n_kv, B = dsm->block, R = dsm->rank;
        Dev& D = dev[(size_t) dsm->dev];
        CK(cudaSetDevice(D.id));
        // row 0: the anchor's embedding, normed as the target's first step does; rows 1 ..: the mask embedding
        std::memcpy(hemb, m->embd + (size_t) anchor * m->embd_row, m->embd_row);
        CK(cudaMemcpyAsync(D.erows, hemb, m->embd_row, cudaMemcpyHostToDevice, D.st));
        K::iq_embed_rows(m->embd_type, D.erows, m->embd_row, D.iota, 1, N, D.o, D.st);
        k_rms<<<1, kThreads, 0, D.st>>>(D.o, m->tok_norm, dsx, N, hp.eps);
        for (int r = 1; r < B; ++r) CK(cudaMemcpyAsync(dsx + (size_t) r * N, dsm->mask, (size_t) N * 4, cudaMemcpyDeviceToDevice, D.st));
        const float scale = 1.0f / std::sqrt((float) kHD);
        const int64_t hi = pos0 + B - 1;
        for (int l = 0; l < dsm->n_layer; ++l) {
            const DsLayer& y = dsm->L[(size_t) l];
            const int ring = dsm->ring(y);
            k_rms<<<B, kThreads, 0, D.st>>>(dsx, y.attn_norm, D.xn[0], N, dsm->eps);
            prep(D, D.xn[0], N, B);
            apply(D, y.wq, D.q, B);
            apply(D, y.wk, D.k, B);
            apply(D, y.wv, D.v, B);
            k_ds_qprep<<<dim3((unsigned) B, (unsigned) hp.n_head), kHD, 0, D.st>>>(D.q, (int) pos0, y.q_norm, D.qb, dsm->eps, dsm->base, G, HKV);
            k_ds_kput<<<dim3((unsigned) B, (unsigned) HKV), kHD, 0, D.st>>>(D.k, D.v, (int) pos0, y.k_norm, ring, st->k[(size_t) l],
                                                                            st->v[(size_t) l], dsm->eps, dsm->base, HKV);
            // every row sees the injected context in the window before the anchor and the whole block (non-causal)
            const int64_t lo = std::max<int64_t>(st->valid_from, y.swa ? pos0 - (dsm->swa - 1) : hi + 1 - ring);
            const int span = (int) (hi - std::max<int64_t>(0, lo) + 1);
            const int lo_i = (int) std::max<int64_t>(0, lo);
            const int dk = std::min(std::max(2048, (span + 64 * 64 - 1) / (64 * 64) * 64), std::max(256, (span + 16 * 64 - 1) / (16 * 64) * 64));
            const int nc = (span - 1) / dk + 1;
            const KvSrc src{st->k[(size_t) l], st->v[(size_t) l], nullptr, nullptr, ring, ring, 0};
            const int GE = B * G;   // one token's "heads": the block's rows x the KV head's queries
            k_attn_dreg<false><<<dim3((unsigned) ((GE + 15) / 16), (unsigned) HKV, (unsigned) nc), 128, kDrSmem, D.st>>>(
                D.qb, nullptr, src, nullptr, 0, 0, (int) hi, 0, 1, HKV * GE, GE, 0, lo_i, (int) hi, lo_i, dk, 0, nc, scale, D.pacc, D.pm,
                D.pl);
            k_ds_merge<<<(unsigned) (HKV * GE), kHD, 0, D.st>>>(D.pacc, D.pm, D.pl, nc, D.att, B, G);
            prep(D, D.att, hp.n_head * kHD, B);
            apply(D, y.wo, D.o, B);
            k_add<<<grid1((int64_t) B * N), kThreads, 0, D.st>>>(dsx, dsx, D.o, (int64_t) B * N);
            k_rms<<<B, kThreads, 0, D.st>>>(dsx, y.ffn_norm, D.xn[0], N, dsm->eps);
            prep(D, D.xn[0], N, B);
            apply(D, y.gate, ds_g, B);
            apply(D, y.up, ds_up, B);
            k_silu_mul<<<grid1((int64_t) B * dsm->n_ff), kThreads, 0, D.st>>>(ds_g, ds_up, ds_hid, (int64_t) B * dsm->n_ff);
            K::native_quantize_q8_1(ds_hid, ds_q8, dsm->n_ff * B, 1, D.st);
            if (K::native_mmvq_supported(y.down.type)) K::native_mmvq(y.down.type, y.down.d, ds_q8, D.o, y.down.n_in, y.down.n_out, B, D.st);
            else K::iq_mmvq(y.down.type, y.down.d, ds_q8, D.o, y.down.n_in, y.down.n_out, B, D.st);
            k_add<<<grid1((int64_t) B * N), kThreads, 0, D.st>>>(dsx, dsx, D.o, (int64_t) B * N);
        }
        k_rms<<<B, kThreads, 0, D.st>>>(dsx, dsm->out_norm, dsh, N, dsm->eps);
        // the head: one read of the target's head for all the draft rows (their logits, each row's top candidates); then
        // a token at a time, each candidate's logit plus its head row times the step's correction, softcapped
        const bool q80 = m->head.type == 8;
        if (q80) {
            K::native_quantize_q8_1(dsh + N, D.q8, N * kmax, 1, D.st);
            K::native_mmvq(m->head.type, m->head.d, D.q8, ds_base, N, m->head.n_out, kmax, D.st);
            k_tk_init<<<1, 32, 0, D.st>>>(ds_tk, kmax);
            const int shifts[3] = {21, 10, 0}, bits[3] = {11, 11, 10};
            for (int ps = 0; ps < 3; ++ps) {
                k_tk_hist<<<dim3(kTkBlocks, (unsigned) kmax), 512, 0, D.st>>>(ds_base, hp.vocab, hp.vocab_pad, ds_tk, ds_tkh, shifts[ps], bits[ps]);
                k_tk_pick<<<(unsigned) kmax, 256, 0, D.st>>>(ds_tk, ds_tkh, shifts[ps], bits[ps]);
            }
            k_tk_take<<<dim3(kTkBlocks, (unsigned) kmax), 512, 0, D.st>>>(ds_base, hp.vocab, hp.vocab_pad, ds_tk, ds_ci, ds_cv);
            if (pick) CK(cudaMemcpyAsync(ds_ci_h, ds_ci, (size_t) kmax * kDsCand * 4, cudaMemcpyDeviceToHost, D.st));
        }
        ds_tok_h[0] = anchor;
        CK(cudaMemcpyAsync(ds_tok, ds_tok_h, 4, cudaMemcpyHostToDevice, D.st));
        CK(cudaMemsetAsync(ds_state, 0, (size_t) R * 4, D.st));
        const float* ctx = st->ctx + (size_t) ((pos0 - 1) % DsState::kCtx) * N;
        const size_t hrow = (size_t) (N / 32) * 34;
        for (int i = 0; i < kmax; ++i) {
            const float* row = dsh + (size_t) (i + 1) * N;
            k_ds_head_in<<<grid1(2 * R + 2 * N), kThreads, 0, D.st>>>(dsm->markov, ds_tok + i, ds_state, row, ctx, ds_x, R, N);
            k_gemv_f32<<<dim3((unsigned) ((3 * R + 7) / 8), 1), kThreads, 0, D.st>>>(dsm->joint, ds_x, ds_j, 2 * R + 2 * N, 3 * R);
            k_ds_state<<<grid1(R), kThreads, 0, D.st>>>(ds_j, dsm->joint_b, ds_state, ds_lat, R);
            k_gemv_f32<<<dim3((unsigned) ((N + 7) / 8), 1), kThreads, 0, D.st>>>(dsm->hidden, ds_lat, ds_corr, R, N);
            if (q80) {   // (the correction's bias with it: one vector)
                k_add<<<grid1(N), kThreads, 0, D.st>>>(ds_u, ds_corr, dsm->hidden_b, (int64_t) N);
                k_ds_cand<<<kDsCand / 8, 256, 0, D.st>>>(ds_ci + (size_t) i * kDsCand, ds_cv + (size_t) i * kDsCand, m->head.d, hrow,
                                                         ds_u, N, dsm->softcap, ds_cl);
                if (pick) {
                    CK(cudaMemcpyAsync(ds_cl_h, ds_cl, kDsCand * 4, cudaMemcpyDeviceToHost, D.st));
                    CK(cudaStreamSynchronize(D.st));
                    ds_tok_h[i + 1] = pick_cand(pick, ds_ci_h + (size_t) i * kDsCand, ds_cl_h, i);
                    CK(cudaMemcpyAsync(ds_tok + i + 1, ds_tok_h + i + 1, 4, cudaMemcpyHostToDevice, D.st));
                } else
                    k_cand_argmax<<<1, kDsCand, 0, D.st>>>(ds_ci + (size_t) i * kDsCand, ds_cl, ds_tok + i + 1);
                continue;
            }
            k_ds_u<<<grid1(N), kThreads, 0, D.st>>>(row, ds_corr, dsm->hidden_b, ds_u, N);
            K::native_quantize_q8_1(ds_u, D.q8, N, 1, D.st);
            if (K::native_mmvq_supported(m->head.type)) K::native_mmvq(m->head.type, m->head.d, D.q8, D.logits, N, m->head.n_out, 1, D.st);
            else K::iq_mmvq(m->head.type, m->head.d, D.q8, D.logits, N, m->head.n_out, 1, D.st);
            if (dsm->softcap > 0.f)
                k_softcap<<<grid1((int64_t) hp.vocab), kThreads, 0, D.st>>>(D.logits, hp.vocab, hp.vocab_pad, 1, dsm->softcap);
            if (pick) {   // (a head the candidates' kernel does not read: the full logits to the host)
                ds_lgv.resize((size_t) hp.vocab);
                std::vector<int32_t> all((size_t) hp.vocab);
                for (int j = 0; j < hp.vocab; ++j) all[(size_t) j] = j;
                CK(cudaMemcpyAsync(ds_lgv.data(), D.logits, (size_t) hp.vocab * 4, cudaMemcpyDeviceToHost, D.st));
                CK(cudaStreamSynchronize(D.st));
                ds_tok_h[i + 1] = pick(all.data(), ds_lgv.data(), hp.vocab, i);
                CK(cudaMemcpyAsync(ds_tok + i + 1, ds_tok_h + i + 1, 4, cudaMemcpyHostToDevice, D.st));
                continue;
            }
            k_argmax<<<kArgBlocks, kThreads, 0, D.st>>>(D.logits, hp.vocab, ds_tok + i + 1, D.amv, D.ami, false);
            k_argmax<<<1, kThreads, 0, D.st>>>(D.logits, kArgBlocks, ds_tok + i + 1, D.amv, D.ami, true);
        }
        if (pick) {
            CK(cudaStreamSynchronize(D.st));
            out.assign(ds_tok_h + 1, ds_tok_h + 1 + kmax);
            return out;
        }
        CK(cudaMemcpyAsync(ds_tok_h, ds_tok, (size_t) (kmax + 1) * 4, cudaMemcpyDeviceToHost, D.st));
        CK(cudaStreamSynchronize(D.st));
        out.assign(ds_tok_h + 1, ds_tok_h + 1 + kmax);
        return out;
    }
    /// A decode's convs (one or two) in one launch; a verify keeps their inputs for commit() as run_conv does.
    void conv_fused(Dev& D, ConvJob a, ConvJob* b, int T) {
        auto keep = [&](ConvJob& j) {
            if (!defer_conv) return;
            const size_t n = (size_t) T * j.C;
            if (D.cv_used + n > D.cv_cap) throw std::runtime_error("inkling2: verify rows exceed the conv save space");
            j.save = D.cv_save + D.cv_used;
            D.cv_used += n;
            conv_recs.push_back(ConvRec{&D, j.hist, j.save, j.C});
        };
        keep(a);
        if (b) keep(*b);
        const int C = std::max(a.C, b ? b->C : 0);
        k_conv_fused<<<dim3(grid1(C), b ? 2u : 1u), kThreads, 0, D.st>>>(a, b ? *b : a, T);
    }
    void run_conv(Dev& D, const float* x, float* hist, const float* w, const float* dyn, int nh, float* out, int C, int T,
              float scale) {
        k_conv<<<grid1((int64_t) C * T), kThreads, 0, D.st>>>(x, hist, w, dyn, nh, out, C, T, scale);
        if (!defer_conv) {
            k_conv_hist<<<grid1(C), kThreads, 0, D.st>>>(x, hist, C, T);
            return;
        }
        const size_t n = (size_t) T * C;
        if (D.cv_used + n > D.cv_cap) throw std::runtime_error("inkling2: verify rows exceed the conv save space");
        float* sv = D.cv_save + D.cv_used;
        D.cv_used += n;
        CK(cudaMemcpyAsync(sv, x, n * 4, cudaMemcpyDeviceToDevice, D.st));
        conv_recs.push_back(ConvRec{&D, hist, sv, C});
    }

    /// A verify forward (defer_conv) leaves each conv's history as it was and keeps its inputs; commit(keep) moves the
    /// histories and the position on by the first `keep` rows only, as if just those had been run.
    bool defer_conv = false;
    struct ConvRec {
        Dev* D;
        float* hist;
        const float* save;
        int C;
    };
    std::vector<ConvRec> conv_recs;
    void commit(int keep) {
        for (const ConvRec& r : conv_recs) {
            CK(cudaSetDevice(r.D->id));
            k_conv_hist<<<grid1(r.C), kThreads, 0, r.D->st>>>(r.save, r.hist, r.C, keep);
        }
        conv_recs.clear();
        S->pos -= T_now - keep;
    }

    /// Prompt chunks on two executors at once (--prefill-pipe): chunk c's visit v waits for chunk c - 1's visit v (its
    /// cache cells and conv state), so while one chunk runs on one card the next runs on the other. Each side records an
    /// event a visit and the chunk it last finished there.
    struct PipeSide {
        std::vector<cudaEvent_t> ev;
        std::unique_ptr<std::atomic<int64_t>[]> done;
    };
    PipeSide* pipe_self = nullptr;
    const PipeSide* pipe_other = nullptr;
    int64_t pipe_chunk = -1, pipe_pos = -1;   // >= 0: this forward is chunk pipe_chunk at position pipe_pos
    void make_pipe_side(PipeSide& ps) const {
        const HP& hp = m->hp;
        ps.ev.resize((size_t) hp.n_vis);
        ps.done.reset(new std::atomic<int64_t>[(size_t) hp.n_vis]);
        for (int il = 0; il < hp.n_vis; ++il) {
            CK(cudaSetDevice(m->dev_ids[(size_t) m->layers[(size_t) hp.phys(il)].dev]));
            CK(cudaEventCreateWithFlags(&ps.ev[(size_t) il], cudaEventDisableTiming));
            ps.done[(size_t) il].store(-1);
        }
    }

    void forward(const int32_t* toks, int T, int logit_rows) {
        const HP& hp = m->hp;
        if (defer_conv) {
            if (T > kGemmT) throw std::runtime_error("inkling2: a verify forward takes at most 8 rows");
            for (int i = 0; i < nd; ++i) dev[i].cv_used = 0;
            conv_recs.clear();
        }
        const int N = hp.n_embd, H = hp.n_head, HKV = hp.n_kv;
        T_now = T;
        const bool piped = pipe_chunk >= 0;
        const int pos0 = piped ? (int) pipe_pos : (int) S->pos;
        int d = m->layers.front().dev;
        Dev* D = &dev[(size_t) d];
        CK(cudaSetDevice(D->id));
        CK(cudaStreamSynchronize(D->st));
        bool pics = false;
        for (int t = 0; t < T; ++t) {
            const bool pic = is_img_id(toks[t]);
            pics = pics || pic;
            std::memcpy(hemb + (size_t) t * m->embd_row, m->embd + (size_t) (pic ? 0 : toks[t]) * m->embd_row, m->embd_row);
        }
        CK(cudaMemcpyAsync(D->erows, hemb, (size_t) T * m->embd_row, cudaMemcpyHostToDevice, D->st));
        K::iq_embed_rows(m->embd_type, D->erows, m->embd_row, D->iota, T, N, D->o, D->st);
        V cur = newv(d);
        k_rms<<<T, kThreads, 0, D->st>>>(D->o, m->tok_norm, at(cur, d), N, hp.eps);
        if (pics) {   // a picture's rows as they are (not RMS-normed: llama.cpp's inkling2 norms token rows only), a run a copy
            if (!img || img->n_embd != N) throw std::runtime_error("inkling2: a picture's row without the request's pictures");
            if (himg_rows < T) {
                if (himg) CK(cudaFreeHost(himg));
                CK(cudaHostAlloc((void**) &himg, (size_t) T * N * 4, cudaHostAllocPortable));
                himg_rows = T;
            }
            float* dst = at(cur, d);
            for (int t = 0; t < T;) {
                if (!is_img_id(toks[t])) { ++t; continue; }
                int e = t;
                for (; e < T && is_img_id(toks[e]); ++e) {
                    const auto it = img->at.find(toks[e]);
                    if (it == img->at.end()) throw std::runtime_error("inkling2: a picture's row not in the request's pictures");
                    std::memcpy(himg + (size_t) e * N, img->rows.data() + it->second * (size_t) N, (size_t) N * 4);
                }
                CK(cudaMemcpyAsync(dst + (size_t) t * N, himg + (size_t) t * N, (size_t) (e - t) * N * 4, cudaMemcpyHostToDevice, D->st));
                t = e;
            }
        }
        std::vector<V> blocks, pre_blocks, bank;
        V pre_cur, bank_last;
        for (int il = 0; il < hp.n_vis; ++il) {
            const int ip = hp.phys(il);
            const Layer& L = m->layers[(size_t) ip];
            if (L.dev != d) {
                d = L.dev;
                D = &dev[(size_t) d];
                CK(cudaSetDevice(D->id));
            }
            if (ds_mode && dsm) {   // a DSpark feature: this visit's input rows
                const int sl = dsm->tap_slot(il);
                if (sl >= 0)   // (the buffer on the drafter's card: peer stores from this one)
                    k_f2h_rows<<<dim3(4, (unsigned) T), kThreads, 0, D->st>>>(at(cur, d), ds_feat + (size_t) sl * N, N, (size_t) 4 * N);
            }
            if (piped && pipe_other && pipe_chunk > 0) {   // the chunk before finished this visit first
                while (pipe_other->done[(size_t) il].load(std::memory_order_acquire) < pipe_chunk - 1) std::this_thread::yield();
                CK(cudaStreamWaitEvent(D->st, pipe_other->ev[(size_t) il], 0));
            }
            const int pass = hp.pass(il);
            const bool core = hp.in_core(ip);
            if (core && ip == hp.core0) {
                if (pass == 0) {
                    pre_blocks = blocks;
                    pre_cur = cur;
                } else {
                    bank.assign(blocks.begin() + (long) pre_blocks.size() + 1, blocks.end());
                    bank_last = cur;
                    blocks = pre_blocks;
                    cur = pre_cur;
                }
            }
            const float* msg[4] = {};
            if (core && pass > 0) {
                mix(*D, d, bank, bank_last, L.msg_q, 4, D->msg[0], T);   // the Q, K, V and FFN messages, [4][T][N]
                for (int k = 0; k < 4; ++k) msg[k] = D->msg[0] + (size_t) k * T * N;
            }
            const V inpSA = cur;
            const float* in[3];
            const float* curp = at(cur, d);
            if (L.attn_res_q) {
                mix(*D, d, blocks, cur, L.attn_res_q, L.n_res_q, D->mixo, T);
                for (int j = 0; j < 3; ++j) in[j] = D->mixo + (size_t) (L.swa ? 0 : j) * T * N;
            } else {
                in[0] = in[1] = in[2] = curp;
            }
            const bool fused_ctl = T <= kGemmT && hp.gate_rank <= 64 && hp.gate_rank % 4 == 0 && hp.n_embd % (kCtlSplits * 2) == 0;
            if (msg[0] && fused_ctl) {
                float* const o3[3] = {D->xn[0], D->xn[1], D->xn[2]};
                controls(*D, L, 0, 3, in, msg, o3, T);
                for (int j = 0; j < 3; ++j) in[j] = D->xn[j];
            } else if (msg[0]) {
                for (int j = 0; j < 3; ++j) {
                    control(*D, L, j, in[j], msg[j], D->xn[j], T);
                    in[j] = D->xn[j];
                }
            }
            const float* xn[3];
            if (L.swa && msg[0]) {
                for (int j = 0; j < 3; ++j) {
                    k_rms<<<T, kThreads, 0, D->st>>>(in[j], L.attn_norm, D->xn[j], N, hp.eps);
                    xn[j] = D->xn[j];
                }
            } else if (L.swa) {
                k_rms<<<T, kThreads, 0, D->st>>>(in[0], L.attn_norm, D->xn[0], N, hp.eps);
                xn[0] = xn[1] = xn[2] = D->xn[0];
            } else {
                const float* w[3] = {L.q_norm, L.k_norm, L.v_norm};
                for (int j = 0; j < 3; ++j) {
                    k_rms<<<T, kThreads, 0, D->st>>>(in[j], w[j], D->xn[j], N, hp.eps);
                    xn[j] = D->xn[j];
                }
            }
            bool seg = T <= 8;   // a decode's q, rel, gate, k and v in one launch (k_q8_seg)
            for (const Mat* w : {&L.wq, &L.wrel, &L.wg, &L.wk, &L.wv}) seg = seg && w->type == 8 && w->n_in == N && w->n_out % 8 == 0;
            if (seg) {
                K::native_quantize_q8_1(xn[0], D->q8, N * T, 1, D->st);
                const void* xk = D->q8;
                if (xn[1] != xn[0]) {
                    K::native_quantize_q8_1(xn[1], D->q8b, N * T, 1, D->st);
                    xk = D->q8b;
                }
                const void* xv = xk;
                if (xn[2] != xn[1]) {
                    K::native_quantize_q8_1(xn[2], D->q8c, N * T, 1, D->st);
                    xv = D->q8c;
                }
                const Mat* ws[5] = {&L.wq, &L.wrel, &L.wg, &L.wk, &L.wv};
                const void* xs[5] = {D->q8, D->q8, D->q8, xk, xv};
                float* os[5] = {D->q, D->r, D->g, D->k, D->v};
                apply_q8_seg(*D, ws, xs, os, 5, T);
                f32mm(*D, L.sink_proj, xn[0], D->sink, N, H, T);
            } else {
                prep(*D, xn[0], N, T);
                apply(*D, L.wq, D->q, T);
                apply(*D, L.wrel, D->r, T);
                apply(*D, L.wg, D->g, T);
                f32mm(*D, L.sink_proj, xn[0], D->sink, N, H, T);
                if (xn[1] != xn[0]) prep(*D, xn[1], N, T);
                apply(*D, L.wk, D->k, T);
                if (xn[2] != xn[1]) prep(*D, xn[2], N, T);
                apply(*D, L.wv, D->v, T);
            }
            if (!L.swa) {
                f32mm(*D, L.kc_proj, xn[1], D->dk, N, 4 * HKV, T);
                f32mm(*D, L.vc_proj, xn[2], D->dv, N, 4 * HKV, T);
            }
            const int KD = HKV * kHD;
            float* cst = S->conv[(size_t) il];
            const float s_dyn = 1.0f / std::sqrt(2.0f * hp.n_conv + 1.0f), s_st = 1.0f / std::sqrt((float) hp.n_conv + 1.0f);
            const bool fconv = T <= kGemmT;   // a decode's convs fused (k_conv_fused)
            if (fconv) {
                ConvJob jk, jv;
                jk.x = D->k; jk.hist = cst; jk.w = L.k_conv; jk.dyn = L.swa ? nullptr : D->dk; jk.nh = HKV; jk.out = D->kc;
                jk.C = KD; jk.scale = L.swa ? s_st : s_dyn;
                jv = jk;
                jv.x = D->v; jv.hist = cst + (size_t) KD * 3; jv.w = L.v_conv; jv.dyn = L.swa ? nullptr : D->dv; jv.out = D->vc;
                conv_fused(*D, jk, &jv, T);
            } else {
                run_conv(*D, D->k, cst, L.k_conv, L.swa ? nullptr : D->dk, HKV, D->kc, KD, T, L.swa ? s_st : s_dyn);
                run_conv(*D, D->v, cst + (size_t) KD * 3, L.v_conv, L.swa ? nullptr : D->dv, HKV, D->vc, KD, T, L.swa ? s_st : s_dyn);
            }
            const Kv& cv = S->kv[(size_t) il];
            const bool q8 = S->kv_q8[(size_t) il] != 0;
            const int cells = cv.cells, ringv = cv.ring;
            const dim3 pg((unsigned) (H + HKV), (unsigned) T);
            if (q8)
                k_prep<true><<<pg, kHD, 0, D->st>>>(D->q, D->kc, D->vc, D->r, L.q_clip, L.rel_clip, D->qb, D->rb, cv, H, HKV,
                                                     hp.n_rel, pos0, ringv, cells, hp.qk_clip, hp.rel_clip, hp.eps,
                                                     L.swa ? 0 : 1, hp.log_alpha, (float) hp.log_floor);
            else
                k_prep<false><<<pg, kHD, 0, D->st>>>(D->q, D->kc, D->vc, D->r, L.q_clip, L.rel_clip, D->qb, D->rb, cv, H, HKV,
                                                      hp.n_rel, pos0, ringv, cells, hp.qk_clip, hp.rel_clip, hp.eps,
                                                      L.swa ? 0 : 1, hp.log_alpha, (float) hp.log_floor);
            const float scale = 1.0f / std::sqrt((float) kHD);
            const int win = L.swa ? hp.swa : 0, G = H / HKV;
            const int pmax = pos0 + T - 1;
            const bool tier = cv.hk != nullptr;
            const int res_lo = tier ? std::max(0, pmax - cv.ring + 1) : 0;   // the cells the VRAM ring still holds
            KvSrc vram{cv.k, cv.v, cv.ks, cv.vs, cv.ring, cv.cells, 0};
            KvSrc vram2{cv.k2, cv.v2, cv.ks2, cv.vs2, cv.ring, cv.cells, 0};   // a split window's other half
            if (cv.split) {
                vram.split = vram2.split = 1;
                vram2.part = 1;
            }
            if (T >= kAttTileT && G <= 3 && hp.n_rel <= 16) {
                const dim3 ag((unsigned) ((T * G + kFaRows - 1) / kFaRows), (unsigned) HKV);
                auto mma = [&](const KvSrc& src, int kmin, int kmax, bool first, bool last) {
                    if (q8)
                        k_attn_fa2<true><<<ag, kFaWarps * 32, kFaSmem, D->st>>>(D->qb, D->rb, src, L.rel_bias, hp.n_rel, pos0, T, H, G,
                                                                       win, L.extent, scale, D->sink, D->g, D->att, kmin, kmax,
                                                                       D->run_o, D->run_m, D->run_l, first, last);
                    else
                        k_attn_fa2<false><<<ag, kFaWarps * 32, kFaSmem, D->st>>>(D->qb, D->rb, src, L.rel_bias, hp.n_rel, pos0, T, H, G,
                                                                        win, L.extent, scale, D->sink, D->g, D->att, kmin, kmax,
                                                                        D->run_o, D->run_m, D->run_l, first, last);
                };
                bool first = true;
                // a split window (a short read: a long one goes back to whole windows): the other card reads its half at
                // the same time, the rows (and their sinks) copied over, its running result back for the join
                if (cv.split) {
                    Dev& P = dev[(size_t) cv.dev2];
                    CK(cudaEventRecord(D->evh, D->st));
                    CK(cudaSetDevice(P.id));
                    CK(cudaStreamWaitEvent(P.ps, D->evh, 0));
                    CK(cudaMemcpyPeerAsync(P.hq, P.id, D->qb, D->id, (size_t) T * H * kHD * 4, P.ps));
                    CK(cudaMemcpyPeerAsync(P.hrb, P.id, D->rb, D->id, (size_t) T * H * hp.n_rel * 4, P.ps));
                    CK(cudaMemcpyPeerAsync(P.hrm, P.id, D->sink, D->id, (size_t) T * H * 4, P.ps));   // its first m
                    k_attn_fa2<true><<<ag, kFaWarps * 32, kFaSmem, P.ps>>>(P.hq, P.hrb, vram2, L.rel_bias2[(size_t) cv.dev2], hp.n_rel,
                                                                    pos0, T, H, G, win, L.extent, scale, P.hrm, nullptr, nullptr,
                                                                    res_lo, pmax, P.hro, P.hrm, P.hrl, 1, 0);
                    CK(cudaMemcpyPeerAsync(D->hro, D->id, P.hro, P.id, (size_t) T * H * kHD * 4, P.ps));
                    CK(cudaMemcpyPeerAsync(D->hrm, D->id, P.hrm, P.id, (size_t) T * H * 4, P.ps));
                    CK(cudaMemcpyPeerAsync(D->hrl, D->id, P.hrl, P.id, (size_t) T * H * 4, P.ps));
                    CK(cudaEventRecord(P.evp, P.ps));
                    CK(cudaSetDevice(D->id));
                }
                each_host_block(*D, cv, res_lo, [&](const KvSrc& src, int k0, int k1) {
                    mma(src, k0, k1, first, false);
                    first = false;
                });
                mma(vram, res_lo, pmax, first, !cv.split);
                if (cv.split) {
                    CK(cudaStreamWaitEvent(D->st, dev[(size_t) cv.dev2].evp, 0));
                    k_attn_join<<<(unsigned) (T * H), kHD, 0, D->st>>>(D->run_o, D->run_m, D->run_l, D->hro, D->hrm, D->hrl, D->sink, D->g,
                                                                       D->att);
                }
            } else if (T >= kAttTileT && G <= kMaxG) {
                k_attn_tile<<<dim3((unsigned) ((T + kQT - 1) / kQT), (unsigned) HKV), 256, attn_tile_smem(G, hp.n_rel), D->st>>>(
                    D->qb, D->rb, (const __half*) cv.k, (const __half*) cv.v, L.rel_bias, hp.n_rel, pos0, T, H, G, ringv,
                    cells, win, L.extent, scale, D->sink, D->g, D->att);
            } else
            for (int t0 = 0; t0 < T; t0 += att_q) {
                const int nq = std::min(att_q, T - t0);
                const int p_first = pos0 + t0, p_last = p_first + nq - 1;
                const int base = (L.swa ? std::max(0, p_first - hp.swa + 1) : res_lo) / kChunkKeys * kChunkKeys;
                // the VRAM part on the tensor cores (dk keys a block), up to 5 tokens a row tile
                const bool dm = G <= 3 && hp.n_rel <= 16;
                // keys a block: 16 blocks a KV head (the card's SMs busy at shorter ranges), 256 at least, and 2048 at most
                // up to 64 blocks a KV head (more keys a block past 128K: the partials' room is kDecSplits a row)
                const int span = p_last - base + 1;
                const int dk = std::min(std::max(2048, (span + 64 * 64 - 1) / (64 * 64) * 64),
                                        std::max(256, (span + 16 * 64 - 1) / (16 * 64) * 64));
                const int nc_r = dm ? (p_last - base) / dk + 1 : (p_last - base) / kChunkKeys + 1;
                // The RAM part: [0, g_end) through VRAM, [g_end, res_lo) on CPU threads at the same time.
                const bool on_cpu = tier && res_lo >= kCpuMin && q8 && G <= kMaxG && t0 / att_q < kAttTileT;
                int g_end = res_lo;
                CpuRec* cr = nullptr;
                if (on_cpu) {
                    if (cpu_rec.empty()) cpu_rec.resize((size_t) hp.n_vis * kAttTileT);
                    cr = &cpu_rec[(size_t) il * kAttTileT + (size_t) (t0 / att_q)];
                    if (cr->pending && t0 == 0) {   // the last split's times set this one
                        float gms = 0.f;
                        if (cudaEventQuery(cr->g1) == cudaSuccess && cudaEventElapsedTime(&gms, cr->g0, cr->g1) == cudaSuccess &&
                            gms > 0.f && cr->cpu_ms > 0 && cr->cpu_keys > 0 && cr->gpu_keys > 0) {
                            const double rc = cr->cpu_keys / cr->cpu_ms, rg = cr->gpu_keys / (double) gms;
                            cpu_frac = std::clamp(0.7f * cpu_frac + 0.3f * (float) (rc / (rc + rg)), 0.05f, 0.9f);
                        }
                    }
                    if (!cr->g0) {
                        CK(cudaEventCreate(&cr->g0));
                        CK(cudaEventCreate(&cr->g1));
                    }
                    g_end = (int) ((double) res_lo * (1.0 - cpu_frac)) / kChunkKeys * kChunkKeys;
                }
                const int hk = dm ? kHostKeys : kChunkKeys;   // keys a staged block's partial
                const int nc_h = g_end > 0 ? (g_end - 1) / hk + 1 : 0;
                const bool two = cv.split && dm;   // the other card reads its half, its result one more partial
                const int nseg = on_cpu ? kCpuSegs : 0, h_slot = nc_r + nc_h, c_cpu = h_slot + (two ? 1 : 0);
                const int nc = c_cpu + nseg;
                if (two) {   // first, so the other card starts at once: the rows there, its half, its partials folded into ours
                    Dev& P = dev[(size_t) cv.dev2];
                    CK(cudaEventRecord(D->evh, D->st));
                    CK(cudaSetDevice(P.id));
                    CK(cudaStreamWaitEvent(P.ps, D->evh, 0));
                    CK(cudaMemcpyPeerAsync(P.hq, P.id, D->qb + (size_t) t0 * H * kHD, D->id, (size_t) nq * H * kHD * 4, P.ps));
                    CK(cudaMemcpyPeerAsync(P.hrb, P.id, D->rb + (size_t) t0 * H * hp.n_rel, D->id, (size_t) nq * H * hp.n_rel * 4, P.ps));
                    const dim3 dg2((unsigned) ((nq * G + 15) / 16), (unsigned) HKV, (unsigned) nc_r);
                    k_attn_dreg<true><<<dg2, 128, kDrSmem, P.ps>>>(P.hq, P.hrb, vram2, L.rel_bias2[(size_t) cv.dev2], hp.n_rel, L.extent,
                                                                pos0 + t0, 0, nq, H, G, win, res_lo, INT32_MAX, base, dk, 0, nc_r, scale,
                                                                P.hpacc, P.hpm, P.hpl);
                    k_attn_fold<<<(unsigned) (nq * H), kHD, 0, P.ps>>>(P.hpacc, P.hpm, P.hpl, nc_r, D->pacc, D->pm, D->pl, nc, h_slot);
                    CK(cudaEventRecord(P.evp, P.ps));
                    CK(cudaSetDevice(D->id));
                }
                if (on_cpu) {
                    CK(cudaMemcpyAsync(D->qh, D->qb + (size_t) t0 * H * kHD, (size_t) nq * H * kHD * 4, cudaMemcpyDeviceToHost, D->st));
                    CK(cudaEventRecord(D->evq, D->st));
                    CK(cudaStreamWaitEvent(D->hs, D->evq, 0));
                    CpuAttnJob& j = cr->job;
                    j = CpuAttnJob{};
                    j.q = D->qh;
                    j.nq = nq; j.H = H; j.G = G; j.HKV = HKV;
                    j.k = cv.hk; j.v = cv.hv; j.ks = cv.hks; j.vs = cv.hvs;
                    j.hcells = cv.hcells; j.kmin = g_end; j.kmax = res_lo - 1;
                    j.scale = scale; j.nseg = nseg; j.seg_stride = nseg;
                    j.acc = D->cacc; j.m = D->cm; j.l = D->cl;
                    cr->cpu_keys = res_lo - g_end;
                    cr->gpu_keys = g_end;
                    cr->cpu_ms = 0;
                    cr->pending = true;
                    CK(cudaLaunchHostFunc(D->hs, cpu_run, cr));
                    const size_t hgt = (size_t) nq * H;
                    CK(cudaMemcpy2DAsync(D->pacc + (size_t) c_cpu * kHD, (size_t) nc * kHD * 4, D->cacc, (size_t) nseg * kHD * 4,
                                         (size_t) nseg * kHD * 4, hgt, cudaMemcpyHostToDevice, D->hs));
                    CK(cudaMemcpy2DAsync(D->pm + c_cpu, (size_t) nc * 4, D->cm, (size_t) nseg * 4, (size_t) nseg * 4, hgt,
                                         cudaMemcpyHostToDevice, D->hs));
                    CK(cudaMemcpy2DAsync(D->pl + c_cpu, (size_t) nc * 4, D->cl, (size_t) nseg * 4, (size_t) nseg * 4, hgt,
                                         cudaMemcpyHostToDevice, D->hs));
                    CK(cudaEventRecord(D->evcpu, D->hs));
                    CK(cudaEventRecord(cr->g0, D->st));
                }
                auto dec = [&](const KvSrc& src, int kmin, int kmax, int b, int c_off, int n) {
                    const dim3 dg((unsigned) nq, (unsigned) HKV, (unsigned) n);
#define STRATA_INK_FD(Q, GG) k_attn_fd<Q, GG><<<dg, 128, 0, D->st>>>(D->qb, D->rb, src, L.rel_bias, hp.n_rel, L.extent, pos0, t0, \
                                                               H, win, kmin, kmax, b, c_off, nc, scale, D->pacc, D->pm, D->pl)
                    switch ((q8 ? 8 : 0) + G) {
                        case 9: STRATA_INK_FD(true, 1); break;
                        case 10: STRATA_INK_FD(true, 2); break;
                        case 11: STRATA_INK_FD(true, 3); break;
                        case 12: STRATA_INK_FD(true, 4); break;
                        case 1: STRATA_INK_FD(false, 1); break;
                        case 2: STRATA_INK_FD(false, 2); break;
                        case 3: STRATA_INK_FD(false, 3); break;
                        default: STRATA_INK_FD(false, 4); break;
                    }
#undef STRATA_INK_FD
                };
                if (dm) {
                    const dim3 dg((unsigned) ((nq * G + 15) / 16), (unsigned) HKV, (unsigned) nc_r);
                    if (q8)
                        k_attn_dreg<true><<<dg, 128, kDrSmem, D->st>>>(
                            D->qb, D->rb, vram, L.rel_bias, hp.n_rel, L.extent, pos0, t0, nq, H, G, win, res_lo, INT32_MAX, base, dk,
                            0, nc, scale, D->pacc, D->pm, D->pl);
                    else
                        k_attn_dreg<false><<<dg, 128, kDrSmem, D->st>>>(
                            D->qb, D->rb, vram, L.rel_bias, hp.n_rel, L.extent, pos0, t0, nq, H, G, win, res_lo, INT32_MAX, base, dk,
                            0, nc, scale, D->pacc, D->pm, D->pl);
                } else
                    dec(vram, res_lo, INT32_MAX, base, 0, nc_r);
                each_host_block(*D, cv, g_end, [&](const KvSrc& src, int k0, int k1) {
                    if (dm) {
                        const dim3 hg((unsigned) ((nq * G + 15) / 16), (unsigned) HKV, (unsigned) ((k1 - k0) / kHostKeys + 1));
                        k_attn_dreg<true><<<hg, 128, kDrSmem, D->st>>>(D->qb, D->rb, src, L.rel_bias, hp.n_rel, L.extent, pos0, t0, nq, H, G,
                                                                      win, k0, k1, k0, kHostKeys, nc_r + k0 / kHostKeys, nc, scale,
                                                                      D->pacc, D->pm, D->pl);
                    } else
                        dec(src, k0, k1, k0, nc_r + k0 / kChunkKeys, (k1 - k0) / kChunkKeys + 1);
                });
                if (on_cpu) {
                    CK(cudaEventRecord(cr->g1, D->st));
                    CK(cudaStreamWaitEvent(D->st, D->evcpu, 0));
                }
                if (two) CK(cudaStreamWaitEvent(D->st, dev[(size_t) cv.dev2].evp, 0));
                k_attn_merge<<<(unsigned) (nq * H), kHD, 0, D->st>>>(D->pacc, D->pm, D->pl, nc, D->sink, D->g, t0, H, D->att);
            }
            prep(*D, D->att, H * kHD, T);
            apply(*D, L.wo, D->o, T);
            V ao = newv(d);
            const bool blk_start = ip % hp.res_block == 0;
            if (fconv) {
                ConvJob ja;
                ja.x = D->o; ja.hist = cst + (size_t) 2 * KD * 3; ja.w = L.a_conv; ja.out = at(ao, d); ja.C = N; ja.scale = s_st;
                ja.res = blk_start ? nullptr : at(inpSA, d);
                conv_fused(*D, ja, nullptr, T);
            } else {
                run_conv(*D, D->o, cst + (size_t) 2 * KD * 3, L.a_conv, nullptr, 1, at(ao, d), N, T, s_st);
                if (!blk_start) {
                    float* a = at(ao, d);
                    k_add<<<grid1((int64_t) T * N), kThreads, 0, D->st>>>(a, a, at(inpSA, d), (int64_t) T * N);
                }
            }
            if (blk_start) blocks.push_back(inpSA);
            cur = ao;
            const V ffn_inp = cur;
            mix(*D, d, blocks, cur, L.ffn_res_q, 1, D->mixo, T);
            const float* fx = D->mixo;
            if (msg[3]) {
                if (fused_ctl) {
                    const float* x1[1] = {D->mixo};
                    const float* m1[1] = {msg[3]};
                    float* const o1[1] = {D->ctl};
                    controls(*D, L, 3, 1, x1, m1, o1, T);
                } else
                    control(*D, L, 3, D->mixo, msg[3], D->ctl, T);
                fx = D->ctl;
            }
            k_rms<<<T, kThreads, 0, D->st>>>(fx, L.ffn_norm, D->xn[0], N, hp.eps);
            f32mm(*D, L.router, D->xn[0], D->rlog, N, hp.n_expert, T);
            k_route<<<T, 32, 0, D->st>>>(D->rlog, L.probs_b, hp.n_expert, hp.top_k, hp.w_scale, D->ids, D->ewts);
            K::native_quantize_q8_1(D->xn[0], D->xq, N * T, 1, D->st);
            const int n_ent = T * hp.top_k;
            bool moe_parts = false;   // the experts' outputs not yet weighted into D->moe
            if (count_routes) k_hist<<<std::min(64, (n_ent + 255) / 256), 256, 0, D->st>>>(D->ids, n_ent, L.hist);
            const bool in_ram = !L.all_on_gpu();
            if (in_ram && T >= kExpertGemmT) {   // a prompt chunk: the experts in RAM copied in, a batch at a time
                experts_staged(*D, L, D->xn[0], T, n_ent);
                if (L.mma) k_combine_sorted<<<T, kThreads, 0, D->st>>>(D->dg, D->ent_inv, D->ewts, hp.top_k, D->moe, N);
                else k_combine<<<T, kThreads, 0, D->st>>>(D->eparts, D->ewts, hp.top_k, D->moe, N);
            } else {
                MoeRec* mr = nullptr;
                if (in_ram) {   // the entries of experts in RAM on CPU threads, while the GPU runs the others
                    if (moe_rec.empty()) {
                        moe_rec.resize((size_t) hp.n_vis);
                        const size_t TC = (size_t) std::min(tmax, kExpertGemmT - 1), Kp = (size_t) hp.top_k;
                        for (auto& r : moe_rec) {
                            const unsigned fl = cudaHostAllocPortable | cudaHostAllocMapped;
                            CK(cudaHostAlloc((void**) &r.ids_h, TC * Kp * 4, fl));
                            CK(cudaHostAlloc((void**) &r.xq_h, TC * (size_t) (N / 32) * 36, fl));
                            CK(cudaHostAlloc((void**) &r.out_h, TC * Kp * (size_t) N * 4, fl));
                            CK(cudaHostAlloc((void**) &r.mask_h, TC * Kp, fl));
                            int* dn = nullptr;
                            CK(cudaHostAlloc((void**) &dn, 4, fl));
                            *dn = 0;
                            r.done = dn;
                        }
                    }
                    mr = &moe_rec[(size_t) il];
                    while (*mr->done != mr->seq) std::this_thread::yield();   // this visit's last job is done with its buffers
                    ++mr->seq;
                    CK(cudaMemcpyAsync(mr->ids_h, D->ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, D->st));
                    CK(cudaMemcpyAsync(mr->xq_h, D->xq, (size_t) T * (N / 32) * 36, cudaMemcpyDeviceToHost, D->st));
                    CK(cudaEventRecord(D->evmx, D->st));
                    CK(cudaStreamWaitEvent(D->hs, D->evmx, 0));
                    mr->job = CpuMoeJob{};
                    mr->job.L = &L.cx;
                    mr->job.on_gpu = L.on_gpu.data();
                    mr->job.ids = mr->ids_h;
                    mr->job.n_ent = n_ent;
                    mr->job.K = hp.top_k;
                    mr->job.xq = mr->xq_h;
                    mr->job.out = mr->out_h;
                    mr->job.mask = mr->mask_h;
                    CK(cudaLaunchHostFunc(D->hs, moe_run, mr));
                }
                k_group<<<1, kThreads, 0, D->st>>>(D->ids, n_ent, hp.top_k, hp.n_expert, L.eptr, nullptr, D->grp_ptr,
                                                   D->grp_start, D->n_groups, D->ent_dst, D->ent_tok, D->ent_inv);
                if (T >= kExpertGemmT && L.mma) {
                    experts_gemm(*D, L, D->xn[0], n_ent);
                    k_combine_sorted<<<T, kThreads, 0, D->st>>>(D->dg, D->ent_inv, D->ewts, hp.top_k, D->moe, N);
                } else {
                    K::native_expert_grouped(L.L, D->grp_ptr, D->grp_start, D->n_groups, D->ent_dst, D->ent_tok,
                                             std::min(n_ent, hp.n_expert), n_ent, D->xq, D->escr, D->eparts, D->st);
                    if (mr) k_fill_cpu_wait<<<n_ent, kThreads, 0, D->st>>>(D->eparts, D->ids, L.eptr, mr->out_h, mr->done, mr->seq, N);
                    if (fconv) moe_parts = true;   // weighted in the conv's launch
                    else k_combine<<<T, kThreads, 0, D->st>>>(D->eparts, D->ewts, hp.top_k, D->moe, N);
                }
            }
            V fo = newv(d);
            float* fp = at(fo, d);
            if (fconv) {
                ConvJob jf;
                if (moe_parts) {
                    jf.parts = D->eparts; jf.wts = D->ewts; jf.K = hp.top_k;
                } else
                    jf.x = D->moe;
                jf.hist = cst + (size_t) (2 * KD + N) * 3; jf.w = L.f_conv; jf.out = fp; jf.C = N; jf.scale = s_st;
                jf.res = at(ffn_inp, d);
                conv_fused(*D, jf, nullptr, T);
            } else {
                run_conv(*D, D->moe, cst + (size_t) (2 * KD + N) * 3, L.f_conv, nullptr, 1, fp, N, T, s_st);
                k_add<<<grid1((int64_t) T * N), kThreads, 0, D->st>>>(fp, fp, at(ffn_inp, d), (int64_t) T * N);
            }
            cur = fo;
            if (ck_out) ckpt_visit(*D, *ck_out, il, L.swa, pos0 + T);
            if (piped && pipe_self) {
                CK(cudaEventRecord(pipe_self->ev[(size_t) il], D->st));
                pipe_self->done[(size_t) il].store(pipe_chunk, std::memory_order_release);
            }
        }
        if (m->head_dev != d) {
            d = m->head_dev;
            D = &dev[(size_t) d];
            CK(cudaSetDevice(D->id));
        }
        if (ds_mode && dsm) {   // the last feature: the normed hidden rows the head reads; then (a prompt) into the drafter
            mix(*D, d, blocks, cur, m->out_res_q, 1, D->mixo, T);
            k_rms<<<T, kThreads, 0, D->st>>>(D->mixo, m->out_norm, D->xn[1], N, hp.eps);
            k_f2h_rows<<<dim3(4, (unsigned) T), kThreads, 0, D->st>>>(D->xn[1], ds_feat + 3 * (size_t) N, N, (size_t) 4 * N);
            if (ds_mode == 1 && ds_from < T) ds_inject(*D, *S->ds, ds_from, T - ds_from, pos0 + ds_from);
        }
        if (logit_rows > 0) {
            mix(*D, d, blocks, cur, m->out_res_q, 1, D->mixo, T);
            const int r0 = T - logit_rows;
            k_rms<<<logit_rows, kThreads, 0, D->st>>>(D->mixo + (size_t) r0 * N, m->out_norm, D->xn[0], N, hp.eps);
            K::native_quantize_q8_1(D->xn[0], D->q8, N * logit_rows, 1, D->st);
            if (K::native_mmvq_supported(m->head.type))
                K::native_mmvq(m->head.type, m->head.d, D->q8, D->logits, N, m->head.n_out, logit_rows, D->st);
            else
                K::iq_mmvq(m->head.type, m->head.d, D->q8, D->logits, N, m->head.n_out, logit_rows, D->st);
            if (hp.softcap > 0.f)
                k_softcap<<<grid1((int64_t) hp.vocab * logit_rows), kThreads, 0, D->st>>>(D->logits, hp.vocab, hp.vocab_pad,
                                                                                          logit_rows, hp.softcap);
        }
        blocks.clear();
        pre_blocks.clear();
        bank.clear();
        pre_cur.reset();
        bank_last.reset();
        cur.reset();
        if (!piped) S->pos += T;
        last_dev = d;
        n_logit_rows = logit_rows;
    }
    int n_logit_rows = 0;
    /// Visit il's state at position `end` into checkpoint k, on the visit's stream (before a later chunk changes it).
    void ckpt_visit(Dev& D, Ckpt& k, int il, bool swa, int64_t end) {
        const Kv& c = S->kv[(size_t) il];
        CK(cudaMemcpyAsync(k.conv[(size_t) il], S->conv[(size_t) il], k.conv_bytes, cudaMemcpyDeviceToHost, D.st));
        if (!swa || !k.swa[(size_t) il]) return;
        const int64_t keep = m->hp.swa, cnt = std::min<int64_t>(end, keep), st0 = end - cnt;
        const size_t HKV = (size_t) m->hp.n_kv, row = (size_t) kHD * 2;
        uint8_t* kd = k.swa[(size_t) il];
        uint8_t* vd = kd + HKV * (size_t) keep * row;
        for (int64_t a = st0; a < end;) {   // cells a .. b - 1 are contiguous in the ring
            const int64_t b = std::min<int64_t>(end, (a / c.ring + 1) * c.ring), n = b - a;
            const size_t so = (size_t) (a % c.ring) * row, dofs = (size_t) (a - st0) * row;
            CK(cudaMemcpy2DAsync(kd + dofs, (size_t) keep * row, (const uint8_t*) c.k + so, (size_t) c.cells * row, (size_t) n * row,
                                 HKV, cudaMemcpyDeviceToHost, D.st));
            CK(cudaMemcpy2DAsync(vd + dofs, (size_t) keep * row, (const uint8_t*) c.v + so, (size_t) c.cells * row, (size_t) n * row,
                                 HKV, cudaMemcpyDeviceToHost, D.st));
            a = b;
        }
    }

    int64_t process(const int32_t* ids, int64_t n, const std::function<bool(int64_t)>& progress) {
        int64_t done = 0;
        while (done < n) {
            const int T = (int) std::min<int64_t>(tmax, n - done);
            forward(ids + done, T, done + T == n ? 1 : 0);
            done += T;
            if (done < n) sync_all();
            if (!progress(done)) break;
        }
        return done;
    }
    int greedy(int row = 0) {
        Dev& D = dev[(size_t) last_dev];
        CK(cudaSetDevice(D.id));
        const float* lg = D.logits + (size_t) row * m->hp.vocab_pad;
        k_argmax<<<kArgBlocks, kThreads, 0, D.st>>>(lg, m->hp.vocab, D.pick, D.amv, D.ami, false);
        k_argmax<<<1, kThreads, 0, D.st>>>(lg, kArgBlocks, D.pick, D.amv, D.ami, true);
        CK(cudaMemcpyAsync(hpin, D.pick, 4, cudaMemcpyDeviceToHost, D.st));
        sync_all();
        return hpin[0];
    }
    void host_logits(std::vector<float>& lg, int row = 0) {
        Dev& D = dev[(size_t) last_dev];
        CK(cudaSetDevice(D.id));
        lg.resize((size_t) m->hp.vocab);
        sync_all();
        CK(cudaMemcpy(lg.data(), D.logits + (size_t) row * m->hp.vocab_pad, lg.size() * 4, cudaMemcpyDeviceToHost));
    }
};

Value::~Value() {
    if (E) E->release(*this);
}

using strata::qwen35::Sampling;

/// The token distribution sampling draws from for a logit row: the kept ids and their probabilities (summing to 1);
/// one id at probability 1 for greedy decoding.
struct Dist {
    std::vector<int> idx;
    std::vector<double> p;
};
Dist dist(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    strata::qwen35::apply_penalties(lg, s, hist);
    Dist d;
    if (s.temperature <= 0.f) {
        d.idx = {(int) (std::max_element(lg.begin(), lg.end()) - lg.begin())};
        d.p = {1.0};
        return d;
    }
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    d.idx.assign(idx.begin(), idx.begin() + (long) keep);
    d.p.resize(keep);
    for (size_t i = 0; i < keep; ++i) d.p[i] = p[i] / tot;
    return d;
}
/// A draw from d, leaving out `skip` (the drafted token a verify turned down: the rest renormalized).
int draw(const Dist& d, std::mt19937_64& rng, int skip = -1) {
    double tot = 0;
    for (size_t i = 0; i < d.idx.size(); ++i) tot += d.idx[i] == skip ? 0.0 : d.p[i];
    if (tot <= 0) return d.idx[0];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    int last = d.idx[0];
    for (size_t i = 0; i < d.idx.size(); ++i) {
        if (d.idx[i] == skip) continue;
        last = d.idx[i];
        if ((r -= d.p[i]) <= 0) return d.idx[i];
    }
    return last;
}
int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    return draw(dist(lg, s, hist), rng);
}

/// Drafts from the conversation itself (prompt lookup): the tokens that followed the latest earlier occurrence of the
/// last 4, 3 or 2 tokens. A verify forward checks them, so a wrong draft costs time and never changes a token. How many
/// to draft comes from the measured acceptance and the measured time of a forward of each row count: none when
/// drafting does not pay.
struct Drafter {
    static constexpr int kMinN = 2, kMaxN = 4;
    std::unordered_map<uint64_t, int32_t> ix[kMaxN - kMinN + 1];   // the n tokens ending at i -> i + 1, the latest
    size_t done = 0;                                               // the tokens of `live` indexed
    double acc = 3, rej = 2;                                       // decayed drafted-token outcomes
    double t_ms[kGemmT + 1] = {};                                  // a step's time by its rows
    int steps = 0;
    int max_k = kGemmT - 1;                                        // --lookup-max-t: the most tokens a step drafts (0: none)

    static uint64_t key(const int32_t* t, int n) {
        uint64_t h = 0x9E3779B97F4A7C15ull * (uint64_t) n;
        for (int i = 0; i < n; ++i) h = (h ^ (uint32_t) t[i]) * 0x100000001B3ull + (h >> 29);
        return h;
    }
    void reindex() {   // the lane holds another conversation now
        for (auto& m : ix) m.clear();
        done = 0;
    }
    void index(const std::vector<int32_t>& live) {
        if (live.size() < done) {   // the state was reset or cut: index it again
            for (auto& m : ix) m.clear();
            done = 0;
        }
        for (size_t i = done; i < live.size(); ++i)
            for (int n = kMinN; n <= kMaxN; ++n)
                if (i + 1 >= (size_t) n) ix[n - kMinN][key(live.data() + i + 1 - n, n)] = (int32_t) (i + 1);
        done = live.size();
    }
    /// Up to kmax tokens that may follow live + x.
    std::vector<int32_t> draft(const std::vector<int32_t>& live, int32_t x, int kmax) {
        std::vector<int32_t> out;
        if (kmax <= 0) return out;
        index(live);
        const size_t L = live.size() + 1;
        auto at = [&](size_t i) { return i < live.size() ? live[i] : x; };
        for (int n = kMaxN; n >= kMinN && out.empty(); --n) {
            if (L < (size_t) n) continue;
            int32_t tail[kMaxN];
            for (int i = 0; i < n; ++i) tail[i] = at(L - (size_t) n + (size_t) i);
            const auto it = ix[n - kMinN].find(key(tail, n));
            if (it == ix[n - kMinN].end()) continue;
            for (size_t j = (size_t) it->second; j < L && (int) out.size() < kmax && !is_img_id(at(j)); ++j) out.push_back(at(j));
        }
        return out;
    }
    /// The drafts worth checking: the k that gives the most tokens per millisecond.
    int choose(int avail) {
        avail = std::min(avail, max_k);
        if (avail <= 0) return 0;
        ++steps;
        const double a = std::min(0.98, acc / (acc + rej));
        const double t1 = t_ms[1] > 0 ? t_ms[1] : 1.0;
        int best = 0;
        double best_rate = 1.0 / t1;
        for (int k = 1; k <= avail; ++k) {
            const double t = t_ms[k + 1] > 0 ? t_ms[k + 1] : t1 * (1.0 + 0.1 * k);   // unmeasured: optimistic, so it is tried
            const double e = (1.0 - std::pow(a, k + 1)) / (1.0 - a);
            if (e / t > best_rate) { best_rate = e / t; best = k; }
        }
        if (best == 0 && steps % 32 == 0) best = 1;   // keep measuring
        return best;
    }
    void time(int rows, double ms) { t_ms[rows] = t_ms[rows] > 0 ? 0.8 * t_ms[rows] + 0.2 * ms : ms; }
    void outcome(int accepted, bool rejected) {
        acc = acc * 0.97 + accepted;
        rej = rej * 0.97 + (rejected ? 1 : 0);
    }
};

/// One decode step from x (the last token written, not yet in the state): x and the drafts in one forward, then each
/// row's token picked as plain decoding would pick it (a draft stands when it is that token; under sampling it stands
/// with its probability and a turned-down draft is left out of the redraw, which leaves the distribution as it was).
/// `emit` takes each token in order and says whether to go on. The state keeps x and the drafts that stood; x becomes
/// the last token emitted. False when emit said stop.
template <class Emit>
bool spec_step(Engine* ex, Session* S, std::vector<int32_t>& live, Drafter& dr, int32_t& x, int budget, int room,
               const Sampling& smp, std::vector<int32_t>& hist, std::mt19937_64& rng, std::vector<float>& buf, Emit&& emit) {
    std::vector<int32_t> toks{x};
    const int kmax = std::min({kGemmT - 1, budget - 1, room - 1});
    std::vector<int32_t> d = dr.draft(live, x, kmax);
    d.resize((size_t) dr.choose((int) d.size()));
    // DSpark (when loaded) drafts unless the conversation itself has a longer run to copy
    const bool dsk = S->ds && ex->dsm;
    bool from_ds = false;
    std::vector<Dist> qd;   // a sampled DSpark draft's distributions, a row each
    // A run of the conversation's own tokens (lookup) goes first; else DSpark drafts the length that gives the most
    // tokens a millisecond by its record (the drafts' time and the verify's by rows; 0: none)
    if (dsk && d.size() < 2 && g_dspark_n > 0) {
        DsState& st = *S->ds;
        const int km = std::min(g_dspark_n, kmax);
        const double t1 = dr.t_ms[1] > 0 ? dr.t_ms[1] : 20.0;
        int best = 0;
        double best_rate = 1.0 / t1, p = 1.0, e = 1.0;
        for (int kk = 1; kk <= km; ++kk) {
            p *= st.rate(kk);
            e += p;
            const double tv = dr.t_ms[kk + 1] > 0 ? dr.t_ms[kk + 1] : t1 * (1.0 + 0.12 * kk);   // (a row more: ~12% measured)
            const double td = st.t_draft[kk] > 0 ? st.t_draft[kk] : 2.0 + 1.5 * kk;
            if (e / (tv + td) > best_rate) { best_rate = e / (tv + td); best = kk; }
        }
        if (best == 0 && ++st.idle % 32 == 0) best = 1;   // keep measuring
        else if (best > 0 && best < km && ++st.idle % 8 == 0) ++best;   // and the next length up (else it is never measured)
        if (best > 0) {
            const auto td0 = std::chrono::steady_clock::now();
            // a sampled request: each draft drawn from the drafter's own distribution under the request's settings (its
            // distribution kept: the verify takes it with min(1, p / q)); greedy: the drafter's argmax
            const bool sampled = smp.temperature > 0.f;
            std::vector<int32_t> dh;
            if (sampled) dh = hist;
            auto pick = [&](const int32_t* ids, const float* lg, int n, int) -> int32_t {
                // the drafter's distribution over its candidates under the request's temperature, top-p and min-p
                std::vector<int> ord((size_t) n);
                for (int j = 0; j < n; ++j) ord[(size_t) j] = j;
                const int kk = std::min(n, smp.top_k > 0 ? smp.top_k : 64);
                std::partial_sort(ord.begin(), ord.begin() + kk, ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });
                Dist q;
                double sum = 0;
                std::vector<double> p((size_t) kk);
                for (int j = 0; j < kk; ++j) sum += (p[(size_t) j] = std::exp((lg[ord[(size_t) j]] - lg[ord[0]]) / smp.temperature));
                int keep = kk;
                double cum = 0;
                for (int j = 0; j < kk; ++j) {
                    if (p[(size_t) j] < smp.min_p * p[0]) { keep = j; break; }
                    cum += p[(size_t) j] / sum;
                    if (cum >= smp.top_p) { keep = j + 1; break; }
                }
                keep = std::max(1, keep);
                double tot = 0;
                for (int j = 0; j < keep; ++j) tot += p[(size_t) j];
                for (int j = 0; j < keep; ++j) {
                    q.idx.push_back(ids[ord[(size_t) j]]);
                    q.p.push_back(p[(size_t) j] / tot);
                }
                const int y = draw(q, rng);
                qd.push_back(std::move(q));
                dh.push_back(y);
                return y;
            };
            std::vector<int32_t> dd = sampled ? ex->ds_draft(*S, x, best, pick) : ex->ds_draft(*S, x, best);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            st.t_draft[best] = st.t_draft[best] > 0 ? 0.8 * st.t_draft[best] + 0.2 * ms : ms;
            if (dd.size() > d.size()) {
                d = std::move(dd);
                from_ds = true;
            } else
                qd.clear();
        }
    }
    toks.insert(toks.end(), d.begin(), d.end());
    const int T = (int) toks.size(), k = T - 1;
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t pos_before = S->pos;
    ex->S = S;
    ex->defer_conv = k > 0;
    ex->ds_mode = dsk ? (k > 0 ? 2 : 1) : 0;   // every kept row into the drafter: a verify's after the check
    ex->ds_from = 0;
    ex->forward(toks.data(), T, T);
    ex->ds_mode = 0;
    ex->defer_conv = false;
    int emitted = 0, stood = 0;
    bool go = true, turned_down = false;
    for (int i = 0; i < T && go; ++i) {
        const int dt = i < k ? d[(size_t) i] : -1;
        int y;
        if (!smp.needs_logits()) {
            y = ex->greedy(i);
        } else {
            ex->host_logits(buf, i);
            const Dist dd = dist(buf, smp, hist);
            auto prob = [](const Dist& v, int t) {
                for (size_t j = 0; j < v.idx.size(); ++j)
                    if (v.idx[j] == t) return v.p[j];
                return 0.0;
            };
            const double pd = dt >= 0 ? prob(dd, dt) : 0.0;
            if (dt >= 0 && i < (int) qd.size()) {   // a sampled draft: it stands with min(1, p / q); else from (p - q)+
                const double qv = prob(qd[(size_t) i], dt);
                if (qv > 0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) * qv < pd) y = dt;
                else {
                    Dist r;
                    double tot = 0;
                    for (size_t j = 0; j < dd.idx.size(); ++j) {
                        const double w = std::max(0.0, dd.p[j] - prob(qd[(size_t) i], dd.idx[j]));
                        if (w > 0) {
                            r.idx.push_back(dd.idx[j]);
                            r.p.push_back(w);
                            tot += w;
                        }
                    }
                    if (tot > 0) {
                        for (auto& v : r.p) v /= tot;
                        y = draw(r, rng);
                    } else
                        y = draw(dd, rng, dt);
                }
            } else if (dt >= 0 && pd >= 1.0) y = dt;
            else if (dt >= 0 && pd > 0 && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < pd) y = dt;
            else y = draw(dd, rng, dt);
        }
        hist.push_back(y);
        ++emitted;
        go = emit(y);
        if (y != dt) {
            turned_down = dt >= 0;
            break;
        }
        ++stood;
    }
    if (k > 0) {
        ex->commit(emitted);
        if (!from_ds) dr.outcome(stood, turned_down);
        if (dsk) ex->ds_inject(ex->dev[(size_t) ex->dsm->dev], *S->ds, 0, emitted, pos_before);
        if (from_ds) {
            S->ds->drafted += k;
            S->ds->accepted += stood;
            ++S->ds->steps;
            S->ds->outcome(k, stood);
        }
    }
    live.push_back(x);
    for (int i = 0; i + 1 < emitted; ++i) live.push_back(toks[(size_t) i + 1]);
    x = hist.back();
    dr.time(T, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return go;
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

std::vector<int32_t> g_eos;   // tokenizer.ggml.eos_token_id / eot_token_id
int g_bench_at = 0;           // --bench-at
int g_lookup_max = kGemmT - 1;   // --lookup-max-t
bool is_eos(int t) { return std::find(g_eos.begin(), g_eos.end(), t) != g_eos.end(); }

/// The VRAM the experts and the sessions' cache windows share (--kv-resident auto). Every expert and every cache cell
/// is in RAM too, so the pool only moves what VRAM holds: a session about to outgrow its window has it grown and the
/// least-used experts on its cards evicted for it (by routings a byte), a session that starts over gives its window
/// back to the most-used experts, and every so often a much-used expert in RAM swaps with a little-used one in VRAM.
/// Forwards hold `mu` shared from launch to their last sync; the pool changes things with it held alone.
struct Pool {
    Model* m = nullptr;
    const Engine* E = nullptr;          // the prompt executor: its window sizes
    bool on = false;
    std::shared_mutex mu;
    std::vector<std::vector<double>> use;   // [layer][expert] decayed routings
    std::atomic<int> ticks{0};
    int adapt_every = 4, swaps = 8;         // a round every adapt_every steps, up to `swaps` a card
    struct Pending {                        // an expert being copied into a slot it will take once the copy is done
        int l = 0, e = 0, slab = 0, slot = 0;
        cudaEvent_t ev = nullptr;
    };
    std::vector<Pending> pending;
    cudaStream_t cs[kMaxDev] = {};
    uint64_t reserve = (uint64_t) 768 << 20;   // VRAM a card keeps free
    uint64_t headroom = (uint64_t) 256 << 20;  // ... and beyond that before an expert comes back
    int grow = 16384;                          // cells a window grows past what it must hold

    void init(Model& mm, const Engine& e, bool enable) {
        m = &mm;
        E = &e;
        on = enable;
        use.assign(m->layers.size(), std::vector<double>((size_t) m->hp.n_expert, 0.0));
        if (on)
            for (int d = 0; d < m->n_dev; ++d) {
                CK(cudaSetDevice(m->dev_ids[(size_t) d]));
                CK(cudaStreamCreateWithFlags(&cs[d], cudaStreamNonBlocking));
            }
    }
    /// The copies that are done: their experts in VRAM from now on.
    void publish(bool wait) {
        for (size_t i = 0; i < pending.size();) {
            Pending& p = pending[i];
            Layer& L = m->layers[(size_t) p.l];
            CK(cudaSetDevice(m->dev_ids[(size_t) L.dev]));
            if (!wait && cudaEventQuery(p.ev) == cudaErrorNotReady) { ++i; continue; }
            CK(cudaEventSynchronize(p.ev));
            CK(cudaEventDestroy(p.ev));
            const Slab& sb = m->slabs[(size_t) p.slab];
            L.eptr_h[(size_t) p.e] = (unsigned long long) (sb.p + (size_t) p.slot * L.stride);
            L.on_gpu[(size_t) p.e] = 1;
            L.slab_of[(size_t) p.e] = p.slab;
            L.slot_of[(size_t) p.e] = p.slot;
            ++L.n_res;
            L.upload_ptrs();
            pending[i] = pending.back();
            pending.pop_back();
        }
    }
    bool is_pending(int l, int e) const {
        for (const Pending& p : pending)
            if (p.l == l && p.e == e) return true;
        return false;
    }
    /// What share of the (decayed) routings land on experts in VRAM.
    double hit_share() const {
        double hit = 0, all = 0;
        for (size_t l = 0; l < use.size(); ++l)
            for (size_t e = 0; e < use[l].size(); ++e) {
                all += use[l][e];
                if (m->layers[l].on_gpu[e]) hit += use[l][e];
            }
        return all > 0 ? hit / all : 0;
    }
    /// What share of the routings each layer's most-used half of the experts takes (the skew a VRAM half can catch).
    double top_half_share() const {
        double top = 0, all = 0;
        for (const auto& u : use) {
            std::vector<double> v = u;
            std::sort(v.begin(), v.end(), std::greater<double>());
            for (size_t i = 0; i < v.size(); ++i) {
                all += v[i];
                if (i < v.size() / 2) top += v[i];
            }
        }
        return all > 0 ? top / all : 0;
    }
    uint64_t free_on(int d) const {
        CK(cudaSetDevice(m->dev_ids[(size_t) d]));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        return fr > reserve ? fr - reserve : 0;
    }
    void refresh() {
        std::vector<uint32_t> h((size_t) m->hp.n_expert);
        for (size_t l = 0; l < m->layers.size(); ++l) {
            Layer& L = m->layers[l];
            CK(cudaSetDevice(m->dev_ids[(size_t) L.dev]));
            CK(cudaMemcpy(h.data(), L.hist, h.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemset(L.hist, 0, h.size() * 4));
            for (size_t e = 0; e < h.size(); ++e) use[l][e] = 0.9 * use[l][e] + h[e];
        }
    }
    // An expert's routings: a routing in RAM costs the CPU its bytes, and its bytes are the VRAM it holds, so the
    // VRAM's worth a byte is the routings whatever the expert's size (per byte would empty the Q8_0 layers first)
    double value(size_t l, int e) const { return use[l][(size_t) e] + 1e-3; }
    /// The resident expert on card d worth least (in) or the missing one worth most (!in); -1 when none.
    std::pair<int, int> pick(int d, bool in) const {
        int bl = -1, be = -1;
        double bv = 0;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.dev != d) continue;
            for (int e = 0; e < m->hp.n_expert; ++e) {
                if ((L.on_gpu[(size_t) e] != 0) != in) continue;
                if (!in && is_pending((int) l, e)) continue;
                const double v = value(l, e);
                if (bl < 0 || (in ? v < bv : v > bv)) { bl = (int) l; be = e; bv = v; }
            }
        }
        return {bl, be};
    }
    void evict(int l, int e) {
        Layer& L = m->layers[(size_t) l];
        CK(cudaSetDevice(m->dev_ids[(size_t) L.dev]));
        m->evict_expert(l, e);
        L.upload_ptrs();
    }
    bool bring(int l, int e) {
        Layer& L = m->layers[(size_t) l];
        CK(cudaSetDevice(m->dev_ids[(size_t) L.dev]));
        if (!m->load_expert(l, e)) return false;
        L.upload_ptrs();
        return true;
    }
    /// The slab on card d whose experts are worth least, all of them back to RAM (its VRAM to the driver); false when
    /// the card has no slab left.
    bool evict_slab(int d) {
        int best = -1;
        double bv = 0;
        for (size_t i = 0; i < m->slabs.size(); ++i) {
            const Slab& sb = m->slabs[i];
            if (!sb.p || sb.dev != d) continue;
            double v = 0;
            for (int k = 0; k < kSlab; ++k)
                if (sb.user_l[k] >= 0) v += value((size_t) sb.user_l[k], sb.user_e[k]);
            if (best < 0 || v < bv) { best = (int) i; bv = v; }
        }
        if (best < 0) return false;
        for (int k = 0; k < kSlab; ++k) {
            const Slab& sb = m->slabs[(size_t) best];
            if (sb.user_l[k] >= 0) evict(sb.user_l[k], sb.user_e[k]);
        }
        return true;
    }
    void fill(int d) {   // the most-used missing experts back while the card has room for them
        for (;;) {
            const auto [l, e] = pick(d, false);
            if (l < 0 || free_on(d) < m->layers[(size_t) l].stride + headroom || !bring(l, e)) return;
        }
    }
    /// Session s is about to hold `need` positions. A prompt (may_evict false) grows the window only into free VRAM: its
    /// chunks run every expert, so an expert in RAM is copied in for each of them, while a cell in RAM costs a chunk a
    /// few hundred bytes; decode is the other way round (8 experts a layer, every cell), so it takes experts' room.
    void before(Session& s, int64_t need, bool may_evict = true) {
        if (!on || s.win == 0 || s.win >= E->ctx || need <= s.win) return;
        std::unique_lock<std::shared_mutex> lk(mu);
        refresh();
        publish(true);
        int want = (int) std::min<int64_t>(E->ctx, (need + grow + 4095) / 4096 * 4096);
        for (int d = 0; d < m->n_dev; ++d) {
            const uint64_t cb = E->window_cell_bytes(s, d);
            if (!cb) continue;
            const uint64_t need_b = (uint64_t) (want - s.win) * cb;
            while (may_evict && free_on(d) < need_b && evict_slab(d)) {
            }
            want = (int) std::min<int64_t>(want, s.win + (int64_t) (free_on(d) / cb));
        }
        if (want > s.win) E->set_window(s, want);
    }
    /// Session s starts over: its window back to the smallest, the room back to experts.
    void shrink(Session& s) {
        if (!on || s.win == 0 || s.win <= E->min_window()) return;
        std::unique_lock<std::shared_mutex> lk(mu);
        publish(true);
        E->set_window(s, E->min_window());
        refresh();
        for (int d = 0; d < m->n_dev; ++d) fill(d);
    }
    std::string state(const Session& s) const {
        int res = 0, all = 0;
        for (const Layer& L : m->layers) {
            res += L.n_res;
            all += (int) L.on_gpu.size();
        }
        return "window " + std::to_string(s.win) + " cells, " + std::to_string(res) + " of " + std::to_string(all) + " experts in VRAM";
    }
    /// Every adapt_every decode steps: finished copies published, experts back into free room, and up to `swaps` a
    /// card where an expert in RAM is worth 1.25x a resident one of its size: the resident one leaves VRAM at once and
    /// the other is copied into its slot in the background (the CPU computes it until the copy is done).
    void tick() {
        if (!on || ++ticks % adapt_every) return;
        std::unique_lock<std::shared_mutex> lk(mu, std::try_to_lock);   // never waits: a prompt chunk holds it for seconds
        if (!lk.owns_lock()) return;
        refresh();
        publish(false);
        for (int d = 0; d < m->n_dev; ++d) {
            fill(d);
            for (int k = 0; k < swaps; ++k) {
                const auto [li, ei] = pick(d, false);
                if (li < 0) break;
                const Layer& Li = m->layers[(size_t) li];
                int lo = -1, eo = -1;   // the coldest resident expert of the same size on this card
                double vo = 0;
                for (size_t l = 0; l < m->layers.size(); ++l) {
                    const Layer& L = m->layers[l];
                    if (L.dev != d || L.stride != Li.stride) continue;
                    for (int e = 0; e < m->hp.n_expert; ++e)
                        if (L.on_gpu[(size_t) e] && (lo < 0 || value(l, e) < vo)) { lo = (int) l; eo = e; vo = value(l, e); }
                }
                if (lo < 0 || value((size_t) li, ei) < 1.25 * vo + 1e-12) break;
                Layer& Lo = m->layers[(size_t) lo];
                Pending p;
                p.l = li;
                p.e = ei;
                p.slab = Lo.slab_of[(size_t) eo];
                p.slot = Lo.slot_of[(size_t) eo];
                Slab& sb = m->slabs[(size_t) p.slab];
                CK(cudaSetDevice(m->dev_ids[(size_t) d]));
                Lo.eptr_h[(size_t) eo] = 0;   // out of VRAM now: nothing in flight (the lock), nothing new reads it
                Lo.on_gpu[(size_t) eo] = 0;
                Lo.slab_of[(size_t) eo] = Lo.slot_of[(size_t) eo] = -1;
                --Lo.n_res;
                Lo.upload_ptrs();
                sb.user_l[p.slot] = li;
                sb.user_e[p.slot] = ei;
                CK(cudaMemcpyAsync(sb.p + (size_t) p.slot * Li.stride, Li.hblobs + (size_t) ei * Li.stride, Li.stride,
                                   cudaMemcpyHostToDevice, cs[d]));
                CK(cudaEventCreateWithFlags(&p.ev, cudaEventDisableTiming));
                CK(cudaEventRecord(p.ev, cs[d]));
                pending.push_back(p);
            }
        }
    }
};

/// A prompt's rows into a session in chunks of the prompt executors (rows the size of a decode step left to the caller):
/// with two executors, chunk c on executor c % 2, each on a thread of its own, so the next chunk runs on one card while
/// this one runs on the other (Engine::PipeSide). Returns the rows read (fewer when `stop` said so) and the executor that
/// ran the last chunk.
/// Which chunk ends of a prompt read keep a checkpoint, and where they go (in order, only chunks the read kept).
struct CkPolicy {
    std::function<bool(int64_t)> want;   // the chunk's end position
    std::function<CkptP()> make;
    std::vector<CkptP> got;
};
constexpr int64_t kWholeRead = 4096;   // a prompt read this long goes back to whole windows for its chunks
struct Prompt {
    Engine* ex[2] = {};
    Pool* pool = nullptr;
    Engine::PipeSide side[2];
    void init(Engine& a, Engine* b, Pool& p) {
        ex[0] = &a;
        ex[1] = b;
        pool = &p;
        if (b) {
            a.make_pipe_side(side[0]);
            b->make_pipe_side(side[1]);
            a.pipe_self = &side[0];
            a.pipe_other = &side[1];
            b->pipe_self = &side[1];
            b->pipe_other = &side[0];
        }
    }
    int64_t ds_window() const { return ex[0]->dsm ? ex[0]->dsm->window : 0; }
    /// A chunk [p, p + T) of a read ending at `end`: its rows within the drafter's window give it their features.
    void ds_rows(Engine& e, Session& s, int64_t p, int T, int64_t end) const {
        e.ds_mode = 0;
        if (!s.ds || !e.dsm || p + T <= end - ds_window()) return;
        e.ds_mode = 1;
        e.ds_from = (int) std::max<int64_t>(0, end - ds_window() - p);
    }
    /// The chunks of rows [0, n) (all but a tail of at most kGemmT rows when `leave_tail`).
    int64_t read(Session& s, const int32_t* ids, int64_t n, bool leave_tail, Engine*& last,
                 const std::function<bool(int64_t)>& progress, CkPolicy* ckp = nullptr) {
        std::vector<std::pair<int64_t, int>> ch;   // (first row, rows)
        for (int64_t o = 0; n - o > (leave_tail ? kGemmT : 0);) {
            const int T = (int) std::min<int64_t>(ex[0]->tmax, n - o);
            ch.push_back({o, T});
            o += T;
        }
        if (ch.empty()) return 0;
        const int64_t pos0 = s.pos;
        if (s.ds && n > ds_window()) s.ds->valid_from = std::max(s.ds->valid_from, pos0 + n - ds_window());   // (rows before: no features)
        // a long read on whole windows (both cards are busy with its chunks: a split window would have each wait on the
        // other), split again for the decode after it; both take the same VRAM on each card
        struct Layout {
            Prompt& p;
            Session& s;
            bool on = false;
            Layout(Prompt& p_, Session& s_, bool flip) : p(p_), s(s_), on(flip) {
                if (on) set(0);
            }
            ~Layout() {
                if (on) set(1);
            }
            void set(int l) {
                std::unique_lock<std::shared_mutex> lk(p.pool->mu);
                p.ex[0]->set_window(s, s.win, true, l);
            }
        } layout(*this, s, s.split && n >= kWholeRead && n * 16 >= s.pos);   // (the flip copies the window from RAM twice)
        if (!ex[1]) {   // one executor: a chunk at a time
            int64_t got = 0;
            for (const auto& [o, T] : ch) {
                pool->before(s, s.pos + T, false);
                CkptP ck = ckp && ckp->want(pos0 + o + T) ? ckp->make() : nullptr;
                {
                    std::shared_lock<std::shared_mutex> lk(pool->mu);
                    ex[0]->S = &s;
                    ds_rows(*ex[0], s, pos0 + o, T, pos0 + n);
                    ex[0]->ck_out = ck.get();
                    ex[0]->forward(ids + o, T, o + T == ch.back().first + ch.back().second ? 1 : 0);
                    ex[0]->ds_mode = 0;
                    ex[0]->ck_out = nullptr;
                    ex[0]->sync_all();
                }
                if (ck) {
                    ck->pos = pos0 + o + T;
                    ckp->got.push_back(ck);
                }
                got = o + T;
                last = ex[0];
                if (!progress(got)) break;
            }
            return got;
        }
        pool->before(s, pos0 + n, false);   // the window for all of it (free VRAM only), before either side starts
        for (auto& sd : side)
            for (size_t i = 0; i < sd.ev.size(); ++i) sd.done[i].store(-1);
        std::mutex mu;
        std::condition_variable cv;
        std::vector<char> done(ch.size(), 0);
        std::vector<CkptP> cks(ch.size());
        if (ckp)   // made here: make() is not thread-safe
            for (size_t c = 0; c < ch.size(); ++c)
                if (ckp->want(pos0 + ch[c].first + ch[c].second)) cks[c] = ckp->make();
        std::atomic<bool> halt{false};
        auto run = [&](int w) {
            Engine& e = *ex[w];
            for (size_t c = (size_t) w; c < ch.size(); c += 2) {
                if (halt.load()) break;
                const auto [o, T] = ch[c];
                {
                    std::shared_lock<std::shared_mutex> lk(pool->mu);
                    e.S = &s;
                    e.pipe_chunk = (int64_t) c;
                    e.pipe_pos = pos0 + o;
                    ds_rows(e, s, pos0 + o, T, pos0 + n);
                    e.ck_out = cks[c].get();
                    e.forward(ids + o, T, c + 1 == ch.size() ? 1 : 0);
                    e.ds_mode = 0;
                    e.ck_out = nullptr;
                    e.sync_all();
                }
                if (cks[c]) cks[c]->pos = pos0 + o + T;
                std::lock_guard<std::mutex> g(mu);
                done[c] = 1;
                cv.notify_all();
            }
            std::lock_guard<std::mutex> g(mu);
            cv.notify_all();
        };
        std::thread t0(run, 0), t1(run, 1);
        size_t next = 0;
        int64_t got = 0;
        {
            std::unique_lock<std::mutex> g(mu);
            while (next < ch.size()) {
                cv.wait(g, [&] { return done[next] || halt.load(); });
                if (!done[next]) break;
                got = ch[next].first + ch[next].second;
                last = ex[next % 2];
                ++next;
                g.unlock();
                if (!progress(got)) halt.store(true);
                g.lock();
                if (halt.load()) break;
            }
        }
        halt.store(true);
        t0.join();
        t1.join();
        while (next < ch.size() && done[next]) {   // chunks that finished after a stop are in the session too
            got = ch[next].first + ch[next].second;
            last = ex[next % 2];
            ++next;
        }
        for (Engine* e : ex) e->pipe_chunk = -1;
        if (ckp)
            for (size_t c = 0; c < next; ++c)
                if (cks[c]) ckp->got.push_back(cks[c]);
        s.pos = pos0 + got;
        return got;
    }
};

/// Conversations parked in RAM (--conversation-cache-mib / -slots / -min-tokens): a lane about to take another
/// conversation parks the one it holds, and a request that continues a parked one has it copied back into a lane
/// instead of reading it again. A parked state: its tokens, each tiered cache's cells (from the lane's RAM copy), each
/// other cache (the sliding-window rings, from VRAM) and each visit's conv history. Oldest used out first.
struct Parked {
    std::vector<int32_t> ids;
    int64_t pos = 0;
    std::vector<std::vector<uint8_t>> cells;   // [visit]: K, V, K scales, V scales of its cells
    std::vector<std::vector<float>> conv;
    std::vector<CkptP> ck;   // its checkpoints (positions <= pos)
    uint64_t bytes = 0, used = 0;
};
struct ConvCache {
    std::vector<std::unique_ptr<Parked>> list;
    uint64_t bytes = 0, clock = 0, evictions = 0;
    uint64_t budget() const { return (uint64_t) g_cc_mib << 20; }
    /// The parked conversation that starts `ids` and is longest (a proper prefix: a request has a token to read).
    Parked* best(const std::vector<int32_t>& ids, size_t than) {
        Parked* b = nullptr;
        for (auto& p : list)
            if (p->ids.size() > than && p->ids.size() < ids.size() && (!b || p->ids.size() > b->ids.size()) &&
                std::equal(p->ids.begin(), p->ids.end(), ids.begin()))
                b = p.get();
        return b;
    }
    /// P out of the list (so parking another conversation cannot drop it while it is copied from), and back.
    std::unique_ptr<Parked> take(Parked* P) {
        for (size_t i = 0; i < list.size(); ++i)
            if (list[i].get() == P) {
                std::unique_ptr<Parked> u = std::move(list[i]);
                list.erase(list.begin() + (long) i);
                bytes -= u->bytes;
                return u;
            }
        return nullptr;
    }
    void put(std::unique_ptr<Parked> u) {
        bytes += u->bytes;
        list.push_back(std::move(u));
    }
    void drop_oldest() {
        ++evictions;
        size_t o = 0;
        for (size_t i = 1; i < list.size(); ++i)
            if (list[i]->used < list[o]->used) o = i;
        bytes -= list[o]->bytes;
        list.erase(list.begin() + (long) o);
    }
};

/// A conversation's lane: its session, its decode executor, the tokens its state holds, its decode thread.
struct Lane {
    Session sess;
    std::unique_ptr<Engine> ex;
    std::vector<int32_t> live;
    Drafter dr;
    std::atomic<bool> busy{false};
    uint64_t used = 0;
    std::thread th;
    // The state at the end of the lane's last prompt, for another lane to start from while this one decodes (a client
    // that sends a conversation again before its answer is done): its tokens, each visit's conv history, the logits.
    std::vector<int32_t> snap_ids;
    std::vector<float*> snap_conv;
    std::vector<uint8_t*> snap_ring;   // [visit]: a cache that lives in VRAM only (a sliding-window ring), in pinned RAM
    float* snap_logits = nullptr;
    std::vector<CkptP> ck;   // checkpoints of what it holds (positions <= sess.pos, ascending), its RAM copy their cells
};

struct Request {
    long long max_new = 0;
    Sampling smp;
    std::vector<int32_t> ids;
    std::string emb;                  // GENI: the pictures' embeddings file (strata-vision records, in prompt order)
    std::vector<uint64_t> img_keys;   // img_keys=<hex>,...: each picture's key (its normalized bytes' hash)
    std::shared_ptr<ImgRows> img;     // their rows, the ids rewritten (attach_images)
};

/// The engine protocol (serve/server.py).  GEN reads and decodes one request on the control lines; with --batch N,
/// BGEN <slot> reads a request into the lane that holds the longest start of it, then the lane decodes on its own
/// thread (BT / BDONE) while the control lines take the next request.  STOP ends the control lines' request, BSTOP a
/// slot's.  GENI / BGENI: the same with pictures, `<max_new> [key=value ...] <embeddings file> <ids>`.
struct Serve {
    Engine& big;
    Pool* pool = nullptr;
    Prompt* prompt = nullptr;
    ConvCache cc;
    std::vector<std::unique_ptr<Lane>> lanes;
    std::vector<int> slot_lane;
    std::unique_ptr<std::atomic<bool>[]> slot_stop;
    std::mutex mu, out_mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    uint64_t clock = 0;
    std::vector<float> lg;
    int lookup_max = kGemmT - 1;   // --lookup-max-t

    Serve(Engine& b, std::vector<std::unique_ptr<Lane>>&& l) : big(b), lanes(std::move(l)) {
        slot_lane.resize(lanes.size());
        for (size_t i = 0; i < lanes.size(); ++i) slot_lane[i] = (int) i;
        slot_stop.reset(new std::atomic<bool>[lanes.size()]);
        for (size_t i = 0; i < lanes.size(); ++i) slot_stop[i] = false;
    }

    void say(const char* fmt, ...) {
        std::lock_guard<std::mutex> lk(out_mu);
        va_list ap;
        va_start(ap, fmt);
        std::vprintf(fmt, ap);
        va_end(ap);
        std::fflush(stdout);
    }

    bool parse(const char* p0, Request& r, bool pics = false) {
        char* p = const_cast<char*>(p0);
        r.max_new = std::strtoll(p, &p, 10);
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") r.smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") r.smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") r.smp.top_k = std::atoi(v);
            else if (key == "min_p") r.smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") r.smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") r.smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") r.smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") r.smp.present = std::strtof(v, nullptr);
            else if (key == "seed") r.smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys")
                for (const char* q = v; *q;) {
                    char* e = nullptr;
                    r.img_keys.push_back(std::strtoull(q, &e, 16));
                    q = *e == ',' ? e + 1 : e;
                    if (e == q && *q) break;
                }
        }
        if (pics) {   // the embeddings file: the first word without an =
            while (*p == ' ') ++p;
            const char* f0 = p;
            while (*p && *p != ' ') ++p;
            r.emb.assign(f0, (size_t) (p - f0));
            if (r.emb.empty()) {
                say("ERR GENI: no embeddings file\n");
                return false;
            }
        }
        r.ids = parse_ids(p);
        if (r.max_new < 1 || r.ids.empty()) {
            say("ERR bad request: max_new or the ids\n");
            return false;
        }
        if ((int64_t) r.ids.size() + r.max_new > big.ctx) {
            say("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", r.ids.size(), r.max_new, big.ctx);
            return false;
        }
        for (int32_t t : r.ids)
            if (t < 0 || t >= (int32_t) big.m->hp.vocab) {
                say("ERR a token id is outside the vocabulary\n");
                return false;
            }
        return !pics || attach_images(r);
    }
    /// GENI: the file's pictures onto their runs of <|unused_200054|>, in order (one row a token); each row's id made from
    /// the picture's key (the server's, else its rows' hash) and the row.
    bool attach_images(Request& r) {
        auto fail = [&](const std::string& why) {
            say("ERR %s\n", why.c_str());
            return false;
        };
        const int N = big.m->hp.n_embd;
        auto img = std::make_shared<ImgRows>();
        img->n_embd = N;
        struct Pic { int64_t n; size_t off; };
        std::vector<Pic> pics;
        std::FILE* f = std::fopen(r.emb.c_str(), "rb");
        if (!f) return fail("cannot open " + r.emb);
        for (;;) {
            int32_t h[5];
            const size_t got = std::fread(h, 4, 5, f);
            if (got == 0) break;
            if (got != 5 || h[0] != 0x31455653 || h[1] < 1 || h[2] < 1 || h[3] < 1 || (int64_t) h[2] * h[3] != h[1] || h[4] != N) {
                std::fclose(f);
                return fail("bad embeddings file (expected strata-vision records of width " + std::to_string(N) + ")");
            }
            const size_t off = img->rows.size() / (size_t) N, cnt = (size_t) h[1] * (size_t) N;
            img->rows.resize(img->rows.size() + cnt);
            if (std::fread(img->rows.data() + off * (size_t) N, 4, cnt, f) != cnt) {
                std::fclose(f);
                return fail("short embeddings file");
            }
            pics.push_back({h[1], off});
        }
        std::fclose(f);
        auto fnv = [](const void* p, size_t n, uint64_t h) {
            const uint8_t* b = static_cast<const uint8_t*>(p);
            for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
            return h;
        };
        size_t k = 0;
        for (size_t i = 0; i < r.ids.size();) {
            if (r.ids[i] != kImagePad) { ++i; continue; }
            if (k >= pics.size()) return fail("the prompt has more images than the embeddings file");
            const Pic& pc = pics[k];
            uint64_t key = fnv(&pc.n, sizeof pc.n, 0xCBF29CE484222325ull);
            key = r.img_keys.size() == pics.size() ? fnv(&r.img_keys[k], 8, key)
                                                   : fnv(img->rows.data() + pc.off * (size_t) N, (size_t) pc.n * N * 4, key);
            ++k;
            for (int64_t j = 0; j < pc.n; ++j) {
                if (i + (size_t) j >= r.ids.size() || r.ids[i + (size_t) j] != kImagePad)
                    return fail("image " + std::to_string(k) + " has " + std::to_string((long long) pc.n) + " rows but fewer <|unused_200054|> tokens");
                int32_t id = kImgId0 | (int32_t) (fnv(&j, sizeof j, key) & 0x3FFFFFFF);
                while (img->at.count(id)) id = kImgId0 | ((id + 1) & 0x3FFFFFFF);   // (a row the same key and row gave twice)
                img->at[id] = pc.off + (size_t) j;
                r.ids[i + (size_t) j] = id;
            }
            i += (size_t) pc.n;
        }
        if (k != pics.size()) return fail("the embeddings file has more images than the prompt");
        if (is_img_id(r.ids.back())) return fail("the prompt cannot end in an image");
        r.img = std::move(img);
        return true;
    }
    /// The executors a request's prompt reads with: its pictures' rows, or none.
    void show_images(Lane& L, const ImgRows* img) {
        for (Engine* e : prompt->ex)
            if (e) e->img = img;
        L.ex->img = img;
    }

    static size_t match(const Lane& L, const std::vector<int32_t>& ids) {
        if (L.live.empty() || L.live.size() >= ids.size()) return 0;
        return std::equal(L.live.begin(), L.live.end(), ids.begin()) ? L.live.size() : 0;
    }
    /// The free lane for a request: the one holding the longest start of it, else `prefer`, else an empty one, else
    /// the least recently used.  -1: every lane decodes.
    int pick(const std::vector<int32_t>& ids, int prefer) const {
        int best = -1, empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < (int) lanes.size(); ++i) {
            const Lane& L = *lanes[(size_t) i];
            if (L.busy) continue;
            size_t l = reach(L, ids);
            for (auto it = L.ck.rbegin(); it != L.ck.rend(); ++it)
                if ((size_t) (*it)->pos > l && (size_t) (*it)->pos < ids.size() &&
                    std::equal((*it)->ids.begin(), (*it)->ids.end(), ids.begin())) {
                    l = (size_t) (*it)->pos;
                    break;
                }
            if (l > bl) { bl = l; best = i; }
            if (L.live.empty() && empty < 0) empty = i;
            if (lru < 0 || L.used < lanes[(size_t) lru]->used) lru = i;
        }
        if (best >= 0) return best;
        if (prefer >= 0 && !lanes[(size_t) prefer]->busy) return prefer;
        return empty >= 0 ? empty : lru;
    }

    struct Read {
        size_t reuse = 0;
        int64_t fresh = 0, got = 0;
        double ms = 0;
        bool cancelled = false;
        Engine* last = nullptr;
    };
    /// The request's prompt into lane L: the part its state does not hold, in prompt chunks on the shared executor
    /// (the last few rows on the lane's own).
    Read read(Lane& L, const std::vector<int32_t>& ids, int64_t forked = -1) {
        Read r;
        r.reuse = forked >= 0 ? (size_t) forked : match(L, ids);
        L.ex->S = &L.sess;
        if (forked >= 0 && r.reuse == ids.size()) r.last = L.ex.get();   // nothing new: the fork brought the logits
        if (r.reuse == 0) {
            park(L);
            L.ex->reset();
            if (L.sess.ds) L.sess.ds->forget(0);
            L.live.clear();
            L.ck.clear();
            pool->shrink(L.sess);
        }
        ck_trim(L.ck, (int64_t) r.reuse);
        const bool cks = ck_ok();
        const int64_t end_pos = (int64_t) ids.size(), rows = prompt->ex[0]->tmax;
        CkPolicy pol;
        pol.want = [&](int64_t e) { return e + 2048 < end_pos && ck_due(e - rows, e); };
        pol.make = [&] { return make_ckpt(); };
        L.used = ++clock;
        const auto t0 = Clock::now();
        r.fresh = (int64_t) (ids.size() - r.reuse);
        const int32_t* src = ids.data() + r.reuse;
        auto report = [&](int64_t got) {
            say("PP %lld %lld %.0f %.1f\n", (long long) got, (long long) r.fresh, ms_since(t0),
                (double) got / std::max(1e-9, ms_since(t0) / 1000.0));
            return !stop.load() || got >= r.fresh;
        };
        r.got = prompt->read(L.sess, src, r.fresh, true, r.last, report, cks ? &pol : nullptr);   // the whole chunks (two cards at once)
        if (r.got < r.fresh && stop.load() && r.got > 0) r.cancelled = true;
        while (!r.cancelled && r.got < r.fresh) {   // the last few rows on the lane's own executor
            const int T = (int) std::min<int64_t>(L.ex->tmax, r.fresh - r.got);
            pool->before(L.sess, L.sess.pos + T);
            {
                std::shared_lock<std::shared_mutex> lk(pool->mu);
                L.ex->S = &L.sess;
                L.ex->ds_mode = L.sess.ds ? 1 : 0;   // the tail's rows into the drafter too
                L.ex->ds_from = 0;
                L.ex->forward(src + r.got, T, r.got + T == r.fresh ? 1 : 0);
                L.ex->ds_mode = 0;
                L.ex->sync_all();
            }
            r.got += T;
            r.last = L.ex.get();
            if (!report(r.got)) {
                r.cancelled = r.got < r.fresh;
                break;
            }
        }
        L.live.insert(L.live.end(), src, src + r.got);
        for (auto& k : pol.got)
            if (k->pos <= L.sess.pos) ck_add(L, k);
        if (cks && !r.cancelled && r.got == r.fresh && r.fresh >= 256) ck_add(L, ckpt_now(L));   // the prompt's end
        // a prompt read keeps a snapshot; a slot taking over its own conversation (a few rows) keeps the one it has
        if (!r.cancelled && r.got == r.fresh && r.last && (forked >= 0 || r.reuse == 0 || r.fresh >= 256 || L.snap_ids.empty()))
            snapshot(L, *r.last);
        r.ms = ms_since(t0);
        return r;
    }
    /// A VRAM-only cache's bytes (rings whole; a tiered cache has its RAM copy instead).
    size_t ring_bytes(const Lane& L, int il) const {
        const Kv& c = L.sess.kv[(size_t) il];
        const bool q8 = L.sess.kv_q8[(size_t) il] != 0;
        return (size_t) big.m->hp.n_kv * (size_t) c.cells * (q8 ? kHD * 2 + 8 : kHD * 4);
    }
    void ring_copy(const Lane& L, int il, uint8_t* host, bool to_host) const {   // K, V (and scales) of a VRAM-only cache
        const Kv& c = L.sess.kv[(size_t) il];
        const bool q8 = L.sess.kv_q8[(size_t) il] != 0;
        const size_t n = (size_t) big.m->hp.n_kv * (size_t) c.cells, eb = q8 ? 1 : 2;
        const cudaMemcpyKind kd = to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        auto cp = [&](void* dev, size_t b) {
            CK(to_host ? cudaMemcpy(host, dev, b, kd) : cudaMemcpy(dev, host, b, kd));
            host += b;
        };
        cp(c.k, n * kHD * eb);
        cp(c.v, n * kHD * eb);
        if (q8) {
            cp(c.ks, n * 4);
            cp(c.vs, n * 4);
        }
    }
    /// Lane L back to its snapshot (the end of its last prompt): the rings and conv histories copied back, the VRAM window
    /// refilled from the RAM copy (the cells past it are rewritten before they are read). Needs tiered caches.
    int64_t rewind(Lane& L) {
        const HP& hp = big.m->hp;
        const int64_t n = (int64_t) L.snap_ids.size();
        const size_t cb = (size_t) big.conv_c * 3 * 4;
        const auto t0 = Clock::now();
        std::unique_lock<std::shared_mutex> lk(pool->mu);
        for (int il = 0; il < hp.n_vis; ++il) {
            CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
            if (!L.sess.kv[(size_t) il].hk) ring_copy(L, il, L.snap_ring[(size_t) il], false);
            CK(cudaMemcpy(L.sess.conv[(size_t) il], L.snap_conv[(size_t) il], cb, cudaMemcpyDeviceToDevice));
        }
        L.sess.pos = n;
        if (L.sess.win > 0) big.set_window(L.sess, L.sess.win, true);
        L.live = L.snap_ids;
        ck_trim(L.ck, n);
        L.dr.reindex();
        std::fprintf(stderr, "strata serve: lane back to the end of its last prompt (%lld tokens, %.0f ms)\n", (long long) n, ms_since(t0));
        return n;
    }
    bool tiered(const Lane& L) const {
        for (const Kv& c : L.sess.kv)
            if (c.ring == 0 && !c.hk) return false;   // a whole cache in VRAM only: no RAM copy to rewind from
        return true;
    }
    /// How much of `ids` lane L can start from: what it holds, or its snapshot.
    size_t reach(const Lane& L, const std::vector<int32_t>& ids) const {
        const size_t m = match(L, ids);
        const size_t s = L.snap_ids.size();
        if (s > m && s < ids.size() && !L.snap_ring.empty() && tiered(L) && std::equal(L.snap_ids.begin(), L.snap_ids.end(), ids.begin()))
            return s;
        return m;
    }
    // ---- checkpoints: a request that shares a long start with any lane or parked conversation continues from the last
    // checkpoint inside the shared part (exact: the same cells and states as reading it), whatever comes after differs.
    bool ck_ok() const {
        for (int il = 0; il < big.m->hp.n_vis; ++il) {
            const Layer& Ly = big.m->layers[(size_t) big.m->hp.phys(il)];
            if (Ly.swa && lanes[0]->sess.kv_q8[(size_t) il]) return false;   // the copies are FP16 rings
        }
        return tiered(*lanes[0]);
    }
    CkptP make_ckpt() const {
        const HP& hp = big.m->hp;
        auto k = std::make_shared<Ckpt>();
        k->swa_bytes = (size_t) 2 * hp.n_kv * hp.swa * kHD * 2;
        k->conv_bytes = (size_t) big.conv_c * 3 * 4;
        k->swa.assign((size_t) hp.n_vis, nullptr);
        k->conv.assign((size_t) hp.n_vis, nullptr);
        for (int il = 0; il < hp.n_vis; ++il) {
            if (big.m->layers[(size_t) hp.phys(il)].swa) CK(cudaHostAlloc((void**) &k->swa[(size_t) il], k->swa_bytes, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &k->conv[(size_t) il], k->conv_bytes, cudaHostAllocPortable));
        }
        return k;
    }
    /// Spacing: 8K tokens to 64K, 16K beyond (a 188K prompt: 16 of ~170 MB).
    static bool ck_due(int64_t a, int64_t b) {   // a chunk [a, b) crosses a checkpoint line
        auto line = [](int64_t x) { return x <= 65536 ? x / 8192 : 8 + (x - 65536) / 16384; };
        return line(b) > line(a);
    }
    /// L's state now as a checkpoint (its executors idle).
    CkptP ckpt_now(Lane& L) {
        CkptP k = make_ckpt();
        L.ex->S = &L.sess;
        L.ex->sync_all();
        for (int il = 0; il < big.m->hp.n_vis; ++il) {
            const Layer& Ly = big.m->layers[(size_t) big.m->hp.phys(il)];
            Dev& D = L.ex->dev[(size_t) Ly.dev];
            CK(cudaSetDevice(D.id));
            L.ex->ckpt_visit(D, *k, il, Ly.swa, L.sess.pos);
        }
        L.ex->sync_all();
        k->pos = L.sess.pos;
        return k;
    }
    void ck_add(Lane& L, CkptP k) {
        if (!L.ck.empty() && L.ck.back()->pos >= k->pos) return;
        k->ids.assign(L.live.begin(), L.live.begin() + k->pos);
        L.ck.push_back(std::move(k));
        constexpr size_t kMax = 40;   // past it every other one of the older half goes
        if (L.ck.size() > kMax) {
            std::vector<CkptP> kept;
            for (size_t i = 0; i < L.ck.size(); ++i)
                if (i >= L.ck.size() / 2 || i % 2 == 0) kept.push_back(L.ck[i]);
            L.ck.swap(kept);
        }
    }
    static void ck_trim(std::vector<CkptP>& v, int64_t upto) {
        while (!v.empty() && v.back()->pos > upto) v.pop_back();
    }
    struct CkPick {
        CkptP k;
        int lane = -1;
        Parked* P = nullptr;
    };
    /// The longest checkpoint whose tokens start `ids` (with a token after it to read): a lane's or a parked one's.
    CkPick best_ck(const std::vector<int32_t>& ids) {
        CkPick b;
        auto look = [&](const std::vector<CkptP>& v, int lane, Parked* P) {
            for (auto it = v.rbegin(); it != v.rend(); ++it) {
                const Ckpt& k = **it;
                if ((size_t) k.pos >= ids.size() || (b.k && k.pos <= b.k->pos)) continue;
                if (std::equal(k.ids.begin(), k.ids.end(), ids.begin())) {
                    b.k = *it;
                    b.lane = lane;
                    b.P = P;
                    return;
                }
            }
        };
        for (int i = 0; i < (int) lanes.size(); ++i) look(lanes[(size_t) i]->ck, i, nullptr);
        for (auto& P : cc.list) look(P->ck, -1, P.get());
        return b;
    }
    /// Lane li continues from checkpoint c: the cells [0, pos) from where they are (its own RAM copy, another lane's,
    /// a parked conversation's), the rings' last cells and the conv histories from the checkpoint.
    int64_t resume_ck(int li, const CkPick& c) {
        Lane& D = *lanes[(size_t) li];
        const Ckpt& k = *c.k;
        const int64_t n = k.pos;
        std::vector<CkptP> keep = c.P ? c.P->ck : lanes[(size_t) c.lane]->ck;
        ck_trim(keep, n);
        std::unique_ptr<Parked> held = c.P ? cc.take(c.P) : nullptr;
        if (c.lane != li) park(D);
        if (held) {
            held->used = ++cc.clock;
            cc.put(std::move(held));
        }
        std::unique_lock<std::shared_mutex> lk(pool->mu);   // a busy source lane between two steps from here on
        const auto t0 = Clock::now();
        const HP& hp = big.m->hp;
        const size_t HKV = (size_t) hp.n_kv, row = (size_t) kHD * 2;
        std::vector<std::thread> th;
        for (int il = 0; il < hp.n_vis; ++il) {
            Kv& dk = D.sess.kv[(size_t) il];
            if (!dk.hk || c.lane == li) continue;
            th.emplace_back([&, il] {   // the RAM cells, a thread a visit
                Kv& dk2 = D.sess.kv[(size_t) il];
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t o = kh * (size_t) dk2.hcells;
                    if (c.P) {   // a parked conversation's cells: [kh][K n0][V n0][Ks n0][Vs n0]
                        const size_t n0 = (size_t) c.P->pos;
                        const uint8_t* r = c.P->cells[(size_t) il].data() + kh * n0 * (kHD * 2 + 8);
                        std::memcpy(dk2.hk + o * kHD, r, (size_t) n * kHD);
                        std::memcpy(dk2.hv + o * kHD, r + n0 * kHD, (size_t) n * kHD);
                        std::memcpy(dk2.hks + o, r + n0 * kHD * 2, (size_t) n * 4);
                        std::memcpy(dk2.hvs + o, r + n0 * (kHD * 2 + 4), (size_t) n * 4);
                    } else {
                        const Kv& sk = lanes[(size_t) c.lane]->sess.kv[(size_t) il];
                        std::memcpy(dk2.hk + o * kHD, sk.hk + o * kHD, (size_t) n * kHD);
                        std::memcpy(dk2.hv + o * kHD, sk.hv + o * kHD, (size_t) n * kHD);
                        std::memcpy(dk2.hks + o, sk.hks + o, (size_t) n * 4);
                        std::memcpy(dk2.hvs + o, sk.hvs + o, (size_t) n * 4);
                    }
                }
            });
        }
        for (int il = 0; il < hp.n_vis; ++il) {
            const Layer& Ly = big.m->layers[(size_t) hp.phys(il)];
            CK(cudaSetDevice(big.m->dev_ids[(size_t) Ly.dev]));
            CK(cudaMemcpy(D.sess.conv[(size_t) il], k.conv[(size_t) il], k.conv_bytes, cudaMemcpyHostToDevice));
            if (!Ly.swa || !k.swa[(size_t) il]) continue;
            const Kv& r = D.sess.kv[(size_t) il];
            const int64_t keep_c = hp.swa, cnt = std::min<int64_t>(n, keep_c), st0 = n - cnt;
            const uint8_t* ks = k.swa[(size_t) il];
            const uint8_t* vs = ks + HKV * (size_t) keep_c * row;
            for (int64_t a = st0; a < n;) {
                const int64_t b = std::min<int64_t>(n, (a / r.ring + 1) * r.ring), m = b - a;
                const size_t so = (size_t) (a - st0) * row, dofs = (size_t) (a % r.ring) * row;
                CK(cudaMemcpy2D((uint8_t*) r.k + dofs, (size_t) r.cells * row, ks + so, (size_t) keep_c * row, (size_t) m * row, HKV,
                                cudaMemcpyHostToDevice));
                CK(cudaMemcpy2D((uint8_t*) r.v + dofs, (size_t) r.cells * row, vs + so, (size_t) keep_c * row, (size_t) m * row, HKV,
                                cudaMemcpyHostToDevice));
                a = b;
            }
        }
        for (auto& t : th) t.join();
        D.sess.pos = n;
        if (D.sess.ds && c.lane != li) D.sess.ds->forget(n);   // the drafter did not see these positions made
        if (D.sess.win > 0) big.set_window(D.sess, D.sess.win, true);   // the VRAM window from the RAM cells
        D.live = k.ids;
        D.snap_ids.clear();
        D.dr.reindex();
        D.ck = std::move(keep);
        const char* from = c.P ? "a parked conversation" : c.lane == li ? "its own" : "another lane";
        std::fprintf(stderr, "strata serve: lane %d continues from a checkpoint at %lld tokens (%s, %.0f ms)\n", li,
                     (long long) n, from, ms_since(t0));
        return n;
    }
    /// The bytes a lane's state at `n` positions takes parked.
    uint64_t park_bytes(const Lane& L, int64_t n) const {
        const HP& hp = big.m->hp;
        uint64_t b = 0;
        for (int il = 0; il < hp.n_vis; ++il) {
            const Kv& c = L.sess.kv[(size_t) il];
            const bool q8 = L.sess.kv_q8[(size_t) il] != 0;
            const uint64_t cells = c.hk || c.ring == 0 ? (uint64_t) n : (uint64_t) c.cells;
            b += (uint64_t) hp.n_kv * cells * (q8 ? kHD + 4 : kHD * 2) * 2 + (uint64_t) big.conv_c * 3 * 4;
        }
        return b + L.live.size() * 4;
    }
    /// L's conversation into the cache (when it is long enough and fits; an older parked copy of a prefix of it goes).
    void park(Lane& L) {
        // the state at the end of its last prompt (what a client's next turn starts with: the answer it sends back is
        // rendered again, not the tokens the model wrote) when it has one, else what it holds
        const bool snap = !L.snap_ids.empty() && !L.snap_ring.empty() && tiered(L);
        const std::vector<int32_t>& ids = snap ? L.snap_ids : L.live;
        const int64_t n = (int64_t) ids.size();
        if (g_cc_mib == 0 || g_cc_slots == 0 || n < g_cc_min || (!snap && L.sess.pos != n)) return;
        for (size_t i = 0; i < cc.list.size(); ++i)   // already there (it was restored and read nothing new since)
            if ((int64_t) cc.list[i]->ids.size() == n && cc.list[i]->ids == ids) {
                cc.list[i]->used = ++cc.clock;
                return;
            }
        std::vector<CkptP> cks = L.ck;
        ck_trim(cks, n);
        uint64_t need = park_bytes(L, n);
        for (auto& k : cks) need += k->bytes();
        if (need > cc.budget()) {   // its checkpoints first
            for (auto& k : cks) need -= k->bytes();
            cks.clear();
            if (need > cc.budget()) return;
        }
        for (size_t i = 0; i < cc.list.size();) {   // the copies this one makes redundant (prefixes of it)
            Parked& q = *cc.list[i];
            if (q.ids.size() < ids.size() && std::equal(q.ids.begin(), q.ids.end(), ids.begin())) {
                cc.bytes -= q.bytes;
                cc.list.erase(cc.list.begin() + (long) i);
            } else
                ++i;
        }
        while (!cc.list.empty() && (cc.bytes + need > cc.budget() || (int) cc.list.size() >= g_cc_slots)) cc.drop_oldest();
        const auto t0 = Clock::now();
        const HP& hp = big.m->hp;
        const size_t HKV = (size_t) hp.n_kv, cb = (size_t) big.conv_c * 3;
        auto P = std::make_unique<Parked>();
        P->ids = ids;
        P->pos = n;
        P->ck = std::move(cks);
        P->cells.resize((size_t) hp.n_vis);
        P->conv.resize((size_t) hp.n_vis);
        for (int il = 0; il < hp.n_vis; ++il) {
            const Kv& c = L.sess.kv[(size_t) il];
            const bool q8 = L.sess.kv_q8[(size_t) il] != 0;
            CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
            std::vector<uint8_t>& out = P->cells[(size_t) il];
            const size_t eb = q8 ? 1 : 2;
            if (c.hk) {   // the cells [0, n) of its RAM copy
                out.resize(HKV * (size_t) n * (kHD * 2 + 8));
                uint8_t* w = out.data();
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t o = kh * (size_t) c.hcells;
                    std::memcpy(w, c.hk + o * kHD, (size_t) n * kHD); w += (size_t) n * kHD;
                    std::memcpy(w, c.hv + o * kHD, (size_t) n * kHD); w += (size_t) n * kHD;
                    std::memcpy(w, c.hks + o, (size_t) n * 4); w += (size_t) n * 4;
                    std::memcpy(w, c.hvs + o, (size_t) n * 4); w += (size_t) n * 4;
                }
            } else if (c.ring > 0) {   // a ring, whole: from the snapshot in RAM, or as it is in VRAM
                out.resize(ring_bytes(L, il));
                if (snap) std::memcpy(out.data(), L.snap_ring[(size_t) il], out.size());
                else ring_copy(L, il, out.data(), true);
            } else {   // a whole cache in VRAM only: its cells [0, n)
                const size_t cells = (size_t) n;
                out.resize(HKV * cells * kHD * eb * 2 + (q8 ? HKV * cells * 8 : 0));
                uint8_t* w = out.data();
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t ro = kh * (size_t) c.cells;
                    CK(cudaMemcpy(w, (const uint8_t*) c.k + ro * kHD * eb, cells * kHD * eb, cudaMemcpyDeviceToHost)); w += cells * kHD * eb;
                    CK(cudaMemcpy(w, (const uint8_t*) c.v + ro * kHD * eb, cells * kHD * eb, cudaMemcpyDeviceToHost)); w += cells * kHD * eb;
                    if (q8) {
                        CK(cudaMemcpy(w, c.ks + ro, cells * 4, cudaMemcpyDeviceToHost)); w += cells * 4;
                        CK(cudaMemcpy(w, c.vs + ro, cells * 4, cudaMemcpyDeviceToHost)); w += cells * 4;
                    }
                }
            }
            P->conv[(size_t) il].resize(cb);
            CK(cudaMemcpy(P->conv[(size_t) il].data(), snap ? L.snap_conv[(size_t) il] : L.sess.conv[(size_t) il], cb * 4,
                          cudaMemcpyDeviceToHost));
        }
        P->bytes = need;
        P->used = ++cc.clock;
        cc.bytes += need;
        cc.list.push_back(std::move(P));
        std::fprintf(stderr, "strata serve: conversation cache: parked %lld tokens (%.2f GiB, %.0f ms; %.1f of %d GiB); "
                     "parked=%zu bytes=%llu evictions=%llu\n", (long long) n, (double) need / (1 << 30), ms_since(t0),
                     (double) cc.bytes / (1 << 30), g_cc_mib >> 10, cc.list.size(), (unsigned long long) cc.bytes,
                     (unsigned long long) cc.evictions);
    }
    /// Parked conversation P back into lane L (idle; what it held is parked first). Returns its positions.
    int64_t restore(Lane& L, Parked& P) {
        std::unique_ptr<Parked> held = cc.take(&P);
        park(L);
        if (held) cc.put(std::move(held));
        const auto t0 = Clock::now();
        const HP& hp = big.m->hp;
        const size_t HKV = (size_t) hp.n_kv, cb = (size_t) big.conv_c * 3;
        const int64_t n = P.pos;
        std::unique_lock<std::shared_mutex> lk(pool->mu);
        for (int il = 0; il < hp.n_vis; ++il) {
            Kv& c = L.sess.kv[(size_t) il];
            const bool q8 = L.sess.kv_q8[(size_t) il] != 0;
            const size_t eb = q8 ? 1 : 2;
            CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
            const uint8_t* r = P.cells[(size_t) il].data();
            if (c.hk) {
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t o = kh * (size_t) c.hcells;
                    std::memcpy(c.hk + o * kHD, r, (size_t) n * kHD); r += (size_t) n * kHD;
                    std::memcpy(c.hv + o * kHD, r, (size_t) n * kHD); r += (size_t) n * kHD;
                    std::memcpy(c.hks + o, r, (size_t) n * 4); r += (size_t) n * 4;
                    std::memcpy(c.hvs + o, r, (size_t) n * 4); r += (size_t) n * 4;
                }
            } else if (c.ring > 0) {   // a whole ring
                ring_copy(L, il, const_cast<uint8_t*>(r), false);
            } else {
                const size_t cells = (size_t) n;
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t ro = kh * (size_t) c.cells;
                    CK(cudaMemcpy((uint8_t*) c.k + ro * kHD * eb, r, cells * kHD * eb, cudaMemcpyHostToDevice)); r += cells * kHD * eb;
                    CK(cudaMemcpy((uint8_t*) c.v + ro * kHD * eb, r, cells * kHD * eb, cudaMemcpyHostToDevice)); r += cells * kHD * eb;
                    if (q8) {
                        CK(cudaMemcpy(c.ks + ro, r, cells * 4, cudaMemcpyHostToDevice)); r += cells * 4;
                        CK(cudaMemcpy(c.vs + ro, r, cells * 4, cudaMemcpyHostToDevice)); r += cells * 4;
                    }
                }
            }
            CK(cudaMemcpy(L.sess.conv[(size_t) il], P.conv[(size_t) il].data(), cb * 4, cudaMemcpyHostToDevice));
        }
        L.sess.pos = n;
        if (L.sess.ds) L.sess.ds->forget(n);
        if (L.sess.win > 0) big.set_window(L.sess, L.sess.win, true);
        L.live = P.ids;
        L.snap_ids.clear();
        L.dr.reindex();
        L.ck = P.ck;
        P.used = ++cc.clock;
        std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%.0f ms); parked=%zu bytes=%llu\n",
                     (long long) n, ms_since(t0), cc.list.size(), (unsigned long long) cc.bytes);
        return n;
    }
    /// The state request `ids` can start from in lane li, other than the lane's own: a busy lane's (fork) or a parked one.
    int64_t start_from(int li, const std::vector<int32_t>& ids) {
        Lane& L = *lanes[(size_t) li];
        const size_t own = reach(L, ids);
        Parked* P = cc.best(ids, own);
        if (ck_ok()) {   // a checkpoint inside a longer shared start than any whole state gives
            const int fs = fork_src(li, ids);
            const size_t whole = std::max({own, P ? P->ids.size() : (size_t) 0,
                                           fs >= 0 ? lanes[(size_t) fs]->snap_ids.size() : (size_t) 0});
            const CkPick c = best_ck(ids);
            if (c.k && (size_t) c.k->pos > whole + 256) return resume_ck(li, c);
        }
        if (!P && own > match(L, ids)) return rewind(L);   // its own snapshot is the most there is
        const int64_t f = fork(li, ids);                   // a busy lane's snapshot, when longer than its own
        if (f >= 0) return f;
        if (P) return restore(L, *P);
        if (own > match(L, ids)) return rewind(L);
        return -1;
    }
    /// L's state now (the end of its prompt) kept for a fork: the conv histories and the last row's logits copied.
    void snapshot(Lane& L, Engine& last) {
        const HP& hp = big.m->hp;
        const size_t cb = (size_t) big.conv_c * 3 * 4;
        alloc_snapshot(L);
        last.sync_all();
        for (int il = 0; il < hp.n_vis; ++il) {
            CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
            CK(cudaMemcpy(L.snap_conv[(size_t) il], L.sess.conv[(size_t) il], cb, cudaMemcpyDeviceToDevice));
        }
        for (int il = 0; il < hp.n_vis; ++il)
            if (!L.sess.kv[(size_t) il].hk && L.snap_ring[(size_t) il]) {
                CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
                ring_copy(L, il, L.snap_ring[(size_t) il], true);
            }
        const Dev& hd = last.dev[(size_t) last.last_dev];
        CK(cudaSetDevice(hd.id));
        CK(cudaMemcpy(L.snap_logits, hd.logits, (size_t) hp.vocab_pad * 4, cudaMemcpyDeviceToDevice));
        L.snap_ids = L.live;
    }
    void alloc_snapshot(Lane& L) {
        const HP& hp = big.m->hp;
        const size_t cb = (size_t) big.conv_c * 3 * 4;
        if (L.snap_conv.empty()) {
            L.snap_conv.resize((size_t) hp.n_vis);
            for (int il = 0; il < hp.n_vis; ++il) {
                CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
                CK(cudaMalloc(&L.snap_conv[(size_t) il], cb));
            }
            CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->head_dev]));
            CK(cudaMalloc(&L.snap_logits, (size_t) hp.vocab_pad * 4));
            L.snap_ring.assign((size_t) hp.n_vis, nullptr);
            for (int il = 0; il < hp.n_vis; ++il)
                if (!L.sess.kv[(size_t) il].hk && L.sess.kv[(size_t) il].ring > 0)
                    CK(cudaHostAlloc((void**) &L.snap_ring[(size_t) il], ring_bytes(L, il), cudaHostAllocPortable));
        }
    }
    /// A busy lane whose last prompt starts `ids` (long enough to be worth it, and its sliding-window rings still holding
    /// that prompt's last cells): its state copied into free lane `li` instead of reading the prompt again. The cache
    /// cells come from the RAM copy, the rings and conv histories from VRAM. Returns the positions it brought, -1 for none.
    int fork_src(int li, const std::vector<int32_t>& ids) const {
        constexpr size_t kForkMin = 4096;
        const Lane& D = *lanes[(size_t) li];
        int si = -1;
        for (int i = 0; i < (int) lanes.size(); ++i) {
            const Lane& B = *lanes[(size_t) i];
            if (i == li || !B.busy || B.snap_ids.size() < kForkMin || B.snap_ids.size() > ids.size() || B.snap_ring.empty() ||
                !tiered(B) || B.snap_ids.size() <= reach(D, ids) || !std::equal(B.snap_ids.begin(), B.snap_ids.end(), ids.begin()))
                continue;
            if (si < 0 || B.snap_ids.size() > lanes[(size_t) si]->snap_ids.size()) si = i;
        }
        return si;
    }
    int64_t fork(int li, const std::vector<int32_t>& ids) {
        Lane& D = *lanes[(size_t) li];
        const int si = fork_src(li, ids);
        if (si < 0) return -1;
        Lane& B = *lanes[(size_t) si];
        const int64_t n = (int64_t) B.snap_ids.size();
        park(D);
        std::unique_lock<std::shared_mutex> lk(pool->mu);   // B between two steps from here on
        const auto t0 = Clock::now();
        const HP& hp = big.m->hp;
        const size_t HKV = (size_t) hp.n_kv, cb = (size_t) big.conv_c * 3 * 4;
        for (int il = 0; il < hp.n_vis; ++il) {
            Kv& sk = B.sess.kv[(size_t) il];
            Kv& dk = D.sess.kv[(size_t) il];
            CK(cudaSetDevice(big.m->dev_ids[(size_t) big.m->layers[(size_t) hp.phys(il)].dev]));
            if (sk.hk) {   // a tiered cache: the RAM copy's cells [0, n)
                for (size_t kh = 0; kh < HKV; ++kh) {
                    const size_t o = kh * (size_t) sk.hcells;
                    std::memcpy(dk.hk + o * kHD, sk.hk + o * kHD, (size_t) n * kHD);
                    std::memcpy(dk.hv + o * kHD, sk.hv + o * kHD, (size_t) n * kHD);
                    std::memcpy(dk.hks + o, sk.hks + o, (size_t) n * 4);
                    std::memcpy(dk.hvs + o, sk.hvs + o, (size_t) n * 4);
                }
            } else {   // a ring: as B's snapshot has it in RAM
                ring_copy(D, il, B.snap_ring[(size_t) il], false);
            }
            CK(cudaMemcpy(D.sess.conv[(size_t) il], B.snap_conv[(size_t) il], cb, cudaMemcpyDeviceToDevice));
        }
        D.sess.pos = n;
        if (D.sess.ds) D.sess.ds->forget(n);
        if (D.sess.win > 0) big.set_window(D.sess, D.sess.win, true);   // its VRAM window refilled from the copied cells
        Engine& ex = *D.ex;   // the logits of the prompt's last row, for a request with nothing more to read
        Dev& hd = ex.dev[(size_t) big.m->head_dev];
        CK(cudaSetDevice(hd.id));
        CK(cudaMemcpy(hd.logits, B.snap_logits, (size_t) hp.vocab_pad * 4, cudaMemcpyDeviceToDevice));
        ex.last_dev = big.m->head_dev;
        ex.n_logit_rows = 1;
        D.live = B.snap_ids;
        D.snap_ids.clear();
        D.dr.reindex();
        D.ck = B.ck;
        ck_trim(D.ck, n);
        std::fprintf(stderr, "strata serve: lane %d starts from lane %d's state at %lld tokens (copied in %.0f ms)\n", li, si,
                     (long long) n, ms_since(t0));
        return n;
    }
    /// The drafter's record over the reply, then cleared.
    static void ds_report(Lane& L, const char* who, int id) {
        DsState* st = L.sess.ds;
        if (!st || !st->steps) return;
        std::fprintf(stderr, "%s %d: DSpark: %lld of %lld drafted tokens stood (%.0f%%), %.2f tokens a step\n", who, id,
                     (long long) st->accepted, (long long) st->drafted, 100.0 * st->accepted / std::max<int64_t>(1, st->drafted),
                     1.0 + (double) st->accepted / (double) st->steps);
        st->accepted = st->drafted = st->steps = 0;
    }
    int pick_token(Engine* ex, const Sampling& smp, const std::vector<int32_t>& hist, std::mt19937_64& rng,
                   std::vector<float>& buf) {
        if (!smp.needs_logits()) return ex->greedy();
        ex->host_logits(buf);
        return sample(buf, smp, hist, rng);
    }
    std::vector<int32_t> history(const std::vector<int32_t>& ids) {
        static const bool pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
        std::vector<int32_t> text;   // (the pictures' rows are not tokens)
        text.reserve(ids.size());
        for (int32_t t : ids)
            if (!is_img_id(t)) text.push_back(t);
        return strata::qwen35::penalty_history_start(text, pen_prompt);
    }

    void solo(const std::string& line) {
        Request q;
        const bool pics = line.rfind("GENI ", 0) == 0;
        if (!parse(line.c_str() + (pics ? 5 : 4), q, pics)) return;
        const int li = pick(q.ids, -1);
        if (li < 0) { say("ERR every lane is decoding\n"); return; }
        Lane& L = *lanes[(size_t) li];
        const int64_t forked = start_from(li, q.ids);
        L.busy = true;
        stop.store(false);
        show_images(L, q.img.get());
        const Read r = read(L, q.ids, forked);
        show_images(L, nullptr);
        if (r.cancelled) {
            say("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", q.ids.size(), r.ms, r.reuse, (long long) r.got);
            L.busy = false;
            return;
        }
        std::mt19937_64 rng(q.smp.seed);
        std::vector<int32_t> hist = history(q.ids);
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        Engine* ex = r.last;
        int32_t nx = pick_token(ex, q.smp, hist, rng, lg);
        say("T %d\n", nx);
        ++produced;
        hist.push_back(nx);
        auto emit = [&](int y) {
            say("T %d\n", y);
            ++produced;
            if (is_eos(y)) { finish = "stop"; return false; }
            if (produced >= q.max_new) return false;
            if (stop.load()) { finish = "cancel"; return false; }
            return true;
        };
        if (is_eos(nx)) finish = "stop";
        else if (stop.load()) finish = "cancel";
        else if (produced < q.max_new)
            while (L.sess.pos + 1 <= big.ctx) {
                L.dr.max_k = lookup_max;
                pool->before(L.sess, L.sess.pos + kGemmT);
                pool->tick();
                std::shared_lock<std::shared_mutex> lk(pool->mu);
                if (!spec_step(L.ex.get(), &L.sess, L.live, L.dr, nx, (int) (q.max_new - produced),
                               (int) (big.ctx - L.sess.pos), q.smp, hist, rng, lg, emit))
                    break;
            }
        const double dms = ms_since(d0);
        info();
        say("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, q.ids.size(), r.ms, dms, finish, r.reuse,
            (long long) r.fresh);
        if (pool->on) std::fprintf(stderr, "strata serve: lane %d: pool: %s\n", li, pool->state(L.sess).c_str());
        ds_report(L, "strata serve: lane", li);
        std::fprintf(stderr, "strata serve: lane %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s) (%s)\n", li, q.ids.size(), r.reuse, (long long) r.fresh, r.ms,
                     r.ms > 0 ? 1000.0 * (double) r.fresh / r.ms : 0.0, produced, dms,
                     dms > 0 ? 1000.0 * (double) produced / dms : 0.0, finish);
        L.busy = false;
    }

    void admit(const std::string& line) {
        const bool pics = line.rfind("BGENI ", 0) == 0;
        char* p = const_cast<char*>(line.c_str()) + (pics ? 6 : 5);
        const long sl = std::strtol(p, &p, 10);
        if (sl < 0 || sl >= (long) lanes.size() || lanes.size() < 2 || lanes[(size_t) slot_lane[(size_t) sl]]->busy) {
            say("ERR BGEN: no such free slot (--batch %zu)\n", lanes.size() < 2 ? (size_t) 0 : lanes.size());
            return;
        }
        Request q;
        if (!parse(p, q, pics)) return;
        const int li = pick(q.ids, slot_lane[(size_t) sl]);
        if (li < 0) { say("ERR BGEN: every lane is decoding\n"); return; }
        for (size_t s2 = 0; s2 < slot_lane.size(); ++s2)   // the slot takes the lane; the slot that had it, this one's
            if (slot_lane[s2] == li) slot_lane[s2] = slot_lane[(size_t) sl];
        slot_lane[(size_t) sl] = li;
        Lane& L = *lanes[(size_t) li];
        if (L.th.joinable()) L.th.join();
        const int64_t forked = start_from(li, q.ids);
        L.busy = true;
        slot_stop[(size_t) sl] = false;
        stop.store(false);
        show_images(L, q.img.get());
        const Read r = read(L, q.ids, forked);
        show_images(L, nullptr);
        if (r.cancelled) {
            say("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", q.ids.size(), r.ms, r.reuse, (long long) r.got);
            say("BADM %ld 0\n", sl);
            L.busy = false;
            return;
        }
        auto rng = std::make_shared<std::mt19937_64>(q.smp.seed);
        auto hist = std::make_shared<std::vector<int32_t>>(history(q.ids));
        const int nx = pick_token(r.last, q.smp, *hist, *rng, lg);
        hist->push_back(nx);
        say("T %d\n", nx);
        const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : q.max_new <= 1 ? "length"
                          : L.sess.pos + 1 > big.ctx ? "length" : nullptr;
        say("DONE 1 %zu %.1f 0.0 %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", q.ids.size(), r.ms, fin ? fin : "length", r.reuse,
            (long long) r.fresh);
        std::fprintf(stderr, "strata batch: slot %ld (lane %d) takes a prompt of %zu tokens = %zu reused + %lld read in "
                     "%.0f ms\n", sl, li, q.ids.size(), r.reuse, (long long) r.fresh, r.ms);
        if (fin) {
            say("BADM %ld 0\n", sl);
            L.busy = false;
            return;
        }
        say("BADM %ld 1\n", sl);
        L.th = std::thread([this, &L, sl, nx, q, rng, hist] { decode(L, (int) sl, nx, q, *rng, *hist); });
    }

    /// A slot's tokens on its lane's own thread and executor, until its end.
    void decode(Lane& L, int sl, int32_t x, const Request& q, std::mt19937_64& rng, std::vector<int32_t>& hist) {
        const auto t0 = Clock::now();
        long long produced = 1;
        std::vector<float> buf;
        Engine* ex = L.ex.get();
        const char* fin = nullptr;
        auto emit = [&](int y) {
            say("BT %d %d\n", sl, y);
            ++produced;
            fin = is_eos(y) ? "stop" : slot_stop[(size_t) sl] ? "cancel" : produced >= q.max_new ? "length" : nullptr;
            return fin == nullptr;
        };
        while (!fin) {
            if (slot_stop[(size_t) sl]) { fin = "cancel"; break; }
            if (L.sess.pos + 1 > big.ctx) { fin = "length"; break; }
            L.dr.max_k = lookup_max;
            pool->before(L.sess, L.sess.pos + kGemmT);
            pool->tick();
            std::shared_lock<std::shared_mutex> lk(pool->mu);
            spec_step(ex, &L.sess, L.live, L.dr, x, (int) (q.max_new - produced), (int) (big.ctx - L.sess.pos), q.smp, hist,
                      rng, buf, emit);
        }
        info();
        say("BDONE %d %lld %s %.1f\n", sl, produced, fin, ms_since(t0));
        if (pool->on) std::fprintf(stderr, "strata batch: slot %d: pool: %s\n", sl, pool->state(L.sess).c_str());
        ds_report(L, "strata batch: slot", sl);
        std::fprintf(stderr, "strata batch: slot %d: %lld generated in %.0f ms (%.1f tok/s) (%s)\n", sl, produced, ms_since(t0),
                     produced / std::max(1e-9, ms_since(t0) / 1000.0), fin);
        L.used = ++clock;
        L.busy = false;
    }

    /// The Monitor's figures (the server reads INFO lines at start and between requests): the experts in VRAM now, the
    /// largest full-attention window a lane has in VRAM, the conversation cache's settings.
    void info() {
        long long res = 0, all = 0, bytes = 0;
        for (const Layer& L : big.m->layers) {
            res += L.n_res;
            all += (long long) L.on_gpu.size();
            bytes += (long long) L.n_res * (long long) L.stride;
        }
        long long win = 0;
        for (const auto& L : lanes) win = std::max<long long>(win, L->sess.win);
        say("INFO context=%d kv=%s kv_resident=%lld expert_slots=%lld experts=%lld expert_cache_mib=%lld lookup=%d "
            "conversation_cache_mib=%d conversation_cache_slots=%d conversation_cache_min_tokens=%d\n",
            big.ctx, big.q8_kv ? "int8" : "fp16", win, res, all, bytes >> 20, lookup_max > 0 ? 1 : 0, g_cc_mib, g_cc_slots,
            g_cc_min);
    }
    int run() {
        info();
        if (lanes.size() > 1) say("INFO batch_slots=%zu slot_cache=1\n", lanes.size());
        say("READY %d stop\n", big.ctx);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                if (l.rfind("BSTOP ", 0) == 0) {
                    const int b = std::atoi(l.c_str() + 6);
                    if (b >= 0 && b < (int) lanes.size()) slot_stop[(size_t) b] = true;
                    continue;
                }
                if (l.rfind("BYIELD", 0) == 0) continue;
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) break;
                line = lines.front();
                lines.pop_front();
            }
            if (line.empty()) continue;
            if (line.rfind("GEN ", 0) == 0 || line.rfind("GENI ", 0) == 0) solo(line);
            else if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) admit(line);
            else say("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...] <file> <id,id,...>\n");
        }
        for (int i = 0; i < (int) lanes.size(); ++i) slot_stop[(size_t) i] = true;
        for (auto& L : lanes)
            if (L->th.joinable()) L->th.join();
        return 0;
    }
};

}  // namespace

int strata::inkling2::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, dspark_path;
    float cli_temp = 0.f;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0;
    bool serve_mode = false, ppl = false, q8_kv = true;
    int n_lanes = 1, kv_res = 0, ppl_from = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-inkling2: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA_VISIBLE_DEVICES)
        else if (a == "--greedy") {}
        else if (a == "--serve") serve_mode = true;
        else if (a == "--vision") {}   // GENI is always taken (the pictures come from strata-vision)
        else if (a == "--ppl") ppl = true;
        else if (a == "--dspark") dspark_path = next();
        else if (a == "--temperature") cli_temp = (float) std::atof(next().c_str());
        else if (a == "--dspark-n") g_dspark_n = std::max(0, std::min(7, std::atoi(next().c_str())));
        else if (a == "--dspark-window") g_dspark_window = std::max(4096, std::atoi(next().c_str())) / 64 * 64;
        else if (a == "--lookup-max-t")   // the most tokens a step drafts from the conversation (verified; 0: none)
            g_lookup_max = std::max(0, std::min(kGemmT - 1, std::atoi(next().c_str())));
        else if (a == "--ppl-from") { ppl = true; ppl_from = std::atoi(next().c_str()); }   // read N as a prompt, score the rest
        else if (a == "--batch" || a == "--slots") n_lanes = std::max(1, std::min(kMaxLanes, std::atoi(next().c_str())));
        else if (a == "--kv") q8_kv = next() != "fp16";   // the full-attention caches: int8 (default) or fp16
        else if (a == "--kv-resident") {   // cells of each full-attention cache in VRAM; auto: the pool sizes them
            const std::string v = next();
            kv_res = v == "auto" ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = std::isdigit((unsigned char) v[0]) ? std::atoi(v.c_str()) : 0; }
        else if (a == "--tokens") {
            tokens = next();
            if (!tokens.empty() && tokens[0] == '@') {   // a file of ids
                std::FILE* f = std::fopen(tokens.c_str() + 1, "rb");
                if (!f) { std::fprintf(stderr, "strata-inkling2: cannot open %s\n", tokens.c_str() + 1); return 2; }
                tokens.clear();
                char buf[1 << 16];
                for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) tokens.append(buf, n);
                std::fclose(f);
            }
        }
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--bench-at") g_bench_at = std::atoi(next().c_str());
        else if (a == "--experts-vram") g_experts_vram = std::atoi(next().c_str());   // experts a layer keeps in VRAM
        else if (a == "--vram-reserve-mib") g_reserve_mib = std::atoi(next().c_str());
        else if (a == "--prefill-pipe") g_prefill_pipe = std::atoi(next().c_str());
        else if (a == "--conversation-cache-mib") g_cc_mib = std::max(0, std::atoi(next().c_str()));
        else if (a == "--conversation-cache-slots") g_cc_slots = std::max(0, std::atoi(next().c_str()));
        else if (a == "--conversation-cache-min-tokens") g_cc_min = std::max(0, std::atoi(next().c_str()));
        else if (serve_mode && a.rfind("--", 0) == 0) {
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-inkling2: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-inkling2 --model GGUF [--max-context 32768] [--chunk 512]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-inkling2: --model is required\n"); return 2; }
    const bool auto_chunk = chunk == 0;
    if (auto_chunk) chunk = 512;
    if (kv_res >= ctx || !q8_kv) kv_res = 0;
    if (kv_res < 0 && (int) window_cells(-1, ctx, chunk > 0 ? chunk : 2048, true) >= ctx) kv_res = 0;   // the pool: a context past one window
    if (kv_res > 0) kv_res = std::max(kv_res, std::max(chunk, 4096) + 1024);   // a whole prompt chunk and the bias range
    if (chunk < kGemmT || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-inkling2: bad --chunk / --max-context\n"); return 2; }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    Model m;
    std::string err;
    try {
        if (!dspark_path.empty()) {   // the drafter goes beside the head: the split leaves it room
            std::error_code ec;
            const auto fsz = std::filesystem::file_size(dspark_path, ec);
            if (!ec) g_head_extra = (uint64_t) fsz;
        }
        if (!load(model, m, ctx, chunk, q8_kv, n_lanes, kv_res, err)) { std::fprintf(stderr, "strata-inkling2: %s\n", err.c_str()); return 1; }
        for (const char* k : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"})
            if (const strata::MetaValue* v = m.file->meta().get(k))
                if (!is_eos((int) v->u)) g_eos.push_back((int32_t) v->u);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-inkling2: %s\n", e.what());
        return 1;
    }
    if (auto_chunk) {   // a chunk reads every expert once: the largest whose buffers fit beside the caches on every card
        for (int c : {2048, 1024}) {
            bool fits = true;
            for (int d = 0; d < m.n_dev; ++d) {
                CK(cudaSetDevice(m.dev_ids[(size_t) d]));
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                const uint64_t n_prompt = g_prefill_pipe != 0 && m.n_dev > 1 ? 2 : 1;   // the prompt executors' buffers
                const uint64_t need = cache_bytes(m, d, ctx, c, q8_kv, kv_res) * (uint64_t) n_lanes +
                                      n_prompt * (uint64_t) c * row_bytes_est(m.hp) + (640ull << 20);
                fits = fits && fr > need + (kv_res < 0 ? (256ull << 20) : (768ull << 20));   // the pool: spare VRAM is its anyway
            }
            if (fits) {
                chunk = c;
                break;
            }
        }
    }
    const int ring = m.hp.swa + std::max(chunk, kGemmT);
    if (kv_res > 0)
        std::fprintf(stderr, "strata-inkling2: full-attention caches tiered: the last %d cells in VRAM, all %d in RAM "
                     "(%.2f GiB of pinned memory per session)\n", kv_res, ctx,
                     (double) ctx * m.hp.n_kv * (kHD + 4) * 2 * 8 / (1 << 30));
    if (kv_res < 0)
        std::fprintf(stderr, "strata-inkling2: VRAM pool: each session's full-attention window grows as it needs, experts "
                     "give way to it (all %d cells in RAM, %.2f GiB of pinned memory per session)\n", ctx,
                     (double) ctx * m.hp.n_kv * (kHD + 4) * 2 * 8 / (1 << 30));
    Engine E;
    E.q8_kv = q8_kv;
    E.kv_resident = kv_res;
    E.init(m, ctx, chunk, ring);
    Pool pool;
    pool.init(m, E, kv_res < 0);
    if (g_reserve_mib >= 0) pool.reserve = (uint64_t) g_reserve_mib << 20;
    g_make_room = [&](int d, size_t bytes) {
        for (int k = 0; k < 4096 && pool.on; ++k) {
            CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            if (fr >= bytes + pool.reserve || !pool.evict_slab(d)) break;
        }
    };
    static DsModel dsm;   // the DSpark drafter (--dspark), on the target's head card
    const DsModel* dsp = nullptr;
    if (!dspark_path.empty()) {
        dsm.window = std::min(g_dspark_window, ctx);
        if (!load_dspark(dspark_path, m, dsm, err)) {
            std::fprintf(stderr, "strata-inkling2: %s\n", err.c_str());
            return 1;
        }
        dsp = &dsm;
        E.dsm = dsp;
        E.ds_alloc();
    }
    std::unique_ptr<Engine> E2;   // the second prompt executor: chunks on both cards at once
    Prompt pp;
    // after the lanes and the sessions: the second executor only where its buffers leave the cards room to spare
    auto second_executor = [&]() {
        if (g_prefill_pipe == 0 || m.n_dev < 2) return;
        // its buffers (measured about twice the row estimate: ~1.05 GiB a card at 1024 rows) and the pool's reserve after
        // them; the pool moves up to 15% of a card's experts to RAM for them where needed (routing is skewed: its adapt
        // rounds bring the used ones back)
        // (+ a split window's room: the other half's query rows and running result, its partials)
        const uint64_t split_b = m.n_dev == 2 && kv_res != 0 ? 2 * (uint64_t) chunk * m.hp.n_head * kHD * 4 + (32ull << 20) : 0;
        // (+ the drafter's features and context rows for its chunks)
        const uint64_t ds_b = dsp ? 3 * (uint64_t) chunk * m.hp.n_embd * 4 + (16ull << 20) : 0;
        const uint64_t want = 2 * (uint64_t) chunk * row_bytes_est(m.hp) + split_b + ds_b + pool.reserve;
        int moved = 0;
        for (int d = 0; d < m.n_dev; ++d) {
            CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            auto free_now = [&] {
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                return (uint64_t) fr;
            };
            int on_card = 0;
            for (const Layer& L : m.layers)
                if (L.dev == d) on_card += L.n_res;
            const int cap = on_card * 15 / 100;
            int here = 0;
            while (free_now() < want && pool.on && here < cap) {
                int before = 0;
                for (const Layer& L : m.layers)
                    if (L.dev == d) before += L.n_res;
                if (!pool.evict_slab(d)) break;
                int after = 0;
                for (const Layer& L : m.layers)
                    if (L.dev == d) after += L.n_res;
                here += before - after;
                CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            }
            moved += here;
            if (free_now() < want) {
                std::fprintf(stderr, "strata-inkling2: prompt chunks one executor at a time: GPU %d has %.2f GiB free, a second "
                             "executor needs %.2f GiB\n", d, (double) free_now() / (1 << 30), (double) want / (1 << 30));
                return;
            }
        }

        E2 = std::make_unique<Engine>();
        E2->q8_kv = q8_kv;
        E2->kv_resident = kv_res;
        E2->dsm = dsp;
        E2->init(m, ctx, chunk, ring);
        for (int d = 0; d < m.n_dev && pool.on; ++d) {   // what it took, measured: each card back to the pool's reserve
            CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            for (int k = 0; k < 256; ++k) {
                size_t fr = 0, tot = 0;
                CK(cudaMemGetInfo(&fr, &tot));
                int before = 0;
                for (const Layer& L : m.layers)
                    if (L.dev == d) before += L.n_res;
                if (fr >= pool.reserve || !pool.evict_slab(d)) break;
                for (const Layer& L : m.layers)
                    if (L.dev == d) before -= L.n_res;
                moved += before;
                CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            }
        }
        std::string fr_s;
        for (int d = 0; d < m.n_dev; ++d) {
            CK(cudaSetDevice(m.dev_ids[(size_t) d]));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            fr_s += (d ? ", GPU " : "GPU ") + std::to_string(d) + " " + std::to_string(fr >> 20) + " MiB";
        }
        std::fprintf(stderr, "strata-inkling2: %d experts to RAM for a second prompt executor; free VRAM now: %s\n", moved,
                     fr_s.c_str());
        std::fprintf(stderr, "strata-inkling2: prompt chunks of %d tokens on two executors at once (--prefill-pipe 0: one)\n", chunk);
    };
    if (serve_mode) {
        std::vector<std::unique_ptr<Lane>> lanes;
        for (int i = 0; i < n_lanes; ++i) {
            auto L = std::make_unique<Lane>();
            L->ex = std::make_unique<Engine>();
            L->ex->q8_kv = q8_kv;
            L->ex->kv_resident = kv_res;
            L->ex->decode_first = true;
            L->ex->dsm = dsp;
            L->ex->init(m, ctx, kGemmT, ring);
            E.alloc_session(L->sess);
            L->ex->S = &L->sess;
            L->ex->reset();
            lanes.push_back(std::move(L));
        }
        if (n_lanes > 1) std::fprintf(stderr, "strata-inkling2: %d lanes, each its own session and decode thread\n", n_lanes);
        Serve sv(E, std::move(lanes));
        for (auto& L : sv.lanes) sv.alloc_snapshot(*L);   // up front, so a prompt's end never needs VRAM
        sv.lookup_max = g_lookup_max;
        second_executor();
        pp.init(E, E2.get(), pool);
        sv.pool = &pool;
        sv.prompt = &pp;
        return sv.run();
    }
    Session s0;
    E.alloc_session(s0);
    E.S = &s0;
    E.reset();
    second_executor();
    pp.init(E, E2.get(), pool);
    if (bench_pp > 0 || bench_tg > 0) {
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 199000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            if (g_bench_at > 0) {   // decode speed at a length, nothing read (the pool sizes the window for it)
                E.S->pos = std::min(g_bench_at, ctx - bench_tg - 2);
                pool.before(*E.S, E.S->pos + bench_tg + (int64_t) toks.size(), bench_tg > 0);   // a prompt evicts no expert
                if (pool.on) std::fprintf(stderr, "strata-inkling2: pool: %s\n", pool.state(*E.S).c_str());
            }
            {
                Engine* last = &E;
                pp.read(*E.S, toks.data(), (int64_t) toks.size(), false, last, [](int64_t) { return true; });
                if (last) last->sync_all();
            }
            E.sync_all();
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) { return true; });
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-inkling2: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (ppl) {   // the mean negative log-likelihood of every token after the first, 8 rows a forward
        double nll = 0;
        int64_t cnt = 0;
        std::vector<float> row;
        size_t start = 0;
        if (ppl_from > 0 && (size_t) ppl_from < prompt.size()) {
            E.process(prompt.data(), ppl_from, [](int64_t) { return true; });
            start = (size_t) ppl_from;
        }
        for (size_t done = start; done + 1 < prompt.size();) {
            const int T = (int) std::min<size_t>(kGemmT, prompt.size() - done);
            E.forward(prompt.data() + done, T, T);
            for (int r = 0; r < T && done + (size_t) r + 1 < prompt.size(); ++r) {
                E.host_logits(row, r);
                const double mx = *std::max_element(row.begin(), row.end());
                double z = 0;
                for (float v : row) z += std::exp((double) v - mx);
                nll += mx + std::log(z) - row[(size_t) prompt[done + (size_t) r + 1]];
                ++cnt;
            }
            done += (size_t) T;
        }
        std::printf("ppl: %lld tokens, mean nll %.5f, ppl %.4f\n", (long long) cnt, nll / (double) cnt, std::exp(nll / (double) cnt));
        return 0;
    }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    const auto t0 = Clock::now();
    {
        Engine* last = &E;
        const int64_t got = pp.read(*E.S, prompt.data(), (int64_t) prompt.size(), false, last, [](int64_t) { return true; });
        (void) got;
        if (last != &E) {   // the logits are on the executor that read the last chunk
            last->sync_all();
            E.last_dev = last->last_dev;
            E.n_logit_rows = last->n_logit_rows;
            for (int d = 0; d < m.n_dev; ++d) {
                CK(cudaSetDevice(E.dev[(size_t) d].id));
                if (E.dev[(size_t) d].logits && last->dev[(size_t) d].logits)
                    CK(cudaMemcpy(E.dev[(size_t) d].logits, last->dev[(size_t) d].logits,
                                  (size_t) m.hp.vocab_pad * 4, cudaMemcpyDeviceToDevice));
            }
        }
    }
    E.sync_all();
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (forced.empty() && !ff && n_gen > 1) {   // plain generation: through the verify step, as the server runs it
        std::vector<int32_t> live(prompt.begin(), prompt.end() - 1), hist;
        Drafter dr;
        dr.max_k = g_lookup_max;
        Sampling greedy_smp;
        if (cli_temp > 0.f) {   // --temperature: sampled generation (top-k 0 and top-p 1, as the model card)
            greedy_smp.temperature = cli_temp;
            greedy_smp.top_k = 0;
            greedy_smp.top_p = 1.f;
        }
        std::mt19937_64 rng(0);
        int32_t x = E.greedy();
        gen.push_back(x);
        live.push_back(prompt.back());
        int steps = 0;
        while ((int) gen.size() < n_gen && E.S->pos + 1 <= ctx) {
            ++steps;
            pool.before(*E.S, E.S->pos + kGemmT);
            pool.tick();
            spec_step(&E, E.S, live, dr, x, n_gen - (int) gen.size(), (int) (ctx - E.S->pos), greedy_smp, hist, rng, lg,
                      [&](int y) { gen.push_back(y); return (int) gen.size() < n_gen; });
        }
        std::fprintf(stderr, "strata-inkling2: %zu tokens in %d forwards (%.2f a forward), drafted-token acceptance %.0f%%, "
                     "step ms by rows:", gen.size() - 1, steps, (double) (gen.size() - 1) / std::max(1, steps),
                     100.0 * dr.acc / (dr.acc + dr.rej));
        for (int r = 1; r <= kGemmT; ++r) std::fprintf(stderr, " %d:%.1f", r, dr.t_ms[r]);
        std::fprintf(stderr, "\n");
        n_gen = 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) { return true; });
    }
    if (ff) std::fclose(ff);
    if (E.S && E.S->ds && E.S->ds->steps)
        std::fprintf(stderr, "strata-inkling2: DSpark: %lld of %lld drafted tokens stood (%.0f%%) over %lld steps, %.2f tokens a step\n",
                     (long long) E.S->ds->accepted, (long long) E.S->ds->drafted, 100.0 * E.S->ds->accepted / std::max<int64_t>(1, E.S->ds->drafted),
                     (long long) E.S->ds->steps, 1.0 + (double) E.S->ds->accepted / (double) E.S->ds->steps);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-inkling2: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    if (kv_res > 0) std::fprintf(stderr, "strata-inkling2: CPU share of the RAM part %.0f%%\n", 100.0 * E.cpu_frac);
    if (pool.on) {
        int res = 0, all = 0;
        for (const Layer& L : m.layers) {
            res += L.n_res;
            all += (int) L.on_gpu.size();
        }
        std::fprintf(stderr, "strata-inkling2: pool: a window of %d cells, %d of %d experts in VRAM; a layer's most-used half "
                     "of the experts takes %.0f%% of its routings; %.1f%% of the routings hit VRAM\n", E.S->win, res, all,
                     100.0 * pool.top_half_share(), 100.0 * pool.hit_share());
    }
    return 0;
}
