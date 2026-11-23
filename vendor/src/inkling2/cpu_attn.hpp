// src/inkling2/cpu_attn.hpp - attention over the host copy of a tiered cache on CPU threads, as split-K partials
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace strata::inkling2 {

/// Keys [kmin, kmax] of one full-attention visit's host copy for nq query rows (every key is before every row, and
/// beyond the bias range, so the score is q.k * scale alone). The output is one partial per (row, head, segment) in
/// the split-K format of the GPU kernels: acc unnormalized against m, l the sum of exponentials.
struct CpuAttnJob {
    const float* q = nullptr;   // [nq][H][128]
    int nq = 0, H = 0, G = 0, HKV = 0;
    const int8_t *k = nullptr, *v = nullptr;   // [HKV][hcells][128]
    const float *ks = nullptr, *vs = nullptr;  // [HKV][hcells]
    int hcells = 0, kmin = 0, kmax = -1;
    float scale = 1.f;
    int nseg = 1;                // the key range in this many segments
    int seg_stride = 1;          // slots a (row, head) has in the output; segment s is slot s
    float *acc = nullptr, *m = nullptr, *l = nullptr;   // [nq * H][seg_stride][128], [nq * H][seg_stride]
};

/// Worker threads shared by every engine of the process; run() takes one job at a time.
class CpuPool {
public:
    static CpuPool& get();
    int threads() const { return n_; }
    void run(int n_tasks, const std::function<void(int)>& fn);

private:
    explicit CpuPool(int n);
    struct Impl;
    Impl* impl_;
    int n_;
};

void cpu_attn(const CpuAttnJob& job);

/// A layer's experts as the CPU reads them: every expert's blob (the GPU's layout: gate rows, up rows at up_off, down
/// rows at down_off, NVFP4's {s_gate, s_up, s_down} at tail_off) in pinned host memory, `stride` bytes apart.
struct CpuExperts {
    const uint8_t* blobs = nullptr;
    size_t stride = 0, gu_row = 0, d_row = 0, up_off = 0, down_off = 0, tail_off = 0;
    int n_embd = 0, n_ff = 0, gu_type = 0, d_type = 0;   // GGUF types: 40 NVFP4, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 6 Q5_0
    float situ_beta = 0.f, situ_lb = 0.f;
};

/// The routed entries whose expert is not in VRAM (on_gpu[expert] == 0), computed on CPU threads as the GPU's grouped
/// kernels compute them: gate/up from the GPU's q8_1 activations, SITU, the product re-quantized to q8_1, down. Writes
/// out[entry] (n_embd floats, before the routing weight) and mask[entry] = 1 for each one it computes, 0 for the others.
struct CpuMoeJob {
    const CpuExperts* L = nullptr;
    const uint8_t* on_gpu = nullptr;   // [n_expert]
    const int32_t* ids = nullptr;      // [n_ent]: entry e is token e / K's expert
    int n_ent = 0, K = 1;
    const void* xq = nullptr;          // [tokens][n_embd / 32] q8_1 blocks (36 bytes: fp16 d, fp16 sum, 32 int8)
    float* out = nullptr;              // [n_ent][n_embd]
    uint8_t* mask = nullptr;           // [n_ent]
    int n_cpu = 0;                     // set by cpu_moe: the entries it computed
};
bool cpu_experts_supported(int gu_type, int d_type);
void cpu_moe(CpuMoeJob& job);

}  // namespace strata::inkling2
