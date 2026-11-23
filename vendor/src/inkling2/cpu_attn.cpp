// src/inkling2/cpu_attn.cpp - attention over the host copy of a tiered cache on CPU threads, as split-K partials
#include "cpu_attn.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define STRATA_INK_X86 1
#endif

namespace strata::inkling2 {

namespace {
constexpr int kHD = 128, kKB = 32;

struct Row {   // one query row's running result over a segment
    const float* q;
    float m, l;
    float* acc;
    int t, h;
};

void finish(const CpuAttnJob& j, std::vector<Row>& rows, int seg) {
    for (size_t r = 0; r < rows.size(); ++r) {
        const size_t o = ((size_t) rows[r].t * j.H + rows[r].h) * j.seg_stride + seg;
        std::memcpy(j.acc + o * kHD, rows[r].acc, kHD * 4);
        j.m[o] = rows[r].m;
        j.l[o] = rows[r].l;
    }
}

void segment_scalar(const CpuAttnJob& j, std::vector<Row>& rows, int kvh, int a, int b) {
    float kf[kHD], vf[kHD];
    for (int kp = a; kp <= b; ++kp) {
        const size_t ro = (size_t) kvh * j.hcells + kp;
        const float ksc = j.ks[ro] * j.scale, vsc = j.vs[ro];
        for (int i = 0; i < kHD; ++i) {
            kf[i] = (float) j.k[ro * kHD + i] * ksc;
            vf[i] = (float) j.v[ro * kHD + i] * vsc;
        }
        for (Row& R : rows) {
            float s = 0.f;
            for (int i = 0; i < kHD; ++i) s += R.q[i] * kf[i];
            const float mn = std::max(R.m, s), c = std::exp(R.m - mn), e = std::exp(s - mn);
            for (int i = 0; i < kHD; ++i) R.acc[i] = R.acc[i] * c + e * vf[i];
            R.l = R.l * c + e;
            R.m = mn;
        }
    }
}

#ifdef STRATA_INK_X86
__attribute__((target("avx2,fma"))) inline __m256 exp8(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

__attribute__((target("avx2,fma"))) inline float hsum(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

__attribute__((target("avx2,fma"))) inline void to_f32(const int8_t* src, float sc, float* dst) {
    const __m256 s = _mm256_set1_ps(sc);
    for (int i = 0; i < kHD; i += 8) {
        const __m256i w = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i)));
        _mm256_store_ps(dst + i, _mm256_mul_ps(_mm256_cvtepi32_ps(w), s));
    }
}

__attribute__((target("avx2,fma"))) void segment_avx2(const CpuAttnJob& j, std::vector<Row>& rows, int kvh, int a, int b) {
    alignas(32) float kf[kKB][kHD], vf[kKB][kHD], s[kKB];
    for (int k0 = a; k0 <= b; k0 += kKB) {
        const int n = std::min(kKB, b - k0 + 1);
        for (int kk = 0; kk < n; ++kk) {
            const size_t ro = (size_t) kvh * j.hcells + k0 + kk;
            to_f32(j.k + ro * kHD, j.ks[ro] * j.scale, kf[kk]);
            to_f32(j.v + ro * kHD, j.vs[ro], vf[kk]);
        }
        for (int kk = n; kk < kKB; ++kk) s[kk] = -INFINITY;
        for (Row& R : rows) {
            __m256 q[kHD / 8];
            for (int i = 0; i < kHD / 8; ++i) q[i] = _mm256_loadu_ps(R.q + 8 * i);
            float bm = -INFINITY;
            for (int kk = 0; kk < n; ++kk) {
                __m256 d0 = _mm256_mul_ps(q[0], _mm256_load_ps(kf[kk])), d1 = _mm256_mul_ps(q[1], _mm256_load_ps(kf[kk] + 8));
                for (int i = 2; i < kHD / 8; i += 2) {
                    d0 = _mm256_fmadd_ps(q[i], _mm256_load_ps(kf[kk] + 8 * i), d0);
                    d1 = _mm256_fmadd_ps(q[i + 1], _mm256_load_ps(kf[kk] + 8 * i + 8), d1);
                }
                s[kk] = hsum(_mm256_add_ps(d0, d1));
                bm = std::max(bm, s[kk]);
            }
            const float mn = std::max(R.m, bm);
            const float c = std::exp(R.m - mn);
            float esum = 0.f;
            const __m256 mv = _mm256_set1_ps(mn);
            for (int kk = 0; kk < kKB; kk += 8) _mm256_storeu_ps(s + kk, exp8(_mm256_sub_ps(_mm256_loadu_ps(s + kk), mv)));
            for (int kk = 0; kk < n; ++kk) esum += s[kk];
            __m256 acc[kHD / 8];
            const __m256 cv = _mm256_set1_ps(c);
            for (int i = 0; i < kHD / 8; ++i) acc[i] = _mm256_mul_ps(_mm256_loadu_ps(R.acc + 8 * i), cv);
            for (int kk = 0; kk < n; ++kk) {
                const __m256 e = _mm256_set1_ps(s[kk]);
                for (int i = 0; i < kHD / 8; ++i) acc[i] = _mm256_fmadd_ps(e, _mm256_load_ps(vf[kk] + 8 * i), acc[i]);
            }
            for (int i = 0; i < kHD / 8; ++i) _mm256_storeu_ps(R.acc + 8 * i, acc[i]);
            R.l = R.l * c + esum;
            R.m = mn;
        }
    }
}

bool have_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return ok;
}
#endif
}  // namespace

struct CpuPool::Impl {
    std::vector<std::thread> th;
    std::mutex mu, run_mu;
    std::condition_variable cv, done_cv;
    const std::function<void(int)>* fn = nullptr;
    int n_tasks = 0;
    std::atomic<int> next{0}, done{0};
    uint64_t gen = 0;

    void work() {
        for (int i; (i = next.fetch_add(1)) < n_tasks;) {
            (*fn)(i);
            if (done.fetch_add(1) + 1 == n_tasks) {
                std::lock_guard<std::mutex> lk(mu);
                done_cv.notify_all();
            }
        }
    }
};

CpuPool::CpuPool(int n) : impl_(new Impl), n_(n) {
    for (int t = 1; t < n; ++t)
        impl_->th.emplace_back([this] {
            uint64_t seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(impl_->mu);
                    impl_->cv.wait(lk, [&] { return impl_->gen != seen; });
                    seen = impl_->gen;
                }
                impl_->work();
            }
        });
}

CpuPool& CpuPool::get() {
    // Physical cores, the reader included (the host copy is read at memory speed, so hyperthreads add nothing).
    static CpuPool pool(std::max(1, std::min(8, (int) std::thread::hardware_concurrency() / 2)));
    return pool;
}

void CpuPool::run(int n_tasks, const std::function<void(int)>& fn) {
    std::lock_guard<std::mutex> one(impl_->run_mu);
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->fn = &fn;
        impl_->n_tasks = n_tasks;
        impl_->next = 0;
        impl_->done = 0;
        ++impl_->gen;
    }
    impl_->cv.notify_all();
    impl_->work();
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->done_cv.wait(lk, [&] { return impl_->done.load() >= impl_->n_tasks; });
}

namespace {
void range(const CpuAttnJob& j, std::vector<Row>& rows, int kvh, int a, int b) {
    if (a > b) return;
#ifdef STRATA_INK_X86
    if (have_avx2()) return segment_avx2(j, rows, kvh, a, b);
#endif
    segment_scalar(j, rows, kvh, a, b);
}
}  // namespace

void cpu_attn(const CpuAttnJob& j) {
    const int keys = j.kmax - j.kmin + 1;
    if (keys <= 0) return;
    const int R = j.nq * j.G;
    CpuPool::get().run(j.HKV * j.nseg, [&](int task) {
        const int kvh = task % j.HKV, seg = task / j.HKV;
        const int a = j.kmin + (int) ((int64_t) keys * seg / j.nseg), b = j.kmin + (int) ((int64_t) keys * (seg + 1) / j.nseg) - 1;
        thread_local std::vector<float> accs;
        accs.assign((size_t) R * kHD, 0.f);
        std::vector<Row> rows((size_t) R);
        for (int r = 0; r < R; ++r) {
            const int t = r / j.G, h = kvh * j.G + r % j.G;
            rows[(size_t) r] = Row{j.q + ((size_t) t * j.H + h) * kHD, -INFINITY, 0.f, accs.data() + (size_t) r * kHD, t, h};
        }
        range(j, rows, kvh, a, b);
        finish(j, rows, seg);
    });
}

}  // namespace strata::inkling2

// ---- experts on CPU threads --------------------------------------------------------------------------------------------

namespace strata::inkling2 {
namespace {
constexpr int kQ81 = 36;   // a q8_1 block: fp16 d, fp16 sum, 32 int8
constexpr int kRowChunk = 64;

float h2f(const uint8_t* p) {
    const uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do { ++ee; mm <<= 1; } while (!(mm & 0x400));
            f = s | (uint32_t) (127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) f = s | 0x7f800000u | m << 13;
    else f = s | (e + 112) << 23 | m << 13;
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}
uint16_t f2h(float v) {   // round to nearest even
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (uint16_t) (s | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000;
        const int shift = 14 - e;
        uint32_t hm = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) ++hm;
        return (uint16_t) (s | hm);
    }
    uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1fff;
    uint32_t h = s | (uint32_t) e << 10 | hm;
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) ++h;
    return (uint16_t) h;
}
struct Ue4m3 {   // NVFP4's sub-block scales, halved as the GPU reads them
    float v[256];
    Ue4m3() {
        for (int x = 0; x < 256; ++x) {
            const int e = (x >> 3) & 0xF, m = x & 7;
            float raw = e == 0 ? (float) m * std::ldexp(1.0f, -9) : (1.0f + (float) m / 8.0f) * std::ldexp(1.0f, e - 7);
            if ((x & 0x7F) == 0 || (x & 0x7F) == 0x7F) raw = 0.f;
            v[x] = raw * 0.5f;
        }
    }
};
const Ue4m3 kUe;
const int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/// One weight row against up to kMaxE activation rows at once (the row's blocks decoded once): x[e] the int8 codes of
/// q8_1 blocks (kQ81 bytes apart), dx[e] their scales as floats.
constexpr int kMaxE = 8;
/// A Q4_K / Q5_K super-block's sub-block j scale and min (ggml's get_scale_min_k4).
inline void kq_scale_min(const uint8_t* q, int j, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
void dots_scalar(int type, const uint8_t* w, int n_in, const uint8_t* const* x, const float* const* dx, int ne, float* out) {
    for (int e = 0; e < ne; ++e) {
        float acc = 0.f;
        const uint8_t* wr = w;
        if (type == 12 || type == 13) {   // Q4_K / Q5_K: d sc q - dmin m a sub-block of 32 (the min against the codes' sum, as ggml)
            const int bs = type == 12 ? 144 : 176;
            for (int b = 0; b < n_in / 256; ++b, wr += bs) {
                const float d = h2f(wr), dmin = h2f(wr + 2);
                const uint8_t* qs = wr + (type == 12 ? 16 : 48);
                const uint8_t* qh = wr + 16;
                for (int j = 0; j < 8; ++j) {
                    int sc, m;
                    kq_scale_min(wr + 4, j, sc, m);
                    const uint8_t* xb = x[e] + (size_t) (8 * b + j) * kQ81;
                    const int8_t* xv = (const int8_t*) (xb + 4);
                    int sm = 0, sx = 0;
                    for (int l = 0; l < 32; ++l) {
                        int q = (j & 1) ? qs[32 * (j / 2) + l] >> 4 : qs[32 * (j / 2) + l] & 0xF;
                        if (type == 13) q += ((qh[l] >> j) & 1) << 4;
                        sm += q * xv[l];
                        sx += xv[l];
                    }
                    acc += dx[e][8 * b + j] * (d * (float) sc * (float) sm - dmin * (float) m * (float) sx);
                }
            }
        } else if (type == 14) {   // Q6_K: d sc (q - 32), a scale every 16
            for (int b = 0; b < n_in / 256; ++b, wr += 210) {
                const float d = h2f(wr + 208);
                const int8_t* scl = (const int8_t*) (wr + 192);
                for (int hf = 0; hf < 2; ++hf)
                    for (int grp = 0; grp < 4; ++grp) {   // 32 values: one q8_1 block
                        const uint8_t* ql = wr + 64 * hf + (grp & 1) * 32;
                        const uint8_t* qh = wr + 128 + 32 * hf;
                        const int blk = 8 * b + 4 * hf + grp;
                        const int8_t* xv = (const int8_t*) (x[e] + (size_t) blk * kQ81 + 4);
                        for (int h16 = 0; h16 < 2; ++h16) {
                            int sm = 0;
                            for (int l = 16 * h16; l < 16 * h16 + 16; ++l) {
                                const int q = (int) (((ql[l] >> ((grp >> 1) * 4)) & 0xF) | (((qh[l] >> (2 * grp)) & 3) << 4)) - 32;
                                sm += q * xv[l];
                            }
                            acc += d * (float) scl[8 * hf + h16 + 2 * grp] * dx[e][blk] * (float) sm;
                        }
                    }
            }
        } else if (type == 6) {   // Q5_0: d (q - 16)
            for (int b = 0; b < n_in / 32; ++b, wr += 22) {
                const float d = h2f(wr);
                uint32_t qh;
                std::memcpy(&qh, wr + 2, 4);
                const int8_t* xv = (const int8_t*) (x[e] + (size_t) b * kQ81 + 4);
                int sm = 0;
                for (int j = 0; j < 16; ++j) {
                    const int q0 = ((wr[6 + j] & 0xF) | (((qh >> j) & 1) << 4)) - 16;
                    const int q1 = ((wr[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16;
                    sm += q0 * xv[j] + q1 * xv[j + 16];
                }
                acc += d * dx[e][b] * (float) sm;
            }
        } else if (type == 40) {
            for (int b = 0; b < n_in / 64; ++b, wr += 36)
                for (int sb = 0; sb < 4; ++sb) {
                    const int8_t* xv = (const int8_t*) (x[e] + (size_t) (2 * b + sb / 2) * kQ81 + 4) + (sb % 2) * 16;
                    int sm = 0;
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t q = wr[4 + sb * 8 + j];
                        sm += kFp4[q & 0xF] * xv[j] + kFp4[q >> 4] * xv[8 + j];
                    }
                    acc += (float) sm * kUe.v[wr[sb]] * dx[e][2 * b + sb / 2];
                }
        } else {
            for (int b = 0; b < n_in / 32; ++b, wr += 34) {
                const int8_t* wv = (const int8_t*) (wr + 2);
                const int8_t* xv = (const int8_t*) (x[e] + (size_t) b * kQ81 + 4);
                int sm = 0;
                for (int j = 0; j < 32; ++j) sm += wv[j] * xv[j];
                acc += (float) sm * h2f(wr) * dx[e][b];
            }
        }
        out[e] = acc;
    }
}

#ifdef STRATA_INK_X86
__attribute__((target("avx2,fma,f16c"))) void dots_avx2(int type, const uint8_t* w, int n_in, const uint8_t* const* x,
                                                       const float* const* dx, int ne, float* out) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[kMaxE];
    for (int e = 0; e < ne; ++e) acc[e] = _mm256_setzero_ps();
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    if (type == 12 || type == 13) {   // Q4_K / Q5_K: u8 codes (maddubs) per 32, d sc dx, the min on the codes' sum (as ggml)
        const int bs = type == 12 ? 144 : 176;
        const __m256i m1 = _mm256_set1_epi8(1);
        for (int b = 0; b < n_in / 256; ++b, w += bs) {
            uint16_t hd, hm;
            std::memcpy(&hd, w, 2);
            std::memcpy(&hm, w + 2, 2);
            const float d = _cvtsh_ss(hd), dmin = _cvtsh_ss(hm);
            const uint8_t* qs = w + (type == 12 ? 16 : 48);
            const __m256i qhv = type == 13 ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16)) : _mm256_setzero_si256();
            int scs[8], mns[8];
            for (int j = 0; j < 8; ++j) kq_scale_min(w + 4, j, scs[j], mns[j]);
            for (int g = 0; g < 4; ++g) {
                const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32 * g));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                if (type == 13) {
                    lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2 * g), m1), 4));
                    hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2 * g + 1), m1), 4));
                }
                for (int e = 0; e < ne; ++e) {
                    const uint8_t* x0 = x[e] + (size_t) (8 * b + 2 * g) * kQ81;
                    const __m256i xa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + 4));
                    const __m256i xb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + kQ81 + 4));
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xa), ones);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xb), ones);
                    const float dx0 = dx[e][8 * b + 2 * g], dx1 = dx[e][8 * b + 2 * g + 1];
                    // the mins against the codes' sums (madd of 1 x codes), in the same lanes
                    const __m256i s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(m1, xa), ones);
                    const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(m1, xb), ones);
                    acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(d * (float) scs[2 * g] * dx0), acc[e]);
                    acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(d * (float) scs[2 * g + 1] * dx1), acc[e]);
                    acc[e] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_set1_ps(dmin * (float) mns[2 * g] * dx0), acc[e]);
                    acc[e] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_set1_ps(dmin * (float) mns[2 * g + 1] * dx1), acc[e]);
                }
            }
        }
    } else if (type == 14) {   // Q6_K: u8 codes 0..63 (maddubs), less 32 x the codes' sum, a scale every 16 (4 lanes)
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m256i u1 = _mm256_set1_epi8(1);
        for (int b = 0; b < n_in / 256; ++b, w += 210) {
            uint16_t hd;
            std::memcpy(&hd, w + 208, 2);
            const float d = _cvtsh_ss(hd);
            const int8_t* scl = (const int8_t*) (w + 192);
            for (int hf = 0; hf < 2; ++hf) {
                const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 128 + 32 * hf));
                const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * hf));
                const __m256i qb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * hf + 32));
                for (int grp = 0; grp < 4; ++grp) {
                    const __m256i ql = grp & 1 ? qb : qa;
                    const __m256i lo4 = grp >> 1 ? _mm256_and_si256(_mm256_srli_epi16(ql, 4), m4) : _mm256_and_si256(ql, m4);
                    const __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * grp), m3), 4);
                    const __m256i q = _mm256_or_si256(lo4, h2);   // 0..63
                    const int blk = 8 * b + 4 * hf + grp;
                    const __m256 sv = _mm256_setr_ps(scl[8 * hf + 2 * grp], scl[8 * hf + 2 * grp], scl[8 * hf + 2 * grp], scl[8 * hf + 2 * grp],
                                                     scl[8 * hf + 2 * grp + 1], scl[8 * hf + 2 * grp + 1], scl[8 * hf + 2 * grp + 1],
                                                     scl[8 * hf + 2 * grp + 1]);
                    for (int e = 0; e < ne; ++e) {
                        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[e] + (size_t) blk * kQ81 + 4));
                        const __m256i pq = _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones);
                        const __m256i px = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, xv), ones);   // the codes' sums
                        const __m256i p = _mm256_sub_epi32(pq, _mm256_slli_epi32(px, 5));
                        acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_mul_ps(sv, _mm256_set1_ps(d * dx[e][blk])), acc[e]);
                    }
                }
            }
        }
    } else if (type == 6) {   // Q5_0: u8 codes 0..31 (maddubs), less 16 x the codes' sum
        const __m256i u1 = _mm256_set1_epi8(1);
        const __m256i bitsel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
                                                1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
        const __m256i bshuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        for (int b = 0; b < n_in / 32; ++b, w += 22) {
            uint16_t hd;
            std::memcpy(&hd, w, 2);
            const float d = _cvtsh_ss(hd);
            uint32_t qhw;
            std::memcpy(&qhw, w + 2, 4);
            const __m128i q16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 6));
            const __m256i qq = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q16, 4), _mm_set1_epi8(0x0F)), _mm_and_si128(q16, _mm_set1_epi8(0x0F)));
            // the fifth bits: byte i of the 32 gets bit i of qh
            const __m256i hb = _mm256_shuffle_epi8(_mm256_set1_epi32((int) qhw), bshuf);
            const __m256i hbit = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hb, bitsel), bitsel), _mm256_set1_epi8(16));
            const __m256i q = _mm256_or_si256(qq, hbit);   // 0..31
            for (int e = 0; e < ne; ++e) {
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[e] + (size_t) b * kQ81 + 4));
                const __m256i pq = _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), ones);
                const __m256i px = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, xv), ones);
                const __m256i p = _mm256_sub_epi32(pq, _mm256_slli_epi32(px, 4));
                acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[e][b]), acc[e]);
            }
        }
    } else if (type == 40) {
        const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12,
                                             0, -1, -2, -3, -4, -6, -8, -12);
        for (int b = 0; b < n_in / 64; ++b, w += 36) {
            const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 4));
            const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(q, m4));
            const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
            const __m256i A = _mm256_unpacklo_epi64(lo, hi), B = _mm256_unpackhi_epi64(lo, hi);
            const __m256i v0 = _mm256_permute2x128_si256(A, B, 0x20), v1 = _mm256_permute2x128_si256(A, B, 0x31);
            const __m256i a0 = _mm256_sign_epi8(v0, v0), a1 = _mm256_sign_epi8(v1, v1);
            const __m256 s0 = _mm256_set_m128(_mm_set1_ps(kUe.v[w[1]]), _mm_set1_ps(kUe.v[w[0]]));
            const __m256 s1 = _mm256_set_m128(_mm_set1_ps(kUe.v[w[3]]), _mm_set1_ps(kUe.v[w[2]]));
            for (int e = 0; e < ne; ++e) {
                const uint8_t* x0 = x[e] + (size_t) (2 * b) * kQ81;
                const __m256i xa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + 4));
                const __m256i xb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x0 + kQ81 + 4));
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(xa, v0)), ones);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(xb, v1)), ones);
                acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_mul_ps(s0, _mm256_set1_ps(dx[e][2 * b])), acc[e]);
                acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_mul_ps(s1, _mm256_set1_ps(dx[e][2 * b + 1])), acc[e]);
            }
        }
    } else {
        for (int b = 0; b < n_in / 32; ++b, w += 34) {
            const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2));
            const __m256i aw = _mm256_sign_epi8(wv, wv);
            uint16_t hw;
            std::memcpy(&hw, w, 2);
            const float dw = _cvtsh_ss(hw);
            for (int e = 0; e < ne; ++e) {
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[e] + (size_t) b * kQ81 + 4));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, wv)), ones);
                acc[e] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(dw * dx[e][b]), acc[e]);
            }
        }
    }
    for (int e = 0; e < ne; ++e) out[e] = hsum(acc[e]);
}
bool have_f16c() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return ok;
}
#endif

void row_dots(int type, const uint8_t* w, int n_in, const uint8_t* const* x, const float* const* dx, int ne, float* out) {
#ifdef STRATA_INK_X86
    if (have_f16c()) return dots_avx2(type, w, n_in, x, dx, ne, out);
#endif
    dots_scalar(type, w, n_in, x, dx, ne, out);
}

/// The GPU's quantize_q8_1 on one row: a block a 32 values, d = amax / 127 (kept finite), codes rounded.
void quantize_q8_1(const float* x, uint8_t* y, int n) {
    constexpr float kHalfMax = 65504.0f;
    for (int b = 0; b < n / 32; ++b, x += 32, y += kQ81) {
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(x[i]));
            sum += x[i];
        }
        float d = amax / 127.0f;
        d = std::fabs(d) > kHalfMax ? std::copysign(kHalfMax, d) : d;
        for (int i = 0; i < 32; ++i) {
            float q = amax == 0.f ? 0.f : std::round(x[i] / d);
            q = q > 127.f ? 127.f : q < -127.f ? -127.f : q;
            ((int8_t*) (y + 4))[i] = (int8_t) q;
        }
        const float sf = std::fabs(sum) > kHalfMax ? std::copysign(kHalfMax, sum) : sum;
        const uint16_t hd = f2h(d), hs = f2h(sf);
        std::memcpy(y, &hd, 2);
        std::memcpy(y + 2, &hs, 2);
    }
}
}  // namespace

bool cpu_experts_supported(int gu_type, int d_type) {
    auto ok = [](int t) { return t == 40 || t == 8 || t == 12 || t == 13 || t == 14 || t == 6; };
    return ok(gu_type) && ok(d_type);
}

void cpu_moe(CpuMoeJob& j) {
    const CpuExperts& L = *j.L;
    const int N = L.n_embd, F = L.n_ff;
    std::vector<int> miss;
    for (int e = 0; e < j.n_ent; ++e) {
        const bool m = !j.on_gpu[j.ids[e]];
        j.mask[e] = m ? 1 : 0;
        if (m) miss.push_back(e);
    }
    j.n_cpu = (int) miss.size();
    if (miss.empty()) return;
    std::sort(miss.begin(), miss.end(), [&](int a, int b) { return j.ids[a] != j.ids[b] ? j.ids[a] < j.ids[b] : a < b; });
    std::vector<int> g0{0};   // the entries of each expert, contiguous in `miss`
    for (size_t i = 1; i < miss.size(); ++i)
        if (j.ids[miss[i]] != j.ids[miss[i - 1]]) g0.push_back((int) i);
    g0.push_back((int) miss.size());
    const int ng = (int) g0.size() - 1;
    const size_t xrow = (size_t) (N / 32) * kQ81;
    const uint8_t* xq = (const uint8_t*) j.xq;
    std::vector<float> gu((size_t) miss.size() * 2 * F), dxi((size_t) miss.size() * (N / 32)), dxh((size_t) miss.size() * (F / 32));
    std::vector<uint8_t> hq((size_t) miss.size() * (F / 32) * kQ81);
    for (size_t i = 0; i < miss.size(); ++i) {   // the activation scales as floats, once
        const uint8_t* xr = xq + (size_t) (miss[i] / j.K) * xrow;
        for (int b = 0; b < N / 32; ++b) dxi[i * (N / 32) + (size_t) b] = h2f(xr + (size_t) b * kQ81);
    }
    const int gu_chunks = (2 * F + kRowChunk - 1) / kRowChunk, d_chunks = (N + kRowChunk - 1) / kRowChunk;
    CpuPool::get().run(ng * gu_chunks, [&](int task) {
        const int g = task / gu_chunks, c = task % gu_chunks;
        const uint8_t* blob = L.blobs + (size_t) j.ids[miss[(size_t) g0[g]]] * L.stride;
        const float* tail = L.tail_off ? (const float*) (blob + L.tail_off) : nullptr;
        for (int i0 = g0[g]; i0 < g0[g + 1]; i0 += kMaxE) {
            const int ne = std::min(kMaxE, g0[g + 1] - i0);
            const uint8_t* xs[kMaxE];
            const float* ds[kMaxE];
            for (int k = 0; k < ne; ++k) {
                xs[k] = xq + (size_t) (miss[(size_t) (i0 + k)] / j.K) * xrow;
                ds[k] = dxi.data() + (size_t) (i0 + k) * (N / 32);
            }
            float o[kMaxE];
            for (int r = c * kRowChunk; r < std::min(2 * F, (c + 1) * kRowChunk); ++r) {
                const bool up = r >= F;
                const uint8_t* w = blob + (up ? L.up_off + (size_t) (r - F) * L.gu_row : (size_t) r * L.gu_row);
                const float sc = tail ? tail[up ? 1 : 0] : 1.0f;
                row_dots(L.gu_type, w, N, xs, ds, ne, o);
                for (int k = 0; k < ne; ++k) gu[(size_t) (i0 + k) * 2 * F + r] = o[k] * sc;
            }
        }
    });
    std::vector<float> h((size_t) F);
    for (size_t i = 0; i < miss.size(); ++i) {
        const float* g = gu.data() + i * 2 * F;
        for (int r = 0; r < F; ++r) {
            const float u = L.situ_lb > 0.f ? L.situ_lb * std::tanh(g[F + r] / L.situ_lb) : g[F + r];
            h[(size_t) r] = L.situ_beta * std::tanh(g[r] / L.situ_beta) / (1.0f + std::exp(-g[r])) * u;
        }
        uint8_t* hr = hq.data() + i * (F / 32) * kQ81;
        quantize_q8_1(h.data(), hr, F);
        for (int b = 0; b < F / 32; ++b) dxh[i * (F / 32) + (size_t) b] = h2f(hr + (size_t) b * kQ81);
    }
    CpuPool::get().run(ng * d_chunks, [&](int task) {
        const int g = task / d_chunks, c = task % d_chunks;
        const uint8_t* blob = L.blobs + (size_t) j.ids[miss[(size_t) g0[g]]] * L.stride;
        const float sd = L.tail_off ? ((const float*) (blob + L.tail_off))[2] : 1.0f;
        for (int i0 = g0[g]; i0 < g0[g + 1]; i0 += kMaxE) {
            const int ne = std::min(kMaxE, g0[g + 1] - i0);
            const uint8_t* xs[kMaxE];
            const float* ds[kMaxE];
            for (int k = 0; k < ne; ++k) {
                xs[k] = hq.data() + (size_t) (i0 + k) * (F / 32) * kQ81;
                ds[k] = dxh.data() + (size_t) (i0 + k) * (F / 32);
            }
            float o[kMaxE];
            for (int r = c * kRowChunk; r < std::min(N, (c + 1) * kRowChunk); ++r) {
                row_dots(L.d_type, blob + L.down_off + (size_t) r * L.d_row, F, xs, ds, ne, o);
                for (int k = 0; k < ne; ++k) j.out[(size_t) miss[(size_t) (i0 + k)] * N + r] = o[k] * sd;
            }
        }
    });
}

}  // namespace strata::inkling2
