#include <cuda_fp16.h>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"
#include <cstdlib>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

/*

Fused mHC HyperConnection mix() kernel

mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D):

  flat = rmsnorm_unweighted(streams.flatten)          (row of H * D values)
  mixv = flat @ fn.T                                  (M = 2H + H^2 outputs)
  pre  = sigmoid(mixv[0:H]  * s0 + base[0:H]) + eps
  post = 2 sigmoid(mixv[H:2H] * s1 + base[H:2H])
  comb = sinkhorn(softmax(mixv[2H:] * s2 + base[2H:]))   (iters alternating row/col norms)
  collapsed = sum_h pre[h] * streams[h]

Two launches, no grid-wide sync, deterministic (fixed reduction order, no atomics):
  K1 partials: grid (chunksA, R); each block reduces its column chunk to M + 1 partials.
  K2 finalize: grid (chunksC, R); EVERY block re-reduces the tiny partial matrix and
    derives rmr + pre redundantly (removes the cross-block dependency), then streams its
    chunk of collapsed; the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0
    (row sums: shfl_xor 1|2, col sums: shfl_xor 4|8 for H = 4) and writes post/comb.

*/

// Partials blocks are small: at R = 1 the grid is the only parallelism, so favor many
// blocks (row_len / (4 * 64) chunks) over wide ones; the M + 1 block reduce also shrinks
#define NUM_THREADS 256
#define NUM_THREADS_A 64

__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const FN_T* __restrict__ fn,         // (M, H * D) float, or half (opt-in, halves traffic)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            if constexpr (std::is_same_v<FN_T, half>)
            {
                // Single vectorized 8-byte load (two half2 loads would double the LDGs)
                int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                float2 lo = __half22float2(*(const half2*) &pk.x);
                float2 hi = __half22float2(*(const half2*) &pk.y);
                w = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = acc[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

template <int H, int M_, bool HEAD, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            // softmax over rows
            float m = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            v = __expf(v - m);
            float s = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = v;
            #pragma unroll
            for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = v;
                #pragma unroll
                for (int o = 1; o < H; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                v = __fdividef(v, rs + hc_eps);
                cs = v;
                #pragma unroll
                for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
        }
        return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }
}


/*

GatedResidual (Qwen4Exp, the low-rank elementwise cousin of mHC) fused mix, decode form:

  mix(streams (R, H, D) fp32) -> post (R, H), mixed (R, D):
    normed[h] = rmsnorm(streams[h]) * w[h]           (per-STREAM norm, weighted, w incl +1)
    dots      = cat(down, inject) @ normed.flatten   (M = LR + H outputs, LR = low rank ~320)
    t         = silu(dots[:LR] / H)
    post      = 2 sigmoid(dots[LR:] / H)
    g[h, d]   = sigmoid(up[h * D + d, :] @ t)        (per-CHANNEL gate through the low rank)
    mixed[d]  = mean_h g[h, d] * normed[h, d]

  Same two-launch, no-grid-sync, deterministic shape as hc_mix, restructured for the wide low
  rank (LR >> mHC's M = 24, so per-thread accumulator arrays don't fit):
    K1 gr_dots: one block per (fn row | sum-of-squares), computing per-STREAM partial dots
      against the RAW streams -- by linearity the per-stream rms scale applies in K2, and the
      norm WEIGHT is folded into the fn rows at load time.
    K2 gr_finalize: every block re-derives rmr / t redundantly from the K1 output (the mHC
      finalize pattern), then streams its chunk of mixed, evaluating the per-channel up-gate
      inline -- upT is laid out (LR, H * D) so the serial rank loop reads coalesced and stays
      L2-resident at decode R. NOT for large R (the untiled up/down reads defeat the L2);
      the python side runs a plain half-GEMM path for prefill.

  apply_ is hc_apply with no comb: x[h] += post[h] * y.

*/

#define GR_THREADS_A 128

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;            // fn row, or M for the sum-of-squares row
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 16-byte fn loads (8 halves) against two 16-byte stream quads
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,                        // low rank; M = LR + (post ? H : 0)
    const int chunk_cols,                // multiple of 4
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Redundant per-block head derivation (no cross-block dependency): rmr from the sumsq
    // row, then the silu'd low-rank activations into shared memory
    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Streamed mixed chunk: one WARP per column quad, the LR-long up-gate dots split across
    // the lanes (lane l covers ranks l, l+32, ...) and shfl-reduced -- at decode R the total
    // column count is small (R * D / 4), so per-thread columns would leave the GPU nearly
    // idle with each thread serializing the rank loop
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // upx layout (H, D / 4, LR, 4): consecutive lanes (consecutive i) read
                // consecutive 8-byte quads, so the rank loop is fully coalesced
                half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x);
                g[h].y = fmaf(ti, HIGH_TO_FLOAT(u.x), g[h].y);
                g[h].z = fmaf(ti, LOW_TO_FLOAT(u.y), g[h].z);
                g[h].w = fmaf(ti, HIGH_TO_FLOAT(u.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}



// ---------------------------------------------------------------------------------------------
// int8-weight variants (EXL3_GR_INT8). Same math and reduction order as gr_dots_kernel /
// gr_finalize_kernel; only the weight load differs: int8 x per-row fp32 scale instead of half.
// fn_q: (M, H * D) int8, fn_s: (M) float   -- row j dequantizes as fn_q[j, :] * fn_s[j]
// up_q: (H, D / 4, LR, 4) int8, up_s: (H, D) float -- channel (h, d) scale, constant over LR
// Because each scale is constant along the contracted dim, it factors out of the k-loop:
//   sum_k s[k] * (q[k] * scale) == scale * sum_k s[k] * q[k]   (exact in real arithmetic; in
// fp32 the rounding pattern differs slightly from multiplying inside the loop, but this is the
// SAME trajectory whether the reference is "fp16 kernel on dequantized weights" or not -- the
// reference to compare against is gr_parity's fp32 torch path on the dequantized weights).

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_i8_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const int8_t* __restrict__ fn_q,     // (M, H * D) int8
    const float* __restrict__ fn_s,      // (M) per-row scale
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float sj = (j < M) ? fn_s[j] : 1.0f;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 8-byte int8 loads (8 weights) against two 16-byte stream quads: same 8-wide
            // step as the half kernel's 16-byte load, half the weight bytes
            const int2* f8 = (const int2*) (fn_q + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int2 pk = f8[c];
                const int8_t* q = (const int8_t*) &pk;
                a = fmaf(s0.x, (float) q[0], a);
                a = fmaf(s0.y, (float) q[1], a);
                a = fmaf(s0.z, (float) q[2], a);
                a = fmaf(s0.w, (float) q[3], a);
                a = fmaf(s1.x, (float) q[4], a);
                a = fmaf(s1.y, (float) q[5], a);
                a = fmaf(s1.z, (float) q[6], a);
                a = fmaf(s1.w, (float) q[7], a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v * sj;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_i8_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H)
    const int8_t* __restrict__ up_q,     // (H, D / 4, LR, 4) int8
    const float* __restrict__ up_s,      // (H, D) per-channel scale
    const half* __restrict__ w,          // (H * D) half norm weight
    float* __restrict__ post,
    void* __restrict__ mixed,
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // (H, D/4, LR, 4) int8: consecutive lanes read consecutive 4-byte quads
                int u = *(const int*) (up_q + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                const int8_t* q = (const int8_t*) &u;
                g[h].x = fmaf(ti, (float) q[0], g[h].x);
                g[h].y = fmaf(ti, (float) q[1], g[h].y);
                g[h].z = fmaf(ti, (float) q[2], g[h].z);
                g[h].w = fmaf(ti, (float) q[3], g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float4 us = *(const float4*) (up_s + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x * us.x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y * us.y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z * us.z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w * us.w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}

// Decode mix, bytes-in-flight version (benchmarks/gr_mix_decode): the fn table is read once per
// site for all rows, with each lane's chunks of GR_J rows issued together, one warp per stream;
// the finalize issues all its 16-byte up loads before use and runs rows in groups. Cold on a PRO
// 6000: 16.1 -> 14.9 us at one row, 33 -> 26 at eight. Shape-templated on the chunks per lane
// (D / 256) and rank iterations (LR / 64); other shapes take the generic kernels above.
#define GR_THREADS_A2 128           // 4 warps: one per stream
#define GR_THREADS_C 128            // 4 warps: one column quad each
#define GR_RB 4                     // rows per accumulator pass
#define GR_MAX_R 32
#define GR_J 2                      // fn rows per dots block (share the stream loads)

// Phase A: dots[r, j, h] = <streams[r, h, :], fn[j, h, :]> for every row r (row M: sum of squares).
// One block per fn row j, one warp per stream h: each lane holds its 10 x 16-byte chunks of the
// fn row in registers (issued together: the whole 20 KB row is in flight per block), then runs
// the rows in groups of GR_RB against the L2-resident streams. The table is read once per site
// regardless of R.
template <int H, int NCH>
__device__ __forceinline__ void gr_dots_block
(
    const float* __restrict__ streams, const half* __restrict__ fn, float* __restrict__ dots,
    const int M, const int D, const int R, const int block
)
{
    const int j0 = block * GR_J;
    const int h = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int D8 = D / 8;                // chunks per stream row
    if (j0 < M)
    {
        int4 f[GR_J][NCH];
        #pragma unroll
        for (int jj = 0; jj < GR_J; ++jj)
        {
            const int4* f8 = (const int4*) (fn + ((size_t) min(j0 + jj, M - 1) * H + h) * D);
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                f[jj][k] = (c < D8) ? f8[c] : make_int4(0, 0, 0, 0);
            }
        }
        for (int r0 = 0; r0 < R; r0 += GR_RB)
        {
            float acc[GR_RB][GR_J];
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
                #pragma unroll
                for (int jj = 0; jj < GR_J; ++jj) acc[q][jj] = 0.0f;
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                const bool ok = c < D8;
                #pragma unroll
                for (int q = 0; q < GR_RB; ++q)
                {
                    const int r = r0 + q;
                    const float4* s4 = (const float4*) (streams + ((size_t) min(r, R - 1) * H + h) * D + (ok ? c : 0) * 8);
                    float4 s0 = s4[0], s1 = s4[1];
                    #pragma unroll
                    for (int jj = 0; jj < GR_J; ++jj)
                    {
                        const half2* w2 = (const half2*) &f[jj][k];
                        float2 w0 = __half22float2(w2[0]), w1 = __half22float2(w2[1]), w2_ = __half22float2(w2[2]), w3 = __half22float2(w2[3]);
                        float a = acc[q][jj];
                        a = fmaf(s0.x, w0.x, a); a = fmaf(s0.y, w0.y, a); a = fmaf(s0.z, w1.x, a); a = fmaf(s0.w, w1.y, a);
                        a = fmaf(s1.x, w2_.x, a); a = fmaf(s1.y, w2_.y, a); a = fmaf(s1.z, w3.x, a); a = fmaf(s1.w, w3.y, a);
                        acc[q][jj] = a;
                    }
                }
            }
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
                #pragma unroll
                for (int jj = 0; jj < GR_J; ++jj)
                {
                    float a = acc[q][jj];
                    for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
                    if (lane == 0 && r0 + q < R && j0 + jj < M) dots[((size_t) (r0 + q) * (M + 1) + j0 + jj) * H + h] = a;
                }
        }
    }
    else
    {
        // sum of squares per (row, stream)
        for (int r = 0; r < R; ++r)
        {
            const float4* s4 = (const float4*) (streams + ((size_t) r * H + h) * D);
            float a = 0.0f;
            for (int c = lane; c < D / 4; c += 32)
            {
                float4 s = s4[c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
            for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
            if (lane == 0) dots[((size_t) r * (M + 1) + M) * H + h] = a;
        }
    }
}

// Phase B: rmr and the silu'd latent per row (redundantly per block, into shared memory), then
// one warp per column quad: the up-gate dots over the low rank with 16-byte loads (two ranks per
// lane, LR / 64 unrolled iterations, all four streams issued together), rows in groups of GR_RB
// re-reading the (L2-resident) up columns; epilogue: sigmoid gates, stream mean, half/float out.
template <int H, bool HALF_OUT, int NIT>
__device__ __forceinline__ void gr_finalize_block
(
    const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt,
    const half* __restrict__ w, float* __restrict__ post, void* __restrict__ mixed,
    const int D, const int LR, const int R, const float rms_eps, const int block
)
{
    const int M = LR + (post ? H : 0);
    extern __shared__ float t_s[];       // (R, LR) then rmr (R, H)
    float* rmr_s = t_s + (size_t) R * LR;
    const int tid = threadIdx.x;
    if (tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        rmr_s[tid] = rsqrtf(dots[((size_t) r * (M + 1) + M) * H + h] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int idx = tid; idx < R * LR; idx += GR_THREADS_C)
    {
        const int r = idx / LR, i = idx % LR;
        const float* dr = dots + ((size_t) r * (M + 1) + i) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r * H + h], dr[h], v);
        v *= inv_h;
        t_s[idx] = v * sigmoidf_(v);
    }
    if (post && block == 0 && tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        const float* dr = dots + ((size_t) r * (M + 1) + LR + h) * H;
        float v = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh) v = fmaf(rmr_s[r * H + hh], dr[hh], v);
        post[tid] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int lane = tid % 32, warp = tid / 32;
    const int c = block * (GR_THREADS_C / 32) + warp;           // column quad
    if (c >= D / 4) return;
    const int D4 = D / 4;
    for (int r0 = 0; r0 < R; r0 += GR_RB)
    {
        float4 g[GR_RB][H];
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h) g[q][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int4 u4[NIT][H];
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;
            #pragma unroll
            for (int h = 0; h < H; ++h)
                u4[it][h] = (i0 < LR) ? *(const int4*) (upt + ((((size_t) h * D4 + c) * LR + i0) * 4)) : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;              // ranks i0, i0 + 1
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
            {
                const int r = r0 + q;
                if (r >= R) continue;
                const float t0 = i0 < LR ? t_s[(size_t) r * LR + i0] : 0.0f;
                const float t1 = i0 + 1 < LR ? t_s[(size_t) r * LR + i0 + 1] : 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    const half2* u2 = (const half2*) &u4[it][h];
                    float2 a0 = __half22float2(u2[0]), a1 = __half22float2(u2[1]);   // rank i0: 4 columns
                    float2 b0 = __half22float2(u2[2]), b1 = __half22float2(u2[3]);   // rank i0 + 1
                    g[q][h].x = fmaf(t0, a0.x, fmaf(t1, b0.x, g[q][h].x));
                    g[q][h].y = fmaf(t0, a0.y, fmaf(t1, b0.y, g[q][h].y));
                    g[q][h].z = fmaf(t0, a1.x, fmaf(t1, b1.x, g[q][h].z));
                    g[q][h].w = fmaf(t0, a1.y, fmaf(t1, b1.y, g[q][h].w));
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    g[q][h].x += __shfl_xor_sync(0xffffffffu, g[q][h].x, offset);
                    g[q][h].y += __shfl_xor_sync(0xffffffffu, g[q][h].y, offset);
                    g[q][h].z += __shfl_xor_sync(0xffffffffu, g[q][h].z, offset);
                    g[q][h].w += __shfl_xor_sync(0xffffffffu, g[q][h].w, offset);
                }
        if (lane != 0) continue;
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
        {
            const int r = r0 + q;
            if (r >= R) continue;
            const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
            float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float4 sv = s4[(size_t) h * D4 + c];
                const half2* wq = (const half2*) (w + (size_t) h * D + 4 * c);
                float2 w0 = __half22float2(wq[0]), w1 = __half22float2(wq[1]);
                const float coef = rmr_s[r * H + h] * inv_h;
                o.x = fmaf(sigmoidf_(g[q][h].x) * coef * w0.x, sv.x, o.x);
                o.y = fmaf(sigmoidf_(g[q][h].y) * coef * w0.y, sv.y, o.y);
                o.z = fmaf(sigmoidf_(g[q][h].z) * coef * w1.x, sv.z, o.z);
                o.w = fmaf(sigmoidf_(g[q][h].w) * coef * w1.y, sv.w, o.w);
            }
            if (HALF_OUT)
            {
                half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
                out2[c * 2] = __floats2half2_rn(o.x, o.y);
                out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
            }
            else
                ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
        }
    }
}


template <int H, int NCH>
__global__ __launch_bounds__(GR_THREADS_A2)
void gr_dots_kernel2(const float* __restrict__ streams, const half* __restrict__ fn, float* __restrict__ dots, const int M, const int D, const int R)
{
    gr_dots_block<H, NCH>(streams, fn, dots, M, D, R, blockIdx.x);
}
template <int H, bool HALF_OUT, int NIT>
__global__ __launch_bounds__(GR_THREADS_C)
void gr_finalize_kernel2(const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt, const half* __restrict__ w,
                         float* __restrict__ post, void* __restrict__ mixed, const int D, const int LR, const int R, const float rms_eps)
{
    gr_finalize_block<H, HALF_OUT, NIT>(streams, dots, upt, w, post, mixed, D, LR, R, rms_eps, blockIdx.x);
}

// Residual update for one sublayer site: x[h, d] <- post[h] * y[d] + sum_h' comb[h', h] *
// x[h', d]. Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column
// mix of the H stream rows, so it runs in place: each thread loads all H values of its columns
// into registers before writing any back.

template <int H, typename Y_T, bool HAS_COMB>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
        }
    }
}

// Shared launch logic. mode: fn rows M = 2H + H^2 (mix) or H (head)

static void hc_mix_launch
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    at::Tensor& partials,
    at::Tensor* post,
    at::Tensor* comb,
    at::Tensor& collapsed,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous(), "hc_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int row_len = H * D;
    int M = fn.size(0);
    bool head = M == H;
    TORCH_CHECK(H == 4, "hc_mix: H = 4 only");
    TORCH_CHECK(head || M == 2 * H + H * H, "hc_mix: fn rows must be H or 2H + H^2");
    TORCH_CHECK(fn.size(1) == row_len && D % 4 == 0, "hc_mix: dims");
    bool fn_half = fn.dtype() == at::kHalf;
    if (!fn_half) TORCH_CHECK_DTYPE(fn, kFloat);
    TORCH_CHECK_DTYPE(base, kFloat);
    TORCH_CHECK_DTYPE(scale, kFloat);

    int n_chunks_a = partials.size(1);
    int chunk_cols = ((row_len / n_chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    n_chunks_a = (row_len + chunk_cols - 1) / chunk_cols;
    TORCH_CHECK(n_chunks_a <= partials.size(1), "hc_mix: partials workspace too small");

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols_c = ((D / chunks_c + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks_c = (D + chunk_cols_c - 1) / chunk_cols_c;

    bool half_out = collapsed.dtype() == at::kHalf;

    dim3 grid_a(n_chunks_a, R);
    dim3 grid_c(n_chunks_c, R);
    #define ARGS_A(FN_T) \
        (const float*) streams.data_ptr(), (const FN_T*) fn.data_ptr(), \
        (float*) partials.data_ptr(), row_len, chunk_cols
    #define ARGS_C(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c, rms_eps, hc_eps, sinkhorn_iters
    if (!head)
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (half_out)
            hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
        else
            hc_mix_finalize_kernel<4, 24, false, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
    }
    else
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 4, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 4, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        if (half_out)
            hc_mix_finalize_kernel<4, 4, true, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
        else
            hc_mix_finalize_kernel<4, 4, true, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
    }
    #undef ARGS_A
    #undef ARGS_C
    cuda_check(cudaPeekAtLastError());
}

int hc_mix_num_chunks(int R, int row_len)
{
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    int chunk_cols = ((row_len / chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    return (row_len + chunk_cols - 1) / chunk_cols;
}

void hc_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float
    const at::Tensor& base,              // (2H + H^2) float
    const at::Tensor& scale,             // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        (int) sinkhorn_iters,
        partials,
        &post,
        &comb,
        collapsed,
        nullptr
    );
}

void hc_head
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (H, H * D) float
    const at::Tensor& base,              // (H) float
    const at::Tensor& scale,             // (1) float
    double rms_eps,
    double hc_eps,
    at::Tensor partials,                 // (R, chunks, H + 1) float workspace
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        0,
        partials,
        nullptr,
        nullptr,
        collapsed,
        nullptr
    );
}

void hc_apply
(
    at::Tensor x,                        // (R, H, D) float, updated IN PLACE
    const at::Tensor& y,                 // (R, D) float or half
    const at::Tensor& post,              // (R, H) float
    const c10::optional<at::Tensor>& comb   // (R, H, H) float, or none (x[h] += post[h] * y)
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(post, kFloat);
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && post.is_contiguous(), "hc_apply: contiguous inputs required");
    int R = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    TORCH_CHECK(H == 4, "hc_apply: H = 4 only");
    TORCH_CHECK(D % 4 == 0, "hc_apply: dims");
    TORCH_CHECK(y.size(0) == R && y.size(-1) == D, "hc_apply: y shape");
    TORCH_CHECK(post.size(0) == R, "hc_apply: gate shapes");
    const float* comb_p = nullptr;
    if (comb)
    {
        TORCH_CHECK_DTYPE(comb.value(), kFloat);
        TORCH_CHECK(comb.value().is_contiguous() && comb.value().size(0) == R, "hc_apply: comb shape");
        comb_p = (const float*) comb.value().data_ptr();
    }

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols = ((D / chunks_c + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;

    dim3 grid(n_chunks, R);
    #define ARGS(Y_T) \
        (float*) x.data_ptr(), (const Y_T*) y.data_ptr(), \
        (const float*) post.data_ptr(), comb_p, D, chunk_cols
    if (y.dtype() == at::kHalf)
    {
        if (comb_p) hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
        else        hc_apply_kernel<4, half, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
    }
    else
    {
        if (comb_p) hc_apply_kernel<4, float, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
        else        hc_apply_kernel<4, float, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
    }
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------
// GR_ROW_BATCHED: row-batched variants for decode/verify (R <= GR_RB_MAX_R). Same math as
// gr_dots_kernel / gr_finalize_kernel, but one block (dots) or one warp (finalize) covers ALL
// R rows so `fn` and `upt` are streamed from memory once per call instead of once per row.
// The R stream rows (R * H * D fp32 = 40 KB each) stay L2-resident and are re-read per row.

#define GR_RB_MAX_R 8
#define GR_RB_THREADS 256

template <int H, int RB>
__global__ __launch_bounds__(GR_RB_THREADS)
void gr_dots_rb_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D)
    float* __restrict__ dots,            // (R, M + 1, H)
    const int R,
    const int M,
    const int D
)
{
    const int j = blockIdx.x;            // fn row, or M for sum-of-squares
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ float red[RB][H][GR_RB_THREADS / 32];

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a[RB];
        #pragma unroll
        for (int r = 0; r < RB; ++r) a[r] = 0.0f;

        if (j < M)
        {
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_RB_THREADS)
            {
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                float f0 = LOW_TO_FLOAT(w0.x), f1 = HIGH_TO_FLOAT(w0.x);
                float f2 = LOW_TO_FLOAT(w0.y), f3 = HIGH_TO_FLOAT(w0.y);
                float f4 = LOW_TO_FLOAT(w1.x), f5 = HIGH_TO_FLOAT(w1.x);
                float f6 = LOW_TO_FLOAT(w1.y), f7 = HIGH_TO_FLOAT(w1.y);
                #pragma unroll
                for (int r = 0; r < RB; ++r)
                {
                    if (r < R)
                    {
                        const float4* s4 = (const float4*) (streams + (size_t) r * H * D) + (size_t) h * D4;
                        float4 s0 = s4[2 * c];
                        float4 s1 = s4[2 * c + 1];
                        float v = a[r];
                        v = fmaf(s0.x, f0, v); v = fmaf(s0.y, f1, v);
                        v = fmaf(s0.z, f2, v); v = fmaf(s0.w, f3, v);
                        v = fmaf(s1.x, f4, v); v = fmaf(s1.y, f5, v);
                        v = fmaf(s1.z, f6, v); v = fmaf(s1.w, f7, v);
                        a[r] = v;
                    }
                }
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_RB_THREADS)
            {
                #pragma unroll
                for (int r = 0; r < RB; ++r)
                {
                    if (r < R)
                    {
                        const float4* s4 = (const float4*) (streams + (size_t) r * H * D) + (size_t) h * D4;
                        float4 sv = s4[c];
                        a[r] = fmaf(sv.x, sv.x, fmaf(sv.y, sv.y, fmaf(sv.z, sv.z, fmaf(sv.w, sv.w, a[r]))));
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < RB; ++r)
        {
            float v = a[r];
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, offset);
            if (lane == 0) red[r][h][warp] = v;
        }
    }
    __syncthreads();
    // RB * H outputs: thread t -> (r = t / H, h = t % H)
    if (threadIdx.x < RB * H)
    {
        const int r = threadIdx.x / H;
        const int h = threadIdx.x % H;
        if (r < R)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_RB_THREADS / 32; ++w)
                v += red[r][h][w];
            dots[((size_t) r * (M + 1) + j) * H + h] = v;
        }
    }
}

// One WARP per column quad, all R rows: the rank loop loads each upt quad once and applies it
// to R t-vectors held in shared memory. Row groups of RG keep the accumulator count bounded
// (RG * H float4 = 64 registers at RG = 4); the second group re-reads upt from L1/L2.
template <int H, bool HALF_OUT, int RG>
__global__ __launch_bounds__(GR_RB_THREADS)
void gr_finalize_rb_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H)
    const half* __restrict__ upt,        // (H, D / 4, LR, 4)
    const half* __restrict__ w,          // (H * D)
    float* __restrict__ post,            // (R, H) or nullptr
    void* __restrict__ mixed,            // (R, D)
    const int R,
    const int D,
    const int LR,
    const int chunk_cols,
    const float rms_eps
)
{
    const int M = LR + (post ? H : 0);
    __shared__ float rmr_s[GR_RB_MAX_R][H];
    extern __shared__ float t_s[];       // (R, LR)
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float inv_h = 1.0f / (float) H;

    if (threadIdx.x < R * H)
    {
        const int r = threadIdx.x / H, h = threadIdx.x % H;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        rmr_s[r][h] = rsqrtf(dr[(size_t) M * H + h] / (float) D + rms_eps);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < R * LR; idx += GR_RB_THREADS)
    {
        const int r = idx / LR, i = idx % LR;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[r][h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[idx] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < R * H)
    {
        const int r = threadIdx.x / H, h = threadIdx.x % H;
        const float* dr = dots + (size_t) r * (M + 1) * H;
        float v = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
            v = fmaf(rmr_s[r][hh], dr[(size_t) (LR + h) * H + hh], v);
        post[(size_t) r * H + h] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    for (int c = c0 / 4 + warp; c < c1 / 4; c += GR_RB_THREADS / 32)
    {
        for (int r0 = 0; r0 < R; r0 += RG)
        {
            float4 g[RG][H];
            #pragma unroll
            for (int q = 0; q < RG; ++q)
                #pragma unroll
                for (int h = 0; h < H; ++h) g[q][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            for (int i = lane; i < LR; i += 32)
            {
                float t[RG];
                #pragma unroll
                for (int q = 0; q < RG; ++q)
                    t[q] = (r0 + q < R) ? t_s[(size_t) (r0 + q) * LR + i] : 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                    float u0 = LOW_TO_FLOAT(u.x), u1 = HIGH_TO_FLOAT(u.x);
                    float u2 = LOW_TO_FLOAT(u.y), u3 = HIGH_TO_FLOAT(u.y);
                    #pragma unroll
                    for (int q = 0; q < RG; ++q)
                    {
                        g[q][h].x = fmaf(t[q], u0, g[q][h].x);
                        g[q][h].y = fmaf(t[q], u1, g[q][h].y);
                        g[q][h].z = fmaf(t[q], u2, g[q][h].z);
                        g[q][h].w = fmaf(t[q], u3, g[q][h].w);
                    }
                }
            }
            #pragma unroll
            for (int q = 0; q < RG; ++q)
                #pragma unroll
                for (int h = 0; h < H; ++h)
                    for (int offset = 16; offset > 0; offset >>= 1)
                    {
                        g[q][h].x += __shfl_xor_sync(0xffffffffu, g[q][h].x, offset);
                        g[q][h].y += __shfl_xor_sync(0xffffffffu, g[q][h].y, offset);
                        g[q][h].z += __shfl_xor_sync(0xffffffffu, g[q][h].z, offset);
                        g[q][h].w += __shfl_xor_sync(0xffffffffu, g[q][h].w, offset);
                    }
            if (lane != 0) continue;
            #pragma unroll
            for (int q = 0; q < RG; ++q)
            {
                const int r = r0 + q;
                if (r >= R) break;
                const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
                float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    float4 sv = s4[(size_t) h * D4 + c];
                    half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
                    float coef = rmr_s[r][h] * inv_h;
                    o.x = fmaf(sigmoidf_(g[q][h].x) * coef * LOW_TO_FLOAT(wq.x),  sv.x, o.x);
                    o.y = fmaf(sigmoidf_(g[q][h].y) * coef * HIGH_TO_FLOAT(wq.x), sv.y, o.y);
                    o.z = fmaf(sigmoidf_(g[q][h].z) * coef * LOW_TO_FLOAT(wq.y),  sv.z, o.z);
                    o.w = fmaf(sigmoidf_(g[q][h].w) * coef * HIGH_TO_FLOAT(wq.y), sv.w, o.w);
                }
                if (HALF_OUT)
                {
                    half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
                    out2[c * 2] = __floats2half2_rn(o.x, o.y);
                    out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
                }
                else
                    ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
            }
        }
    }
}

static int gr_rb_mode()
{
    static int mode = -1;
    if (mode < 0)
    {
        const char* env = std::getenv("EXL3_GR_RB");
        mode = env ? atoi(env) : 0;   // default OFF: ~1.3 ms/round, perturbs greedy trajectory
    }
    return mode;
}

void gr_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) half: cat(down, inject) * w
    const at::Tensor& upt,               // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && dots.is_contiguous(), "gr_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn.size(0);
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix: upt must be the (H, D / 4, LR, 4) repacked layout");
    int LR = upt.size(2);
    TORCH_CHECK(H == 4, "gr_mix: H = 4 only");
    TORCH_CHECK(D % 8 == 0, "gr_mix: dims");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix: dims");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix: dots shape");

    if (gr_rb_mode() != 0 && R <= GR_RB_MAX_R)
    {
        // Row-batched path (GR_ROW_BATCHED): weights read once per call
        dim3 grid_a(M + 1);
        gr_dots_rb_kernel<4, GR_RB_MAX_R><<<grid_a, GR_RB_THREADS, 0, stream>>>
        (
            (const float*) streams.data_ptr(), (const half*) fn.data_ptr(),
            (float*) dots.data_ptr(), R, M, D
        );
        cuda_check(cudaPeekAtLastError());

        const int gran = 4 * (GR_RB_THREADS / 32);
        int chunks_c = std::max(1, (D + gran - 1) / gran);
        int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
        int n_chunks = (D + chunk_cols - 1) / chunk_cols;
        dim3 grid_c(n_chunks);
        int smem = R * LR * sizeof(float);
        float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
        #define ARGS_RB \
            (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
            (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
            post_p, mixed.data_ptr(), R, D, LR, chunk_cols, (float) rms_eps
        if (mixed.dtype() == at::kHalf)
            gr_finalize_rb_kernel<4, true, 4><<<grid_c, GR_RB_THREADS, smem, stream>>>(ARGS_RB);
        else
            gr_finalize_rb_kernel<4, false, 4><<<grid_c, GR_RB_THREADS, smem, stream>>>(ARGS_RB);
        #undef ARGS_RB
        cuda_check(cudaPeekAtLastError());
        return;
    }

    const int nch = D / 256, nit = (LR + 63) / 64;
    const bool fast = R <= GR_MAX_R && D % 256 == 0 && (nch == 10 || nch == 16 || nch == 20) && (nit == 5 || nit == 8 || nit == 16) && LR % 2 == 0;
    if (fast)
    {
        const float* s_p = (const float*) streams.data_ptr(); const half* fn_p = (const half*) fn.data_ptr(); float* dots_p = (float*) dots.data_ptr();
        const int grid_a2 = (M + GR_J - 1) / GR_J + 1;
        switch (nch)
        {
            case 10: gr_dots_kernel2<4, 10><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, fn_p, dots_p, M, D, R); break;
            case 16: gr_dots_kernel2<4, 16><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, fn_p, dots_p, M, D, R); break;
            case 20: gr_dots_kernel2<4, 20><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, fn_p, dots_p, M, D, R); break;
        }
        cuda_check(cudaPeekAtLastError());
        const int grid_c2 = D / 4 / (GR_THREADS_C / 32);
        const int smem2 = (R * LR + R * 4) * sizeof(float);
        float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
        #define ARGS2 \
            (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
            post_p, mixed.data_ptr(), D, LR, R, (float) rms_eps
        const bool hout = mixed.dtype() == at::kHalf;
        switch (nit)
        {
            case 5:  if (hout) gr_finalize_kernel2<4, true, 5><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else gr_finalize_kernel2<4, false, 5><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); break;
            case 8:  if (hout) gr_finalize_kernel2<4, true, 8><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else gr_finalize_kernel2<4, false, 8><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); break;
            case 16: if (hout) gr_finalize_kernel2<4, true, 16><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else gr_finalize_kernel2<4, false, 16><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); break;
        }
        #undef ARGS2
        cuda_check(cudaPeekAtLastError());
        return;
    }

    dim3 grid_a(M + 1, R);
    gr_dots_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
    (
        (const float*) streams.data_ptr(), (const half*) fn.data_ptr(),
        (float*) dots.data_ptr(), M, D
    );
    cuda_check(cudaPeekAtLastError());

    // Phase C is warp-per-column-quad: chunk at warp granularity (4 * NUM_THREADS / 32
    // columns) so small R still fills the device
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    if (mixed.dtype() == at::kHalf)
        gr_finalize_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    else
        gr_finalize_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}


// ---------------------------------------------------------------------------
// Fused dots+finalize, gated by EXL3_GR_TUNED=1. One cooperative launch:
// warp-per-(r, j, h) dots, L2 prefetch of the up_q columns between phases,
// grid.sync(), per-block t_s/rmr, warp-per-(r, c-quad) finalize. Reassociated
// vs the stock two-kernel path (fixed deterministic order, no atomics), so
// results differ from stock within float tolerance but are run-to-run stable.
//
// Gated to R <= GR_FUSED_MAX_R: beyond that the stock pair wins because the
// fused mapping re-reads the int8 tables per row through L2 (cold-cache
// microbench on GB10, us/site -- stock: R1 39.9, R2 40.8, R3 43.9, R4 55.3,
// R5 60.4, R6 65.4; fused: R1 34.0, R2 37.5, R3 43.1, R4 50.8, R5 59.1,
// R6 77.1).
// ---------------------------------------------------------------------------

#define GR_FUSED_THREADS 256
#define GR_FUSED_WARPS (GR_FUSED_THREADS / 32)
#define GR_FUSED_MAX_R 4

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(GR_FUSED_THREADS, 4)
void gr_mix_fused_i8_kernel
(
    const float* __restrict__ streams,
    const int8_t* __restrict__ fn_q,
    const float* __restrict__ fn_s,
    const int8_t* __restrict__ up_q,
    const float* __restrict__ up_s,
    const half* __restrict__ w,
    float* __restrict__ dots,
    float* __restrict__ post,
    void* __restrict__ mixed,
    const int R,
    const int M,
    const int D,
    const int LR,
    const float rms_eps
)
{
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int gwarp = blockIdx.x * GR_FUSED_WARPS + threadIdx.x / 32;
    const int nwarps = gridDim.x * GR_FUSED_WARPS;

    extern __shared__ float fsmem[];
    float* t_s = fsmem;              // R * LR
    float* rmr_s = fsmem + R * LR;   // R * H

    // Phase A: dots, item p = (r * (M + 1) + j) * H + h (flat dots index)
    const int npairs1 = (M + 1) * H;
    const int npairs = R * npairs1;
    const int D8 = D / 8;
    for (int p = gwarp; p < npairs; p += nwarps)
    {
        const int r = p / npairs1;
        const int jl = p - r * npairs1;
        const int j = jl / H;
        const int h = jl % H;
        const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
        const float sj = (j < M) ? __ldg(fn_s + j) : 1.0f;
        float a = 0.0f;
        if (j < M)
        {
            const int2* f8 = (const int2*) (fn_q + ((size_t) j * H + h) * D);
            const float4* sh = s4 + (size_t) h * D4;
            for (int c = lane; c < D8; c += 32)
            {
                int2 pk = __ldcs(f8 + c);
                float4 s0 = __ldg(sh + 2 * c);
                float4 s1 = __ldg(sh + 2 * c + 1);
                const int8_t* q = (const int8_t*) &pk;
                a = fmaf(s0.x, (float) q[0], a);
                a = fmaf(s0.y, (float) q[1], a);
                a = fmaf(s0.z, (float) q[2], a);
                a = fmaf(s0.w, (float) q[3], a);
                a = fmaf(s1.x, (float) q[4], a);
                a = fmaf(s1.y, (float) q[5], a);
                a = fmaf(s1.z, (float) q[6], a);
                a = fmaf(s1.w, (float) q[7], a);
            }
        }
        else
        {
            for (int c = lane; c < D4; c += 32)
            {
                float4 s = __ldg(s4 + (size_t) h * D4 + c);
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) dots[p] = a * sj;
    }

    // L2 prefetch of this warp's phase-B up_q columns (shared across rows;
    // 3.3 MB at the Qwen3.8 shape fits L2, so phase B reads hit L2)
    for (int c = gwarp; c < D4; c += nwarps)
    {
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
        {
            const char* base = (const char*) (up_q + (((size_t) hh * D4 + c) * LR) * 4);
            for (int off = lane * 128; off < LR * 4; off += 32 * 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(base + off));
        }
    }

    cg::this_grid().sync();

    // t_s / rmr / post, redundantly per block (dots reads are L2-hot)
    const int nrows = M + 1;
    if (threadIdx.x < (unsigned) (R * H))
    {
        const int r = threadIdx.x / H;
        const int hh = threadIdx.x % H;
        rmr_s[threadIdx.x] = rsqrtf(dots[((size_t) r * nrows + M) * H + hh] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int idx = threadIdx.x; idx < R * LR; idx += GR_FUSED_THREADS)
    {
        const int r = idx / LR;
        const int i = idx - r * LR;
        const float* dr = dots + (size_t) r * nrows * H;
        float v = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
            v = fmaf(rmr_s[r * H + hh], dr[(size_t) i * H + hh], v);
        v *= inv_h;
        t_s[idx] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < (unsigned) (R * H))
    {
        const int r = threadIdx.x / H;
        const int hh = threadIdx.x % H;
        const float* dr = dots + (size_t) r * nrows * H;
        float v = 0.0f;
        #pragma unroll
        for (int hhh = 0; hhh < H; ++hhh)
            v = fmaf(rmr_s[r * H + hhh], dr[(size_t) (LR + hh) * H + hhh], v);
        post[threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Phase B: finalize, warp per (r, c-quad)
    const int nq = R * D4;
    for (int qq = gwarp; qq < nq; qq += nwarps)
    {
        const int r = qq / D4;
        const int c = qq - r * D4;
        const float* t_r = t_s + r * LR;
        const float* rmr_r = rmr_s + r * H;
        const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
        float4 g[H];
        #pragma unroll
        for (int hh = 0; hh < H; ++hh) g[hh] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int i = lane;
        for (; i + 96 < LR; i += 128)
        {
            int u[4][H];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                #pragma unroll
                for (int hh = 0; hh < H; ++hh)
                    u[k][hh] = __ldcs((const int*) (up_q + ((((size_t) hh * D4 + c) * LR + i + 32 * k) * 4)));
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                float ti = t_r[i + 32 * k];
                #pragma unroll
                for (int hh = 0; hh < H; ++hh)
                {
                    const int8_t* qb = (const int8_t*) &u[k][hh];
                    g[hh].x = fmaf(ti, (float) qb[0], g[hh].x);
                    g[hh].y = fmaf(ti, (float) qb[1], g[hh].y);
                    g[hh].z = fmaf(ti, (float) qb[2], g[hh].z);
                    g[hh].w = fmaf(ti, (float) qb[3], g[hh].w);
                }
            }
        }
        for (; i < LR; i += 32)
        {
            float ti = t_r[i];
            #pragma unroll
            for (int hh = 0; hh < H; ++hh)
            {
                int u = __ldcs((const int*) (up_q + ((((size_t) hh * D4 + c) * LR + i) * 4)));
                const int8_t* qb = (const int8_t*) &u;
                g[hh].x = fmaf(ti, (float) qb[0], g[hh].x);
                g[hh].y = fmaf(ti, (float) qb[1], g[hh].y);
                g[hh].z = fmaf(ti, (float) qb[2], g[hh].z);
                g[hh].w = fmaf(ti, (float) qb[3], g[hh].w);
            }
        }
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[hh].x += __shfl_xor_sync(0xffffffffu, g[hh].x, offset);
                g[hh].y += __shfl_xor_sync(0xffffffffu, g[hh].y, offset);
                g[hh].z += __shfl_xor_sync(0xffffffffu, g[hh].z, offset);
                g[hh].w += __shfl_xor_sync(0xffffffffu, g[hh].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
        {
            float4 s = s4[(size_t) hh * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) hh * D + 4 * c);
            float4 us = *(const float4*) (up_s + (size_t) hh * D + 4 * c);
            float coef = rmr_r[hh] * inv_h;
            o.x = fmaf(sigmoidf_(g[hh].x * us.x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[hh].y * us.y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[hh].z * us.z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[hh].w * us.w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}

static bool gr_tuned_enabled()
{
    static const bool v = []()
    {
        const char* e = getenv("EXL3_GR_TUNED");
        return e && e[0] == '1';
    }();
    return v;
}

// Resident-block count for the fused kernel, or 0 when cooperative launch is
// unavailable (caller falls back to the stock two-kernel path). Queried with
// the maximum smem footprint (GR_FUSED_MAX_R) so one count covers all R.
static int gr_fused_blocks(int max_smem, bool half_out)
{
    static int cached[2] = { -1, -1 };
    int slot = half_out ? 1 : 0;
    if (cached[slot] >= 0) return cached[slot];
    int dev, nsm, nb = 0, coop = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    if (coop)
    {
        if (half_out)
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &nb, gr_mix_fused_i8_kernel<4, true>, GR_FUSED_THREADS, max_smem);
        else
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &nb, gr_mix_fused_i8_kernel<4, false>, GR_FUSED_THREADS, max_smem);
    }
    cached[slot] = nb * nsm;
    return cached[slot];
}

void gr_mix_int8
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn_q,              // (M, H * D) int8
    const at::Tensor& fn_s,              // (M) float
    const at::Tensor& up_q,              // (H, D / 4, LR, 4) int8
    const at::Tensor& up_s,              // (H, D) float
    const at::Tensor& w,                 // (H * D) half
    double rms_eps,
    at::Tensor dots,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK(fn_q.dtype() == at::kChar && up_q.dtype() == at::kChar, "gr_mix_int8: int8 weights required");
    TORCH_CHECK_DTYPE(fn_s, kFloat);
    TORCH_CHECK_DTYPE(up_s, kFloat);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn_q.is_contiguous() && up_q.is_contiguous() &&
                fn_s.is_contiguous() && up_s.is_contiguous() && w.is_contiguous() && dots.is_contiguous(),
                "gr_mix_int8: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn_q.size(0);
    TORCH_CHECK(up_q.dim() == 4 && up_q.size(0) == H && up_q.size(1) == D / 4 && up_q.size(3) == 4,
                "gr_mix_int8: up_q must be the (H, D / 4, LR, 4) layout");
    int LR = up_q.size(2);
    TORCH_CHECK(H == 4, "gr_mix_int8: H = 4 only");
    TORCH_CHECK(D % 8 == 0, "gr_mix_int8: dims");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix_int8: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn_q.size(1) == H * D && w.numel() == H * D, "gr_mix_int8: dims");
    TORCH_CHECK(fn_s.numel() == M && up_s.numel() == H * D, "gr_mix_int8: scale shapes");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix_int8: dots shape");

    if (gr_tuned_enabled() && R <= GR_FUSED_MAX_R && D % 16 == 0)
    {
        const bool hout = mixed.dtype() == at::kHalf;
        int fused_smem = (R * LR + R * 4) * sizeof(float);
        int nblocks = gr_fused_blocks((GR_FUSED_MAX_R * LR + GR_FUSED_MAX_R * 4) * sizeof(float), hout);
        if (nblocks > 0)
        {
            const float* s_p = (const float*) streams.data_ptr();
            const int8_t* fq_p = (const int8_t*) fn_q.data_ptr();
            const float* fs_p = (const float*) fn_s.data_ptr();
            const int8_t* uq_p = (const int8_t*) up_q.data_ptr();
            const float* us_p = (const float*) up_s.data_ptr();
            const half* w_p = (const half*) w.data_ptr();
            float* d_p = (float*) dots.data_ptr();
            float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
            void* mx_p = mixed.data_ptr();
            float eps = (float) rms_eps;
            void* args[] = { &s_p, &fq_p, &fs_p, &uq_p, &us_p, &w_p, &d_p, &post_p, &mx_p,
                             (void*) &R, (void*) &M, (void*) &D, (void*) &LR, &eps };
            cudaError_t err;
            if (hout)
                err = cudaLaunchCooperativeKernel(
                    (void*) gr_mix_fused_i8_kernel<4, true>, dim3(nblocks),
                    dim3(GR_FUSED_THREADS), args, fused_smem, stream);
            else
                err = cudaLaunchCooperativeKernel(
                    (void*) gr_mix_fused_i8_kernel<4, false>, dim3(nblocks),
                    dim3(GR_FUSED_THREADS), args, fused_smem, stream);
            TORCH_CHECK(err == cudaSuccess, "gr_mix_int8 fused launch: ", cudaGetErrorString(err));
            cuda_check(cudaPeekAtLastError());
            return;
        }
    }

    dim3 grid_a(M + 1, R);
    gr_dots_i8_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
    (
        (const float*) streams.data_ptr(), (const int8_t*) fn_q.data_ptr(),
        (const float*) fn_s.data_ptr(), (float*) dots.data_ptr(), M, D
    );
    cuda_check(cudaPeekAtLastError());

    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS_I8 \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const int8_t*) up_q.data_ptr(), (const float*) up_s.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    if (mixed.dtype() == at::kHalf)
        gr_finalize_i8_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS_I8);
    else
        gr_finalize_i8_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS_I8);
    #undef ARGS_I8
    cuda_check(cudaPeekAtLastError());
}
