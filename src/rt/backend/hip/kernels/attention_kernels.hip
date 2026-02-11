#include "gcore/rt/hip/kernels/attention_kernels.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static inline bool attn_align_trace_enabled() {
  const char *v = std::getenv("GRETA_TRACE_ATTN_ALIGN");
  if (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y'))
    return true;
  const char *v2 = std::getenv("GRETA_TRACE_ATTN_DECODE_VERIFY");
  return v2 && (v2[0] == '1' || v2[0] == 'y' || v2[0] == 'Y');
}

static inline void attn_log_align(const char *tag, const void *ptr,
                                  size_t align) {
  if (!ptr)
    return;
  if (reinterpret_cast<uintptr_t>(ptr) % align != 0) {
    std::fprintf(stderr,
                 "[ATTN_ALIGN] %s ptr=%p not %zub aligned\n", tag, ptr,
                 align);
  }
}

namespace gcore::rt::hip::kernels {

__device__ void rope_logic(float *x, uint32_t seq_len, uint32_t num_heads,
                           uint32_t head_dim, float base, uint32_t pos_offset) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t half_dim = head_dim / 2;
  uint32_t total_pairs = seq_len * num_heads * half_dim;
  
  if (idx >= total_pairs) return;

  uint32_t pair_idx = idx % half_dim;
  uint32_t head_idx = (idx / half_dim) % num_heads;
  uint32_t s_batch = idx / (half_dim * num_heads);
  uint32_t pos = pos_offset + s_batch;

  float theta = (float)pos * powf(base, -2.0f * (float)pair_idx / (float)head_dim);
  float cos_val = cosf(theta);
  float sin_val = sinf(theta);

  uint32_t base_idx = s_batch * (num_heads * head_dim) + head_idx * head_dim + pair_idx;
  
  float v0 = x[base_idx];
  float v1 = x[base_idx + half_dim];

  x[base_idx] = v0 * cos_val - v1 * sin_val;
  x[base_idx + half_dim] = v0 * sin_val + v1 * cos_val;
}

__global__ void rope_kernel(float *x, uint32_t seq_len, uint32_t num_heads, 
                            uint32_t head_dim, float base, uint32_t pos_offset) {
  rope_logic(x, seq_len, num_heads, head_dim, base, pos_offset);
}

__global__ void rope_kernel_p(float *x, uint32_t seq_len, uint32_t num_heads,
                               uint32_t head_dim, float base,
                               const uint32_t *pos_ptr) {
  rope_logic(x, seq_len, num_heads, head_dim, base, *pos_ptr);
}

void launch_rope(hipStream_t stream, float *x, uint32_t seq_len, 
                 uint32_t num_heads, uint32_t head_dim, float base, uint32_t pos_offset) {
  uint32_t half_dim = head_dim / 2;
  uint32_t total_pairs = seq_len * num_heads * half_dim;
  int block_size = 256;
  int grid_size = (total_pairs + block_size - 1) / block_size;
  
  rope_kernel<<<grid_size, block_size, 0, stream>>>(x, seq_len, num_heads, head_dim, base, pos_offset);
}

void launch_rope(hipStream_t stream, float *x, uint32_t seq_len,
                 uint32_t num_heads, uint32_t head_dim, float base,
                 const uint32_t *d_pos) {
  uint32_t half_dim = head_dim / 2;
  uint32_t total_pairs = seq_len * num_heads * half_dim;
  int block_size = 256;
  int grid_size = (total_pairs + block_size - 1) / block_size;

  rope_kernel_p<<<grid_size, block_size, 0, stream>>>(x, seq_len, num_heads, head_dim, base, d_pos);
}

__global__ void causal_mask_kernel(float *data, uint32_t seq_len, float mask_val) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len) return;

  uint32_t base = row * seq_len;
  for (uint32_t col = row + 1; col < seq_len; ++col) {
    data[base + col] = mask_val;
  }
}

void launch_causal_mask(hipStream_t stream, float *data, uint32_t seq_len, float mask_val) {
  int block_size = 256;
  int grid_size = (seq_len + block_size - 1) / block_size;
  
  causal_mask_kernel<<<grid_size, block_size, 0, stream>>>(data, seq_len, mask_val);
}

__device__ void kv_update_logic(float *cache_k, float *cache_v, const float *new_k,
                                const float *new_v, uint32_t pos,
                                uint32_t max_seq_len, uint32_t num_heads,
                                uint32_t head_dim) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = num_heads * head_dim;

  if (idx >= total) return;

  uint32_t head_idx = idx / head_dim;
  uint32_t feat_idx = idx % head_dim;

  uint32_t src_off = head_idx * head_dim + feat_idx;
  uint32_t dst_off = head_idx * (max_seq_len * head_dim) + pos * head_dim + feat_idx;

  cache_k[dst_off] = new_k[src_off];
  cache_v[dst_off] = new_v[src_off];
}

__global__ void kv_update_kernel(float *cache_k, float *cache_v, const float *new_k,
                               const float *new_v, uint32_t pos,
                               uint32_t max_seq_len, uint32_t num_heads,
                               uint32_t head_dim) {
  kv_update_logic(cache_k, cache_v, new_k, new_v, pos, max_seq_len, num_heads, head_dim);
}

__global__ void kv_update_kernel_p(float *cache_k, float *cache_v, const float *new_k,
                                 const float *new_v, const uint32_t *pos_ptr,
                                 uint32_t max_seq_len, uint32_t num_heads,
                                 uint32_t head_dim) {
  kv_update_logic(cache_k, cache_v, new_k, new_v, *pos_ptr, max_seq_len, num_heads, head_dim);
}

void launch_kv_update(hipStream_t stream, float *cache_k, float *cache_v,
                      const float *new_k, const float *new_v, uint32_t pos,
                      uint32_t max_seq_len, uint32_t num_heads,
                      uint32_t head_dim) {
  uint32_t total = num_heads * head_dim;
  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;

  kv_update_kernel<<<blocks, threads, 0, stream>>>(
      cache_k, cache_v, new_k, new_v, pos, max_seq_len, num_heads, head_dim);
}

void launch_kv_update(hipStream_t stream, float *cache_k, float *cache_v,
                      const float *new_k, const float *new_v, 
                      const uint32_t *d_pos, uint32_t max_seq_len,
                      uint32_t num_heads, uint32_t head_dim) {
  uint32_t total = num_heads * head_dim;
  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;

  kv_update_kernel_p<<<blocks, threads, 0, stream>>>(
      cache_k, cache_v, new_k, new_v, d_pos, max_seq_len, num_heads, head_dim);
}

#define FLASH_BLOCK_SIZE 32
#define FLASH_HEAD_DIM 128

__device__ __forceinline__ float round_fp16_device(float x) {
  return __half2float(__float2half_rn(x));
}

__device__ void flash_attention_decode_logic(
    const float *__restrict__ Q,      // [num_heads, head_dim]
    const float *__restrict__ K,      // [num_heads_kv, max_seq, head_dim]
    const float *__restrict__ V,      // [num_heads_kv, max_seq, head_dim]
    float *__restrict__ O,            // [num_heads, head_dim]
    uint32_t num_heads, uint32_t num_heads_kv, uint32_t seq_len,
    uint32_t max_seq_len, uint32_t head_dim, float scale, int accum_mode) {

  uint32_t head = blockIdx.x;
  uint32_t tid = threadIdx.x;
  
  if (head >= num_heads) return;
  if (num_heads_kv == 0) return;
  uint32_t group = num_heads / num_heads_kv;
  uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (kv_head >= num_heads_kv) return;
  
  __shared__ float s_score[FLASH_BLOCK_SIZE];
  __shared__ float s_max;
  __shared__ float s_sum;
  __shared__ float s_output[FLASH_HEAD_DIM];
  
  if (tid < head_dim) s_output[tid] = 0.0f;
  if (tid == 0) { s_max = -INFINITY; s_sum = 0.0f; }
  __syncthreads();

  const float *q_ptr = Q + head * head_dim;
  const float *k_ptr = K + kv_head * max_seq_len * head_dim;
  const float *v_ptr = V + kv_head * max_seq_len * head_dim;
  float *o_ptr = O + head * head_dim;

  for (uint32_t k_start = 0; k_start < seq_len; k_start += FLASH_BLOCK_SIZE) {
    uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, seq_len);
    uint32_t block_len = k_end - k_start;
    
    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = q_ptr[d] * k_ptr[k_idx * head_dim + d];
        if (accum_mode == 1) {
          prod = round_fp16_device(prod);
          dot = round_fp16_device(dot + prod);
        } else {
          dot = fmaf(q_ptr[d], k_ptr[k_idx * head_dim + d], dot);
        }
      }
      float s = dot * scale;
      s_score[tid] = (accum_mode == 1) ? round_fp16_device(s) : s;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = -INFINITY;
    }
    __syncthreads();
    
    for (uint32_t stride = FLASH_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_score[tid] = fmaxf(s_score[tid], s_score[tid + stride]);
      __syncthreads();
    }
    float new_max = fmaxf(s_max, s_score[0]);
    __syncthreads();
    
    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = q_ptr[d] * k_ptr[k_idx * head_dim + d];
        if (accum_mode == 1) {
          prod = round_fp16_device(prod);
          dot = round_fp16_device(dot + prod);
        } else {
          dot = fmaf(q_ptr[d], k_ptr[k_idx * head_dim + d], dot);
        }
      }
      float s = expf(dot * scale - new_max);
      s_score[tid] = (accum_mode == 1) ? round_fp16_device(s) : s;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = 0.0f;
    }
    __syncthreads();
    
    for (uint32_t stride = FLASH_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) s_score[tid] += s_score[tid + stride];
      __syncthreads();
    }
    
    float correction = expf(s_max - new_max);
    if (tid == 0) {
      s_sum = correction * s_sum + s_score[0];
    }
    __syncthreads();
    
    if (tid == 0) s_max = new_max;
    __syncthreads();

    if (tid < block_len) {
      uint32_t k_idx = k_start + tid;
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        float prod = q_ptr[d] * k_ptr[k_idx * head_dim + d];
        if (accum_mode == 1) {
          prod = round_fp16_device(prod);
          dot = round_fp16_device(dot + prod);
        } else {
          dot = fmaf(q_ptr[d], k_ptr[k_idx * head_dim + d], dot);
        }
      }
      float s = expf(dot * scale - s_max);
      s_score[tid] = (accum_mode == 1) ? round_fp16_device(s) : s;
    } else if (tid < FLASH_BLOCK_SIZE) {
      s_score[tid] = 0.0f;
    }
    __syncthreads();
    
    if (tid < head_dim) {
      float o_update = correction * s_output[tid];
      for (uint32_t ki = 0; ki < block_len; ++ki) {
        float term = s_score[ki] * v_ptr[(k_start + ki) * head_dim + tid];
        if (accum_mode == 1) {
          term = round_fp16_device(term);
          o_update = round_fp16_device(o_update + term);
        } else {
          o_update = fmaf(s_score[ki], v_ptr[(k_start + ki) * head_dim + tid],
                          o_update);
        }
      }
      s_output[tid] = o_update;
    }
    __syncthreads();
  }

  if (tid < head_dim) o_ptr[tid] = s_output[tid] / s_sum;
}

__global__ void flash_attention_decode_kernel(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    float *__restrict__ O, uint32_t num_heads, uint32_t num_heads_kv,
    uint32_t seq_len, uint32_t max_seq_len, uint32_t head_dim, float scale,
    int accum_mode) {
  flash_attention_decode_logic(Q, K, V, O, num_heads, num_heads_kv, seq_len,
                               max_seq_len, head_dim, scale, accum_mode);
}

__global__ void flash_attention_decode_kernel_p(
    const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
    float *__restrict__ O, uint32_t num_heads, uint32_t num_heads_kv,
    const uint32_t *pos_ptr, uint32_t max_seq_len, uint32_t head_dim,
    float scale, int accum_mode) {
  flash_attention_decode_logic(Q, K, V, O, num_heads, num_heads_kv,
                               *pos_ptr + 1,
                               max_seq_len, head_dim, scale, accum_mode);
}

void launch_flash_attention_decode(hipStream_t stream, 
                                   const float *Q, const float *K, const float *V,
                                   float *O, uint32_t num_heads,
                                   uint32_t num_heads_kv, uint32_t seq_len,
                                   uint32_t max_seq_len, uint32_t head_dim,
                                   float scale, int accum_mode) {
  if (head_dim > FLASH_HEAD_DIM) {
    if (attn_align_trace_enabled()) {
      std::fprintf(stderr,
                   "[ATTN_DECODE] head_dim=%u exceeds FLASH_HEAD_DIM=%u\n",
                   head_dim, FLASH_HEAD_DIM);
    }
    return;
  }
  if (attn_align_trace_enabled()) {
    attn_log_align("Q", Q, 16);
    attn_log_align("K", K, 16);
    attn_log_align("V", V, 16);
    attn_log_align("O", O, 16);
    attn_log_align("Q", Q, 32);
    attn_log_align("K", K, 32);
    attn_log_align("V", V, 32);
    attn_log_align("O", O, 32);
  }
  dim3 grid(num_heads);
  dim3 block(max(FLASH_BLOCK_SIZE, (int)head_dim));
  flash_attention_decode_kernel<<<grid, block, 0, stream>>>(
      Q, K, V, O, num_heads, num_heads_kv, seq_len, max_seq_len, head_dim,
      scale, accum_mode);
}

void launch_flash_attention_decode(hipStream_t stream, 
                                   const float *Q, const float *K, const float *V,
                                   float *O, uint32_t num_heads,
                                   uint32_t num_heads_kv,
                                   const uint32_t *d_pos,
                                   uint32_t max_seq_len, uint32_t head_dim,
                                   float scale, int accum_mode) {
  if (head_dim > FLASH_HEAD_DIM) {
    if (attn_align_trace_enabled()) {
      std::fprintf(stderr,
                   "[ATTN_DECODE] head_dim=%u exceeds FLASH_HEAD_DIM=%u\n",
                   head_dim, FLASH_HEAD_DIM);
    }
    return;
  }
  if (attn_align_trace_enabled()) {
    attn_log_align("Q", Q, 16);
    attn_log_align("K", K, 16);
    attn_log_align("V", V, 16);
    attn_log_align("O", O, 16);
    attn_log_align("Q", Q, 32);
    attn_log_align("K", K, 32);
    attn_log_align("V", V, 32);
    attn_log_align("O", O, 32);
  }
  dim3 grid(num_heads);
  dim3 block(max(FLASH_BLOCK_SIZE, (int)head_dim));
  flash_attention_decode_kernel_p<<<grid, block, 0, stream>>>(
      Q, K, V, O, num_heads, num_heads_kv, d_pos, max_seq_len, head_dim,
      scale, accum_mode);
}

__global__ void attn_softmax_trace_kernel(
    const float *__restrict__ Q, const float *__restrict__ K_cache,
    uint32_t num_heads, uint32_t num_heads_kv, uint32_t head_dim,
    uint32_t seq_len, uint32_t max_seq_len, uint32_t head,
    uint32_t window_start, uint32_t window_len, float scale,
    float *__restrict__ qk_out, float *__restrict__ softmax_out,
    float *__restrict__ stats_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (!Q || !K_cache || !qk_out || !softmax_out || !stats_out)
    return;
  if (num_heads == 0 || head_dim == 0 || seq_len == 0)
    return;
  if (num_heads_kv == 0)
    return;

  const uint32_t group = num_heads / num_heads_kv;
  const uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (head >= num_heads || kv_head >= num_heads_kv)
    return;

  const float *q_ptr = Q + head * head_dim;
  const float *k_ptr = K_cache + kv_head * max_seq_len * head_dim;

  float max_qk = -INFINITY;
  float second_qk = -INFINITY;
  for (uint32_t t = 0; t < seq_len; ++t) {
    const float *k_t = k_ptr + t * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot = fmaf(q_ptr[d], k_t[d], dot);
    }
    float qk = dot * scale;
    if (qk > max_qk) {
      second_qk = max_qk;
      max_qk = qk;
    } else if (qk > second_qk) {
      second_qk = qk;
    }
  }

  double sumexp = 0.0;
  double sumexp_log = 0.0;
  for (uint32_t t = 0; t < seq_len; ++t) {
    const float *k_t = k_ptr + t * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot = fmaf(q_ptr[d], k_t[d], dot);
    }
    float qk = dot * scale;
    float e = expf(qk - max_qk);
    sumexp += static_cast<double>(e);
    sumexp_log += static_cast<double>(e) * static_cast<double>(qk - max_qk);
  }

  float sumexp_f = static_cast<float>(sumexp);
  float inv_sum = (sumexp > 0.0) ? static_cast<float>(1.0 / sumexp) : 0.0f;
  float top1_prob = inv_sum;
  float top2_prob =
      (sumexp > 0.0 && isfinite(second_qk))
          ? expf(second_qk - max_qk) * inv_sum
          : 0.0f;
  float entropy =
      (sumexp > 0.0)
          ? static_cast<float>(log(sumexp) - sumexp_log / sumexp)
          : 0.0f;

  for (uint32_t i = 0; i < window_len; ++i) {
    uint32_t t = window_start + i;
    if (t >= seq_len) {
      qk_out[i] = 0.0f;
      softmax_out[i] = 0.0f;
      continue;
    }
    const float *k_t = k_ptr + t * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot = fmaf(q_ptr[d], k_t[d], dot);
    }
    float qk = dot * scale;
    qk_out[i] = qk;
    softmax_out[i] = (sumexp > 0.0) ? expf(qk - max_qk) * inv_sum : 0.0f;
  }

  stats_out[0] = max_qk;
  stats_out[1] = sumexp_f;
  stats_out[2] = top1_prob;
  stats_out[3] = top2_prob;
  stats_out[4] = entropy;
}

void launch_attn_softmax_trace(hipStream_t stream, const float *Q,
                               const float *K_cache, uint32_t num_heads,
                               uint32_t num_heads_kv, uint32_t head_dim,
                               uint32_t seq_len, uint32_t max_seq_len,
                               uint32_t head, uint32_t window_start,
                               uint32_t window_len, float scale,
                               float *qk_out, float *softmax_out,
                               float *stats_out) {
  dim3 grid(1);
  dim3 block(1);
  attn_softmax_trace_kernel<<<grid, block, 0, stream>>>(
      Q, K_cache, num_heads, num_heads_kv, head_dim, seq_len, max_seq_len, head,
      window_start, window_len, scale, qk_out, softmax_out, stats_out);
}

__global__ void attn_vacc_vsample_kernel(
    const float *__restrict__ V_cache, uint32_t num_heads,
    uint32_t num_heads_kv, uint32_t head_dim, uint32_t seq_len,
    uint32_t max_seq_len, uint32_t head, uint32_t window_start,
    uint32_t window_len, uint32_t dims_sample, float *__restrict__ v_row_out,
    float *__restrict__ v_col_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (!V_cache || !v_row_out || !v_col_out)
    return;
  if (num_heads == 0 || num_heads_kv == 0 || head_dim == 0 || seq_len == 0)
    return;
  if (head >= num_heads)
    return;
  if (dims_sample == 0 || window_len == 0)
    return;

  const uint32_t group = num_heads / num_heads_kv;
  const uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (kv_head >= num_heads_kv)
    return;

  const float *v_ptr = V_cache + kv_head * max_seq_len * head_dim;

  for (uint32_t i = 0; i < window_len; ++i) {
    uint32_t t = window_start + i;
    for (uint32_t d = 0; d < dims_sample; ++d) {
      float row_val = 0.0f;
      float col_val = 0.0f;
      if (t < seq_len) {
        row_val = v_ptr[t * head_dim + d];
        col_val = v_ptr[d * max_seq_len + t];
      }
      v_row_out[i * dims_sample + d] = row_val;
      v_col_out[i * dims_sample + d] = col_val;
    }
  }
}

void launch_attn_vacc_vsample(hipStream_t stream, const float *V_cache,
                              uint32_t num_heads, uint32_t num_heads_kv,
                              uint32_t head_dim, uint32_t seq_len,
                              uint32_t max_seq_len, uint32_t head,
                              uint32_t window_start, uint32_t window_len,
                              uint32_t dims_sample, float *v_row_out,
                              float *v_col_out) {
  dim3 grid(1);
  dim3 block(1);
  attn_vacc_vsample_kernel<<<grid, block, 0, stream>>>(
      V_cache, num_heads, num_heads_kv, head_dim, seq_len, max_seq_len, head,
      window_start, window_len, dims_sample, v_row_out, v_col_out);
}



#ifdef GRETA_PREFILL_Q_LDS
// [V3] Q-in-LDS + Segmented Output to eliminate scratch spilling
// ... existing V3 code ...
// (I will need to verify if I can just modify the existing block or duplicate)
// To keep minimal diff, I will use #if defined(GRETA_PREFILL_Q_LDS) || defined(GRETA_PREFILL_Q_LDS_V4)
// and inside use #ifdef GRETA_PREFILL_Q_LDS_V4 for the V logic.
#define FLASH_BLOCK_SIZE 32
#define FLASH_HEAD_DIM 128

__global__ void flash_attention_prefill_kernel(
    const float *__restrict__ Q,      // [seq_len, num_heads, head_dim]
    const float *__restrict__ K,      // [seq_len, num_heads_kv, head_dim]
    const float *__restrict__ V,      // [seq_len, num_heads_kv, head_dim]
    float *__restrict__ O,            // [seq_len, num_heads, head_dim]
    uint32_t seq_len, uint32_t num_heads, uint32_t num_heads_kv,
    uint32_t head_dim, float scale, bool causal) {

  // LDS Layout
  __shared__ float sQ[FLASH_BLOCK_SIZE][FLASH_HEAD_DIM];
  __shared__ float sK[FLASH_BLOCK_SIZE][FLASH_HEAD_DIM];
  
#ifdef GRETA_PREFILL_Q_LDS_V4
  // V4: SEG=32 to reduce K-reloads (128/32 = 4 passes)
  constexpr int SEG = 32;
#else
  // V3: SEG=16 (128/16 = 8 passes)
  constexpr int SEG = 16;
#endif
  __shared__ float sV[FLASH_BLOCK_SIZE][SEG];

  uint32_t head = blockIdx.x;
  // ... (rest of preamble is same) ...
  // (I will use multi_replace to be precise)

  uint32_t head = blockIdx.x;
  uint32_t q_block = blockIdx.y;
  uint32_t tid = threadIdx.x;
  
  if (num_heads_kv == 0) return;
  uint32_t group = num_heads / num_heads_kv;
  uint32_t kv_head = (group > 0) ? (head / group) : 0;
  
  uint32_t q_start = q_block * FLASH_BLOCK_SIZE;
  if (q_start >= seq_len) return;
  
  uint32_t q_idx = q_start + tid;
  uint32_t q_end = min(q_start + FLASH_BLOCK_SIZE, seq_len);
  bool valid_q = (q_idx < q_end);
  
  // 1. Cooperative Load Q into LDS
  {
      uint32_t total_elements = FLASH_BLOCK_SIZE * 128;
      for (uint32_t i = tid; i < total_elements; i += FLASH_BLOCK_SIZE) {
          uint32_t row = i / 128;
          uint32_t col = i % 128;
          uint32_t abs_q = q_start + row;
          if (abs_q < seq_len) {
             sQ[row][col] = Q[abs_q * num_heads * 128 + head * 128 + col];
          } else {
             sQ[row][col] = 0.0f;
          }
      }
  }
  __syncthreads();
  
  // Outer Loop: Segments of Output
  for (uint32_t d_start = 0; d_start < 128; d_start += SEG) {
      if (d_start >= head_dim) break;
      
      // Scalar Accumulators
      float o0=0, o1=0, o2=0, o3=0, o4=0, o5=0, o6=0, o7=0;
      float o8=0, o9=0, o10=0, o11=0, o12=0, o13=0, o14=0, o15=0;
#ifdef GRETA_PREFILL_Q_LDS_V4
      float o16=0, o17=0, o18=0, o19=0, o20=0, o21=0, o22=0, o23=0;
      float o24=0, o25=0, o26=0, o27=0, o28=0, o29=0, o30=0, o31=0;
#endif
      
      float m = -INFINITY;
      float l = 0.0f;
      
      uint32_t block_max_k = causal ? q_end : seq_len;
      
      // Inner Loop: K/V blocks
      for (uint32_t k_start = 0; k_start < block_max_k; k_start += FLASH_BLOCK_SIZE) {
          __syncthreads(); 
          
          // Load K (Full)
          uint32_t k_elems = FLASH_BLOCK_SIZE * 128;
          for (uint32_t i = tid; i < k_elems; i += FLASH_BLOCK_SIZE) {
              uint32_t row = i / 128;
              uint32_t col = i % 128;
              uint32_t abs_k = k_start + row;
              uint32_t k_off = abs_k * num_heads_kv * 128 + kv_head * 128 + col;
              if (abs_k < seq_len) sK[row][col] = K[k_off];
              else sK[row][col] = 0.0f;
          }
          
          // Load V (Segment)
          uint32_t v_elems = FLASH_BLOCK_SIZE * SEG;
          for (uint32_t i = tid; i < v_elems; i += FLASH_BLOCK_SIZE) {
              uint32_t row = i / SEG;
              uint32_t col = i % SEG;
              uint32_t abs_k = k_start + row;
              uint32_t k_off = abs_k * num_heads_kv * 128 + kv_head * 128 + (d_start + col);
              if (abs_k < seq_len && (d_start + col) < head_dim) sV[row][col] = V[k_off];
              else sV[row][col] = 0.0f;
          }
          __syncthreads();
          
          if (valid_q) {
              uint32_t k_end_in_tile = min(FLASH_BLOCK_SIZE, (causal ? (q_idx + 1) : seq_len) - k_start);
              if (causal && k_start > q_idx) k_end_in_tile = 0;
              
              for (uint32_t k_inner = 0; k_inner < k_end_in_tile; ++k_inner) {
                  float dot = 0.0f;
                  #pragma unroll 16
                  for(int d=0; d<128; ++d) {
                      dot += sQ[tid][d] * sK[k_inner][d];
                  }
                  
                  float s = dot * scale;
                  float m_new = fmaxf(m, s);
                  float p = expf(s - m_new);
                  float correction = expf(m - m_new);
                  l = correction * l + p;
                   
                  // Update Accumulators
                  o0 = correction * o0 + p * sV[k_inner][0];
                  o1 = correction * o1 + p * sV[k_inner][1];
                  o2 = correction * o2 + p * sV[k_inner][2];
                  o3 = correction * o3 + p * sV[k_inner][3];
                  o4 = correction * o4 + p * sV[k_inner][4];
                  o5 = correction * o5 + p * sV[k_inner][5];
                  o6 = correction * o6 + p * sV[k_inner][6];
                  o7 = correction * o7 + p * sV[k_inner][7];
                  o8 = correction * o8 + p * sV[k_inner][8];
                  o9 = correction * o9 + p * sV[k_inner][9];
                  o10 = correction * o10 + p * sV[k_inner][10];
                  o11 = correction * o11 + p * sV[k_inner][11];
                  o12 = correction * o12 + p * sV[k_inner][12];
                  o13 = correction * o13 + p * sV[k_inner][13];
                  o14 = correction * o14 + p * sV[k_inner][14];
                  o15 = correction * o15 + p * sV[k_inner][15];
#ifdef GRETA_PREFILL_Q_LDS_V4
                  o16 = correction * o16 + p * sV[k_inner][16];
                  o17 = correction * o17 + p * sV[k_inner][17];
                  o18 = correction * o18 + p * sV[k_inner][18];
                  o19 = correction * o19 + p * sV[k_inner][19];
                  o20 = correction * o20 + p * sV[k_inner][20];
                  o21 = correction * o21 + p * sV[k_inner][21];
                  o22 = correction * o22 + p * sV[k_inner][22];
                  o23 = correction * o23 + p * sV[k_inner][23];
                  o24 = correction * o24 + p * sV[k_inner][24];
                  o25 = correction * o25 + p * sV[k_inner][25];
                  o26 = correction * o26 + p * sV[k_inner][26];
                  o27 = correction * o27 + p * sV[k_inner][27];
                  o28 = correction * o28 + p * sV[k_inner][28];
                  o29 = correction * o29 + p * sV[k_inner][29];
                  o30 = correction * o30 + p * sV[k_inner][30];
                  o31 = correction * o31 + p * sV[k_inner][31];
#endif
                  m = m_new;
              }
          }
      } // End K Loop
      
      // Store Output Segment
      if (valid_q && l > 0.0f) {
           uint32_t base = q_idx * num_heads * 128 + head * 128 + d_start;
           float inv_l = 1.0f / l;
           if (d_start + 0 < head_dim) O[base + 0] = o0 * inv_l;
           if (d_start + 1 < head_dim) O[base + 1] = o1 * inv_l;
           if (d_start + 2 < head_dim) O[base + 2] = o2 * inv_l;
           if (d_start + 3 < head_dim) O[base + 3] = o3 * inv_l;
           if (d_start + 4 < head_dim) O[base + 4] = o4 * inv_l;
           if (d_start + 5 < head_dim) O[base + 5] = o5 * inv_l;
           if (d_start + 6 < head_dim) O[base + 6] = o6 * inv_l;
           if (d_start + 7 < head_dim) O[base + 7] = o7 * inv_l;
           if (d_start + 8 < head_dim) O[base + 8] = o8 * inv_l;
           if (d_start + 9 < head_dim) O[base + 9] = o9 * inv_l;
           if (d_start + 10 < head_dim) O[base + 10] = o10 * inv_l;
           if (d_start + 11 < head_dim) O[base + 11] = o11 * inv_l;
           if (d_start + 12 < head_dim) O[base + 12] = o12 * inv_l;
           if (d_start + 13 < head_dim) O[base + 13] = o13 * inv_l;
           if (d_start + 14 < head_dim) O[base + 14] = o14 * inv_l;
           if (d_start + 15 < head_dim) O[base + 15] = o15 * inv_l;
#ifdef GRETA_PREFILL_Q_LDS_V4
           if (d_start + 16 < head_dim) O[base + 16] = o16 * inv_l;
           if (d_start + 17 < head_dim) O[base + 17] = o17 * inv_l;
           if (d_start + 18 < head_dim) O[base + 18] = o18 * inv_l;
           if (d_start + 19 < head_dim) O[base + 19] = o19 * inv_l;
           if (d_start + 20 < head_dim) O[base + 20] = o20 * inv_l;
           if (d_start + 21 < head_dim) O[base + 21] = o21 * inv_l;
           if (d_start + 22 < head_dim) O[base + 22] = o22 * inv_l;
           if (d_start + 23 < head_dim) O[base + 23] = o23 * inv_l;
           if (d_start + 24 < head_dim) O[base + 24] = o24 * inv_l;
           if (d_start + 25 < head_dim) O[base + 25] = o25 * inv_l;
           if (d_start + 26 < head_dim) O[base + 26] = o26 * inv_l;
           if (d_start + 27 < head_dim) O[base + 27] = o27 * inv_l;
           if (d_start + 28 < head_dim) O[base + 28] = o28 * inv_l;
           if (d_start + 29 < head_dim) O[base + 29] = o29 * inv_l;
           if (d_start + 30 < head_dim) O[base + 30] = o30 * inv_l;
           if (d_start + 31 < head_dim) O[base + 31] = o31 * inv_l;
#endif
      }
  } // End Segment Loop
}
#elif defined(GRETA_PREFILL_SEGMENTED)
// V1: Full LDS Tiling (Regressions due to Spill)
__global__ void flash_attention_prefill_kernel(
    const float *__restrict__ Q,      // [seq_len, num_heads, head_dim]
    const float *__restrict__ K,      // [seq_len, num_heads_kv, head_dim]
    const float *__restrict__ V,      // [seq_len, num_heads_kv, head_dim]
    float *__restrict__ O,            // [seq_len, num_heads, head_dim]
    uint32_t seq_len, uint32_t num_heads, uint32_t num_heads_kv,
    uint32_t head_dim, float scale, bool causal) {

  __shared__ float sK[FLASH_BLOCK_SIZE][FLASH_HEAD_DIM];
  __shared__ float sV[FLASH_BLOCK_SIZE][FLASH_HEAD_DIM];

  uint32_t head = blockIdx.x;
  uint32_t q_block = blockIdx.y;
  uint32_t tid = threadIdx.x;
  if (num_heads_kv == 0) return;
  uint32_t group = num_heads / num_heads_kv;
  uint32_t kv_head = (group > 0) ? (head / group) : 0;
  
  uint32_t q_start = q_block * FLASH_BLOCK_SIZE;
  if (q_start >= seq_len) return;
  uint32_t q_end = min(q_start + FLASH_BLOCK_SIZE, seq_len);
  
  uint32_t q_idx = q_start + tid;
  bool valid_q = (q_idx < q_end);
  
  float m = -INFINITY;
  float l = 0.0f;
  float o[FLASH_HEAD_DIM];
  for (uint32_t d = 0; d < FLASH_HEAD_DIM; ++d) o[d] = 0.0f;
  
  // Stage Q into registers
  float q_local[FLASH_HEAD_DIM];
  for (uint32_t d = 0; d < FLASH_HEAD_DIM; ++d) q_local[d] = 0.0f;
  
  if (valid_q) {
    for (uint32_t d = 0; d < head_dim; ++d) {
      q_local[d] = Q[q_idx * num_heads * head_dim + head * head_dim + d];
    }
  }

  uint32_t block_max_k = causal ? q_end : seq_len;

  for (uint32_t k_start = 0; k_start < block_max_k; k_start += FLASH_BLOCK_SIZE) {
    // Cooperative load K & V into LDS
    uint32_t total_elements = FLASH_BLOCK_SIZE * head_dim;
    for (uint32_t i = tid; i < total_elements; i += FLASH_BLOCK_SIZE) {
        uint32_t row = i / head_dim;
        uint32_t col = i % head_dim;
        uint32_t abs_k = k_start + row;
        uint32_t k_off = abs_k * num_heads_kv * head_dim + kv_head * head_dim + col;
        if (abs_k < seq_len) {
            sK[row][col] = K[k_off];
            sV[row][col] = V[k_off];
        } else {
            sK[row][col] = 0.0f;
            sV[row][col] = 0.0f;
        }
    }
    __syncthreads();

    if (valid_q) {
        uint32_t k_end_in_tile = min(FLASH_BLOCK_SIZE, (causal ? (q_idx + 1) : seq_len) - k_start);
        if (causal && k_start > q_idx) k_end_in_tile = 0;

        for (uint32_t k_inner = 0; k_inner < k_end_in_tile; ++k_inner) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += q_local[d] * sK[k_inner][d];
            }
            float s = dot * scale;
            
            float m_new = fmaxf(m, s);
            float p = expf(s - m_new);
            float correction = expf(m - m_new);
            l = correction * l + p;
            
            for (uint32_t d = 0; d < head_dim; ++d) {
                o[d] = correction * o[d] + p * sV[k_inner][d];
            }
            m = m_new;
        }
    }
    __syncthreads();
  }
  
  if (valid_q && l > 0) {
    for (uint32_t d = 0; d < head_dim; ++d) {
      uint32_t o_off = q_idx * num_heads * head_dim + head * head_dim + d;
      O[o_off] = o[d] / l;
    }
  }
}
#else
__global__ void flash_attention_prefill_kernel(
    const float *__restrict__ Q,      // [seq_len, num_heads, head_dim]
    const float *__restrict__ K,      // [seq_len, num_heads_kv, head_dim]
    const float *__restrict__ V,      // [seq_len, num_heads_kv, head_dim]
    float *__restrict__ O,            // [seq_len, num_heads, head_dim]
    uint32_t seq_len, uint32_t num_heads, uint32_t num_heads_kv,
    uint32_t head_dim, float scale, bool causal) {

  uint32_t head = blockIdx.x;
  uint32_t q_block = blockIdx.y;
  uint32_t tid = threadIdx.x;
  if (num_heads_kv == 0) return;
  uint32_t group = num_heads / num_heads_kv;
  uint32_t kv_head = (group > 0) ? (head / group) : 0;
  if (kv_head >= num_heads_kv) return;
  
  uint32_t q_start = q_block * FLASH_BLOCK_SIZE;
  if (q_start >= seq_len) return;
  uint32_t q_end = min(q_start + FLASH_BLOCK_SIZE, seq_len);
  
  uint32_t q_idx = q_start + tid;
  bool valid_q = (q_idx < q_end);
  
  float m = -INFINITY;
  float l = 0.0f;
  float o[FLASH_HEAD_DIM] = {0}; // Fixed buffer size
  
  uint32_t max_k = causal ? (q_idx + 1) : seq_len;
  
  for (uint32_t k_start = 0; k_start < max_k; k_start += FLASH_BLOCK_SIZE) {
    uint32_t k_end = min(k_start + FLASH_BLOCK_SIZE, max_k);
    
    for (uint32_t k_idx = k_start; k_idx < k_end; ++k_idx) {
      if (!valid_q) continue;
      
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_dim; ++d) {
        uint32_t q_off = q_idx * num_heads * head_dim + head * head_dim + d;
        uint32_t k_off =
            k_idx * num_heads_kv * head_dim + kv_head * head_dim + d;
        dot += Q[q_off] * K[k_off];
      }
      float s = dot * scale;
      
      float m_new = fmaxf(m, s);
      float p = expf(s - m_new);
      float correction = expf(m - m_new);
      l = correction * l + p;
      
      for (uint32_t d = 0; d < head_dim; ++d) {
        uint32_t v_off =
            k_idx * num_heads_kv * head_dim + kv_head * head_dim + d;
        o[d] = correction * o[d] + p * V[v_off];
      }
      m = m_new;
    }
  }
  
  if (valid_q) {
    for (uint32_t d = 0; d < head_dim; ++d) {
      uint32_t o_off = q_idx * num_heads * head_dim + head * head_dim + d;
      O[o_off] = o[d] / l;
    }
  }
}
#endif

void launch_flash_attention_prefill(hipStream_t stream,
                                    const float *Q, const float *K, const float *V,
                                    float *O, uint32_t seq_len, uint32_t num_heads,
                                    uint32_t num_heads_kv, uint32_t head_dim,
                                    float scale, bool causal) {
  dim3 grid(num_heads, (seq_len + FLASH_BLOCK_SIZE - 1) / FLASH_BLOCK_SIZE);
  dim3 block(FLASH_BLOCK_SIZE);
  flash_attention_prefill_kernel<<<grid, block, 0, stream>>>(
      Q, K, V, O, seq_len, num_heads, num_heads_kv, head_dim, scale, causal);
}

} // namespace gcore::rt::hip::kernels
