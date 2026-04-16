# llama.cpp Apple Silicon / Metal Deep Analysis

**Source**: https://github.com/ggml-org/llama.cpp (master branch, April 2026)
**Focus**: Metal backend, GGML graph, quantization, attention, KV cache, and all Apple Silicon optimizations

---

## 1. Metal Backend (`ggml/src/ggml-metal/ggml-metal.metal`)

### 1.1 Architecture Overview

The Metal backend is a single massive `.metal` shader file (~10,000+ lines) containing all compute kernels. It uses:

- **SIMD group operations** (`simd_sum`, `simd_max`, `simd_prefix_inclusive_sum`) for warp-level reductions
- **SIMD group matrix operations** (`simdgroup_load`, `simdgroup_multiply_accumulate`, `simdgroup_store`) for 8x8 block matrix math
- **Metal Performance Primitives (MPP)** via `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` — a newer path using `tensor<T>` and `mpp::tensor_ops::matmul2d` for matrix multiply
- **Threadgroup memory** for shared scratchpad data between SIMD groups
- **Function constants** for compile-time kernel specialization (avoiding runtime branching)

**Key constant**: `N_SIMDWIDTH = 32` (Apple GPU SIMD group size is always 32)

### 1.2 Complete Kernel Inventory (~95+ distinct kernels)

#### Core Computation Kernels
| Kernel | Purpose |
|--------|---------|
| `kernel_mul_mm` | Matrix-matrix multiplication (GEMM) — the main weight multiply kernel |
| `kernel_mul_mm_id` | GEMM with index mapping (for Mixture of Experts) |
| `kernel_mul_mv_*` | Matrix-vector multiplication (per quantization type) — the token generation kernel |
| `kernel_flash_attn_ext` | Flash attention (fused Q*K^T softmax * V) — multiple DK/DV specializations |
| `kernel_flash_attn_ext_pad` | Flash attention padding kernel |
| `kernel_flash_attn_ext_blk` | Flash attention block processing |
| `kernel_flash_attn_ext_vec` | Flash attention vector path |
| `kernel_flash_attn_ext_vec_reduce` | Flash attention vector reduction |

#### Normalization Kernels
| Kernel | Purpose |
|--------|---------|
| `kernel_rms_norm_fuse_impl` | Fused RMS norm (with optional bias/fuse) |
| `kernel_norm_fuse_impl` | Fused Layer norm |
| `kernel_group_norm_f32` | Group normalization |
| `kernel_l2_norm_impl` | L2 normalization |

#### Activation Kernels
| Kernel | Purpose |
|--------|---------|
| `kernel_swiglu_f32` | SwiGLU activation |
| `kernel_swiglu_oai_f32` | OpenAI SwiGLU variant |
| `kernel_geglu_f32` | GeGLU activation |
| `kernel_geglu_erf_f32` | GeGLU with erf approximation |
| `kernel_geglu_quick_f32` | Quick GeGLU |
| `kernel_reglu_f32` | ReGLU activation |

#### RoPE (Rotary Position Embedding) Kernels
| Kernel | Purpose |
|--------|---------|
| `kernel_rope_norm` | Standard RoPE (Llama 1 style) |
| `kernel_rope_neox` | Neox-style RoPE (Llama 2/3 style) |
| `kernel_rope_multi` | Multi-dimensional RoPE (for vision/multimodal) |
| `kernel_rope_vision` | Vision-specific RoPE with different rotation logic |

#### Softmax & Attention Helpers
| Kernel | Purpose |
|--------|---------|
| `kernel_soft_max` | Standard softmax with mask support |
| `kernel_soft_max_4` | Optimized 4-wide softmax |

#### Data Movement
| Kernel | Purpose |
|--------|---------|
| `kernel_cpy_t_t` | Copy tensor-to-tensor (type preserving) |
| `kernel_cpy_f32_q` | Copy + quantize F32 → quantized |
| `kernel_cpy_q_f32` | Copy + dequantize quantized → F32 |
| `kernel_get_rows_f` | Gather rows (float types) |
| `kernel_get_rows_q` | Gather rows (quantized types) |
| `kernel_set_rows_f` | Scatter rows (float types) |
| `kernel_set_rows_q32` | Scatter rows (quantized types) |
| `kernel_concat` | Tensor concatenation |
| `kernel_repeat` | Tensor repeat/tiling |
| `kernel_add_id` | Add with index mapping |
| `kernel_pad_f32` | Padding |
| `kernel_memset` | GPU memset |

#### Pooling & Convolution
| Kernel | Purpose |
|--------|---------|
| `kernel_pool_1d_avg_f32` | 1D average pooling |
| `kernel_pool_1d_max_f32` | 1D max pooling |
| `kernel_pool_2d_avg_f32` | 2D average pooling |
| `kernel_pool_2d_max_f32` | 2D max pooling |
| `kernel_conv_2d` | 2D convolution (template for float/half) |
| `kernel_conv_3d` | 3D convolution (template for float/half) |
| `kernel_conv_transpose_1d` | Transposed 1D convolution |
| `kernel_conv_transpose_2d` | Transposed 2D convolution |
| `kernel_im2col` | Image-to-column for convolution |

#### SSM / Mamba / RWKV Support
| Kernel | Purpose |
|--------|---------|
| `kernel_ssm_conv_f32_f32` | SSM convolution (base) |
| `kernel_ssm_conv_f32_f32_4` | SSM convolution (4-wide) |
| `kernel_ssm_conv_f32_f32_batched` | Batched SSM convolution |
| `kernel_ssm_scan_f32` | SSM selective scan |
| `kernel_rwkv_wkv6_f32` | RWKV v6 Time-Mixing |
| `kernel_rwkv_wkv7_f32` | RWKV v7 Time-Mixing |
| `kernel_gated_delta_net_impl` | Gated DeltaNet attention (2 variants by dimension) |

#### Other
| Kernel | Purpose |
|--------|---------|
| `kernel_argsort_f32_i32` | Argsort |
| `kernel_argsort_merge_f32_i32` | Argsort merge step |
| `kernel_argmax_f32` | Argmax |
| `kernel_solve_tri_f32` | Triangular solve |
| `kernel_count_equal` | Count equal elements |
| `kernel_tri` | Triangle mask |
| `kernel_cumsum_blk` / `kernel_cumsum_add` | Cumulative sum |
| `kernel_op_sum_f32` / `kernel_sum_rows_impl` | Sum reduction |
| `kernel_unary_impl` | Unary ops (22 types: sin/cos/log/gelu/silu/etc.) |
| `kernel_bin_fuse_impl` | Binary ops with fusion |
| `kernel_arange_f32` | Arange |
| `kernel_timestep_embedding_f32` | Timestep embeddings (for diffusion) |
| `kernel_upscale_nearest/bilinear/bicubic_f32` | Upscaling |
| `kernel_diag_f32` | Diagonal matrix |
| `kernel_opt_step_sgd_f32` | SGD optimizer step |
| `kernel_opt_step_adamw_f32` | AdamW optimizer step |
| `kernel_pad_reflect_1d_f32` | Reflect padding |

### 1.3 Matrix-Vector Multiply (Token Generation Path)

Per-quantization-type kernels, each with tuned SIMD group parameters:

| Kernel | N_R0 (rows/SIMD) | N_SG (SIMD groups) |
|--------|----------|------|
| `kernel_mul_mv_q1_0_f32` | 8 | 2 |
| `kernel_mul_mv_q4_0_f32` | 4 | 2 |
| `kernel_mul_mv_q4_1_f32` | 4 | 2 |
| `kernel_mul_mv_q5_0_f32` | 4 | 2 |
| `kernel_mul_mv_q5_1_f32` | 4 | 2 |
| `kernel_mul_mv_q8_0_f32` | 2 | 4 |
| `kernel_mul_mv_q2_K_f32` | 4 | 2 |
| `kernel_mul_mv_q3_K_f32` | 2 | 2 |
| `kernel_mul_mv_q4_K_f32` | 2 | 2 |
| `kernel_mul_mv_q5_K_f32` | 1 | 2 |
| `kernel_mul_mv_q6_K_f32` | 2 | 2 |
| `kernel_mul_mv_iq1_s_f32` | 4 | 2 |
| `kernel_mul_mv_iq1_m_f32` | 4 | 2 |
| `kernel_mul_mv_iq2_xxs_f32` | 4 | 2 |
| `kernel_mul_mv_iq2_xs_f32` | 4 | 2 |
| `kernel_mul_mv_iq2_s_f32` | 4 | 2 |
| `kernel_mul_mv_iq3_xxs_f32` | 4 | 2 |
| `kernel_mul_mv_iq3_s_f32` | 4 | 2 |
| `kernel_mul_mv_iq4_nl_f32` | 2 | 2 |
| `kernel_mul_mv_iq4_xs_f32` | 2 | 2 |
| `kernel_mul_mv_mxfp4_f32` | 2 | 2 |
| `kernel_mul_mv_t_t_*` | Various | Various |

**Implementation technique**: Each SIMD group processes N_R0 rows simultaneously. The weight data is dequantized on-the-fly using `block_q_n_dot_y()` helper functions that compute the dot product directly from quantized blocks without full dequantization. This saves memory bandwidth — the key bottleneck.

### 1.4 Matrix-Matrix Multiply (`kernel_mul_mm`) — Prefill Path

**Tile size**: 64×32 output blocks, 32-wide K dimension
- Uses 4 SIMD groups per threadgroup
- 8 `simdgroup_float8x8` accumulators → 8×8×8 = 512 output elements
- **Two paths**:
  1. **Classic SIMD group path**: `simdgroup_load` → `simdgroup_multiply_accumulate` → `simdgroup_store`
  2. **Metal Performance Primitives path** (`GGML_METAL_HAS_TENSOR`): Uses `tensor<T>` API + `mpp::tensor_ops::matmul2d` with `execution_simdgroups<4>` — Apple's cooperative matrix multiply
- **Dequantization fusion**: Weight blocks are dequantized directly into threadgroup memory before the SIMD group multiply
- **Padding avoidance**: Boundary checks prevent out-of-bounds reads

### 1.5 Flash Attention (`kernel_flash_attn_ext`)

**Massive kernel template** with compile-time specializations for every combination of:
- **K/V data types**: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- **Head dimensions**: DK/DV = 32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 192/128, 256, 320/256, 512, 576/512

**Total specialized kernels**: 15 head dims × 8 K/V types × 3 Q types = ~360 template instantiations (for f16/bf16/q4_0/q4_1/q5_0/q5_1/q8_0 + f32)

**Algorithm**:
1. Q is loaded into threadgroup memory
2. K/V are streamed in blocks of `C` (=64) cache items
3. For each K/V block:
   - `simdgroup_multiply_accumulate` computes QK^T (8×8 SIMD matrix blocks)
   - Online softmax: track running max and sum
   - Accumulate V contribution using corrected attention weights
4. Final output is rescaled and written

**Features via function constants**:
- `has_mask`: Attention mask (for causal/padding)
- `has_sinks`: Streaming attention sinks
- `has_bias`: Attention bias
- `has_scap`: Soft cap attention
- `has_kvpad`: KV padding
- `bc_mask`: Broadcast mask
- `nsg`: Number of SIMD groups (4 or 8)

**This is the single most important optimization for Apple Silicon** — it fuses the entire attention computation into a single kernel, eliminating intermediate memory reads/writes.

---

## 2. GGML Computation Graph

### 2.1 Graph Structure (`ggml/include/ggml.h`)

- Tensors are nodes, operations are edges in a DAG
- `ggml_new_graph()` creates a context for graph building
- `ggml_build_forward_expand()` traverses the DAG
- `ggml_graph_compute()` executes the graph

### 2.2 Backend Dispatch (`ggml/src/ggml-backend.cpp`)

- Multi-backend architecture: Metal, CPU (BLAS), CUDA, Vulkan, etc.
- `ggml_backend_sched` automatically splits graph operations across backends
- Metal backend claims supported ops, CPU handles the rest
- Buffer type system allows tensors on different backends

### 2.3 Operator Support in Metal

The Metal backend supports these GGML ops (from `ggml-metal-ops.cpp`):
- `GGML_OP_MUL_MAT` (with all quant types)
- `GGML_OP_MUL_MAT_ID` (MoE)
- `GGML_OP_ADD`, `GGML_OP_MUL`, `GGML_OP_DIV` (binary ops)
- `GGML_OP_SILU`, `GGML_OP_GELU`, etc. (activations)
- `GGML_OP_NORM`, `GGML_OP_RMS_NORM` (normalization)
- `GGML_OP_SOFT_MAX` (softmax)
- `GGML_OP_ROPE` (all 4 variants)
- `GGML_OP_FLASH_ATTN_EXT` (flash attention)
- `GGML_OP_CPY` (copy/quantize)
- `GGML_OP_CONV_*` (convolution)
- `GGML_OP_POOL_*` (pooling)
- `GGML_OP_SSM_CONV`, `GGML_OP_SSM_SCAN` (SSM)
- `GGML_OP_RWKV_WKV*` (RWKV)
- `GGML_OP_ARGSORT`, `GGML_OP_ARGMAX`
- And more...

### 2.4 Graph Fusion

Limited but present:
- `kernel_rms_norm_fuse_impl`: RMS norm fused with optional bias/add
- `kernel_norm_fuse_impl`: Layer norm fused
- `kernel_bin_fuse_impl`: Binary operation fusion
- **Flash attention** is the primary fusion: fuses QK^T + softmax + *V into one kernel
- `kernel_mul_mv_*`: Dequantization is fused into the mat-vec multiply

---

## 3. Attention Implementation

### 3.1 Flash Attention (Metal)

See Section 1.5 above. Key innovation: **per-quantization-type flash attention** — Q4_0 and Q8_0 KV caches can be used directly in flash attention without dequantization to F16 first. This saves massive memory bandwidth.

### 3.2 Sliding Window Attention (SWA)

- Supported via `n_swa` parameter in KV cache
- `llama_swa_type` enum for different SWA flavors
- Handled in flash attention via mask application
- Dedicated files: `llama-kv-cache-iswa.cpp/.h` for interleaved SWA

### 3.3 Multi-Query Attention / Grouped Query Attention (GQA)

- Fully supported — `n_kv_head` vs `n_head` distinction in hparams
- The flash attention kernel handles GQA natively by mapping Q heads to KV heads

### 3.4 Attention Sink Support

- `has_sinks` function constant in flash attention
- StreamingLLM-style attention sinks for infinite context

### 3.5 Soft Cap Attention

- `has_scap` function constant — used by Gemma 2 and similar models
- Applies `tanh(x / cap) * cap` to attention scores

---

## 4. KV Cache

### 4.1 Architecture (`src/llama-kv-cache.h`)

**Not paged** — uses a contiguous ring buffer approach:
- `kv_size`: Maximum cache size (number of positions)
- `n_stream`: Number of parallel sequences
- `slot_info`: Tracks which KV cells each token maps to
- Cell-level tracking via `llama_kv_cells` (position, sequence ID)

### 4.2 KV Cache Quantization

- **`type_k` and `type_v` can be independently set to any quantization type**
- Supported: F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, etc.
- `cpy_k`/`cpy_v` operations quantize on write, dequantize on read (fused in flash attention)
- `v_trans`: V cache can be transposed for more efficient reads in attention

### 4.3 Cache Management

- `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`: Sequence operations
- `get_can_shift()`: Supports shifting tokens for continuous batching
- `defrag`: Fragmentation recovery
- `state_write`/`state_read`: Save/load cache state

### 4.4 Hybrid Memory

Three memory types:
1. `llama_kv_cache`: Standard transformer attention cache
2. `llama_memory_hybrid`: Hybrid attention + recurrent memory
3. `llama_memory_recurrent`: Pure recurrent (Mamba/RWKV)

### 4.5 Layer-Level Control

- `layer_filter_cb`: Filter which layers use KV cache
- `layer_reuse_cb`: Reuse KV cache across layers (for models like Gemma with layer sharing)

---

## 5. Quantization Methods

### 5.1 All Supported Types (`ggml/include/ggml.h`)

| Type | Bits/Weight | Block Size | Description |
|------|------------|------------|-------------|
| `GGML_TYPE_F32` | 32 | 1 | Full precision |
| `GGML_TYPE_F16` | 16 | 1 | Half precision |
| `GGML_TYPE_BF16` | 16 | 1 | BFloat16 |
| `GGML_TYPE_Q1_0` | ~1.56 | 32 | 1-bit ternary (new!) |
| `GGML_TYPE_Q4_0` | 4.5 | 32 | 4-bit, asymmetric, single scale |
| `GGML_TYPE_Q4_1` | 5.5 | 32 | 4-bit, asymmetric, scale+min |
| `GGML_TYPE_Q5_0` | 5.5 | 32 | 5-bit, asymmetric |
| `GGML_TYPE_Q5_1` | 6.5 | 32 | 5-bit, asymmetric, scale+min |
| `GGML_TYPE_Q8_0` | 8.5 | 32 | 8-bit, symmetric |
| `GGML_TYPE_Q8_1` | 9.5 | 32 | 8-bit, symmetric+sum |
| `GGML_TYPE_Q2_K` | 2.56 | 256 | 2-bit K-quant (super-block) |
| `GGML_TYPE_Q3_K` | 3.44 | 256 | 3-bit K-quant |
| `GGML_TYPE_Q4_K` | 4.5 | 256 | 4-bit K-quant |
| `GGML_TYPE_Q5_K` | 5.5 | 256 | 5-bit K-quant |
| `GGML_TYPE_Q6_K` | 6.56 | 256 | 6-bit K-quant |
| `GGML_TYPE_Q8_K` | 8.5 | 256 | 8-bit K-quant |
| `GGML_TYPE_IQ2_XXS` | 2.06 | 256 | 2-bit, extra-extra-small |
| `GGML_TYPE_IQ2_XS` | 2.25 | 256 | 2-bit, extra-small |
| `GGML_TYPE_IQ2_S` | 2.5 | 256 | 2-bit, small |
| `GGML_TYPE_IQ3_XXS` | 3.06 | 256 | 3-bit, extra-extra-small |
| `GGML_TYPE_IQ3_S` | 3.44 | 256 | 3-bit, small |
| `GGML_TYPE_IQ1_S` | 1.56 | 256 | 1-bit, small |
| `GGML_TYPE_IQ1_M` | 1.75 | 256 | 1-bit, medium |
| `GGML_TYPE_IQ4_NL` | 4.5 | 32 | 4-bit, non-linear quantization |
| `GGML_TYPE_IQ4_XS` | 4.25 | 256 | 4-bit, extra-small |
| `GGML_TYPE_TQ1_0` | ~1.5 | 32 | Ternary quantization 1-bit |
| `GGML_TYPE_TQ2_0` | ~2.0 | 32 | Ternary quantization 2-bit |
| `GGML_TYPE_MXFP4` | ~4 | 32 | MX FP4 (Microscaling) |

**Total: 28 quantization types**

### 5.2 Fast Dequantization in Metal

Every quantization type has a Metal dequantization template:
```metal
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg);
```

These templates:
- Take a block pointer + `il` (interleave index)
- Output a `float4x4` (16 floats) or `type4` (4 floats) in registers
- Use bitwise operations, `select()`, and SIMD-friendly access patterns
- Are inlined into mat-mul and flash attention kernels for zero-overhead dequantization

### 5.3 K-Quant Super-Block Architecture

K-quants (Q2_K through Q6_K) use a **two-level quantization** scheme:
- Super-block of 256 weights
- Sub-blocks of 32 weights each
- Per-super-block scales + per-sub-block scales
- `QK_K = 256` constant defines super-block size

### 5.4 IQ (Integer Quantization) Variants

The IQ family uses importance-based quantization:
- Requires an **importance matrix** (imatrix) during quantization
- Uses lookup tables (`kvalues_iq4nl_f`, `kvalues_mxfp4_f`) for non-linear quantization
- `best_index_int8()`: Binary search to find nearest quantization level

---

## 6. Batching (`src/llama-batch.cpp`)

### 6.1 Micro-batching

- `llama_ubatch` structure: a unit of work within a batch
- `n_ubatch` parameter controls the micro-batch size
- Each ubatch is processed independently through the graph

### 6.2 Continuous Batching

- Multiple sequences can be interleaved in a single batch
- `slot_info` tracks per-sequence cell assignments
- `s0`/`s1` range defines active streams
- Flash attention mask handles variable-length sequences

### 6.3 Sequence Management

- `llama_seq_id` identifies sequences
- `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div` for manipulation
- Position tracking per sequence (`seq_pos_min`, `seq_pos_max`)

---

## 7. Speculative Decoding

### 7.1 Draft Model Support

llama.cpp supports speculative decoding via its **server tool** (`tools/server`):
- A smaller draft model runs ahead to propose tokens
- The target model verifies them in a single batch forward pass
- Accepted tokens are fast; rejected tokens trigger re-generation

### 7.2 Ngram Speculative Decoding

- **No separate draft model needed**
- Uses n-gram lookup from the prompt itself
- `common/speculative.cpp` implementation
- Looks up n-gram matches in context, proposes continuations

### 7.3 Implementation Details

- The batch system naturally supports this: draft tokens are added as extra positions
- Single forward pass verifies all draft tokens at once
- KV cache naturally extends for draft tokens, rolls back on rejection

---

## 8. Memory Management

### 8.1 mmap Model Loading (`src/llama-mmap.cpp/.h`)

- **Full mmap support** for model files
- `llama_mmap` class wraps platform-specific mmap calls
- On Apple Silicon: uses **shared/system memory** via unified memory architecture
- `GGML_USE_METAL` + `mmap` = zero-copy loading; GPU reads directly from mmap'd pages
- `gguf_set_mmap` controls whether mmap is used

### 8.2 Memory Pinning

Not applicable to Apple Silicon — unified memory means all memory is inherently "pinned" (GPU-accessible).

### 8.3 Metal Buffer Management (`ggml-metal.cpp`)

- `ggml_backend_metal_buffer_type` allocates Metal buffers
- Uses `MTLHeap` for sub-allocation where possible
- `ggml_metal_host_malloc` for host-visible memory
- **Shared memory mode**: On Apple Silicon, Metal buffers are in system memory, accessible by both CPU and GPU without copying

### 8.4 GGUF Format

- Custom tensor container format
- Supports memory-mapped loading
- Alignment-aware for Metal buffer requirements
- Key-value metadata store for model hyperparameters

---

## 9. Token Processing

### 9.1 Tokenizer (`src/llama-vocab.cpp`)

- **BPE tokenizer** implementation
- `llama_vocab` class handles all tokenization/detokenization
- Supports SentencePiece, GPT-2 BPE, and RWKV World tokenizer types
- Unicode normalization via custom `unicode.cpp`

### 9.2 BPE Optimizations

- Pre-compiled regex patterns for pre-tokenization
- Lookup tables for common merges
- `llama_tokenize` API handles full pipeline

---

## 10. Multithreading

### 10.1 CPU Backend Threading (`ggml/src/ggml-threading.cpp`)

- `ggml_threadpool` with configurable thread count
- Work-stealing scheduler
- `ggml_graph_compute_parallel` distributes ops across threads

### 10.2 Apple Silicon Specific

- **No P-core/E-core pinning** — relies on OS scheduler
- **No NUMA awareness** — Apple Silicon has unified memory, no NUMA
- Thread count typically set to number of performance cores
- Metal handles all GPU work; CPU threads only for non-offloaded ops

### 10.3 Metal Pipeline

- Command buffer recording is single-threaded
- Multiple compute encoders for dependency management
- `ggml_backend_metal_graph_compute()` manages the entire pipeline
- Automatic synchronization between Metal and CPU ops

---

## 11. Sampling Optimizations (`src/llama-sampler.cpp/.h`)

### 11.1 Sampler Chain Architecture

```c
struct llama_sampler_chain {
    llama_sampler_chain_params params;
    std::vector<info> samplers;
    std::vector<llama_token_data> cur;  // pre-allocated
};
```

### 11.2 Available Samplers

- **Greedy / Top-K / Top-P / Min-P**: Standard sampling
- **Temperature**: With dynamic scaling
- **Typical P**: Locally typical sampling
- **DRY**: DRY sampler (penalizes repetitive n-grams)
- **Mirostat v1/v2**: Adaptive entropy sampling
- **XTC**: Exclude-top-choice sampler
- **Penalties**: Frequency, presence, repetition
- **Grammar**: CFG-based constrained generation

### 11.3 Grammar-Constrained Generation (`src/llama-grammar.cpp`)

- **Incremental grammar evaluation** — only evaluates new tokens
- Pre-compiled grammar to state machine
- Uses bitsets for efficient token mask computation
- Supports JSON Schema via grammar compilation
- `llama_grammar_init` accepts GBNF grammars

### 11.4 Sampling Optimizations

- **Pre-allocated token data buffer** in sampler chain
- **Sorted candidate array** for top-k/top-p
- **SIMD-optimized softmax** in Metal
- **Batch sampling**: Process multiple sequences per sampling call

---

## 12. LoRA/Adapter Support (`src/llama-adapter.cpp/.h`)

### 12.1 LoRA Implementation

- `llama_adapter_lora` class manages LoRA weights
- LoRA weights stored in GGUF format
- `convert_lora_to_gguf.py` converts HF LoRA to GGUF
- **Applied at inference time** — modifies base model outputs

### 12.2 Adapter Architecture

- Multiple LoRA adapters can be loaded simultaneously
- Per-adapter scale (alpha) control
- Adapters can target specific layers
- `llama_adapter_apply` modifies the computation graph

### 12.3 Efficiency

- LoRA weights are kept in their quantized format
- Applied as additional mat-mul operations in the graph
- No model merging required — applied on-the-fly

---

## 13. Embedding/Reranking

### 13.1 Embedding Generation

- `llama_embedding` API for extracting hidden states
- Pooling modes: last token, mean pooling, CLS token
- Supported via `llama_pooling_type` parameter

### 13.2 Reranking

- Special model architecture type (`LLM_ARCH_GEMMA2` with rerank head)
- Uses logits processing for relevance scoring
- Embedding extraction reuses the same forward pass infrastructure

---

## 14. VLM (Vision-Language Model) Support

### 14.1 Multimodal Processing (`tools/mtmd/`)

- **mtmd** (multimodal tools): Image, audio, and video input processing
- Supports: CLIP, SigLIP, and custom vision encoders
- Audio support: Whisper, MERaLiON-2

### 14.2 Vision Processing

- Image preprocessing via `stb_image.h`
- Vision encoder runs as a separate model
- Vision embeddings injected into the LLM's token sequence
- `kernel_rope_vision` handles vision-specific position embeddings
- `kernel_conv_2d` / `kernel_pool_2d_*` for vision encoder ops
- `kernel_im2col` for convolution preprocessing

### 14.3 Supported VLM Architectures

- LLaVA (1.5 and next)
- MiniCPM-V
- Pixtral
- Gemma 3 / 4
- Qwen 2.5 VL
- And more...

---

## 15. Hardware-Specific Tuning

### 15.1 SIMD Width

- Hardcoded `N_SIMDWIDTH = 32` for all Apple GPUs
- All reduction operations use `simd_sum()`, `simd_max()` with 32-wide groups

### 15.2 Matrix Multiply Tuning

- **Tile sizes**: 64×32 (NR0×NR1) output, 32-wide K
- **4 SIMD groups per threadgroup** for mat-mul
- Threadgroup memory: 4096 bytes per matrix half (sa/sb)
- `FOR_UNROLL` pragma forces full unrolling of inner loops

### 15.3 Register Usage Optimization

- Kernel arguments use `int32_t` (not `int64_t`) for element counts to reduce register pressure
- Strides use `uint64_t` (necessary for large models)
- Explicit comment in code: "to reduce register usage"

### 15.4 Function Constants for Specialization

- Uses Metal function constants (`[[function_constant(N)]]`) for compile-time branching
- Each flash attention head dimension is a separate specialization
- Eliminates runtime branching in hot loops
- Over 160 function constant slots defined

### 15.5 BF16 Support

- Conditionally compiled via `GGML_METAL_HAS_BF16`
- Requires Metal 3.0+ (`__METAL_VERSION__ >= 310`)
- Uses `bfloat4x4`, `simdgroup_bfloat8x8` types

### 15.6 Metal Performance Primitives (MPP)

- Conditionally compiled via `GGML_METAL_HAS_TENSOR`
- Uses `tensor<T>` + `mpp::tensor_ops::matmul2d` for cooperative matrix multiply
- `execution_simdgroups<4>` — uses 4 SIMD groups cooperatively
- This is the newer Apple framework for high-performance matrix operations

### 15.7 M-Series Specific Notes

No explicit M1/M2/M3/M4 differentiation in code — Metal handles GPU family differences automatically. However:
- **Unified memory** is the key advantage — no CPU↔GPU copy overhead
- **Memory bandwidth** is the bottleneck, not compute — hence quantization focus
- BF16 support on M2+ (Metal 3.0+)
- MPP tensor API likely optimized for M3+ GPU architectures

### 15.8 Cache Line Considerations

- `PAD2(x, n)` macro for alignment: `(((x) + (n) - 1) & ~((n) - 1))`
- Threadgroup memory layout optimized for bank conflict avoidance
- Row-major access patterns in mat-mul kernels

---

## 16. Performance Benchmarks & Tuning

### 16.1 Benchmark Suite (`benches/`)

- Contains benchmark results for various hardware including Apple Silicon
- Benchmarks for Nemotron 3 Nano on DGX Spark (Apple Silicon-based)

### 16.2 Key Performance Parameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `-ngl` / `-sm` | Number of GPU layers | All layers on Metal for best perf |
| `-b` | Batch size | Larger = better prefill throughput |
| `-ub` | Micro-batch size | Controls parallelism in attention |
| `-t` | Thread count | CPU threads for non-offloaded ops |
| `--mlock` | Lock model in memory | Prevents page-out on macOS |
| `--no-mmap` | Disable mmap | Load into memory instead |
| `--kv-cache-type` | KV cache quantization | Q8_0 or Q4_0 saves VRAM |
| `--flash-attn` | Enable flash attention | Always beneficial |
| `--parallel` | Parallel sequences | For continuous batching |

### 16.3 Performance Tips (from code/docs)

1. **Use flash attention** (`--flash-attn`) — mandatory for good performance
2. **Offload all layers** (`-ngl 999`) — Metal is always faster than CPU
3. **Quantize KV cache** (`--kv-cache-type q8_0`) — 2x cache size for minimal quality loss
4. **Use Q4_K_M** quantization — best quality/speed tradeoff
5. **Use mmap** (default) — zero-copy model loading on Apple Silicon
6. **mlock** — prevents system from swapping model pages

---

## Summary: Key Innovations for NovaMLX

### Must-Have (Critical Performance)
1. **Per-quantization-type flash attention** — fused QK^T+softmax+*V with direct quantized KV access
2. **Fused dequantize-in-matmul** — never fully dequantize weights; compute dot products from quantized blocks
3. **SIMD group matrix multiply** — 8×8 block matrix operations via Metal SIMD group API
4. **Metal Performance Primitives fallback** — use `tensor<T>` + MPP for cooperative matmul
5. **Function constant specialization** — compile-time kernel variants per head dimension / quant type

### Should-Have (Important)
6. **Quantized KV cache** with on-the-fly dequantization in flash attention
7. **V-transpose** — store V cache transposed for better memory access in attention
8. **RoPE variants** — norm, neox, multi, vision (4 types)
9. **Continuous batching** with micro-batch scheduling
10. **SSM/RWKV support** — for non-transformer architectures

### Nice-to-Have
11. **Speculative decoding** (draft model + ngram)
12. **LoRA adapter** on-the-fly application
13. **Grammar-constrained** generation with incremental evaluation
14. **VLM pipeline** with vision encoder + LLM integration
15. **Tensor parallelism** (experimental in latest llama.cpp)

---

*Analysis generated: April 2026*
*Source: https://github.com/ggml-org/llama.cpp (master branch)*
