# HPC GPU Kernel Lab

A collection of 50 GPU kernel studies spanning array operations, matrix
multiplication, normalization, attention, and inference workloads. The project
explores CUDA and Triton implementations alongside PyTorch and JAX baselines,
with correctness testing, reproducible benchmarks, and hardware profiling.

Each study examines how an algorithm maps to GPU hardware: memory access,
parallel execution, resource limits, and the performance impact of optimization.
Problem documentation includes implementation details, validation status, and
measured results as they become available.

## Roadmap

Validation details are recorded in each problem README.

Status: ⬜ Planned · 🟨 In Progress · ✅ Complete · 🚀 Optimized.

| # | Problem | CUDA | Triton | PyTorch | JAX | Primary lesson | Status |
| ---: | --- | :---: | :---: | :---: | :---: | --- | --- |
| 01 | [Vector Addition](problems/01_vector_add/README.md) | ✅ | ✅ | ✅ | ✅ | Launch overhead vs bandwidth | ✅ Complete |
| 02 | [ReLU](problems/02_relu/README.md) | — | — | — | — | Predication and branching | ⬜ Planned |
| 03 | [Reverse Array](problems/03_reverse_array/README.md) | — | — | — | — | Race-free in-place indexing | ⬜ Planned |
| 04 | [Interleave Arrays](problems/04_interleave_arrays/README.md) | — | — | — | — | Lane-to-output mapping | ⬜ Planned |
| 05 | [RGB to Grayscale](problems/05_rgb_to_grayscale/README.md) | — | — | — | — | Interleaved channel access | ⬜ Planned |
| 06 | [Rainbow Table](problems/06_rainbow_table/README.md) | — | — | — | — | Integer instruction throughput | ⬜ Planned |
| 07 | [Matrix Copy](problems/07_matrix_copy/README.md) | — | — | — | — | Memory bandwidth ceiling | ⬜ Planned |
| 08 | [Matrix Transpose](problems/08_matrix_transpose/README.md) | — | — | — | — | Shared-memory bank conflicts | ⬜ Planned |
| 09 | [1D Convolution](problems/09_convolution_1d/README.md) | — | — | — | — | Halo reuse in shared memory | ⬜ Planned |
| 10 | [Gaussian Blur](problems/10_gaussian_blur/README.md) | — | — | — | — | Separable filtering tradeoffs | ⬜ Planned |
| 11 | [2D Max Pooling](problems/11_max_pooling_2d/README.md) | — | — | — | — | Overlapping window locality | ⬜ Planned |
| 12 | [Weight Dequantization](problems/12_weight_dequantization/README.md) | — | — | — | — | Scale-tile locality | ⬜ Planned |
| 13 | [Reduction](problems/13_reduction/README.md) | — | — | — | — | Warp-level reduction | ⬜ Planned |
| 14 | [Count Array Element](problems/14_count_array_element/README.md) | — | — | — | — | Predicate reduction | ⬜ Planned |
| 15 | [Mean Squared Error](problems/15_mean_squared_error/README.md) | — | — | — | — | Map-reduce fusion | ⬜ Planned |
| 16 | [Dot Product](problems/16_dot_product/README.md) | — | — | — | — | Fused multiply-accumulate reduction | ⬜ Planned |
| 17 | [Prefix Sum](problems/17_prefix_sum/README.md) | — | — | — | — | Cross-block prefix propagation | ⬜ Planned |
| 18 | [Histogramming](problems/18_histogramming/README.md) | — | — | — | — | Atomic contention | ⬜ Planned |
| 19 | [Stream Compaction](problems/19_stream_compaction/README.md) | — | — | — | — | Stable parallel filtering | ⬜ Planned |
| 20 | [Parallel Merge](problems/20_parallel_merge/README.md) | — | — | — | — | Balanced irregular partitioning | ⬜ Planned |
| 21 | [Dense GEMV](problems/21_dense_gemv/README.md) | — | — | — | — | Low-arithmetic-intensity matrix math | ⬜ Planned |
| 22 | [Sparse Matrix-Vector Multiplication](problems/22_sparse_matvec/README.md) | — | — | — | — | Irregular sparse memory access | ⬜ Planned |
| 23 | [Matrix Multiplication](problems/23_matrix_multiplication_fp32/README.md) | — | — | — | — | Shared-memory data reuse | ⬜ Planned |
| 24 | [Batched Matrix Multiplication — FP32](problems/24_batched_matmul_fp32/README.md) | — | — | — | — | Batch scheduling and occupancy | ⬜ Planned |
| 25 | [General Matrix Multiplication (GEMM) — FP16](problems/25_gemm_fp16/README.md) | — | — | — | — | Tensor Core utilization | ⬜ Planned |
| 26 | [FP16 Batched Matrix Multiplication](problems/26_batched_matmul_fp16/README.md) | — | — | — | — | Mixed-precision batch utilization | ⬜ Planned |
| 27 | [INT8 Quantized MatMul](problems/27_int8_quantized_matmul/README.md) | — | — | — | — | Quantized arithmetic contracts | ⬜ Planned |
| 28 | [Sparse Matrix-Dense Matrix Multiplication](problems/28_sparse_dense_matmul/README.md) | — | — | — | — | Sparse reuse across output columns | ⬜ Planned |
| 29 | [Fused GEMM + Bias + Activation](problems/29_fused_gemm_epilogue/README.md) | — | — | — | — | GEMM epilogue fusion | ⬜ Planned |
| 30 | [Sigmoid Linear Unit](problems/30_silu/README.md) | — | — | — | — | Special-function throughput | ⬜ Planned |
| 31 | [Swish-Gated Linear Unit](problems/31_swiglu_activation/README.md) | — | — | — | — | Elementwise gating fusion | ⬜ Planned |
| 32 | [Softmax](problems/32_softmax/README.md) | — | — | — | — | Numerically stable fused reduction | ⬜ Planned |
| 33 | [RMS Normalization](problems/33_rms_norm/README.md) | — | — | — | — | Normalization traffic and precision | ⬜ Planned |
| 34 | [Layer Normalization](problems/34_layer_norm/README.md) | — | — | — | — | Stable variance reduction | ⬜ Planned |
| 35 | [Batch Normalization](problems/35_batch_norm/README.md) | — | — | — | — | Reduction-axis layout | ⬜ Planned |
| 36 | [Fused Residual Add and RMS Norm](problems/36_fused_residual_rms_norm/README.md) | — | — | — | — | Residual-normalization fusion | ⬜ Planned |
| 37 | [Categorical Cross Entropy Loss](problems/37_categorical_cross_entropy/README.md) | — | — | — | — | Fused log-sum-exp loss | ⬜ Planned |
| 38 | [Rotary Positional Embedding](problems/38_rope/README.md) | — | — | — | — | Positional pair layouts | ⬜ Planned |
| 39 | [SwiGLU MLP Block](problems/39_swiglu_mlp/README.md) | — | — | — | — | MLP fusion boundaries | ⬜ Planned |
| 40 | [Softmax Attention](problems/40_softmax_attention/README.md) | — | — | — | — | Attention pipeline decomposition | ⬜ Planned |
| 41 | [Causal Self-Attention](problems/41_causal_attention/README.md) | — | — | — | — | Causal masking and triangular work | ⬜ Planned |
| 42 | [Multi-Head Attention](problems/42_multi_head_attention/README.md) | — | — | — | — | Head layout and scheduling | ⬜ Planned |
| 43 | [Attention with Linear Biases](problems/43_alibi_attention/README.md) | — | — | — | — | Fused attention bias generation | ⬜ Planned |
| 44 | [Grouped Query Attention](problems/44_grouped_query_attention/README.md) | — | — | — | — | Shared K/V head reuse | ⬜ Planned |
| 45 | [Decaying Causal Attention](problems/45_decaying_causal_attention/README.md) | — | — | — | — | Recurrence vs quadratic materialization | ⬜ Planned |
| 46 | [Quantize / Dequantize Pipeline](problems/46_quantization_pipeline/README.md) | — | — | — | — | Quantization error vs bandwidth | ⬜ Planned |
| 47 | [KV-Cache Update and Paged Access](problems/47_kv_cache_operations/README.md) | — | — | — | — | Stateful cache layout | ⬜ Planned |
| 48 | [FlashAttention-Style Online Attention](problems/48_flash_attention/README.md) | — | — | — | — | Numerically stable online reduction | ⬜ Planned |
| 49 | [Paged Attention](problems/49_paged_attention/README.md) | — | — | — | — | Irregular decode-time attention | ⬜ Planned |
| 50 | [MoE Token Routing and Dispatch](problems/50_moe_token_routing/README.md) | — | — | — | — | Balanced scatter and expert dispatch | ⬜ Planned |

## Development

See the [testing methodology](docs/methodology.md),
[benchmarking guidelines](docs/benchmarking.md), and
[profiling workflow](docs/profiling.md) for the study process.
Setup and execution commands are documented in each problem README.

Contribution requirements are described in [CONTRIBUTING.md](CONTRIBUTING.md).
