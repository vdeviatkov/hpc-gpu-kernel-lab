# HPC GPU Kernel Lab

A GPU kernel engineering lab with 50 studies covering foundational array
operations, matrix multiplication, normalization, attention, and inference
workloads. Implementations use CUDA, Triton, PyTorch, and JAX, with emphasis on
correctness, benchmarking, profiling, and understanding hardware behavior.

The goal is to explain how algorithms map onto hardware: what data moves, where
parallelism comes from, which resource limits execution, and why an optimization
changes performance. Workload sources supply starting contracts; evidence and
technical analysis define the portfolio.

## Roadmap

`code` means implemented, with validation status in the problem README.
`plan` means intended implementation, not existing code. `—` means no current
backend plan.
Source and official LeetGPU difficulty are recorded in each problem README. Status: ⬜ Planned · 🟨 In Progress · ✅ Complete · 🚀 Optimized.

| # | Problem | CUDA | Triton | PyTorch | JAX | Primary lesson | Status |
| ---: | --- | :---: | :---: | :---: | :---: | --- | --- |
| 01 | [Vector Addition](problems/01_vector_add/README.md) | code | code | code | code | Launch overhead vs bandwidth | 🟨 In Progress |
| 02 | [ReLU](problems/02_relu/README.md) | plan | plan | plan | — | Predication and branching | ⬜ Planned |
| 03 | [Reverse Array](problems/03_reverse_array/README.md) | plan | plan | plan | — | Race-free in-place indexing | ⬜ Planned |
| 04 | [Interleave Arrays](problems/04_interleave_arrays/README.md) | plan | plan | plan | — | Lane-to-output mapping | ⬜ Planned |
| 05 | [RGB to Grayscale](problems/05_rgb_to_grayscale/README.md) | plan | plan | plan | — | Interleaved channel access | ⬜ Planned |
| 06 | [Rainbow Table](problems/06_rainbow_table/README.md) | plan | — | plan | — | Integer instruction throughput | ⬜ Planned |
| 07 | [Matrix Copy](problems/07_matrix_copy/README.md) | plan | plan | plan | — | Memory bandwidth ceiling | ⬜ Planned |
| 08 | [Matrix Transpose](problems/08_matrix_transpose/README.md) | plan | plan | plan | — | Shared-memory bank conflicts | ⬜ Planned |
| 09 | [1D Convolution](problems/09_convolution_1d/README.md) | plan | plan | plan | plan | Halo reuse in shared memory | ⬜ Planned |
| 10 | [Gaussian Blur](problems/10_gaussian_blur/README.md) | plan | plan | plan | plan | Separable filtering tradeoffs | ⬜ Planned |
| 11 | [2D Max Pooling](problems/11_max_pooling_2d/README.md) | plan | plan | plan | — | Overlapping window locality | ⬜ Planned |
| 12 | [Weight Dequantization](problems/12_weight_dequantization/README.md) | plan | plan | plan | — | Scale-tile locality | ⬜ Planned |
| 13 | [Reduction](problems/13_reduction/README.md) | plan | plan | plan | plan | Warp-level reduction | ⬜ Planned |
| 14 | [Count Array Element](problems/14_count_array_element/README.md) | plan | plan | plan | — | Predicate reduction | ⬜ Planned |
| 15 | [Mean Squared Error](problems/15_mean_squared_error/README.md) | plan | plan | plan | plan | Map-reduce fusion | ⬜ Planned |
| 16 | [Dot Product](problems/16_dot_product/README.md) | plan | plan | plan | plan | Fused multiply-accumulate reduction | ⬜ Planned |
| 17 | [Prefix Sum](problems/17_prefix_sum/README.md) | plan | plan | plan | — | Cross-block prefix propagation | ⬜ Planned |
| 18 | [Histogramming](problems/18_histogramming/README.md) | plan | plan | plan | — | Atomic contention | ⬜ Planned |
| 19 | [Stream Compaction](problems/19_stream_compaction/README.md) | plan | plan | plan | — | Stable parallel filtering | ⬜ Planned |
| 20 | [Parallel Merge](problems/20_parallel_merge/README.md) | plan | — | plan | — | Balanced irregular partitioning | ⬜ Planned |
| 21 | [Dense GEMV](problems/21_dense_gemv/README.md) | plan | plan | plan | plan | Low-arithmetic-intensity matrix math | ⬜ Planned |
| 22 | [Sparse Matrix-Vector Multiplication](problems/22_sparse_matvec/README.md) | plan | plan | plan | — | Irregular sparse memory access | ⬜ Planned |
| 23 | [Matrix Multiplication](problems/23_matrix_multiplication_fp32/README.md) | plan | plan | plan | plan | Shared-memory data reuse | ⬜ Planned |
| 24 | [Batched Matrix Multiplication — FP32](problems/24_batched_matmul_fp32/README.md) | plan | plan | plan | plan | Batch scheduling and occupancy | ⬜ Planned |
| 25 | [General Matrix Multiplication (GEMM) — FP16](problems/25_gemm_fp16/README.md) | plan | plan | plan | plan | Tensor Core utilization | ⬜ Planned |
| 26 | [FP16 Batched Matrix Multiplication](problems/26_batched_matmul_fp16/README.md) | plan | plan | plan | plan | Mixed-precision batch utilization | ⬜ Planned |
| 27 | [INT8 Quantized MatMul](problems/27_int8_quantized_matmul/README.md) | plan | plan | plan | — | Quantized arithmetic contracts | ⬜ Planned |
| 28 | [Sparse Matrix-Dense Matrix Multiplication](problems/28_sparse_dense_matmul/README.md) | plan | plan | plan | — | Sparse reuse across output columns | ⬜ Planned |
| 29 | [Fused GEMM + Bias + Activation](problems/29_fused_gemm_epilogue/README.md) | plan | plan | plan | plan | GEMM epilogue fusion | ⬜ Planned |
| 30 | [Sigmoid Linear Unit](problems/30_silu/README.md) | plan | plan | plan | plan | Special-function throughput | ⬜ Planned |
| 31 | [Swish-Gated Linear Unit](problems/31_swiglu_activation/README.md) | plan | plan | plan | plan | Elementwise gating fusion | ⬜ Planned |
| 32 | [Softmax](problems/32_softmax/README.md) | plan | plan | plan | plan | Numerically stable fused reduction | ⬜ Planned |
| 33 | [RMS Normalization](problems/33_rms_norm/README.md) | plan | plan | plan | plan | Normalization traffic and precision | ⬜ Planned |
| 34 | [Layer Normalization](problems/34_layer_norm/README.md) | plan | plan | plan | plan | Stable variance reduction | ⬜ Planned |
| 35 | [Batch Normalization](problems/35_batch_norm/README.md) | plan | plan | plan | plan | Reduction-axis layout | ⬜ Planned |
| 36 | [Fused Residual Add and RMS Norm](problems/36_fused_residual_rms_norm/README.md) | plan | plan | plan | plan | Residual-normalization fusion | ⬜ Planned |
| 37 | [Categorical Cross Entropy Loss](problems/37_categorical_cross_entropy/README.md) | plan | plan | plan | plan | Fused log-sum-exp loss | ⬜ Planned |
| 38 | [Rotary Positional Embedding](problems/38_rope/README.md) | plan | plan | plan | plan | Positional pair layouts | ⬜ Planned |
| 39 | [SwiGLU MLP Block](problems/39_swiglu_mlp/README.md) | plan | plan | plan | plan | MLP fusion boundaries | ⬜ Planned |
| 40 | [Softmax Attention](problems/40_softmax_attention/README.md) | plan | plan | plan | plan | Attention pipeline decomposition | ⬜ Planned |
| 41 | [Causal Self-Attention](problems/41_causal_attention/README.md) | plan | plan | plan | plan | Causal masking and triangular work | ⬜ Planned |
| 42 | [Multi-Head Attention](problems/42_multi_head_attention/README.md) | plan | plan | plan | plan | Head layout and scheduling | ⬜ Planned |
| 43 | [Attention with Linear Biases](problems/43_alibi_attention/README.md) | plan | plan | plan | plan | Fused attention bias generation | ⬜ Planned |
| 44 | [Grouped Query Attention](problems/44_grouped_query_attention/README.md) | plan | plan | plan | plan | Shared K/V head reuse | ⬜ Planned |
| 45 | [Decaying Causal Attention](problems/45_decaying_causal_attention/README.md) | plan | plan | plan | plan | Recurrence vs quadratic materialization | ⬜ Planned |
| 46 | [Quantize / Dequantize Pipeline](problems/46_quantization_pipeline/README.md) | plan | plan | plan | — | Quantization error vs bandwidth | ⬜ Planned |
| 47 | [KV-Cache Update and Paged Access](problems/47_kv_cache_operations/README.md) | plan | plan | plan | — | Stateful cache layout | ⬜ Planned |
| 48 | [FlashAttention-Style Online Attention](problems/48_flash_attention/README.md) | plan | plan | plan | plan | Numerically stable online reduction | ⬜ Planned |
| 49 | [Paged Attention](problems/49_paged_attention/README.md) | plan | plan | plan | — | Irregular decode-time attention | ⬜ Planned |
| 50 | [MoE Token Routing and Dispatch](problems/50_moe_token_routing/README.md) | plan | plan | plan | — | Balanced scatter and expert dispatch | ⬜ Planned |

For implementation, follow the [methodology and tests](docs/methodology.md),
[benchmarking and hardware plan](docs/benchmarking.md), and [profiling plan](docs/profiling.md).
No latency, bandwidth, or speedup is claimed before a real hardware run.

## Repository architecture

```text
hpc-gpu-kernel-lab/
├── README.md                 # Roadmap and progress
├── LICENSE
├── CONTRIBUTING.md           # Status and contribution rules
├── docs/
│   ├── methodology.md        # Workflow and correctness testing
│   ├── benchmarking.md       # Timing, hardware, and reproducibility
│   └── profiling.md          # Nsight investigation plan
└── problems/
    ├── 01_vector_add/         # Implementations, tests, benchmarks, study notes
    ├── ...                   # 50 study plans
    └── 50_moe_token_routing/README.md
```

Backend/test/result subdirectories are documented per problem and created only
when their first meaningful file exists. There are no empty `.cu`/`.py` files,
no generated solution placeholders. Vector addition builds its CUDA extension
on first use; see its README for setup, tests, benchmarks, and profiling commands.

## Evidence and attribution

Status is maintained in this roadmap and each problem README. See the
[contribution guide](CONTRIBUTING.md) for completion criteria and the
[benchmarking plan](docs/benchmarking.md) for reproducibility requirements.
**Results pending hardware benchmark.**

LeetGPU names and links provide attribution; this independent lab is not
affiliated with LeetGPU or an employer. Original material is MIT licensed;
third-party content retains its own terms.
