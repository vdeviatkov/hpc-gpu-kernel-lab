"""CUDA backend for 47 · KV-Cache Update and Paged Access. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def kv_cache_append(k_new, v_new, k_cache, v_cache, block_table, positions):
    if not IMPLEMENTED:
        raise NotImplementedError("kv_cache_append: CUDA backend not implemented (cuda/kernels.cu)")
    load_extension("gpu_lab_kv_cache_operations", Path(__file__).parent).kv_cache_append(
        k_new, v_new, k_cache, v_cache, block_table, positions
    )
    return (k_cache, v_cache)
