"""CUDA backend for 50 · MoE Token Routing and Dispatch. Not implemented yet.

Implement cuda/kernels.cu, then set IMPLEMENTED = True. The extension builds lazily
on the first call and launches on PyTorch's current stream.
"""

from pathlib import Path

import torch

from lab.cuda_ext import load_extension

IMPLEMENTED = False


def route_tokens(x, router_logits, top_k):
    if not IMPLEMENTED:
        raise NotImplementedError("route_tokens: CUDA backend not implemented (cuda/kernels.cu)")
    dispatched = torch.empty((x.shape[0] * top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    expert_offsets = torch.empty(router_logits.shape[1] + 1, dtype=torch.int32, device=x.device)
    token_index = torch.empty(x.shape[0] * top_k, dtype=torch.int32, device=x.device)
    weights = torch.empty(x.shape[0] * top_k, dtype=torch.float32, device=x.device)
    load_extension("gpu_lab_moe_token_routing", Path(__file__).parent).route_tokens(
        x, router_logits, top_k, dispatched, expert_offsets, token_index, weights
    )
    return (dispatched, expert_offsets, token_index, weights)
