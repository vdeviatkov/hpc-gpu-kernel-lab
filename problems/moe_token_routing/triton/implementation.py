"""Triton backend for 50 · MoE Token Routing and Dispatch. Not implemented yet.

Implement route_tokens_kernel and choose the grid, then set IMPLEMENTED = True. Tile size and
warp count are explicit arguments; there is no autotuning.
"""

import torch
import triton
import triton.language as tl

IMPLEMENTED = False


@triton.jit
def route_tokens_kernel(
    x_ptr,
    router_logits_ptr,
    top_k,
    dispatched_ptr,
    expert_offsets_ptr,
    token_index_ptr,
    weights_ptr,
    t,
    d,
    e,
    BLOCK: tl.constexpr,
):
    pass  # TODO: implement.


def route_tokens(x, router_logits, top_k, *, block_size=1024, num_warps=4):
    if not IMPLEMENTED:
        raise NotImplementedError(
            "route_tokens: Triton backend not implemented (triton/implementation.py)"
        )
    dispatched = torch.empty((x.shape[0] * top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    expert_offsets = torch.empty(router_logits.shape[1] + 1, dtype=torch.int32, device=x.device)
    token_index = torch.empty(x.shape[0] * top_k, dtype=torch.int32, device=x.device)
    weights = torch.empty(x.shape[0] * top_k, dtype=torch.float32, device=x.device)
    t = x.shape[0]
    d = x.shape[1]
    e = router_logits.shape[1]
    grid = (triton.cdiv(t, block_size),)  # TODO: choose the grid for this problem.
    with torch.cuda.device(x.device):
        route_tokens_kernel[grid](
            x,
            router_logits,
            top_k,
            dispatched,
            expert_offsets,
            token_index,
            weights,
            t,
            d,
            e,
            BLOCK=block_size,
            num_warps=num_warps,
        )
    return (dispatched, expert_offsets, token_index, weights)
