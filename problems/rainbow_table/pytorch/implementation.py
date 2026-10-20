"""PyTorch reference and compiled baseline for 06 · Rainbow Table.

method="reference" is the oracle every other backend is tested against: the source's own
algorithm, in int64 with an explicit mask to 32 bits after every multiply. In eager mode
each operation is a separate kernel, so one round is about 20 launches, each a full pass
over 8-byte intermediates.

method="compile": torch.compile of the same hash written in int32, where multiplication
wraps modulo 2^32 on its own and no masks are needed; the result is reinterpreted as
uint32. All R rounds fuse into one generated kernel. It is compiled once per length and
round count (R becomes a constant and the loop is unrolled); lab.compile explains why not
dynamic=True. Not used on the CPU: there, signed int32 overflow in generated C++ code is
undefined behaviour.
"""

import torch

from lab.compile import StaticCompiled

FNV_PRIME = 16777619
OFFSET_BASIS = 2166136261


def _fnv1a_int64(x):
    """The source's fnv1a_hash: 32-bit FNV-1a over the 4 bytes of x, held in int64."""
    hash_val = torch.full_like(x, OFFSET_BASIS)
    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
        hash_val = hash_val & 0xFFFFFFFF
    return hash_val


def _reference(x, rounds):
    current = x.to(torch.int64)
    for _ in range(rounds):
        current = _fnv1a_int64(current)
    # current holds values in [0, 2^32) after any round (or the sign-extended input after
    # none); the low 32 bits, reinterpreted, are the uint32 result.
    return current.to(torch.int32).view(torch.uint32)


# The same constants as int32 bit patterns: 2166136261 - 2^32.
_OFFSET_BASIS_INT32 = OFFSET_BASIS - 2**32


def _hash_int32(x, rounds):
    h = x
    for _ in range(rounds):
        state = torch.full_like(h, _OFFSET_BASIS_INT32)
        for byte_pos in range(4):
            # The arithmetic shift copies the sign bit, but the mask keeps only byte k.
            state = (state ^ ((h >> (byte_pos * 8)) & 0xFF)) * FNV_PRIME
        h = state
    return h.view(torch.uint32)


_compiled = StaticCompiled(_hash_int32)


def rainbow_table(x, rounds, *, method="reference"):
    if method == "compile":
        # clone: with rounds = 0 the compiled function would return a view of x.
        return _compiled(x, rounds) if rounds else x.view(torch.uint32).clone()
    if method == "int32":
        return _hash_int32(x, rounds) if rounds else x.view(torch.uint32).clone()
    if method != "reference":
        raise ValueError(f"unknown rainbow_table method: {method}")
    return _reference(x, rounds)
