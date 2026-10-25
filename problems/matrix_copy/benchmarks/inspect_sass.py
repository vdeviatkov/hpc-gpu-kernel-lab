"""Count global loads and stores per kernel in static SASS, by width and cache qualifier.

Usage: python -m problems.matrix_copy.benchmarks.inspect_sass <sass_dir>

Reads <sass_dir>/cuda.sass (cuobjdump of the CUDA extension), compiles the Triton kernels
for FP32 and FP16 and disassembles them, and writes <sass_dir>/counts.json. Counts are
static instructions, not executed ones: an unrolled loop shows each copy of its body.
"""

import collections
import json
import re
import sys
from pathlib import Path

import torch

MEMORY = re.compile(r"\b((?:LDG|STG)[A-Z0-9.]*)")


def count(text):
    """{kernel: {opcode: count}} for every Function block in a SASS dump."""
    result = {}
    for name, body in re.findall(r"Function : (\S+)(.*?)(?=Function :|$)", text, re.S):
        result[name] = dict(sorted(collections.Counter(MEMORY.findall(body)).items()))
    return result


def triton_sass():
    """SASS of each Triton kernel and element width, compiled for a 1024 x 1024 copy."""
    from problems.matrix_copy.triton import implementation as triton_impl

    out = {}
    for dtype in (torch.float32, torch.float16):
        a = torch.zeros(1024, 1024, dtype=dtype, device="cuda")
        src = a.view(triton_impl.BITS[a.element_size()])
        dst = torch.empty_like(src)
        flat = triton_impl.flat_kernel[(1024,)](src, dst, a.numel(), BLOCK=1024, num_warps=4)
        tile = triton_impl.tile_kernel[(16, 16)](
            src, dst, 1024, 1024, BLOCK_M=64, BLOCK_N=64, num_warps=4
        )
        width = 8 * a.element_size()
        out[f"triton_flat_{width}bit"] = flat.asm["sass"]
        out[f"triton_tile_{width}bit"] = tile.asm["sass"]
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/matrix_copy/sass")
    counts = count((root / "cuda.sass").read_text())
    for name, sass in triton_sass().items():
        (root / f"{name}.sass").write_text(sass)
        counts[name] = dict(sorted(collections.Counter(MEMORY.findall(sass)).items()))
    (root / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
