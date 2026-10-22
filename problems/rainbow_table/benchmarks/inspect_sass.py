"""Summarize CUDA SASS. Counts are static, not dynamic profiler counters.

Usage: python -m problems.rainbow_table.benchmarks.inspect_sass cuda.sass
Backward BRA.U determines the runtime round body. Unrolled kernels count only
hash IMADs (prime immediate), LOP3, and 32-bit right SHFs, excluding addressing.
"""

import collections
import json
import re
import sys
from pathlib import Path


def inspect(text):
    result = []
    for name, body in re.findall(r"Function : (\S+)(.*?)(?=Function :|$)", text, re.S):
        instructions = [
            (int(a, 16), op, arg)
            for a, op, arg in re.findall(
                r"/\*([0-9a-f]+)\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9.]*)([^;]*);", body
            )
        ]
        loops = []
        for address, op, arg in instructions:
            target = re.search(r",\s*0x([0-9a-f]+)", arg)
            if op == "BRA.U" and target and int(target[1], 16) < address:
                loops.append((int(target[1], 16), address))
        if loops:
            start, end = loops[-1]
            selected = [(op, arg) for a, op, arg in instructions if start <= a <= end]
            scope = f"loop 0x{start:x}..0x{end:x}"
        else:
            selected = [
                (op, arg)
                for _, op, arg in instructions
                if (op == "IMAD" and "0x1000193" in arg) or op == "LOP3.LUT" or op == "SHF.R.U32.HI"
            ]
            scope = "unrolled hash only; excludes setup/addressing/padding"
        counts = collections.Counter(op for op, _ in selected)
        result.append(
            dict(kernel=name, scope=scope, instructions=len(selected), counts=dict(counts))
        )
    return result


if __name__ == "__main__":
    print(json.dumps(inspect(Path(sys.argv[1]).read_text()), indent=2))
