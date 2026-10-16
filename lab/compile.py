"""torch.compile baselines with one static compilation per input signature.

dynamic=True would compile one kernel for every shape, but Inductor sizes the kernel's blocks
from the first input it sees and keeps that choice in its on-disk cache: in RGB to Grayscale,
4096 x 4096 FP32 ran 6x slower after a 2 x 2 image had been compiled first. Static
compilation instead recompiles for each new signature (tensor shapes, dtypes and devices,
plus the values of non-tensor arguments, which become constants).

Dynamo's default recompile limit (8) would silently switch to eager execution after a few
signatures. The first call for each signature raises the limit and makes reaching it an
error; later calls go straight to the compiled code, because entering the config patch on
every call would add about 12 µs.
"""

import torch

# Distinct signatures one compiled function may see in a process.
RECOMPILE_LIMIT = 256


def _signature(args):
    return tuple(
        (a.shape, a.dtype, a.device) if isinstance(a, torch.Tensor) else (type(a), a) for a in args
    )


class StaticCompiled:
    """Call `fn` through torch.compile(dynamic=False), compiling once per signature."""

    def __init__(self, fn):
        self._fn = fn
        self._compiled = None
        self._signatures = set()

    def __call__(self, *args):
        if self._compiled is None:
            self._compiled = torch.compile(self._fn, dynamic=False)
        signature = _signature(args)
        if signature in self._signatures:
            return self._compiled(*args)
        with torch._dynamo.config.patch(
            recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True
        ):
            out = self._compiled(*args)
        self._signatures.add(signature)
        return out

    @property
    def signatures(self):
        """Signatures compiled so far in this process."""
        return len(self._signatures)
