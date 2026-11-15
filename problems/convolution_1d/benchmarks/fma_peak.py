"""Measure the FP32 FMA ceiling and the SM clock it runs at: the roofline's compute roof.

python -m problems.convolution_1d.benchmarks.fma_peak artifacts/convolution_1d/fma_peak_1.json

Each thread runs 8 independent FFMA chains (enough to hide the 4-cycle FMA latency at full
occupancy) for a fixed number of iterations; FLOP/s = 2 · FMAs / median kernel time over
--seconds of back-to-back launches. nvidia-smi samples the SM clock during the timed loop,
and the theoretical peak at that clock (SMs · 128 FP32 lanes · 2 · f) is recorded beside
the measurement.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

SOURCE = r"""
#include <torch/extension.h>
__global__ void fma_loop(float *out, int iters) {
  float a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  float a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const float b = 1.0000001f, c = 1e-7f;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
      a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
    }
  }
  const float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (s == -1.0f) out[0] = s;  // never true; keeps the chains live
}
void run(torch::Tensor out, int64_t blocks, int64_t iters) {
  fma_loop<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(), (int)iters);
}
"""
FP32_LANES_PER_SM = 128


def sample_clocks(samples, stop):
    while not stop.is_set():
        line = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if line:
            samples.append([int(v) for v in line.split(",")])
        time.sleep(0.1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--seconds", type=float, default=4.0)
    args = p.parse_args()
    ext = load_inline(
        "conv1d_fma_peak",
        cpp_sources="void run(torch::Tensor out, int64_t blocks, int64_t iters);",
        cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + SOURCE,
        functions=["run"],
    )
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    blocks, iters = sms * 6 * 32, 2048
    fmas = blocks * 256 * iters * 16 * 8
    out = torch.zeros(1, device="cuda")
    for _ in range(5):
        ext.run(out, blocks, iters)
    torch.cuda.synchronize()

    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample_clocks, args=(samples, stop))
    sampler.start()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        start.record()
        ext.run(out, blocks, iters)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    stop.set()
    sampler.join()

    median = sorted(times)[len(times) // 2]
    loaded = sorted(s for s, _ in samples[2:]) or [0]  # skip samples from before the loop
    sm_mhz = loaded[len(loaded) // 2]
    result = dict(
        gpu=props.name,
        sm_count=sms,
        launches=len(times),
        median_kernel_s=median,
        achieved_tflop_s=2 * fmas / median / 1e12,
        sm_clock_mhz_median=sm_mhz,
        sm_clock_mhz_range=[loaded[0], loaded[-1]],
        mem_clock_mhz_median=sorted(m for _, m in samples[2:])[len(samples[2:]) // 2]
        if len(samples) > 2
        else None,
        theoretical_tflop_s_at_clock=sms * FP32_LANES_PER_SM * 2 * sm_mhz * 1e6 / 1e12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
