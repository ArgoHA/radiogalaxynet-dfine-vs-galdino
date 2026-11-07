"""Engine-only TensorRT latency, the same loop for any .engine (D-FINE-seg and Gal-DINO).

TRT execute only on a dummy input already on the GPU: no preprocessing, no H2D / D2H, no CUDA
graph. Median over 300 timed runs after 50 warmup runs, batch 1.

  python measure_engine.py weights/model.engine galdino/Gal-DINO/export/galdino_fp16.engine
"""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_TRT2TORCH = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
              trt.int64: torch.int64, trt.int8: torch.int8, trt.bool: torch.bool}


def engine_ms(path, iters=300, warmup=50):
    with trt.Runtime(trt.Logger(trt.Logger.ERROR)) as rt:
        eng = rt.deserialize_cuda_engine(Path(path).read_bytes())
    ctx = eng.create_execution_context()
    bufs, in_shape = [], None
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = tuple(1 if d < 0 else d for d in eng.get_tensor_shape(n))
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(n, shape)
            in_shape = shape
        buf = torch.empty(shape, dtype=_TRT2TORCH[eng.get_tensor_dtype(n)], device="cuda")
        ctx.set_tensor_address(n, buf.data_ptr())
        bufs.append(buf)  # keep alive

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    lat = []
    with torch.cuda.stream(stream):
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    return f"{in_shape[2]}x{in_shape[3]}", float(np.median(lat))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="+")
    for p in ap.parse_args().engines:
        hw, ms = engine_ms(p)
        print(f"{p}  input {hw}  engine {ms:.2f} ms (median)")
