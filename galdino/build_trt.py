"""Build a TensorRT engine from export/galdino.onnx with the TensorRT 10.13.3.9 Python API.

  .venv/bin/python build_trt.py --fp16 --out export/galdino_fp16.engine
  .venv/bin/python build_trt.py --out export/galdino_fp32.engine            # fp32 (TF32 off)
  .venv/bin/python build_trt.py --fp16 --fp32-layers GridSample --out ...    # pin op types fp32

--fp32-layers takes ONNX op types or layer-name regexes; matching layers get
precision/output type FP32 under OBEY_PRECISION_CONSTRAINTS.
"""

import argparse
import re
import time
from pathlib import Path

import tensorrt as trt

HERE = Path(__file__).resolve().parent

OP_TO_LAYER = {
    "GridSample": trt.LayerType.GRID_SAMPLE,
    "Softmax": trt.LayerType.SOFTMAX,
    "LayerNormalization": trt.LayerType.NORMALIZATION,
    "InstanceNormalization": trt.LayerType.NORMALIZATION,
    "TopK": trt.LayerType.TOPK,
    "Gemm": trt.LayerType.MATRIX_MULTIPLY,
    "MatMul": trt.LayerType.MATRIX_MULTIPLY,
}

# regex pins touch only arithmetic layers (data movement / index layers carry no precision)
COMPUTE = {trt.LayerType.CONVOLUTION, trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
           trt.LayerType.UNARY, trt.LayerType.ACTIVATION, trt.LayerType.SOFTMAX,
           trt.LayerType.NORMALIZATION, trt.LayerType.REDUCE, trt.LayerType.GRID_SAMPLE,
           trt.LayerType.SCALE}


def build(onnx_path, out, fp16, fp32_layers, tf32, workspace_gb, opt_level):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = opt_level
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)

    pinned = 0
    if fp32_layers:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        types = {OP_TO_LAYER[p] for p in fp32_layers if p in OP_TO_LAYER}
        regs = [re.compile(p) for p in fp32_layers if p not in OP_TO_LAYER]
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            by_name = layer.type in COMPUTE and any(r.search(layer.name) for r in regs)
            if layer.type in types or by_name:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype in (trt.float32, trt.float16):
                        layer.set_output_type(j, trt.float32)
                pinned += 1
        print(f"pinned {pinned} layers to fp32 ({fp32_layers})")

    for i in range(network.num_outputs):
        t = network.get_output(i)
        print("output", t.name, t.shape, t.dtype)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("build failed")
    Path(out).write_bytes(bytes(blob))
    print(f"built {out} in {time.time() - t0:.0f}s, {Path(out).stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(HERE / "export/galdino.onnx"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 for fp32 layers")
    ap.add_argument("--fp32-layers", nargs="*", default=[])
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--opt-level", type=int, default=3)
    a = ap.parse_args()
    build(a.onnx, a.out, a.fp16, a.fp32_layers, a.tf32, a.workspace_gb, a.opt_level)


if __name__ == "__main__":
    main()
