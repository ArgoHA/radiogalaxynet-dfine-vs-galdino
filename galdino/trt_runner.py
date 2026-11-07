"""Minimal TensorRT 10 runner: persistent torch IO buffers bound once, a dedicated stream and an
optional CUDA graph of the engine (same structure as D-FINE-seg's TRTInference)."""

import tensorrt as trt
import torch

_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
       trt.int64: torch.int64, trt.bool: torch.bool}


class TrtRunner:
    def __init__(self, engine_path, use_cuda_graph=True, device="cuda"):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            t = torch.empty(shape, dtype=_DT[self.engine.get_tensor_dtype(name)], device=device)
            self.context.set_tensor_address(name, t.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input = t
            else:
                self.outputs[name] = t
        self.names = ["scores", "labels", "boxes", "kpts"]
        assert set(self.outputs) == set(self.names), self.outputs.keys()
        self.graph = None
        if use_cuda_graph:
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self.stream):
                self.context.execute_async_v3(self.stream.cuda_stream)
            self.graph = g

    def execute(self):
        """Enqueue the engine on self.stream (graph replay if captured). No sync."""
        if self.graph is not None:
            self.graph.replay()
        else:
            self.context.execute_async_v3(self.stream.cuda_stream)

    def infer(self):
        """input already written -> run -> outputs (scores, labels, boxes, kpts), synced."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.execute()
        torch.cuda.current_stream().wait_stream(self.stream)
        return [self.outputs[n] for n in self.names]
