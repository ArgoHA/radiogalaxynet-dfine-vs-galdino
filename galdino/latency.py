"""Gal-DINO TensorRT latency, batch 1, over the RadioGalaxyNET test set (images pre-decoded in RAM).

Timing boundaries mirror D-FINE-seg (dfine_seg/dl/bench.py + infer/trt_model.py): synchronize,
t0, full call on a decoded uint8 HxWx3 BGR image, synchronize, t1. The engine runs as a CUDA-graph
replay on a dedicated stream; preprocessing goes through a pinned host buffer.

  e2e      preprocess (pinned copy -> H2D uint8 -> GPU bilinear resize to 800x800 + RGB +
           normalize, = Gal-DINO's test transform) + engine (incl. in-graph PostProcess top-300)
           + postprocess (conf filter, scale to original pixels) + D2H -> numpy boxes/scores/
           labels/keypoints on CPU.
  e2e_gpu  same, but stops at final tensors on GPU + synchronize (D-FINE-seg's bench boundary:
           its .cpu() copies happen after the timer).
  engine   engine only (CUDA-graph replay), CUDA events, input already in place.

  .venv/bin/python latency.py [--engine export/galdino_fp16.engine] [--conf 0.5]
                                 [--preproc gpu|cv2] [--split test]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import galdino_common as gc  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402


class GalDinoTRT:
    def __init__(self, engine, h, w, conf, preproc="gpu"):
        self.r = TrtRunner(engine, use_cuda_graph=True)
        self.stream = self.r.stream
        self.h, self.w, self.conf, self.preproc = h, w, conf, preproc
        with torch.cuda.stream(self.stream):
            self.pre = gc.GpuPreprocess(h, w, self.r.input)
            self.scale4 = torch.tensor([w, h, w, h], dtype=torch.float32, device="cuda")
            self.scale2 = torch.tensor([w, h], dtype=torch.float32, device="cuda")
        oh, ow = gc.out_size(h, w)
        self.oh, self.ow = oh, ow
        # cv2 path: CPU resize into a pinned HWC buffer, then H2D + normalize on GPU
        self.pin_big = torch.empty((oh, ow, 3), dtype=torch.uint8, pin_memory=True)
        self.gpu_big = torch.empty((oh, ow, 3), dtype=torch.uint8, device="cuda")

    def _preprocess(self, img):
        if self.preproc == "gpu":
            self.pre(img)
        else:
            cv2.resize(img, (self.ow, self.oh), dst=self.pin_big.numpy(),
                       interpolation=cv2.INTER_LINEAR)
            self.gpu_big.copy_(self.pin_big, non_blocking=True)
            x = self.gpu_big.permute(2, 0, 1)[None].float()
            self.r.input.copy_(torch.addcmul(self.pre.shift, x, self.pre.scale).flip(1))

    def __call__(self, img, to_cpu=True):
        with torch.cuda.stream(self.stream):
            self._preprocess(img)
            self.r.execute()
            scores, labels, boxes, kpts = (self.r.outputs[n] for n in self.r.names)
            keep = torch.nonzero(scores[0] >= self.conf, as_tuple=True)[0]
            out = (scores[0][keep], labels[0][keep], boxes[0][keep] * self.scale4,
                   kpts[0][keep] * self.scale2)
            if to_cpu:
                out = tuple(t.cpu().numpy() for t in out)
        torch.cuda.default_stream().wait_stream(self.stream)
        return out


def gpu_busy():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return q, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(HERE / "export/galdino_fp16.engine"))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--preproc", default="gpu", choices=["gpu", "cv2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    images = json.load(open(gc.DATA / "annotations" / f"{a.split}.json"))["images"]
    imgs = [cv2.imread(str(gc.DATA / a.split / im["file_name"])) for im in images]
    h, w = imgs[0].shape[:2]
    assert all(i.shape == (h, w, 3) for i in imgs)
    procs_before = gpu_busy()

    m = GalDinoTRT(a.engine, h, w, a.conf, a.preproc)
    for i in range(a.warmup):
        m(imgs[i % len(imgs)])
    torch.cuda.synchronize()

    res = {}
    for name, to_cpu in (("e2e", True), ("e2e_gpu", False)):
        lat = []
        for img in imgs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(img, to_cpu=to_cpu)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1e3)
        res[name] = lat

    # engine only: graph replay timed with CUDA events, input in place
    lat = []
    s = m.stream
    for img in imgs:
        m(img)  # puts this image's tensor in the engine input
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            e0.record(s)
            m.r.execute()
            e1.record(s)
        e1.synchronize()
        lat.append(e0.elapsed_time(e1))
    res["engine"] = lat

    procs_after = gpu_busy()
    summary = {k: {"mean_ms": float(np.mean(v)), "median_ms": float(np.median(v)),
                   "p90_ms": float(np.percentile(v, 90))} for k, v in res.items()}
    print(f"Gal-DINO TRT {Path(a.engine).name} | input 1x3x{m.oh}x{m.ow} (from {h}x{w}) | "
          f"preproc={a.preproc} conf={a.conf} | {len(imgs)} {a.split} images, warmup {a.warmup}")
    for k, v in summary.items():
        print(f"  {k:8s} mean {v['mean_ms']:.3f} ms | median {v['median_ms']:.3f} | "
              f"p90 {v['p90_ms']:.3f}")
    print(f"  GPU compute apps before: {procs_before[0].replace(chr(10), '; ') or 'none'}")
    print(f"  GPU compute apps after:  {procs_after[0].replace(chr(10), '; ') or 'none'}")
    print(f"  GPU util, SM clock, power at end: {procs_after[1]}")
    others = [ln for ln in (procs_before[0] + "\n" + procs_after[0]).splitlines()
              if ln.strip() and str(__import__('os').getpid()) not in ln]
    if others:
        print("  WARNING: other processes were on the GPU -> numbers are PROVISIONAL")
    if a.json:
        json.dump({"summary": summary, "args": vars(a), "provisional": bool(others)},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
