"""Evaluate Gal-DINO through torch / ONNX Runtime / TensorRT with a chosen preprocessing, write a
COCO results JSON and score it with score.py.

  .venv/bin/python eval_backends.py --backend torch_op   --preproc pil --split test
  .venv/bin/python eval_backends.py --backend torch_grid --preproc pil --split test
  .venv/bin/python eval_backends.py --backend trt --engine export/galdino_fp16.engine \
      --preproc gpu --split test
backends: torch_op (custom CUDA MSDA op, = main.py), torch_grid (pure-PyTorch grid_sample MSDA,
the exported path), ort (ONNX Runtime CUDA/CPU), trt (TensorRT engine).
preproc: pil (Gal-DINO's own PIL transform), cv2 (cv2 INTER_LINEAR on CPU), gpu (torch bilinear on
GPU from the uint8 BGR image; only for trt/torch).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import galdino_common as gc  # noqa: E402
from score import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["torch_op", "torch_grid", "ort", "trt"])
    ap.add_argument("--preproc", default="pil", choices=["pil", "cv2", "gpu", "pil_fp16in"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--onnx", default=str(HERE / "export/galdino.onnx"))
    ap.add_argument("--engine", default=str(HERE / "export/galdino_fp16.engine"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--amp", action="store_true", help="torch backends: autocast fp16 reference")
    a = ap.parse_args()

    gt = gc.DATA / "annotations" / f"{a.split}.json"
    images = json.load(open(gt))["images"]
    H, W = 450, 450
    assert all((im["height"], im["width"]) == (H, W) for im in images)
    oh, ow = gc.out_size(H, W)

    if a.backend in ("torch_op", "torch_grid"):
        torch.backends.cudnn.benchmark = False
        if a.backend == "torch_grid":
            gc.use_pytorch_msda()
        model, _ = gc.build_model()
        wrapper = gc.ExportWrapper(model).eval()

        def run(x):
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=a.amp):
                return [t[0].float().cpu().numpy() for t in wrapper(x)]
    elif a.backend == "ort":
        import onnxruntime as ort

        sess = ort.InferenceSession(a.onnx, providers=["CUDAExecutionProvider",
                                                       "CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        assert names == ["scores", "labels", "boxes", "kpts"], names

        def run(x):
            return [o[0] for o in sess.run(None, {"image": x.cpu().numpy()})]
    else:
        from trt_runner import TrtRunner

        runner = TrtRunner(a.engine)

        def run(x):
            runner.input.copy_(x)
            return [t[0].cpu().numpy() for t in runner.infer()]

    gpu_pre = None
    if a.preproc == "gpu":
        assert a.backend != "ort"
        buf = torch.empty((1, 3, oh, ow), dtype=torch.float32, device="cuda")
        gpu_pre = gc.GpuPreprocess(H, W, buf)

    results = []
    for im in images:
        path = gc.DATA / a.split / im["file_name"]
        if a.preproc == "pil":
            x = gc.preprocess_pil(path)[None].cuda()
        elif a.preproc == "pil_fp16in":  # noise probe: PIL input rounded to fp16
            x = gc.preprocess_pil(path)[None].cuda().half().float()
        elif a.preproc == "cv2":
            x = gc.preprocess_cv2(cv2.imread(str(path)))[None].cuda()
        else:
            gpu_pre(cv2.imread(str(path)))
            x = gpu_pre.dst
        scores, labels, boxes, kpts = run(x)
        results += gc.to_coco(im["id"], scores, labels, boxes, kpts, im["width"], im["height"])

    amp = "_amp" if a.amp else ""
    out = Path(a.out or HERE / f"runs/{a.backend}{amp}_{a.preproc}_{a.split}")
    if a.backend == "trt":
        out = Path(a.out or HERE / f"runs/trt_{Path(a.engine).stem}_{a.preproc}_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / "results.json", "w"))
    m = score(str(gt), results, verbose=False)
    json.dump(m, open(out / "score.json", "w"), indent=1)
    print(f"{out.name}: {len(results)} dets")
    for t in ("bbox", "keypoints"):
        print(f"  {t:10s}" + "".join(f"{k}={100 * m[t][k]:.2f} "
                                     for k in ("AP", "AP50", "AP75", "APS", "APM", "APL")))


if __name__ == "__main__":
    main()
