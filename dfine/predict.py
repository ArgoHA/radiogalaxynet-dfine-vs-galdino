"""D-FINE-seg on RadioGalaxyNET through its own shipped inference wrapper (`load_model`:
.pt -> TorchModel, .engine -> TRTModel). Writes a COCO results JSON for score.py, or times e2e.

  uv run python dfine/predict.py --model weights/model.pt --out preds.json
  uv run python dfine/predict.py --model weights/model.engine --out preds.json
  uv run python dfine/predict.py --model weights/model.engine --latency --conf 0.5 --repeats 3

Same protocol as Gal-DINO: top-300 over queries x classes, no NMS (`--nms 1` turns on D-FINE-seg's
class-agnostic NMS at IoU 0.7, used only for the robustness row; `score.py --nms` does the same
post hoc for any results file). Scoring uses conf 0.001 (COCO AP needs the low-score tail;
Gal-DINO submits its full top-300). Latency uses conf 0.5, the operating point, matching
galdino/latency.py.
e2e latency = decoded uint8 BGR image in RAM -> final boxes / scores / labels / keypoints on CPU.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from dfine_seg import load_model

DATA = Path(__file__).resolve().parents[1] / "data" / "RadioGalaxyNET"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(DATA), help="raw RadioGalaxyNET root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms", type=int, default=0, help="1 = class-agnostic NMS (not the protocol)")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="latency passes over the split")
    args = ap.parse_args()
    data = Path(args.data)

    model = load_model(args.model, conf_thresh=args.conf, apply_nms=bool(args.nms))
    images = json.load(open(data / "annotations" / f"{args.split}.json"))["images"]
    imgs = [cv2.imread(str(data / args.split / i["file_name"])) for i in images]

    for _ in range(20):
        model(imgs[0])
    dets, lat = [], []
    for _ in range(args.repeats if args.latency else 1):
        for info, img in zip(images, imgs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = model(img)[0]
            r = {k: v.cpu() if torch.is_tensor(v) else v for k, v in r.items()}  # results on CPU
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
            if args.latency:
                continue
            for box, score, label, kp in zip(r["boxes"], r["scores"], r["labels"], r["kpts"]):
                x1, y1, x2, y2 = box.tolist()
                x, y = kp[0, :2].tolist()
                dets.append(
                    {
                        "image_id": info["id"],
                        "category_id": int(label) + 1,  # D-FINE labels 0..3 -> original ids 1..4
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                        "keypoints": [round(x, 2), round(y, 2), 1],
                    }
                )
    print(
        json.dumps(
            {
                "latency_ms": round(float(np.mean(lat)), 3),
                "median_ms": round(float(np.median(lat)), 3),
                "n": len(lat),
                "dets": len(dets),
            }
        )
    )
    if args.out:
        json.dump(dets, open(args.out, "w"))
