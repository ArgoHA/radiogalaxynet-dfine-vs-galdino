"""Score the bundled prediction files (predictions/*.json.gz) and check them against the reported
numbers. Needs only the test annotations and pycocotools, no GPU and no model.

  python verify.py [--gt data/RadioGalaxyNET/annotations/test.json]
"""

import argparse
from pathlib import Path

from score import score

HERE = Path(__file__).resolve().parent
# (file, post-hoc class-agnostic NMS IoU or None, box AP, IR-host keypoint AP) in %, two decimals.
# The protocol is no NMS for both models (Gal-DINO's own); the NMS rows are the robustness check.
EXPECTED = [
    ("galdino_torch_800_test.json.gz", None, 53.73, 48.14),
    ("galdino_trt_fp16_800_test.json.gz", None, 53.75, 48.20),
    ("galdino_trt_fp16_640_test.json.gz", None, 52.65, 47.85),
    ("dfine_s_torch_640_test.json.gz", None, 67.21, 70.45),
    ("dfine_s_trt_fp16_640_test.json.gz", None, 67.07, 70.50),
    ("galdino_trt_fp16_800_test.json.gz", 0.7, 43.99, 39.50),
    ("galdino_trt_fp16_640_test.json.gz", 0.7, 43.26, 39.19),
    ("dfine_s_trt_fp16_640_test.json.gz", 0.7, 58.33, 61.55),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(HERE / "data/RadioGalaxyNET/annotations/test.json"))
    gt = ap.parse_args().gt
    ok = True
    print(f"{'predictions':34s} {'NMS':>4s} {'box AP':>7s} {'kpt AP':>7s}  expected")
    for name, nms, eb, ek in EXPECTED:
        m = score(gt, str(HERE / "predictions" / name), verbose=False, nms_iou=nms)
        b, k = 100 * m["bbox"]["AP"], 100 * m["keypoints"]["AP"]
        good = abs(b - eb) < 0.006 and abs(k - ek) < 0.006
        ok &= good
        print(f"{name:34s} {nms or '-':>4} {b:7.2f} {k:7.2f}  {eb:.2f} / {ek:.2f}  "
              f"{'OK' if good else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)
