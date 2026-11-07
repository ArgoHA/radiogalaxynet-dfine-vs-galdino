"""Score the bundled prediction files (predictions/*.json.gz) and check them against the reported
numbers. Needs only the test annotations and pycocotools, no GPU and no model.

  python verify.py [--gt data/RadioGalaxyNET/annotations/test.json]
"""

import argparse
from pathlib import Path

from score import score

HERE = Path(__file__).resolve().parent
# file -> (box AP, IR-host keypoint AP) in %, two decimals
EXPECTED = {
    "galdino_torch_800_test.json.gz": (53.73, 48.14),
    "galdino_trt_fp16_800_test.json.gz": (53.75, 48.20),
    "galdino_trt_fp16_640_test.json.gz": (52.65, 47.85),
    "dfine_s_torch_640_test.json.gz": (58.41, 61.51),
    "dfine_s_trt_fp16_640_test.json.gz": (58.33, 61.55),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(HERE / "data/RadioGalaxyNET/annotations/test.json"))
    gt = ap.parse_args().gt
    ok = True
    print(f"{'predictions':38s} {'box AP':>7s} {'kpt AP':>7s}  expected")
    for name, (eb, ek) in EXPECTED.items():
        m = score(gt, str(HERE / "predictions" / name), verbose=False)
        b, k = 100 * m["bbox"]["AP"], 100 * m["keypoints"]["AP"]
        good = abs(b - eb) < 0.006 and abs(k - ek) < 0.006
        ok &= good
        print(f"{name:38s} {b:7.2f} {k:7.2f}  {eb:.2f} / {ek:.2f}  {'OK' if good else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)
