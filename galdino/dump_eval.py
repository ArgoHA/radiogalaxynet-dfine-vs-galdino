"""Run Gal-DINO's own `main.py --eval` unchanged and also dump the predictions it scores.

The COCO results JSON is captured inside CocoEvaluator.update from the exact lists that
Gal-DINO's prepare_for_coco_detection / prepare_for_coco_keypoint hand to pycocotools, merged per
detection (both iterate the same top-k list in the same order). main.py prints its usual stats.

Usage (from this directory):
  .venv/bin/python dump_eval.py --split test --out runs/torch_test
  .venv/bin/python dump_eval.py --split val  --out runs/torch_val
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import datasets.coco_eval as ce  # noqa: E402
import main as galmain  # noqa: E402

cli = argparse.ArgumentParser()
cli.add_argument("--split", choices=["test", "val"], default="test")
cli.add_argument("--out", required=True)
cli.add_argument("--ckpt", default="outputs_gal/checkpoint.pth")
a = cli.parse_args()

# main.py always evaluates image_set='test'; for val point --coco_path at a root whose
# test2017/instances_test2017.json are symlinks to the val split (no code change).
coco_path = {"test": "RadioGalaxyNET", "val": "RadioGalaxyNET_val_as_test"}[a.split]

captured = []
_orig_update = ce.CocoEvaluator.update


def update(self, predictions):
    bb = self.prepare_for_coco_detection(predictions)
    kp = self.prepare_for_coco_keypoint(predictions)
    assert len(bb) == len(kp)
    for b, k in zip(bb, kp):
        assert b["image_id"] == k["image_id"] and b["category_id"] == k["category_id"]
        assert b["score"] == k["score"]
        captured.append({**b, "keypoints": k["keypoints"]})
    return _orig_update(self, predictions)


ce.CocoEvaluator.update = update

Path(a.out).mkdir(parents=True, exist_ok=True)
argv = ["-c", "config/DINO/DINO_4scale.py", "--eval", "--resume", a.ckpt,
        "--output_dir", a.out, "--coco_path", coco_path]
parser = argparse.ArgumentParser(parents=[galmain.get_args_parser()])
galmain.main(parser.parse_args(argv))

for r in captured:  # plain python types for JSON
    r["image_id"] = int(r["image_id"])
    r["category_id"] = int(r["category_id"])
with open(Path(a.out) / "results.json", "w") as f:
    json.dump(captured, f)
log = [json.loads(line) for line in open(Path(a.out) / "log.txt")][-1]
json.dump({"bbox": log["test_coco_eval_bbox"], "keypoints": log["test_coco_eval_keypoints"]},
          open(Path(a.out) / "main_stats.json", "w"), indent=1)
print(f"wrote {len(captured)} detections to {a.out}/results.json")
