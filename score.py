"""Standalone RadioGalaxyNET scorer that reproduces Gal-DINO's `main.py --eval` numbers.

Protocol (copied from Gal-DINO datasets/coco_eval.py + engine.evaluate, see SCORING.md):
  * pycocotools COCOeval, useCats=True, one eval for iouType='bbox' and one for 'keypoints'.
  * params overridden exactly like Gal-DINO's evaluate():
      areaRng    = [[0, 1e10], [0, 24**2], [24**2, 48**2], [48**2, 1e10]]
      areaRngLbl = ['all', 'small', 'medium', 'large']
      kpt_oks_sigmas = [0.107]
    everything else is the pycocotools default (bbox maxDets [1, 10, 100]; keypoints maxDets [20]).
  * bbox results are loaded with only {image_id, category_id, bbox, score} and keypoint results
    with only {image_id, category_id, keypoints, score} (no bbox), as Gal-DINO does. This matters:
    COCO.loadRes then gives a keypoint detection area 0 (single keypoint), which decides which
    unmatched detections are ignored in the medium/large rows.

Input results JSON (optionally .json.gz): list of {image_id, category_id, bbox [x, y, w, h], score, keypoints}.
`keypoints` is [x, y, v] (or Gal-DINO's [x, y]); only x and y are used.

Protocol: no NMS (Gal-DINO's own: full top-300 per image). `--nms 0.7` instead applies greedy
class-agnostic NMS per image at that IoU before scoring, identical to D-FINE-seg's built-in NMS
(torchvision.ops.nms over all classes); it is used only for the "both with NMS" robustness rows.

Usage: python score.py data/RadioGalaxyNET/annotations/test.json RESULTS.json[.gz] [--nms IOU] [--json OUT.json] [--quiet]
"""

import argparse
import contextlib
import gzip
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

AREA_RNG = [[0**2, 1e5**2], [0**2, 24**2], [24**2, 48**2], [48**2, 1e5**2]]
AREA_LBL = ["all", "small", "medium", "large"]
KPT_SIGMAS = np.array([1.07]) / 10


def _run(coco_gt, dets, iou_type, img_ids, verbose):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets) if dets else COCO()
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.params.imgIds = img_ids
    ev.params.areaRng = AREA_RNG
    ev.params.areaRngLbl = AREA_LBL
    ev.params.kpt_oks_sigmas = KPT_SIGMAS
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    if verbose:
        print(f"IoU metric: {iou_type}")
        print(buf.getvalue(), end="")
    return ev


def _ap(ev, area):
    """AP@[.5:.95] for one area row, max maxDets (same formula as pycocotools summarize)."""
    a = AREA_LBL.index(area)
    p = ev.eval["precision"][:, :, :, a, -1]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else -1.0


def nms(results, iou_thr):
    """Greedy class-agnostic NMS per image (keeps the higher score; IoU on [x, y, w, h] boxes)."""
    by_img = {}
    for r in results:
        by_img.setdefault(r["image_id"], []).append(r)
    out = []
    for dets in by_img.values():
        dets.sort(key=lambda r: -r["score"])
        b = np.array([r["bbox"] for r in dets], dtype=np.float64).reshape(-1, 4)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        area = b[:, 2] * b[:, 3]
        alive = np.ones(len(dets), bool)
        for i in range(len(dets)):
            if not alive[i]:
                continue
            out.append(dets[i])
            iw = np.clip(np.minimum(x2[i], x2) - np.maximum(x1[i], x1), 0, None)
            ih = np.clip(np.minimum(y2[i], y2) - np.maximum(y1[i], y1), 0, None)
            inter = iw * ih
            with np.errstate(invalid="ignore"):  # 0/0 for zero-area boxes -> NaN -> kept
                iou = inter / (area[i] + area - inter)
            alive &= ~(iou > iou_thr)
    return out


def score(gt_path, results, img_ids=None, verbose=True, nms_iou=None):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_path)
    if isinstance(results, str):
        opener = gzip.open if results.endswith(".gz") else open
        with opener(results, "rt") as f:
            results = json.load(f)
    img_ids = sorted(coco_gt.getImgIds()) if img_ids is None else sorted(img_ids)
    keep = set(img_ids)
    results = [r for r in results if r["image_id"] in keep]
    if nms_iou is not None:
        results = nms(results, nms_iou)
    bb = [{k: r[k] for k in ("image_id", "category_id", "bbox", "score")} for r in results]
    kp = [{k: r[k] for k in ("image_id", "category_id", "keypoints", "score")} for r in results]

    ev_b = _run(coco_gt, bb, "bbox", img_ids, verbose)
    ev_k = _run(coco_gt, kp, "keypoints", img_ids, verbose)
    s_b, s_k = ev_b.stats, ev_k.stats
    out = {
        "bbox": dict(AP=s_b[0], AP50=s_b[1], AP75=s_b[2], APS=s_b[3], APM=s_b[4], APL=s_b[5]),
        # pycocotools' keypoint summary has no small row; APS is the same formula on area 'small'.
        "keypoints": dict(AP=s_k[0], AP50=s_k[1], AP75=s_k[2], APS=_ap(ev_k, "small"),
                          APM=s_k[3], APL=s_k[4]),
        "stats_bbox": [float(x) for x in s_b],
        "stats_keypoints": [float(x) for x in s_k],
    }
    for t in ("bbox", "keypoints"):
        out[t] = {k: float(v) for k, v in out[t].items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("results")
    ap.add_argument("--json", help="write metrics here")
    ap.add_argument("--quiet", action="store_true", help="skip the pycocotools printout")
    ap.add_argument("--nms", type=float, default=None, metavar="IOU",
                    help="class-agnostic NMS before scoring (robustness rows only, e.g. 0.7)")
    a = ap.parse_args()
    out = score(a.gt, a.results, verbose=not a.quiet, nms_iou=a.nms)
    print("\n            AP    AP50  AP75  APS   APM   APL")
    for t in ("bbox", "keypoints"):
        print(f"{t:10s}" + "".join(f"{100 * out[t][k]:6.1f}"
                                   for k in ("AP", "AP50", "AP75", "APS", "APM", "APL")))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
