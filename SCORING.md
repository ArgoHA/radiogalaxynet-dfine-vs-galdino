# RadioGalaxyNET scoring protocol (Gal-DINO `main.py --eval`)

`score.py` reproduces the numbers that Gal-DINO's `main.py --eval` prints, bit for bit: all 12 bbox and 10 keypoint `COCOeval.stats` match to 0.0 on test and val (compare `main_stats.json` and the scored `results.json` written by `galdino/dump_eval.py`). Every model in the comparison must be scored with it:

```
python score.py data/RadioGalaxyNET/annotations/test.json RESULTS.json[.gz] [--json metrics.json]
```

## Source of truth

- In the Gal-DINO repo (`galdino/Gal-DINO` after setup): `datasets/coco_eval.py` (`CocoEvaluator`, `evaluate()` override), `engine.py::evaluate`, `models/dino/dino.py::PostProcess` / `PostProcesskeypoint`.
- pycocotools 2.0.11 (pip). Gal-DINO uses the stock `COCOeval` and changes only `params` inside its own `evaluate()`.
- GT is the original annotation JSON (`RadioGalaxyNET/annotations/{test,val}.json`), loaded as is through `COCO(...)`. Gal-DINO's `base_ds` is `dataset.coco`, so no converted or clipped GT.

## Protocol

| item | value |
|---|---|
| evaluator | pycocotools `COCOeval`, `useCats=True`, run once with `iouType='bbox'` and once with `iouType='keypoints'` |
| images | all images in the GT JSON (420 test, 420 val) |
| area ranges | all `[0, 1e10]`, small `[0, 24²]`, medium `[24², 48²]`, large `[48², 1e10]`. pycocotools uses the GT `area` field, which in RadioGalaxyNET equals the GT box `w*h`. These ranges apply to both bbox and keypoints. |
| IoU thresholds | 0.50:0.05:0.95 (default) |
| bbox maxDets | `[1, 10, 100]` (default). AP rows use 100 per image **per category** |
| keypoint maxDets | `[20]` (pycocotools keypoint default). AP uses 20 per image per category, and `computeOks` also truncates to the top 20 by score |
| OKS | one keypoint, `kpt_oks_sigmas = [0.107]`, so `OKS = exp(-d² / (2 · area · (2σ)²))` with `(2σ)² = 0.045796`, `d` the pixel distance, `area` the GT `area` (= GT box area). All GT keypoints have `v=2` and `num_keypoints=1`, so none are ignored |
| category handling | GT category ids 1..4 = FR-II, FR-I, FR-X, R. Detections with any other id (Gal-DINO emits label 0 from its 5-logit head) never match and do not affect the metrics |
| crowd | none (`iscrowd=0` everywhere) |

Reported numbers: bbox AP/AP50/AP75/APS/APM/APL = `stats[0..5]`. Keypoint AP/AP50/AP75 = `stats[0..2]`, APM = `stats[3]`, APL = `stats[4]`. pycocotools prints no small row for keypoints. The paper's keypoint **APS** (17.6) is the same AP formula on the `small` area row of the keypoint eval: `mean(precision[:, :, :, 1, 0] > -1)`. `score.py` computes it that way and reproduces 17.6.

### How detections are formed (Gal-DINO PostProcess)

- `sigmoid` over the `[900 queries × 5 classes]` logits, top-300 over the flattened scores. Each of the 300 picks becomes one detection: `score` = that sigmoid, `category_id` = class index, box and keypoint from the query. No NMS and no confidence threshold, so 300 detections per image are written.
- The keypoint result uses the **same** `score` and `category_id` as its box. There is no separate keypoint confidence. Keypoints are `[x, y]` in original-image pixels. Gal-DINO writes no visibility value. `score.py` accepts `[x, y]` or `[x, y, v]` and ignores `v`.

### Quirk that matters: keypoint detections have area 0

Gal-DINO loads the keypoint results **without** a `bbox` key, so `COCO.loadRes` takes its keypoint branch and sets each detection's `area` to the extent of its keypoints, which is 0 for a single point. In `evaluateImg`, an unmatched detection is ignored when its area is outside the row's range. As a result:

- `all` and `small` rows (range starts at 0): unmatched keypoint detections count as false positives.
- `medium` and `large` rows: unmatched keypoint detections are **ignored**, so these rows measure recall-ordered precision with no false-positive penalty. That is why keypoint APM/APL (81/83) far exceed keypoint AP (48).

`score.py` reproduces this by stripping `bbox` from the keypoint results. If you score keypoints with a `bbox` present, loadRes uses the bbox branch and gives a different (non-Gal-DINO) APM/APL. Bbox detections get the normal `area = w*h`.

## What a model's results file must contain

Results JSON: a list with one entry per detection.

```json
{"image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "score": 0.93, "keypoints": [x, y, 2]}
```

- `image_id`: the id from the original GT JSON (D-FINE-seg's dataset keeps them).
- `category_id`: the **original** id, 1..4. For D-FINE-seg, `category_id = label + 1`.
- `bbox`: `[x, y, w, h]` in original 450×450 pixel coordinates (not the letterboxed/resized input).
- `score`: the detection confidence. The keypoint metric uses this same score, so a separate keypoint presence/visibility score must not replace it.
- `keypoints`: **required on every detection** (one IR-host point per box). `[x, y]` or `[x, y, v]` in original pixels. `v` is ignored.
- Do not threshold on confidence, or AP is underestimated. Gal-DINO submits its full top-300 per image with no NMS and no threshold. The protocol counts up to 100 per (image, category) for bbox and 20 per (image, category) for keypoints. Adding lower-scored detections never lowers COCO AP (precision is interpolated), so submit the model's full top-K (D-FINE-seg: conf_thresh 0). Whether to apply NMS is the model's own choice; Gal-DINO does not.
- Images with no detections may simply be absent. `score.py` evaluates every GT image regardless.
