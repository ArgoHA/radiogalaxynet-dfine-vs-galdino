"""Build the D-FINE-seg COCO layout from raw RadioGalaxyNET.

Usage: python data/build_dfine_dataset.py [<RadioGalaxyNET root> <out dataset dir>]
(defaults: data/RadioGalaxyNET -> data/dataset)

- <out>/images/<file_name> -> absolute symlink to <root>/<split>/<file_name>
- <out>/{train,val,test}.json = the original JSONs with the same images/ids/categories/area and
  only these geometric fixes (D-FINE-seg's albumentations pipeline rejects boxes outside the image):
  * bbox and polygon coordinates clipped to [0, width] x [0, height] (originals overshoot by <= 1 px)
  * keypoints outside the image -> [0, 0, 0] (unlabeled), num_keypoints -> 0
  D-FINE-seg remaps sorted category ids to 0..N-1 itself, reads `area` (kept = original) as the
  OKS scale and resolves file_name against images/.
Fails if a file_name occurs in more than one split or a referenced image is missing.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:3] or [os.path.join(HERE, "RadioGalaxyNET"), os.path.join(HERE, "dataset")]
root, out = map(os.path.abspath, args)
img_dir = os.path.join(out, "images")
os.makedirs(img_dir, exist_ok=True)


def clip(v, hi):
    return min(max(v, 0.0), hi)


seen = {}
for split in ("train", "val", "test"):
    coco = json.load(open(os.path.join(root, "annotations", f"{split}.json")))
    sizes = {}
    for img in coco["images"]:
        name = img["file_name"]
        if name in seen:
            sys.exit(f"file_name collision: {name} in {seen[name]} and {split}")
        seen[name] = split
        src = os.path.join(root, split, name)
        if not os.path.isfile(src):
            sys.exit(f"missing image: {src}")
        dst = os.path.join(img_dir, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        sizes[img["id"]] = (img["width"], img["height"])

    n_box = n_poly = n_kpt = 0
    for a in coco["annotations"]:
        w, h = sizes[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        x1, y1, x2, y2 = clip(x, w), clip(y, h), clip(x + bw, w), clip(y + bh, h)
        new = [x1, y1, x2 - x1, y2 - y1]
        n_box += new != a["bbox"]
        a["bbox"] = new
        for i, p in enumerate(a["segmentation"]):
            q = [clip(v, w if j % 2 == 0 else h) for j, v in enumerate(p)]
            n_poly += q != p
            a["segmentation"][i] = q
        kp = a["keypoints"]
        for k in range(0, len(kp), 3):
            if kp[k + 2] and not (0 <= kp[k] <= w and 0 <= kp[k + 1] <= h):
                kp[k : k + 3] = [0, 0, 0]
                n_kpt += 1
        a["num_keypoints"] = sum(1 for k in range(2, len(kp), 3) if kp[k] > 0)
    with open(os.path.join(out, f"{split}.json"), "w") as f:
        json.dump(coco, f)
    print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} anns | clipped "
          f"{n_box} boxes, {n_poly} polygon parts | {n_kpt} out-of-image keypoints -> unlabeled")
print(f"{len(seen)} symlinks in {img_dir}")
