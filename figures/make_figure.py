"""Qualitative figure: infrared-host predictions of D-FINE-seg and Gal-DINO on RadioGalaxyNET test.

  uv run python figures/make_figure.py      -> figures/host_examples.{png,pdf} + printed counts

Uses the bundled TensorRT predictions (predictions/*.json.gz). Each model runs at its own
F1-optimal confidence: the two models calibrate scores very differently (at 0.5 Gal-DINO keeps
only ~0.5 boxes per image), so one shared threshold would be unfair to one of them.
Matching is one-to-one per image: predictions in score order take the unmatched same-class GT
box with the highest IoU >= 0.5. A matched source's host is "correct" when the keypoint has
OKS >= 0.5 with the GT host (sigma 0.107, scale = GT box area, the benchmark's own rule), so the
allowed distance grows with source size.

Panels: the infrared image (inverted grayscale) with radio contours, cropped around the source.
Rows: D-FINE-seg right where Gal-DINO misplaces the host; both right; D-FINE-seg failures
(two wrong hosts, one missed source that Gal-DINO gets, one that both miss).
"""

import gzip
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "RadioGalaxyNET"
PRED = {
    "D-FINE-seg": ROOT / "predictions" / "dfine_s_trt_fp16_640_test.json.gz",
    "Gal-DINO": ROOT / "predictions" / "galdino_trt_fp16_800_test.json.gz",
}
SIGMA2X2 = (2 * 0.107) ** 2
STYLE = {  # validated categorical pair (blue, orange) + shape as the second encoding
    "GT": dict(marker="o", s=150, facecolors="none", edgecolors="#111111", linewidths=1.6),
    "D-FINE-seg": dict(marker="x", s=90, c="#2a78d6", linewidths=2.2),
    "Gal-DINO": dict(marker="+", s=130, c="#eb6834", linewidths=2.2),
}
CLASS = {1: "FR-II", 2: "FR-I", 3: "FR-X", 4: "R"}
N_COLS = 4


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def load_preds(path):
    by_img = {}
    with gzip.open(path, "rt") as f:
        for d in json.load(f):
            by_img.setdefault(d["image_id"], []).append(d)
    for v in by_img.values():
        v.sort(key=lambda d: -d["score"])
    return by_img


def match(gts, preds, conf):
    """Greedy one-to-one matching in one image -> ({gt id: pred}, n false positives)."""
    out, fp = {}, 0
    for p in preds:
        if p["score"] < conf:
            break
        best, best_iou = None, 0.5
        for g in gts:
            if g["id"] in out or g["category_id"] != p["category_id"]:
                continue
            v = iou(g["bbox"], p["bbox"])
            if v >= best_iou:
                best, best_iou = g, v
        if best is None:
            fp += 1
        else:
            out[best["id"]] = p
    return out, fp


def best_conf(gts_by_img, preds):
    """Detection F1-optimal confidence (all GT boxes count, as in the box metric)."""
    n_gt = sum(len(v) for v in gts_by_img.values())
    best = (0.0, 0.5)
    for conf in np.arange(0.02, 0.96, 0.01):
        tp = fp = 0
        for img_id, gts in gts_by_img.items():
            m, f = match(gts, preds.get(img_id, []), conf)
            tp, fp = tp + len(m), fp + f
        f1 = 2 * tp / (2 * tp + fp + (n_gt - tp)) if tp else 0.0
        best = max(best, (f1, float(conf)))
    return best


def host(gt, p):
    """-> (host xy or None, host error px or None, correct bool) for one GT source."""
    if p is None:
        return None, None, False
    kx, ky = p["keypoints"][:2]
    d = math.dist((kx, ky), gt["keypoints"][:2])
    oks = math.exp(-(d**2) / (2 * gt["area"] * SIGMA2X2))
    return (kx, ky), d, oks >= 0.5


def render(ax, img, gt, res):
    x, y, w, h = gt["bbox"]
    # square crop around the source box + true host + predicted hosts, shifted to stay inside
    pts = [(x, y), (x + w, y + h), tuple(gt["keypoints"][:2])]
    pts += [r[0] for r in res.values() if r[0] is not None]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    side = int(min(img.shape[0], span * 1.35 + 24))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x0 = int(min(max(0, cx - side / 2), img.shape[1] - side))
    y0 = int(min(max(0, cy - side / 2), img.shape[0] - side))
    x1, y1 = x0 + side, y0 + side
    ir = img[y0:y1, x0:x1, 2].astype(np.float32)
    radio = img[y0:y1, x0:x1, 0].astype(np.float32) * 256 + img[y0:y1, x0:x1, 1]
    ext = (x0, x1, y1, y0)
    ax.imshow(ir, cmap="Greys", vmin=0, vmax=320, extent=ext, interpolation="bilinear")
    if radio.max() > 0:
        lv = np.geomspace(max(radio[radio > 0].min(), 256), radio.max(), 5)[:-1]
        ax.contour(radio, levels=np.unique(lv), colors="#7a7a7a", linewidths=0.8,
                   extent=(x0, x1, y0, y1), origin="upper")
    gx, gy = gt["keypoints"][:2]
    ax.scatter([gx], [gy], zorder=3, **STYLE["GT"])
    for name in ("Gal-DINO", "D-FINE-seg"):
        if res[name][0] is not None:
            ax.scatter(*res[name][0], zorder=4, **STYLE[name])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])

    def lbl(name):
        _, d, ok = res[name]
        return f"{name}: {'not detected' if d is None else f'{d:.0f} px'}{' ✓' if ok else ''}"

    ax.set_title(f"{CLASS[gt['category_id']]}\n{lbl('D-FINE-seg')}\n{lbl('Gal-DINO')}",
                 fontsize=8, loc="left", color="#222222")


def main():
    gt_json = json.load(open(DATA / "annotations" / "test.json"))
    files = {i["id"]: i["file_name"] for i in gt_json["images"]}
    preds = {k: load_preds(v) for k, v in PRED.items()}
    gts_by_img = {i: [] for i in files}
    for a in gt_json["annotations"]:
        gts_by_img[a["image_id"]].append(a)

    matched, confs = {}, {}
    for k in PRED:
        f1, confs[k] = best_conf(gts_by_img, preds[k])
        matched[k], fp = {}, 0
        for img_id, gts in gts_by_img.items():
            m, f = match(gts, preds[k].get(img_id, []), confs[k])
            matched[k].update(m)
            fp += f
        print(f"{k:11s} F1-optimal conf {confs[k]:.2f}: box F1 {f1:.3f}, "
              f"TP {len(matched[k])}, FP {fp}")

    rows = []
    for a in gt_json["annotations"]:
        kx, ky = a["keypoints"][:2]
        if not (0 <= kx <= 450 and 0 <= ky <= 450):
            continue  # 3 GT hosts lie outside the image; no model can match them
        rows.append((a, {k: host(a, matched[k].get(a["id"])) for k in PRED}))

    n = len(rows)
    ok = {k: sum(r[1][k][2] for r in rows) for k in PRED}
    both = sum(r[1]["D-FINE-seg"][2] and r[1]["Gal-DINO"][2] for r in rows)
    only_d = sum(r[1]["D-FINE-seg"][2] and not r[1]["Gal-DINO"][2] for r in rows)
    only_g = sum(r[1]["Gal-DINO"][2] and not r[1]["D-FINE-seg"][2] for r in rows)
    print(f"{n} GT sources with in-image hosts; source found + host OKS >= 0.5:")
    for k in PRED:
        print(f"  {k:11s} {ok[k]} / {n} ({100 * ok[k] / n:.1f}%)")
    print(f"  both {both}, only D-FINE-seg {only_d}, only Gal-DINO {only_g}, "
          f"neither {n - both - only_d - only_g}")

    def err(r, k):
        return r[1][k][1] if r[1][k][1] is not None else 1e9

    def pick(cands, key, k, skip=()):
        """k examples by key, at most one per image."""
        out, seen = [], {r[0]["image_id"] for r in skip}
        for r in sorted(cands, key=key):
            if r[0]["image_id"] not in seen:
                out.append(r)
                seen.add(r[0]["image_id"])
            if len(out) == k:
                break
        return out

    def found(r, k):
        return r[1][k][1] is not None

    def big(r):
        return -r[0]["area"]

    d_ok = [r for r in rows if r[1]["D-FINE-seg"][2]]
    win = [r for r in d_ok if found(r, "Gal-DINO") and not r[1]["Gal-DINO"][2]]
    same = [r for r in d_ok if r[1]["Gal-DINO"][2]]
    wrong = [r for r in rows if found(r, "D-FINE-seg") and not r[1]["D-FINE-seg"][2]]
    miss_g_ok = [r for r in rows if not found(r, "D-FINE-seg") and r[1]["Gal-DINO"][2]]
    miss_both = [r for r in rows if not found(r, "D-FINE-seg") and not r[1]["Gal-DINO"][2]]
    used = []
    s1 = pick(win, lambda r: -r[1]["Gal-DINO"][1], N_COLS)
    s2 = pick(same, big, N_COLS, skip=s1)
    used = s1 + s2
    s3 = pick(wrong, big, 2, skip=used)
    s3 += pick(miss_g_ok, big, 1, skip=used + s3)
    s3 += pick(miss_both, big, N_COLS - len(s3), skip=used + s3)
    sections = [
        ("D-FINE-seg finds the host, Gal-DINO finds the source but misplaces the host", s1),
        ("Both find the host", s2),
        ("D-FINE-seg failures: wrong host, then source not detected", s3),
    ]

    fig = plt.figure(figsize=(2.5 * N_COLS, 3.25 * len(sections)), facecolor="white")
    gs = fig.add_gridspec(len(sections), N_COLS, left=0.02, right=0.98, top=0.95, bottom=0.05,
                          wspace=0.08, hspace=0.62)
    for i, (title, picks) in enumerate(sections):
        axes = [fig.add_subplot(gs[i, j]) for j in range(N_COLS)]
        top = axes[0].get_position().y1
        fig.text(0.02, top + 0.068, title, fontsize=10.5, fontweight="bold", color="#111111")
        for ax, r in zip(axes, picks):
            a = r[0]
            img = cv2.imread(str(DATA / "test" / files[a["image_id"]]), cv2.IMREAD_UNCHANGED)
            render(ax, img, a, r[1])
        for ax in axes[len(picks):]:
            ax.axis("off")
    handles = [
        Line2D([], [], ls="none", marker="o", mfc="none", mec="#111111", ms=10, mew=1.6,
               label="true host"),
        Line2D([], [], ls="none", marker="x", color="#2a78d6", ms=9, mew=2.2, label="D-FINE-seg"),
        Line2D([], [], ls="none", marker="+", color="#eb6834", ms=11, mew=2.2, label="Gal-DINO"),
        Line2D([], [], color="#7a7a7a", lw=0.8, label="radio contours"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    out = ROOT / "figures" / "host_examples"
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
