"""Accuracy vs. speed chart for the README, in a light and a dark variant.

  uv run python figures/make_speed_chart.py   -> figures/speed_accuracy{,_dark}.png

The numbers are the TensorRT fp16 rows of the README results table (test split, no NMS for
either model; e2e latency on an RTX 5070 Ti, batch 1). Gal-DINO's two points are the same
released checkpoint run at two test sizes, so they are joined by a line.
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
# name, input, e2e ms, box AP, IR-host keypoint AP
GALDINO = [("Gal-DINO", "640²", 7.76, 52.65, 47.85), ("Gal-DINO", "800²", 11.56, 53.75, 48.20)]
DFINE = [("D-FINE-seg S", "640²", 1.77, 67.07, 70.50)]
THEMES = {  # categorical pairs validated per surface (dataviz validate_palette.js)
    "": dict(bg="#ffffff", text="#18181b", muted="#71717a", grid="#e4e4e7",
             ours="#2a78d6", rival="#eb6834"),
    "_dark": dict(bg="#0d1117", text="#e6edf3", muted="#8b949e", grid="#30363d",
                  ours="#3b82f6", rival="#e2622e"),
}


def r1(v):
    """One decimal, rounded half up like the tables (52.65 -> 52.7, not float's 52.6)."""
    return Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def panel(ax, col, title, th):
    ax.set_facecolor(th["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.grid(axis="y", color=th["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=th["muted"], length=0, labelsize=9)
    ax.set_xlim(0, 14)
    ax.set_ylim(40, 75)
    ax.set_xticks(range(2, 14, 2), [f"{t} ms" for t in range(2, 14, 2)])
    ax.set_title(title, loc="left", fontsize=11, color=th["text"], fontweight="bold", pad=10)

    xs, ys = [r[2] for r in GALDINO], [r[col] for r in GALDINO]
    ax.plot(xs, ys, color=th["rival"], lw=2, zorder=2)
    ax.scatter(xs, ys, s=60, color=th["rival"], edgecolors=th["bg"], linewidths=2, zorder=3)
    for name, inp, ms, *ap in GALDINO:
        ax.annotate(f"{name} {inp}\n{r1(ap[col - 3])} AP · {r1(ms)} ms", (ms, ap[col - 3]),
                    xytext=(0, 12), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=th["muted"])
    for name, inp, ms, *ap in DFINE:
        ax.scatter([ms], [ap[col - 3]], s=90, color=th["ours"], edgecolors=th["bg"],
                   linewidths=2, zorder=4)
        ax.annotate(f"{name} {inp}\n{r1(ap[col - 3])} AP · {r1(ms)} ms", (ms, ap[col - 3]),
                    xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=th["text"], fontweight="bold")


def main():
    for suffix, th in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), facecolor=th["bg"])
        fig.subplots_adjust(left=0.05, right=0.98, top=0.74, bottom=0.08, wspace=0.12)
        panel(axes[0], 3, "Box AP", th)
        panel(axes[1], 4, "Infrared-host keypoint AP", th)
        fig.text(0.05, 0.95, "Accuracy vs. speed, RadioGalaxyNET test split", fontsize=13,
                 color=th["text"], fontweight="bold", va="top")
        fig.text(0.05, 0.885, "TensorRT fp16, batch 1, RTX 5070 Ti, end-to-end latency per "
                 "image. Up and to the left is better.", fontsize=9.5, color=th["muted"],
                 va="top")
        fig.savefig(OUT / f"speed_accuracy{suffix}.png", dpi=200, facecolor=th["bg"])
        plt.close(fig)
        print(f"wrote figures/speed_accuracy{suffix}.png")


if __name__ == "__main__":
    main()
