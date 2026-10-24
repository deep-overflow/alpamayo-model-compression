"""Shared paper-style export for the paper-analysis panels: replicates
experiments/paper/fig_criterion.py --paper-style (fixed 3.5 x 2.8 in canvas, 12 pt axis labels,
10 pt other text, solid lines, y-label kept inside the canvas, png/pdf/svg at 600 dpi)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
sys.path.insert(0, str(REPO / "experiments" / "paper"))
import fig_criterion as fc  # noqa: E402  (applies the shared rcParams on import)

SINGLE = fc.SINGLE
BLUE, YELLOW, BRICK, GREY = "#618BC8", "#ECAE3C", "#C8615F", "#777777"
LABEL, TEXT = 12, 10


def export(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label):
            t.set_fontsize(LABEL)
        ax.tick_params(axis="both", which="both", labelsize=TEXT)
        for t in ax.texts:
            t.set_fontsize(TEXT)
        for line in ax.lines:
            line.set_linestyle("-")
        ax.grid(axis="y")
    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    margin = 3 * fig.dpi / 72
    for ax in fig.axes:  # keep the long rotated y-label inside the fixed canvas
        label = ax.yaxis.label
        box = label.get_window_extent(renderer)
        shift = min(0, fig.bbox.height - margin - box.y1)
        shift += max(0, margin - box.y0 - shift)
        if shift:
            label.set_y(label.get_position()[1] + shift / ax.bbox.height)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf / .svg")
