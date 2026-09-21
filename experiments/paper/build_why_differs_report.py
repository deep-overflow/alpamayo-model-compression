"""Build reports/evaluation/2026-09-21_why-importance-differs.html.

The manuscript panels in figures/ are 600-dpi PNGs (2100 x 1680); inlined as they are the
report would be ~6 MB. This shrinks the ones the template uses to 1050 px wide into a
temporary directory as 128-colour palette PNGs and hands that to build_report.py, which
does the inlining.

Usage:
  .venv/bin/python experiments/paper/build_why_differs_report.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "experiments" / "head_analysis" / "whydiffers_report_template.html"
OUT = REPO / "reports" / "evaluation" / "2026-09-21_why-importance-differs.html"
WIDTH = 1050


def main():
    names = sorted(set(re.findall(r"PLOT::fig::([\w.-]+\.png)", TEMPLATE.read_text())))
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            im = Image.open(REPO / "figures" / name).convert("RGB")
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
            # line charts: a 128-colour palette is visually lossless and a third of the size
            im.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE).save(Path(tmp) / name, optimize=True)
        subprocess.run(
            [sys.executable, str(REPO / "experiments" / "head_analysis" / "build_report.py"),
             str(TEMPLATE), str(OUT), f"fig={tmp}",
             f"ga={REPO / 'outputs' / 'gradanat_v1' / 'plots'}"], check=True)


if __name__ == "__main__":
    main()
