"""Place vector panel PDFs on a caption layer (supp_alpamayo2_figs.py). Run with pypdf from uvx, not the repo venv:

  uvx --with pypdf python iclr27-rebuttal/code/supp_alpamayo2_compose.py LAYER.pdf LAYOUT.json OUT.pdf

LAYOUT.json: {"page_in": [W, H], "place": [{"pdf": path, "x_in": x, "y_in": y}, ...]} (lower-left corners, inches).
Each panel page is merged unscaled (it already has the panel canvas size), so the panels stay exactly as exported.
"""

import json
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation

layer_path, layout_path, out_path = sys.argv[1:4]
layout = json.loads(Path(layout_path).read_text())
writer = PdfWriter(clone_from=layer_path)  # pypdf 7: merge onto a page that belongs to a writer
page = writer.pages[0]
w_pt, h_pt = (v * 72 for v in layout["page_in"])
assert abs(float(page.mediabox.width) - w_pt) < 0.5 and abs(float(page.mediabox.height) - h_pt) < 0.5, page.mediabox
for p in layout["place"]:
    panel = PdfReader(p["pdf"]).pages[0]
    page.merge_transformed_page(panel, Transformation().translate(p["x_in"] * 72, p["y_in"] * 72))
writer.compress_identical_objects()
with open(out_path, "wb") as f:
    writer.write(f)
print(f"  {out_path}: {len(layout['place'])} panels on {layout['page_in'][0]:.2f} x {layout['page_in'][1]:.2f} in")
