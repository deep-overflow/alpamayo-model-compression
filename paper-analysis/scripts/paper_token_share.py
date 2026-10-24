"""Paper-style token-share panels (one per objective): share of each layer's importance earned at
vision / text (prompt + generated CoC, additive) / ego-history positions, MLP channels by default
or Q heads. Style replicates experiments/paper/fig_criterion.py --paper-style: fixed 3.5 x 2.8 in
canvas, 12 pt axis labels, 10 pt other text, solid lines, palette #618BC8 / #ECAE3C, y-label shift.
Data: outputs/<run>/metrics_analysis.json (additive_by_type, share_by_type_and_layer), written by
analyze_gradient_anatomy.py.
A fourth argument "split" plots prompt text and generated CoC as separate series (colours as
fig4_same_position: prompt text yellow, CoC purple) and appends _split to the file names.
Usage: python paper_token_share.py [run=gradanat_v1] [suffix=""] [axis=mlp|q] [split]."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_style import BLUE, BRICK, GREY, REPO, SINGLE, TEXT, YELLOW, export

OUT = Path(__file__).resolve().parent / "paperfigs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "gradanat_v1"  # gradient-anatomy run to read
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""          # appended to the file names
AXIS = sys.argv[3] if len(sys.argv) > 3 else "mlp"         # mlp (channels) or q (heads)
UNIT = {"mlp": "mlp", "q": "q_head"}[AXIS]
SPLIT = "split" in sys.argv[4:]
PURPLE = "#8E6BB3"  # generated CoC, as fig4_same_position
VIS, HIST, PT, COC = 0, 1, 2, 3

m = json.loads((REPO / "outputs" / RUN / "metrics_analysis.json").read_text())["additive_by_type"]
print(f"  {RUN}, {AXIS}:")
for key, name, tag in ((f"{AXIS}_fm", f"fig4_token_share_traj_{UNIT}", "I_traj"),
                       (f"{AXIS}_ce", f"fig4_token_share_coc_{UNIT}", "I_CoC ")):
    s = np.array(m[key]["share_by_type_and_layer"])  # (5, 36), types: vision, hist, prompt, coc, sink
    last = 35 if key.endswith("_fm") else 36  # I_traj is identically zero at layer 35
    L = np.arange(last)
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.axvspan(21.5, 35.5, color=GREY, alpha=0.13, lw=0)
    ax.plot(L, s[VIS][:last], color=BLUE, lw=1.5, label="Vision")
    if SPLIT:
        ax.plot(L, s[PT][:last], color=YELLOW, lw=1.5, label="Prompt text")
        ax.plot(L, s[COC][:last], color=PURPLE, lw=1.5, label="CoC")
    else:
        ax.plot(L, (s[PT] + s[COC])[:last], color=YELLOW, lw=1.5, label="Text")
    ax.plot(L, s[HIST][:last], color=BRICK, lw=1.5, label="Ego history")
    ax.set_xlabel("VLM layer index")
    ax.set_ylabel("Share of layer importance")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 7, 14, 21, 28, 35])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2 if SPLIT else 3, fontsize=TEXT, handlelength=1.2,
              handletextpad=0.45, columnspacing=1.2, borderaxespad=0.2)
    export(fig, OUT, name + ("_split" if SPLIT else "") + SUFFIX)
    if SPLIT:
        print(f"    {tag}: L6-17 prompt {s[PT][6:18].mean():.2f} coc {s[COC][6:18].mean():.2f} | L27-34 prompt {s[PT][27:35].mean():.2f} coc {s[COC][27:35].mean():.2f}")
    v, t, h = s[VIS], s[PT] + s[COC], s[HIST]
    print(f"    {tag}: L6-17 vision {v[6:18].mean():.2f} text {t[6:18].mean():.2f} hist {h[6:18].mean():.2f} | "
          f"L27-34 vision {v[27:35].mean():.2f} text {t[27:35].mean():.2f} hist {h[27:35].mean():.2f}")
