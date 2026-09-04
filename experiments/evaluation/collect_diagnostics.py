"""Flatten the reconstruction-diagnostics runs into one table for the results sheet.

The three evaluation families in `collect_results.py` are all "arm x evaluation set ->
score". The 2026-09-04 diagnostics are a different shape: for one arm they give a
divergence per (quantity, token type), where the quantity is either LOCAL (one
sublayer's output, `run_streamerr.py`) or PROPAGATED (accumulated hidden state and the
KV cache the expert reads, `run_streamprop.py`). They belong in the workbook because
they are the measurement that says a reconstruction arm must NOT be chosen by its fit
error -- but they cannot be squeezed into the master join, so they get their own tab.

One row per (arm, quantity): five stream columns plus the energy-weighted aggregate and
the published capability of that same checkpoint, so a reader can see in one place that
the arm furthest from dense leads LingoQA by 20 points.

Usage:
  .venv/bin/python experiments/evaluation/collect_diagnostics.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STREAMS = ["vision", "prompt_text", "hist", "sink", "coc"]
# val500 minADE@6 mean, LingoQA accuracy -- both already on disk, quoted so the tab is
# self-contained; collect_results.py owns the authoritative copies
CAP = {"dualr": (0.8143, 41.8), "dualr_rep": (0.8437, 52.2),
       "dualr_w": (0.8702, 49.0), "dualr_wl": (0.8271, 72.6)}
FIT = {"dualr": "uniform prefill, CoC 0",
       "dualr_rep": "expert-attention prefill, CoC 0",
       "dualr_w": "expert-attention prefill, CoC 0.16",
       "dualr_wl": "expert-attention prefill, CoC 0.04 + LingoQA train"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="streamerr_v1")
    ap.add_argument("--prop", default="streamprop_v1")
    ap.add_argument("--out", default="results_index")
    args = ap.parse_args()
    L = json.loads((REPO / "outputs" / args.local / "metrics.json").read_text())
    P = json.loads((REPO / "outputs" / args.prop / "metrics.json").read_text())
    out = REPO / "outputs" / args.out
    out.mkdir(parents=True, exist_ok=True)

    def share(den, keys):
        tot = sum(sum(den[k]) for k in keys)
        return {k: sum(den[k]) / tot for k in keys}

    QUANT = [("o_proj (local)", L, "err", "o"), ("down_proj (local)", L, "err", "m"),
             ("hidden (propagated)", P, "rel", "h"), ("cache K (propagated)", P, "rel", "k"),
             ("cache V (propagated)", P, "rel", "v")]
    rows = [["quantity", "kind", "arm", "hessian_fitted_on", *STREAMS,
             "energy_weighted", "energy_share_" + "/".join(STREAMS),
             "val500_minADE6", "lingoqa_acc"]]
    for qname, src, field, key in QUANT:
        w = share(src["den"], [f"{key}_{s}" for s in STREAMS])
        kind = "local" if "local" in qname else "propagated"
        for arm in src[field]:
            per = {s: float(np.nanmedian(src[field][arm][f"{key}_{s}"])) for s in STREAMS}
            agg = float(np.sqrt(sum(w[f"{key}_{s}"] * per[s] ** 2 for s in STREAMS)))
            rows.append([qname, kind, arm, FIT.get(arm, ""),
                         *[f"{per[s]:.4f}" for s in STREAMS], f"{agg:.4f}",
                         "/".join(f"{100 * w[f'{key}_{s}']:.2f}%" for s in STREAMS),
                         f"{CAP[arm][0]:.4f}" if arm in CAP else "",
                         f"{CAP[arm][1]:.1f}" if arm in CAP else ""])
    with (out / "diagnostics.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"{len(rows) - 1} rows -> {out / 'diagnostics.csv'}")


if __name__ == "__main__":
    main()
