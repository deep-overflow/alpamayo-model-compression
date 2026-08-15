"""Side-by-side of every quantization / pruning arm measured on one set, with storage.

The two tracks report different things -- pruning removes parameters and shrinks the
checkpoint for real, quantization is *fake* (quantize->dequantize into bf16) so its
saving is projected from the bit allocation. This puts both on one axis: projected
checkpoint size under the arm's own accounting, next to the quality it costs.

Sizes assume every parameter outside the quantized pool stays bf16. `embed_tokens`
(0.638 B) is never quantized here, which is why no arm reaches its nominal bit ratio.

Usage:
  python experiments/evaluation/compare_quant_arms.py --set test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

import eval_lib as el
from analyze_baseline import load_rows

FULL_PARAMS = 11_078_526_194

# arm -> (exp-id stem, rows tag, spec json or None, slim meta or None)
ARMS = [
    ("baseline", "baseline_ada_{s}", "baseline", None, None),
    ("uniform_w8 (VLM)", "uniform_w8_{s}", "uniform_w8", "uniform_w8", None),
    ("qvla_coc_b8 (VLM)", "qvla_coc_b8_{s}", "qvla_coc_b8", "qvla_coc_b8", None),
    ("uniform_w4 (VLM)", "uniform_w4_{s}", "uniform_w4", "uniform_w4", None),
    ("qvla_coc_b4 (VLM)", "qvla_coc_b4_{s}", "qvla_coc_b4", "qvla_coc_b4", None),
    ("uniform_w8 (+expert)", "w8_all_{s}", "uniform_w8_all", "uniform_w8_all", None),
    ("uniform_w4 (+expert)", "w4_all_{s}", "uniform_w4_all", "uniform_w4_all", None),
    ("dual_u40 (prune 24%)", "dual_u40_v2_{s}", "slim_dual_u40_v2", None, "slim_dual_u40_v2"),
]


def projected_gb(spec_tag, slim_dir):
    """bf16 everywhere except the quantized pool; pruning removes params outright."""
    params = FULL_PARAMS
    if slim_dir:
        meta = json.loads((REPO / "outputs" / slim_dir / "slim_meta.json").read_text())
        params -= meta.get("params", {}).get("removed", 0) if "params" in meta else 0
        cfg = json.loads((REPO / "outputs" / slim_dir / "config.json").read_text())
        params = cfg["params"]["slim"]
    if not spec_tag:
        return params * 2 / 1e9
    # make_uniform_spec writes a flat meta; alloc_lib writes one file per budget holding
    # both the guided ("allocated") and the matched uniform ("uniform") accounting
    j = REPO / "outputs" / "quant_specs" / f"{spec_tag}.json"
    key = "allocated"
    if not j.exists():
        j = REPO / "outputs" / "quant_specs" / f"{spec_tag.replace('uniform_w', 'qvla_coc_b')}.json"
        key = "uniform"
    meta = json.loads(j.read_text())
    sub = meta if "projected_pool_gb" in meta else meta[key]
    pool_gb = sub["projected_pool_gb"]
    pool_par = meta.get("pool_params")
    if pool_par is None:                     # alloc_lib reports the pool in bf16 GB
        pool_par = sub["pool_bf16_gb"] * 1e9 / 2
    return pool_gb + (params - pool_par) * 2 / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="test")
    args = ap.parse_args()

    base = None
    print(f"{'arm':22s} {'n':>4} {'minADE':>8} {'vs base':>9} {'NLL':>7} {'degen':>6} "
          f"{'GB':>6} {'ratio':>6}")
    for name, exp, tag, spec, slim in ARMS:
        rows = load_rows(REPO / "outputs" / exp.format(s=args.which), tag)
        if not rows:
            print(f"{name:22s} {'-':>4}  (no rows)")
            continue
        if base is None:
            base = {r["clip_id"]: r for r in rows}
        ade = np.array([r["minADE_rollout"] for r in rows])
        pair = ""
        if name != "baseline":
            d = np.array([r["minADE_rollout"] - base[r["clip_id"]]["minADE_rollout"]
                          for r in rows if r["clip_id"] in base])
            m, lo, hi = el.paired_bootstrap_ci(d)
            star = "" if lo <= 0 <= hi else "*"
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            pair = f"{m:+.4f}{star}"
            extra = f"   [{lo:+.4f}, {hi:+.4f}] p={p:.1e}"
        else:
            extra = ""
        gb = projected_gb(spec, slim)
        print(f"{name:22s} {len(rows):4d} {ade.mean():8.4f} {pair:>9} "
              f"{np.mean([r['nll_self'] for r in rows]):7.4f} "
              f"{np.mean([r['coc_degenerate'] for r in rows]) * 100:5.1f}% "
              f"{gb:6.2f} {FULL_PARAMS * 2 / 1e9 / gb:5.2f}x{extra}")


if __name__ == "__main__":
    main()
