"""Why LLM-Pruner OOD-Val262 is 1.612 on A100 but 1.662 on Ada (reports/2026-11-09_openloop-lingoqa-ci.md section 6).

1. Checkpoint identity (CPU): the A100 run loaded outputs/cvpr27/slim_lp_r50 (slim_meta.json only, so load_slim slices the
   base weights); the Ada rows use the paper checkpoint data/alpasim/drivers/lp_r50 (slim_state.pt). Same selection in both
   metas -> compare every paper tensor with the base tensor sliced at the kept Q-head rows/cols and MLP channels.
2. Card shift: A100 - Ada per clip for dense, MoOP (same checkpoint on both cards) and LLM-Pruner, Test500 / OOD-Val262,
   clip bootstrap 200k; share of clips whose CoC text differs and where the shift comes from.

  .venv/bin/python iclr27-rebuttal/code/lp_a100_vs_ada.py
writes outputs/iclr27-rebuttal/lp_a100_vs_ada/{summary.txt, metrics.json}
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "iclr27-rebuttal" / "code"))
sys.path.insert(0, str(REPO / "cvpr27" / "code"))
import ci_openloop_lingoqa as ci
from analyze_a100 import load_rows

O = REPO / "outputs"
PAPER = Path("/mnt/nvme1n1/ad_vla/data/alpasim/drivers/lp_r50")
REBUILD = O / "cvpr27" / "slim_lp_r50"
REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
HUB = Path(os.environ.get("HF_HUB_CACHE", "/mnt/nvme1n1/ad_vla/cache/hub"))
SNAP = HUB / "models--nvidia--Alpamayo-1.5-10B" / "snapshots" / REV
HD = 128
A, F = O / "cvpr27" / "a100", O / "iclr27-rebuttal" / "ada_fill"
PAIRS = {  # arm: {set: ((a100_dir, tag), (ada_dir, tag))}
    "dense": {"test500": ((A / "dense_test2000", "baseline"), (O / "baseline_pred_test", "baseline")),
              "oodval262": ((A / "dense_oodval", "baseline"), (O / "baseline_pred_oodval", "baseline"))},
    "moop": {"test500": ((A / "moop_test2000", "slim_dual_u40_v2"), (O / "dual_u40_v2_pred_test", "slim_dual_u40_v2")),
             "oodval262": ((A / "moop_oodval", "slim_dual_u40_v2"), (O / "dual_u40_v2_pred_oodval", "slim_dual_u40_v2"))},
    "lp": {"test500": ((A / "lp_r50_test2000", "slim_lp_r50"), (F / "lp_r50_test500", "lp_r50")),
           "oodval262": ((A / "lp_r50_oodval", "slim_lp_r50"), (F / "lp_r50_oodval262", "lp_r50"))},
}
HZ = ("1.6s", "3.2s", "6.4s")


def checkpoint_identity():
    ma = json.loads((PAPER / "slim_meta.json").read_text())
    mb = json.loads((REBUILD / "slim_meta.json").read_text())
    same_sel = all(ma[k] == mb[k] for k in ("vlm", "expert", "kvonly_layers", "params"))
    paper = torch.load(PAPER / "slim_state.pt", map_location="cpu", weights_only=True, mmap=True)

    def kept(key, n):
        m = re.search(r"(language_model|expert)\.layers\.(\d+)\.", key)
        sel = (ma["vlm"] if m.group(1) == "language_model" else ma["expert"])[int(m.group(2))]
        if "self_attn" in key:
            h = np.asarray(sel["q"])
            idx = (h[:, None] * HD + np.arange(HD)[None]).ravel()  # (n_kept_heads*128,)
        else:
            idx = np.asarray(sel["mlp"])  # (n_kept_mlp,)
        assert len(idx) == n, (key, len(idx), n)
        return torch.as_tensor(idx)

    n_same = n_sliced = 0
    bad, seen = [], set()
    for f in sorted(SNAP.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():  # noqa: SIM118  safe_open handles are not iterable
                seen.add(k)
                b, p = h.get_tensor(k), paper[k]
                if b.shape != p.shape:
                    if p.shape[0] != b.shape[0]:
                        b = b[kept(k, p.shape[0])]
                    if p.ndim > 1 and p.shape[1] != b.shape[1]:
                        b = b[:, kept(k, p.shape[1])]
                    n_sliced += 1
                else:
                    n_same += 1
                if b.shape != p.shape or b.dtype != p.dtype or not torch.equal(b, p):
                    bad.append(k)
    return {"same_selection_in_both_metas": same_sel, "paper_tensors": len(paper), "base_tensors": len(seen),
            "keys_match": set(paper) == seen, "unchanged_equal": n_same, "sliced_compared": n_sliced,
            "mismatches": bad, "smoke_paper": ma["smoke"], "smoke_rebuild": mb["smoke"]}


def card_shift():
    out = {}
    for s in ci.SETS:
        ids = ci.ids_of(s)
        series, arms = {}, {}
        for arm, sets in PAIRS.items():
            (da, ta), (db, tb) = sets[s]
            ra, rb = load_rows(da, ta), load_rows(db, tb)
            assert all(c in ra and c in rb for c in ids), (arm, s)
            changed = np.array([ra[c]["gen_coc"] != rb[c]["gen_coc"] for c in ids])  # (N,)
            degen = np.array([bool(ra[c]["coc_degenerate"]) or bool(rb[c]["coc_degenerate"]) for c in ids])  # (N,)
            for kind in ci.KINDS:
                for h in HZ:
                    series[f"{arm}|{kind}|{h}"] = [ci.metric(ra[c], kind, h) - ci.metric(rb[c], kind, h) for c in ids]
            d6 = np.asarray(series[f"{arm}|ade|6.4s"])  # (N,)
            arms[arm] = {"coc_changed": int(changed.sum()), "coc_changed_and_degenerate": int((changed & degen).sum()),
                         "degenerate_either_card": float(degen.mean()),
                         "sum_dade64_coc_changed": float(d6[changed].sum()),
                         "sum_dade64_coc_changed_degenerate": float(d6[changed & degen].sum()),
                         "sum_dade64_coc_identical": float(d6[~changed].sum())}
        out[s] = {"n": len(ids), "arms": arms, "stats": ci.summarise(series)}
    return out


def main():
    ident = checkpoint_identity()
    shift = card_shift()
    L = ["# LLM-Pruner A100 vs Ada (paper Table 1 checkpoint vs the A100 rebuild)", "",
         "## 1. Checkpoint identity",
         f"- same selection in both slim_meta.json (vlm, expert, kvonly_layers, params): {ident['same_selection_in_both_metas']}",
         f"- tensors: paper {ident['paper_tensors']}, base {ident['base_tensors']}, keys match {ident['keys_match']}",
         (f"- unchanged tensors equal to base: {ident['unchanged_equal']}; pruned tensors compared at kept indices: "
          f"{ident['sliced_compared']}; mismatches {len(ident['mismatches'])} {ident['mismatches'][:5]}"),
         (f"- smoke CoC (build-time, first calib clip): paper {ident['smoke_paper']['coc']!r} / rebuild "
          f"{ident['smoke_rebuild']['coc']!r}"), "",
         "## 2. A100 - Ada per clip (mean [95% clip bootstrap CI], * = excludes 0)"]
    for s, r in shift.items():
        L += ["", f"### {s} (n={r['n']})", "", "| arm | " + " | ".join(f"dADE {h}" for h in HZ) + " | "
              + " | ".join(f"dFDE {h}" for h in HZ) + " | CoC changed (degenerate) | dADE 6.4 s sum: CoC changed / identical |",
              "|---|" + "---|" * 8]
        for arm, a in r["arms"].items():
            cells = []
            for kind in ci.KINDS:
                for h in HZ:
                    e = r["stats"][f"{arm}|{kind}|{h}"]
                    star = "*" if e["ci"][0] > 0 or e["ci"][1] < 0 else ""
                    cells.append(f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]{star}")
            L.append(f"| {arm} | " + " | ".join(cells) + f" | {a['coc_changed']}/{r['n']} ({a['coc_changed_and_degenerate']})"
                     f" | {a['sum_dade64_coc_changed']:+.2f} / {a['sum_dade64_coc_identical']:+.2f} m |")
    dst = O / "iclr27-rebuttal" / "lp_a100_vs_ada"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps({"checkpoint_identity": ident, "card_shift": shift}, indent=1))
    (dst / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
