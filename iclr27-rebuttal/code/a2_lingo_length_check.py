"""Is the u50 LingoQA win a Lingo-Judge length artefact? Paired MoOP - X accuracy within strata of answer length, and accuracy by
question type (yes/no vs open). Segment bootstrap as in lingo_block."""
import json

import numpy as np

O = "/mnt/nvme1n1/ad_vla/outputs/chan/iclr27-rebuttal"


def load(arm):
    return {r["question_id"]: r for r in json.load(open(f"{O}/a2_lingo_{arm}/scored.json"))}


def boot(d, seg, n=50000, seed=0):
    keys, inv = np.unique(seg, return_inverse=True)
    S = np.bincount(inv, weights=d, minlength=len(keys))
    C = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n, len(keys)))
    m = S[idx].sum(1) / C[idx].sum(1)
    return np.percentile(m, [2.5, 97.5])


for alloc in ("_u50", "_u50_q8", ""):
    M = load(f"moop{alloc}")
    for other in ("fm_only", "ce_only"):
        X = load(f"{other}{alloc}")
        qs = sorted(set(M) & set(X))
        seg = np.array([M[q]["segment_id"] for q in qs])
        cm = np.array([float(M[q]["correct"]) for q in qs])
        cx = np.array([float(X[q]["correct"]) for q in qs])
        wm = np.array([len(str(M[q]["prediction"]).split()) for q in qs])
        wx = np.array([len(str(X[q]["prediction"]).split()) for q in qs])
        yn = np.array([str(M[q]["question"]).strip().lower().split()[0] in
                       ("is", "are", "do", "does", "can", "should", "will", "was", "were", "has", "have", "would") for q in qs])
        d = cm - cx
        print(f"\nu{alloc or '40'} MoOP - {other}: all {d.mean() * 100:+.1f}pp {np.round(boot(d, seg) * 100, 1)} "
              f"| median words MoOP {np.median(wm):.0f} vs {np.median(wx):.0f}")
        for lab, sel in (("both answers <=3 words", (wm <= 3) & (wx <= 3)), ("both answers >3 words", (wm > 3) & (wx > 3)),
                         ("|len diff| <= 2 words", np.abs(wm - wx) <= 2), ("yes/no questions", yn), ("open questions", ~yn)):
            if sel.sum() < 30:
                print(f"  {lab:<24} n={sel.sum()} (too few)")
                continue
            print(f"  {lab:<24} n={sel.sum():3d}  {d[sel].mean() * 100:+.1f}pp {np.round(boot(d[sel], seg[sel]) * 100, 1)}  "
                  f"(MoOP {cm[sel].mean() * 100:.1f} vs {cx[sel].mean() * 100:.1f})")
