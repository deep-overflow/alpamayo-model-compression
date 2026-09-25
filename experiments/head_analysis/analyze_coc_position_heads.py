"""Why do the two scores rank Q heads differently at generated-CoC positions only, and who are the
heads they disagree on? Stored calib_100 anatomy only (outputs/gradanat_v1/anatomy_perclip_q.npz).

  0  per-clip structure of the CoC-position score of each objective (length dependence,
     manoeuvre composition, concentration, per-clip vs pooled agreement)
  1  the disagreeing heads (top / bottom 6 per layer by rank(I_traj@CoC) - rank(I_CoC@CoC)):
     overall importance, arm membership, share of their own score earned at CoC positions,
     manoeuvre composition, dominant position under each objective
  2  residual-stream quantities the anatomy stored, by token type: gradient energy of each loss,
     dot product and per-token cosine between the two losses' residual gradients

Usage:
  python experiments/head_analysis/analyze_coc_position_heads.py [--anatomy gradanat_v1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent))
import eval_lib as el
import sample_cache as sc

VIS, HIST, PT, COC, SINK = range(5)
TYPES = ("vision", "hist", "prompt", "CoC", "sink")
MAN = ("cruise", "accel", "decel_stop", "turn")


def rk(x):
    return np.argsort(np.argsort(x, axis=1), axis=1) / (x.shape[1] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomy", default="gradanat_v1")
    args = ap.parse_args()
    d = REPO / "outputs" / args.anatomy
    pq = np.load(d / "anatomy_perclip_q.npz")
    fm, ce = pq["fm_full"].astype(np.float64), pq["ce"].astype(np.float64)  # (N, 5, 36, 32) signed
    met = json.loads((d / "metrics.json").read_text())
    ids = [r["clip_id"] for r in met["per_clip"]]
    coc_len = np.array([r["n_tok"]["coc"] for r in met["per_clip"]])
    t0 = dict(sc.calib_samples(REPO, "calib_100"))
    buckets = np.array([el.bucket(np.asarray(np.load(sc.path_for("calib", c, t0[c]), allow_pickle=True)["ego_future_xyz"]).reshape(-1, 3)[:, :2]) for c in ids])
    Tc, Cc = np.abs(fm[:, COC]).mean(0), np.abs(ce[:, COC]).mean(0)  # same-position scores at CoC (36, 32)
    Tp, Cp = np.abs(fm.sum(1)).mean(0), np.abs(ce.sum(1)).mean(0)  # pooled (shipped)
    kept = {a: json.loads((REPO / "outputs" / f"slim_{a}_u40_v2" / "slim_meta.json").read_text())["vlm"] for a in ("traj", "coc", "dual")}

    def dom(arr, l, h):
        g = arr[:, :, l, h]
        c4 = (g * np.sign(g.sum(1, keepdims=True))).mean(0)[[VIS, PT, COC, HIST]]
        return ["vision", "prompt", "CoC", "hist"][int(c4.argmax())]

    print("0. per-clip structure of the CoC-position score (all heads, layers 0-34)")
    tc_clip = np.abs(fm[:, COC, :35]).sum((1, 2))
    cc_clip = np.abs(ce[:, COC, :35]).sum((1, 2))
    print(f"   Spearman(clip CoC-position mass, CoC length): I_traj {spearmanr(tc_clip, coc_len)[0]:+.2f}, I_CoC {spearmanr(cc_clip, coc_len)[0]:+.2f}")
    print(f"   Spearman across clips between the two objectives' CoC-position mass: {spearmanr(tc_clip, cc_clip)[0]:+.2f}")
    print("   CoC-position mass by manoeuvre  I_traj: " + " ".join(f"{b} {np.abs(fm[:, COC, :35])[buckets == b].sum() / np.abs(fm[:, COC, :35]).sum():.2f}" for b in MAN)
          + " | I_CoC: " + " ".join(f"{b} {np.abs(ce[:, COC, :35])[buckets == b].sum() / np.abs(ce[:, COC, :35]).sum():.2f}" for b in MAN)
          + " | clip share: " + " ".join(f"{b} {(buckets == b).mean():.2f}" for b in MAN))
    print(f"   10 largest clips carry {np.sort(tc_clip)[::-1][:10].sum() / tc_clip.sum():.0%} of I_traj@CoC mass and "
          f"{np.sort(cc_clip)[::-1][:10].sum() / cc_clip.sum():.0%} of I_CoC@CoC mass")
    rho_clip_all = np.nanmean([spearmanr(np.abs(fm[:, COC, l, h]), np.abs(ce[:, COC, l, h]))[0] for l in range(35) for h in range(32)])
    print(f"   per head, across clips, Spearman(|G_traj@CoC|, |G_CoC@CoC|): mean {rho_clip_all:+.2f}")
    for bname, sl in (("0-21", slice(0, 22)), ("22-34", slice(22, 35))):
        per_clip = np.nanmean([[spearmanr(np.abs(fm[c, COC, l]), np.abs(ce[c, COC, l]))[0] for l in range(sl.start, sl.stop)] for c in range(len(ids))])
        pooled = np.nanmean([spearmanr(Tc[l], Cc[l])[0] for l in range(sl.start, sl.stop)])
        vis_pooled = np.nanmean([spearmanr(np.abs(fm[:, VIS]).mean(0)[l], np.abs(ce[:, VIS]).mean(0)[l])[0] for l in range(sl.start, sl.stop)])
        vis_clip = np.nanmean([[spearmanr(np.abs(fm[c, VIS, l]), np.abs(ce[c, VIS, l]))[0] for l in range(sl.start, sl.stop)] for c in range(len(ids))])
        print(f"   within-layer rank agreement, layers {bname}: at CoC positions on the 100-clip means {pooled:+.2f}, single clips {per_clip:+.2f} | "
              f"at vision positions on the means {vis_pooled:+.2f}, single clips {vis_clip:+.2f}")

    # clip-weighting component: give I_traj@CoC the clip weights of I_CoC@CoC (and vice versa)
    print("\n0b. clip-mixture component: agreement after reweighting one objective's clips to the other's clip mass")
    wT = cc_clip / np.maximum(tc_clip, 1e-30)  # weight that makes I_traj's clip masses match I_CoC's
    wC = tc_clip / np.maximum(cc_clip, 1e-30)
    Tc_w = (np.abs(fm[:, COC]) * wT[:, None, None]).mean(0)
    Cc_w = (np.abs(ce[:, COC]) * wC[:, None, None]).mean(0)
    for bname, sl in (("0-21", slice(0, 22)), ("22-34", slice(22, 35))):
        a0 = np.nanmean([spearmanr(Tc[l], Cc[l])[0] for l in range(sl.start, sl.stop)])
        a1 = np.nanmean([spearmanr(Tc_w[l], Cc[l])[0] for l in range(sl.start, sl.stop)])
        a2 = np.nanmean([spearmanr(Tc[l], Cc_w[l])[0] for l in range(sl.start, sl.stop)])
        # same-clip-set control: agreement of I_traj@CoC on the 10 heaviest FM clips vs I_CoC@CoC on the same 10
        top = np.argsort(tc_clip)[::-1][:10]
        a3 = np.nanmean([spearmanr(np.abs(fm[top, COC, l]).mean(0), np.abs(ce[top, COC, l]).mean(0))[0] for l in range(sl.start, sl.stop)])
        print(f"   layers {bname}: raw {a0:+.2f} | I_traj reweighted to I_CoC's clip masses {a1:+.2f} | I_CoC reweighted to I_traj's {a2:+.2f} | both restricted to the 10 heaviest FM clips {a3:+.2f}")

    rT, rC = rk(Tc), rk(Cc)
    diff = rT - rC
    K = 6
    print("\n1. Q heads at CoC positions: T-fav = top-6 per layer by rank(I_traj@CoC) - rank(I_CoC@CoC); C-fav = bottom-6")
    for bname, layers in (("0-21", range(22)), ("22-34", range(22, 35))):
        rows = {"T-fav": [], "C-fav": []}
        for l in layers:
            order = np.argsort(diff[l])
            rows["C-fav"] += [(l, h) for h in order[:K]]
            rows["T-fav"] += [(l, h) for h in order[-K:]]
        print(f"   == layers {bname}")
        for g, hs in rows.items():
            L_, H_ = np.array([l for l, _ in hs]), np.array([h for _, h in hs])
            keptby = {a: np.mean([h in kept[a][l]["q"] for l, h in hs]) for a in ("traj", "coc", "dual")}
            addT = (fm[:, COC][:, L_, H_] * np.sign(fm.sum(1)[:, L_, H_])).mean(0) / Tp[L_, H_]
            addC = (ce[:, COC][:, L_, H_] * np.sign(ce.sum(1)[:, L_, H_])).mean(0) / Cp[L_, H_]
            mT = {b: np.abs(fm[:, COC][:, L_, H_])[buckets == b].sum() / np.abs(fm[:, COC][:, L_, H_]).sum() for b in MAN}
            mC = {b: np.abs(ce[:, COC][:, L_, H_])[buckets == b].sum() / np.abs(ce[:, COC][:, L_, H_]).sum() for b in MAN}
            rho_clip = np.nanmean([spearmanr(np.abs(fm[:, COC, l, h]), np.abs(ce[:, COC, l, h]))[0] for l, h in hs])
            domT = {t: np.mean([dom(fm, l, h) == t for l, h in hs]) for t in ("vision", "prompt", "CoC", "hist")}
            domC = {t: np.mean([dom(ce, l, h) == t for l, h in hs]) for t in ("vision", "prompt", "CoC", "hist")}
            print(f"   {g}: n={len(hs)} | pooled rank I_traj {rk(Tp)[L_, H_].mean():.2f} I_CoC {rk(Cp)[L_, H_].mean():.2f} | "
                  f"kept by traj {keptby['traj']:.0%} coc {keptby['coc']:.0%} dual {keptby['dual']:.0%}")
            print(f"        share of own pooled score earned at CoC positions: under I_traj {np.nanmean(addT):+.2f}, under I_CoC {np.nanmean(addC):+.2f}")
            print("        CoC-position mass by manoeuvre  I_traj: " + " ".join(f"{b} {v:.2f}" for b, v in mT.items()) + " | I_CoC: " + " ".join(f"{b} {v:.2f}" for b, v in mC.items()))
            print(f"        across clips, Spearman(|G_traj@CoC|, |G_CoC@CoC|) mean {rho_clip:+.2f}")
            print("        dominant position under I_traj: " + " ".join(f"{t} {v:.0%}" for t, v in domT.items()) + " | under I_CoC: " + " ".join(f"{t} {v:.0%}" for t, v in domC.items()))

    print("\n2. residual-stream quantities by token type, mean over clips (layers 0-6 / 7-21 / 22-34)")
    for key in ("res_ce", "res_fm", "res_dot", "res_cos", "direct"):
        if key in pq.files:
            m = np.nanmean(pq[key], 0)  # (36, 5)
            print(f"   {key:7s}: " + " | ".join(f"{t} {m[0:7, i].mean():+.3g} / {m[7:22, i].mean():+.3g} / {m[22:35, i].mean():+.3g}" for i, t in enumerate(TYPES)))
    if "res_ce" in pq.files and "res_fm" in pq.files:
        ce_e, fm_e = np.nanmean(pq["res_ce"], 0), np.nanmean(pq["res_fm"], 0)
        n_tok = np.array([[r["n_tok"][t] for t in ("vision", "hist", "prompt_text", "coc", "sink")] for r in met["per_clip"]]).mean(0)
        print("   per-token residual-gradient energy (energy / n_tok), layers 7-21 / 22-34:")
        for i, t in enumerate(TYPES):
            print(f"      {t:7s} CE {ce_e[7:22, i].mean() / n_tok[i]:.3g} / {ce_e[22:35, i].mean() / n_tok[i]:.3g} | FM {fm_e[7:22, i].mean() / n_tok[i]:.3g} / {fm_e[22:35, i].mean() / n_tok[i]:.3g}")


if __name__ == "__main__":
    main()
