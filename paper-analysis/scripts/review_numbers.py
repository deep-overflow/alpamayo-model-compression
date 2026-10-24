"""Numbers for the reply to the Finding-1 review: (4) cross-objective mass loss and bottom-k
Jaccard at the shipped budget, (5) every quantity under one band convention 0-6 / 7-21 / 22-34,
(6) split-half reliabilities by band and a clip-bootstrap CI for the per-layer Q-head rho."""
import json
import numpy as np
from scipy.stats import spearmanr

REPO = "/home/cvlab21/project/chan/alpamayo-model-compression"
BANDS = {"0-6": slice(0, 7), "7-21": slice(7, 22), "22-34": slice(22, 35), "0-21": slice(0, 22)}
z = np.load(f"{REPO}/outputs/importance_v2/importance.npz")
pc = np.load(f"{REPO}/outputs/importance_v2/importance_perclip.npz")
KEEP = {"q": 19, "mlp": 7390}

def rho_layers(a, b):
    return np.array([spearmanr(a[l], b[l])[0] if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan for l in range(a.shape[0])])

print("=== (4) shipped budget: removed sets by each objective, per layer, layers 0-34 ===")
for ax in ("q", "mlp"):
    T, C = z[f"traj_vlm_{ax}"], z[f"coc_vlm_{ax}"]
    n = T.shape[1]; k_rm = n - KEEP[ax]
    jac, own_t, own_c, cross_tc, cross_ct = [], [], [], [], []
    for l in range(35):
        rt, rc = set(np.argsort(T[l])[:k_rm]), set(np.argsort(C[l])[:k_rm])  # bottom-k removed
        jac.append(len(rt & rc) / len(rt | rc))
        own_t.append(T[l][list(rt)].sum() / T[l].sum()); own_c.append(C[l][list(rc)].sum() / C[l].sum())
        cross_tc.append(C[l][list(rt)].sum() / C[l].sum())  # prune by traj -> I_CoC mass removed
        cross_ct.append(T[l][list(rc)].sum() / T[l].sum())  # prune by coc  -> I_traj mass removed
    jac, own_t, own_c, cross_tc, cross_ct = map(np.array, (jac, own_t, own_c, cross_tc, cross_ct))
    rand = k_rm / n
    for name, sl in BANDS.items():
        print(f"  {ax:3s} L{name:5s} Jaccard(removed sets) {jac[sl].mean():.2f} | I_CoC mass removed: by own rank {own_c[sl].mean():.2f}, by traj rank {cross_tc[sl].mean():.2f} | "
              f"I_traj mass removed: by own rank {own_t[sl].mean():.2f}, by coc rank {cross_ct[sl].mean():.2f} | random {rand:.2f}")

print("\n=== (5) rank agreement, raw, one band convention ===")
for ax in ("q", "mlp"):
    r = rho_layers(z[f"traj_vlm_{ax}"], z[f"coc_vlm_{ax}"])
    print(f"  {ax:3s} " + "  ".join(f"L{b} {np.nanmean(r[s]):.2f}" for b, s in BANDS.items()) + f" | L7 {r[7]:.2f} L6 {r[6]:.2f} L0-5 {np.nanmean(r[:6]):.2f}")

m = json.load(open(f"{REPO}/outputs/gradanat_v1/metrics_analysis.json"))
print("\n=== (5) typed shares (additive), one band convention; text = prompt + CoC ===")
for key in ("mlp_fm", "mlp_ce", "q_fm", "q_ce"):
    s = np.array(m["additive_by_type"][key]["share_by_type_and_layer"])
    last = 35 if key.endswith("fm") else 36
    for b, sl in BANDS.items():
        v, h, p, c = (s[i][sl].mean() for i in (0, 1, 2, 3))
        print(f"  {key:6s} L{b:5s} vision {v:.2f} text {p + c:.2f} (prompt {p:.2f} + CoC {c:.2f}) hist {h:.2f}")
    if key == "q_fm":
        print(f"  q_fm layer 34: vision {s[0][34]:.2f} text {s[2][34] + s[3][34]:.2f} hist {s[1][34]:.2f} | layer 33: vision {s[0][33]:.2f}")
    if key == "mlp_fm":
        print(f"  mlp_fm layer 34: vision {s[0][34]:.2f} text {s[2][34] + s[3][34]:.2f} hist {s[1][34]:.2f} | layer 33: vision {s[0][33]:.2f}")

print("\n=== (6) split-half reliabilities (self-agreement) and corrected cross, by band ===")
for ax in ("q", "mlp"):
    for key in ("pooled (the shipped score)", "both at VISION positions", "both at prompt-text positions only", "both at CoC positions only"):
        if key not in m["P2"][ax]:
            continue
        b = m["P2"][ax][key]["per_layer"]; st, sc, cr = (np.array(b[k]) for k in ("self_traj", "self_coc", "cross"))
        ok = (st > 0.2) & (sc > 0.2); corr = np.where(ok, cr / np.sqrt(np.where(ok, st * sc, 1)), np.nan)
        print(f"  {ax:3s} {key[:34]:34s} " + " | ".join(f"L{bn}: self {np.nanmean(st[s]):.2f}/{np.nanmean(sc[s]):.2f} raw {np.nanmean(cr[s]):.2f} corr {np.nanmean(corr[s]):.2f}" for bn, s in BANDS.items() if bn != "0-21"))

print("\n=== (6) Q-head per-layer rho: clip-bootstrap 95% CI (B=300) ===")
rng = np.random.default_rng(0)
T, C = pc["traj_vlm_q"].astype(np.float64), pc["coc_vlm_q"].astype(np.float64)
point = rho_layers(T.mean(0), C.mean(0))
boot = np.stack([rho_layers(T[i].mean(0), C[i].mean(0)) for i in (rng.integers(0, 100, 100) for _ in range(300))])
lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
w = hi - lo
print(f"  CI half-width: median all {np.nanmedian(w[:35]) / 2:.2f}, late {np.nanmedian(w[22:35]) / 2:.2f} | layer-to-layer |diff| late median {np.nanmedian(np.abs(np.diff(point[22:35]))):.2f}")
for l in (23, 26, 28, 32, 33, 34):
    print(f"    L{l}: rho {point[l]:+.2f} [{lo[l]:+.2f}, {hi[l]:+.2f}]")
# band-mean CI
bm = boot[:, 22:35].mean(1); print(f"  late band mean {np.nanmean(point[22:35]):.3f} [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]")
bm = boot[:, 7:22].mean(1); print(f"  7-21 band mean {np.nanmean(point[7:22]):.3f} [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]")
bm = boot[:, 0:7].mean(1); print(f"  0-6  band mean {np.nanmean(point[0:7]):.3f} [{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]")
