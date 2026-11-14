"""Split-half reliability of the within-layer importance ranking, FM (traj) vs CE (coc), A2 vs A1.5 (calib_100 per-clip Taylor).
First identifies the aggregation that reproduces importance.npz, then for 20 random halves computes the per-layer Spearman rho between
the two halves' aggregates and averages over layers. Also the top-50% keep-set overlap between halves (what selection sees)."""
import numpy as np
from scipy.stats import spearmanr

SRC = {"A2": "/home/cvlab21/project/chan/alpamayo2-distill/outputs/a2_importance_v1",
       "A1.5": "/mnt/nvme1n1/ad_vla/outputs/chan/importance_v2"}
AGG = {"abs_sum": lambda x: np.abs(x.sum(0)), "mean_abs": lambda x: np.abs(x).mean(0), "abs_mean": lambda x: np.abs(x.mean(0)),
       "sum": lambda x: x.sum(0), "mean": lambda x: x.mean(0)}
rng = np.random.default_rng(0)
for name, d in SRC.items():
    pc = np.load(f"{d}/importance_perclip.npz")
    ref = np.load(f"{d}/importance.npz")
    for obj in ("traj", "coc"):
        for ax in ("q", "mlp"):
            key = f"{obj}_vlm_{ax}"
            x = pc[key].astype(np.float64)  # (100, L, U)
            r = ref[key].astype(np.float64)  # (L, U)
            agg = None
            for an, f in AGG.items():
                a = f(x)
                if np.allclose(a / np.abs(a).max(), r / np.abs(r).max(), rtol=1e-3, atol=1e-6):
                    agg = (an, f)
                    break
            if agg is None:  # fall back to the closest by rank correlation
                agg = max(AGG.items(), key=lambda kv: spearmanr(kv[1](x).ravel(), r.ravel())[0])
            rhos, ovl = [], []
            for _ in range(20):
                p = rng.permutation(x.shape[0])
                a, b = agg[1](x[p[:50]]), agg[1](x[p[50:]])  # (L, U)
                lr, lo = [], []
                for li in range(x.shape[1]):
                    if np.ptp(a[li]) == 0 or np.ptp(b[li]) == 0:
                        continue
                    lr.append(spearmanr(a[li], b[li])[0])
                    k = x.shape[2] // 2
                    ta, tb = set(np.argsort(-a[li])[:k]), set(np.argsort(-b[li])[:k])
                    lo.append(len(ta & tb) / k)
                rhos.append(np.mean(lr))
                ovl.append(np.mean(lo))
            print(f"{name:<5} {key:<14} agg {agg[0]:<9} split-half rho {np.mean(rhos):.3f}  top-50% overlap {np.mean(ovl):.3f}")
