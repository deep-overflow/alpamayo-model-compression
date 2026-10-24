"""Expert-score checks for the paper's method/analysis sections.
1b  per-step scale: mean_u A[s, l, u], averaged over the 36 expert layers, per step
1c  z-mean vs raw-mean rankings: per-layer overlap of the top-516 channels (8256 - 7740) and
    within-layer Spearman between the two scores; also the shipped znorm file reproduced
2b  MLP: min over the 45 step pairs of the step-to-step within-layer Spearman (36-layer mean)
2c  cumulative importance C(r) of the raw step mean: MLP C(87.5%), C(93.75%); Q C(93.75%)
Data: outputs/stepimp_fm_perstep_v2/step_importance.npz (q_abs_step (10,36,16),
mlp_abs_step (10,36,8256)); outputs/importance_stepexp_znorm/importance.npz."""
import numpy as np
from scipy import stats

R = "/home/cvlab21/project/chan/alpamayo-model-compression"
with np.load(f"{R}/outputs/stepimp_fm_perstep_v2/step_importance.npz") as s:
    Q, M = s["q_abs_step"].astype(np.float64), s["mlp_abs_step"].astype(np.float64)
with np.load(f"{R}/outputs/importance_stepexp_znorm/importance.npz") as z:
    zq_ship, zm_ship = z["traj_exp_q"], z["traj_exp_mlp"]
n_step, n_layer, U = M.shape
KEEP = 8256 - 7740  # 516 channels kept per layer at 93.75%


def zs(a):  # (S, L, U) -> within-(step, layer) z-score
    return (a - a.mean(-1, keepdims=True)) / np.maximum(a.std(-1, keepdims=True), 1e-12)


print("## 1b per-step scale of the expert MLP score: mean over channels of A[s,l,u], averaged over 36 layers")
scale = M.mean(-1).mean(-1)  # (10,)
print("   step :", " ".join(f"{i:>7d}" for i in range(n_step)))
print("   mean :", " ".join(f"{v:7.4f}" for v in scale), f"| max/min {scale.max() / scale.min():.2f}x")
print("   Q head scale per step:", " ".join(f"{v:.4f}" for v in Q.mean(-1).mean(-1)))

print("\n## 1c z-mean vs raw-mean (MLP)")
zmean, rmean = zs(M).mean(0), M.mean(0)  # (36, 8256)
rep = np.abs(zmean - zm_ship).max()
print(f"   shipped importance_stepexp_znorm traj_exp_mlp == mean_s z(A_s)? max |diff| {rep:.2e}; "
      f"Spearman(shipped, reproduced) per layer min {min(stats.spearmanr(zmean[l], zm_ship[l]).statistic for l in range(n_layer)):.4f}")
ov, rho = [], []
for l in range(n_layer):
    tz, tr = set(np.argsort(zmean[l])[::-1][:KEEP]), set(np.argsort(rmean[l])[::-1][:KEEP])
    ov.append(len(tz & tr) / KEEP)
    rho.append(stats.spearmanr(zmean[l], rmean[l]).statistic)
print(f"   top-{KEEP} overlap per layer: mean {np.mean(ov):.3f}, min {np.min(ov):.3f} (layer {int(np.argmin(ov))}), max {np.max(ov):.3f}")
print(f"   within-layer Spearman(z-mean, raw-mean): mean {np.mean(rho):.3f}, min {np.min(rho):.3f}")
zq, rq = zs(Q).mean(0), Q.mean(0)
print(f"   Q heads: shipped znorm reproduced max |diff| {np.abs(zq - zq_ship).max():.2e}; "
      f"top-1 (of 16) same head in {np.mean([np.argmax(zq[l]) == np.argmax(rq[l]) for l in range(n_layer)]):.0%} of layers; Spearman mean {np.mean([stats.spearmanr(zq[l], rq[l]).statistic for l in range(n_layer)]):.3f}")

print("\n## 2a/2b step-to-step rank agreement (within-layer Spearman, 36-layer mean)")
for name, a in (("Q head", Q), ("MLP", M)):
    Mx = np.ones((n_step, n_step))
    for i in range(n_step):
        for j in range(i + 1, n_step):
            Mx[i, j] = Mx[j, i] = np.nanmean([stats.spearmanr(a[i, l], a[j, l]).statistic for l in range(n_layer)])
    adj = [Mx[i, i + 1] for i in range(n_step - 1)]
    off = Mx[np.triu_indices(n_step, 1)]
    print(f"   {name:7s} adjacent mean {np.mean(adj):.3f} | rho(0,9) {Mx[0, 9]:.3f} | 45-pair mean {off.mean():.3f} min {off.min():.3f} "
          f"(pair {[(i, j) for i in range(n_step) for j in range(i + 1, n_step)][int(np.argmin(off))]})")

print("\n## 2c cumulative importance of the raw step mean, C(r) = bottom floor(rU) share, 36-layer mean")
for name, a, fracs in (("MLP", rmean, (0.875, 0.9, 0.9375)), ("Q head", rq, (0.875, 0.9375))):
    for f in fracs:
        k = int(np.floor(f * a.shape[1]))
        per = [np.cumsum(np.sort(a[l]))[k - 1] / a[l].sum() for l in range(n_layer)]
        print(f"   {name:7s} C({f:.4f}) = bottom {k} of {a.shape[1]}: {np.mean(per):.4f} (layer min {np.min(per):.4f}, max {np.max(per):.4f})")
