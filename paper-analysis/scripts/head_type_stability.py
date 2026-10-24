"""Does a Q head's dominant token position (the position type that carries most of its score)
hold on held-out clips? For calib_100 and the three held-out anatomy runs (val100 / OOD / test100)
the per-head additive split C[t,l,u] = E_c[G sign(sum_t G)] gives each head a dominant type
(vision / prompt-text / generated CoC / ego-history) and continuous per-type shares C_t / I.
Reported per objective and band (0-21 / 22-34): (1) share of heads by dominant type in every set;
(2) head-wise agreement of the dominant type with calib_100 -- raw fraction and Cohen's kappa
(chance-corrected), for all heads and for the top-19 heads of calib_100's own ranking -- next to
the within-calib split-half agreement (two disjoint 50-clip halves, 50 draws), which is the noise
ceiling for a label estimated from 50 clips; (3) Spearman across sets of the per-head vision and
text shares (continuous version of the same question), again next to the split-half value.
Data: outputs/{gradanat_v1,gradanat_val100,gradanat_ood,gradanat_test100}/anatomy_perclip_q.npz."""
import numpy as np
from scipy.stats import spearmanr

REPO = "/home/cvlab21/project/chan/alpamayo-model-compression"
VIS, HIST, PT, COC, SINK = range(5)
T4 = (VIS, PT, COC, HIST)
NAMES = ("vision", "prompt-text", "generated CoC", "ego-history")
RUNS = {"calib_100": "gradanat_v1", "val100": "gradanat_val100", "OOD": "gradanat_ood", "test100": "gradanat_test100"}
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, 35)}
KEEP = 19
N_SPLIT = 50


def split(arr):
    """(N, 5, 36, U) -> C4 (4, 36, U) additive parts by type, I (36, U)."""
    g = arr.astype(np.float64)
    C = (g * np.sign(g.sum(1, keepdims=True))).mean(0)
    return np.stack([C[t] for t in T4]), C.sum(0)


def labels(arr):
    C4, I = split(arr)
    sh = C4 / np.where(I > 0, I, np.nan)
    return C4.argmax(0), sh, I


def agree(a, b, mask):
    a, b = a[mask], b[mask]
    po = (a == b).mean()
    pe = sum(((a == k).mean()) * ((b == k).mean()) for k in range(4))
    return po, (po - pe) / (1 - pe)


data = {name: {obj: np.load(f"{REPO}/outputs/{run}/anatomy_perclip_q.npz")[key] for obj, key in (("traj", "fm_full"), ("coc", "ce"))}
        for name, run in RUNS.items()}
lab = {name: {obj: labels(arr) for obj, arr in d.items()} for name, d in data.items()}
sets = list(RUNS)

print("## 1. Share of heads by dominant position type\n")
print("| score | band | set | " + " | ".join(NAMES) + " | median dominant share |")
print("|---|---|---|" + "---:|" * 5)
for obj in ("traj", "coc"):
    for b, sl in BANDS.items():
        for name in sets:
            dom, sh, I = lab[name][obj]
            d = dom[sl]; ds = np.nanmax(sh[:, sl], 0)
            print(f"| I_{obj} | {b} | {name} | " + " | ".join(f"{(d == k).mean():.2f}" for k in range(4)) + f" | {np.nanmedian(ds):.2f} |")

print("\n## 2. Head-wise agreement of the dominant type with calib_100 (raw fraction / Cohen's kappa)\n")
print("| score | band | heads | within-calib split-half (ceiling) | val100 | OOD | test100 |")
print("|---|---|---|---:|---:|---:|---:|")
rng = np.random.default_rng(0)
for obj in ("traj", "coc"):
    arr = data["calib_100"][obj]; n = len(arr)
    dom0, sh0, I0 = lab["calib_100"][obj]
    top = np.zeros_like(dom0, bool)
    for l in range(35):
        top[l, np.argsort(I0[l])[::-1][:KEEP]] = True
    halves = []
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        halves.append((labels(arr[p[:n // 2]])[0], labels(arr[p[n // 2:]])[0]))
    for b, sl in BANDS.items():
        for hname, hmask in (("all", np.ones_like(dom0, bool)), ("top-19 (calib rank)", top)):
            mask = np.zeros_like(dom0, bool); mask[sl] = True; mask &= hmask
            ceil = np.mean([agree(a, c, mask) for a, c in halves], 0)
            cells = [f"{ceil[0]:.2f} / {ceil[1]:.2f}"]
            for name in sets[1:]:
                po, k = agree(dom0, lab[name][obj][0], mask)
                cells.append(f"{po:.2f} / {k:.2f}")
            print(f"| I_{obj} | {b} | {hname} | " + " | ".join(cells) + " |")

print("\n## 3. Per-head shares, Spearman with calib_100 across heads of the band (continuous version)\n")
print("| score | band | share | within-calib split-half (ceiling) | val100 | OOD | test100 |")
print("|---|---|---|---:|---:|---:|---:|")
rng = np.random.default_rng(1)
for obj in ("traj", "coc"):
    arr = data["calib_100"][obj]; n = len(arr)
    sh0 = lab["calib_100"][obj][1]
    hs = []
    for _ in range(N_SPLIT):
        p = rng.permutation(n)
        hs.append((labels(arr[p[:n // 2]])[1], labels(arr[p[n // 2:]])[1]))
    for b, sl in BANDS.items():
        for sname, idx in (("vision", (0,)), ("text (prompt+CoC)", (1, 2)), ("generated CoC", (2,)), ("ego-history", (3,))):
            f = lambda s: np.nansum(s[list(idx)][:, sl], 0).ravel() if len(idx) > 1 else s[idx[0]][sl].ravel()
            ok = lambda x, y: np.isfinite(x) & np.isfinite(y)
            ceil = np.mean([spearmanr(f(a)[ok(f(a), f(c))], f(c)[ok(f(a), f(c))])[0] for a, c in hs])
            cells = [f"{ceil:.2f}"]
            for name in sets[1:]:
                x, y = f(sh0), f(lab[name][obj][1]); m = ok(x, y)
                cells.append(f"{spearmanr(x[m], y[m])[0]:.2f}")
            print(f"| I_{obj} | {b} | {sname} | " + " | ".join(cells) + " |")
