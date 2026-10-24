"""Numbers requested for the Finding-1 rewrite (2026-09-24): (A) 0-21 token shares for paragraph 4
under the same rule as the 22-34 values (mean over the band's layers of each layer's additive
share; a pooled-mass variant is printed for reference only); (B) paragraph-2 rank-agreement
checks; (C) paragraph-5 22-34 share checks incl. the prompt/CoC split; (D) option (b): the raw
(uncorrected, full 100-clip) same-position Spearman next to the split-half corrected value.
Data: outputs/importance_v2 (paper Fig. 1 source), outputs/gradanat_v1 (anatomy on the same clips)."""
import json
import numpy as np
from scipy.stats import spearmanr

REPO = "/home/cvlab21/project/chan/alpamayo-model-compression"
VIS, HIST, PT, COC, SINK = range(5)
LAST = 35
BANDS = {"0-21": slice(0, 22), "22-34": slice(22, LAST)}
TYPE = {"vision": VIS, "prompt": PT, "CoC": COC, "hist": HIST}


def rho_layers(a, b):
    return np.array([spearmanr(a[l], b[l])[0] if np.ptp(a[l]) > 0 and np.ptp(b[l]) > 0 else np.nan for l in range(a.shape[0])])


def split_half(t, c, n_split, rng):
    n = len(t); st, sc_, cr = [], [], []
    for _ in range(n_split):
        p = rng.permutation(n); a, b = p[:n // 2], p[n // 2:]
        ta, tb, ca, cb = t[a].mean(0), t[b].mean(0), c[a].mean(0), c[b].mean(0)
        st.append(rho_layers(ta, tb)); sc_.append(rho_layers(ca, cb)); cr.append(0.5 * (rho_layers(ta, cb) + rho_layers(tb, ca)))
    return np.nanmean(st, 0), np.nanmean(sc_, 0), np.nanmean(cr, 0)


def corrected_band(st, sc_, cr, sl):
    st, sc_, cr = st[sl], sc_[sl], cr[sl]; ok = (st > 0.2) & (sc_ > 0.2)
    return float(np.nanmean(cr[ok] / np.sqrt(st[ok] * sc_[ok]))), int(ok.sum()), float(np.nanmean(cr))


def parts(arr):
    """(N, 5, 36, U) signed typed gradients -> (5, 36) additive parts, mean over clips."""
    part = np.zeros((5, 36))
    for i in range(len(arr)):
        g = arr[i].astype(np.float64)
        part += (g * np.sign(g.sum(0, keepdims=True))).sum(-1)
    return part / len(arr)


m = json.load(open(f"{REPO}/outputs/gradanat_v1/metrics_analysis.json"))
pcq = np.load(f"{REPO}/outputs/gradanat_v1/anatomy_perclip_q.npz")
pcm = np.load(f"{REPO}/outputs/gradanat_v1/anatomy_perclip_mlp.npz")
arrays = {("q", "fm"): pcq["fm_full"], ("q", "ce"): pcq["ce"], ("mlp", "fm"): pcm["fm_type"], ("mlp", "ce"): pcm["ce_type"]}
label = {"fm": "I_traj", "ce": "I_CoC"}

# ---------------------------------------------------------------- A + C: shares, two rules
print("## A. Token shares by band (rule 1 = mean over the band's layers of the per-layer additive share; the rule behind every existing 22-34 number)\n")
print("| axis | score | band | vision | text (= prompt + CoC) | prompt | CoC | ego-history |")
print("|---|---|---|---:|---:|---:|---:|---:|")
share_rule1 = {}
for axis in ("mlp", "q"):
    for obj in ("fm", "ce"):
        s = np.array(m["additive_by_type"][f"{axis}_{obj}"]["share_by_type_and_layer"])  # (5, 36)
        for b, sl in BANDS.items():
            v, h, p, c = (float(s[i][sl].mean()) for i in (VIS, HIST, PT, COC))
            share_rule1[(axis, obj, b)] = (v, p + c, p, c, h)
            print(f"| {axis} | {label[obj]} | {b} | {v:.2f} | {p + c:.2f} | {p:.2f} | {c:.2f} | {h:.2f} |")
print("\n### Rule 2, reference only: pooled mass (sum of the typed additive parts over the band's layers / sum of the layer totals)\n")
print("| axis | score | band | vision | text | prompt | CoC | ego-history |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for axis in ("mlp", "q"):
    for obj in ("fm", "ce"):
        part = parts(arrays[(axis, obj)])  # (5, 36)
        for b, sl in BANDS.items():
            tot = part[:, sl].sum()
            v, h, p, c = (part[i, sl].sum() / tot for i in (VIS, HIST, PT, COC))
            print(f"| {axis} | {label[obj]} | {b} | {v:.2f} | {p + c:.2f} | {p:.2f} | {c:.2f} | {h:.2f} |")

# ---------------------------------------------------------------- B: rank agreement checks
print("\n## B. Rank agreement (raw within-layer Spearman between the two 100-clip mean scores, band mean)\n")
print("| source | axis | 0-21 | 22-34 |")
print("|---|---|---:|---:|")
z = np.load(f"{REPO}/outputs/importance_v2/importance.npz")
for axis in ("q", "mlp"):
    r = rho_layers(z[f"traj_vlm_{axis}"], z[f"coc_vlm_{axis}"])
    print(f"| importance_v2 (Fig. 1 source) | {axis} | {np.nanmean(r[0:22]):.3f} | {np.nanmean(r[22:LAST]):.3f} |")
for axis in ("q", "mlp"):
    t = np.abs(arrays[(axis, "fm")].sum(1)).mean(0); c = np.abs(arrays[(axis, "ce")].sum(1)).mean(0)  # pooled = shipped score
    r = rho_layers(t, c)
    print(f"| gradanat_v1 typed sum (same clips, Ada) | {axis} | {np.nanmean(r[0:22]):.3f} | {np.nanmean(r[22:LAST]):.3f} |")

# ---------------------------------------------------------------- D: same-position agreement, raw vs corrected
print("\n## D. Same-position agreement, 22-34 (and 0-21): raw = Spearman between the 100-clip means of E|G_type| (uncorrected, same rule as B); corrected = split-half ceiling-corrected (analyzer rule, layers with either self-reliability <= 0.2 dropped)\n")
print("| axis | positions | raw 0-21 | raw 22-34 | corrected 22-34 (layers used) | half-sample cross 22-34 |")
print("|---|---|---:|---:|---:|---:|")
for axis, n_split in (("q", 50), ("mlp", 10)):
    rng = np.random.default_rng(0)
    fm, ce = arrays[(axis, "fm")], arrays[(axis, "ce")]
    series = {"pooled (shipped score)": (np.abs(fm.sum(1)), np.abs(ce.sum(1)))}
    for name, ti in TYPE.items():
        series[name] = (np.abs(fm[:, ti]), np.abs(ce[:, ti]))
    for name, (t, c) in series.items():
        raw = rho_layers(t.mean(0), c.mean(0))
        st, sc_, cr = split_half(t[:, :LAST], c[:, :LAST], n_split, rng)
        corr, used, cross = corrected_band(st, sc_, cr, slice(22, LAST))
        print(f"| {axis} | {name} | {np.nanmean(raw[0:22]):.2f} | {np.nanmean(raw[22:LAST]):.2f} | {corr:.2f} ({used}/13) | {cross:.2f} |")
