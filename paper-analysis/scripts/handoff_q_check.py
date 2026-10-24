"""Is the layer-to-layer jitter of the Q-head token shares sampling noise or structure?
Bootstrap the additive shares over the 100 calibration clips and compare the 95% CI width with
the layer-to-layer roughness. Per-clip typed gradients: outputs/gradanat_v1/anatomy_perclip_q.npz
(fm_full, ce: (100, 5, 36, 32) signed)."""
from pathlib import Path

import numpy as np

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
VIS, HIST, PT, COC, SINK = range(5)
NAMES = {VIS: "vision", PT: "prompt", COC: "CoC", HIST: "history"}
LAST = 35


def shares(G):
    """G (N, 5, L, U) signed -> additive share (5, L): sum_c,u G_type * sign(G_full) / sum_c,u |G_full|."""
    full = G.sum(1)  # (N, L, U)
    num = (G * np.sign(full)[:, None]).sum((0, 3))  # (5, L)
    den = np.abs(full).sum((0, 2))  # (L,)
    return num / np.where(den > 0, den, np.nan)


with np.load(REPO / "outputs" / "gradanat_v1" / "anatomy_perclip_q.npz") as z:
    data = {"I_traj": z["fm_full"].astype(np.float64), "I_CoC": z["ce"].astype(np.float64)}
rng = np.random.default_rng(0)
B = 500
for loss, G in data.items():
    point = shares(G)
    boot = np.stack([shares(G[rng.integers(0, len(G), len(G))]) for _ in range(B)])  # (B, 5, L)
    lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
    last = LAST if loss == "I_traj" else 36
    print(f"== {loss}  (95% CI over clips, B={B})")
    for t in (VIS, PT, COC, HIST):
        w = (hi[t] - lo[t])[:last]
        rough = np.abs(np.diff(point[t][:last]))
        late = slice(22, last)
        print(f"  {NAMES[t]:8s} CI half-width: median {np.nanmedian(w) / 2:.3f}  late {np.nanmedian(w[late]) / 2:.3f} | "
              f"layer-to-layer |diff|: median {np.nanmedian(rough):.3f}  late {np.nanmedian(rough[21:]):.3f} | "
              f"ratio diff/CI-width late {np.nanmedian(rough[21:]) / np.nanmedian(w[late]):.2f}")
    # the spikes the eye picks out
    for t, l in ((COC, 27), (VIS, 24), (HIST, 33), (COC, 22)):
        if l < last:
            print(f"    {NAMES[t]} @L{l}: {point[t][l]:.3f} [{lo[t][l]:.3f}, {hi[t][l]:.3f}]  neighbours {point[t][l-1]:.3f} / {point[t][l+1]:.3f}")
