"""Two-channel decomposition on OOD: d_own (own rollout) = d_tf (GT CoC teacher-forced; action channel)
+ lang (own minus teacher-forced; language channel), per arm, all / healthy / degenerate / manoeuvre / own tail."""
import numpy as np, pandas as pd, json
rng = np.random.default_rng(0); df = pd.read_parquet("openloop_ood.parquet")
def ci(x, n=2000):
    x = np.asarray(x, float); m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
rows = {}
for a in ("coc", "traj", "dual"):
    df[f"lang_{a}"] = df[f"d_{a}"] - df[f"dtf_{a}"]; k = len(df) // 10
    groups = [("all", np.ones(len(df), bool)), ("healthy", ~df[f"deg_{a}"].values), ("degenerate", df[f"deg_{a}"].values)] + [(b, (df["bucket"] == b).values) for b in ("cruise", "accel", "decel_stop", "turn")] + [("own worst decile", (df[f"d_{a}"].rank(ascending=False) <= k).values)]
    for g, m in groups:
        o, t, l = ci(df.loc[m, f"d_{a}"]), ci(df.loc[m, f"dtf_{a}"]), ci(df.loc[m, f"lang_{a}"])
        rows[f"{a}|{g}"] = dict(n=int(m.sum()), d_own=o, d_tf=t, lang=l, lang_share=l[0] / o[0] if o[0] else None)
        print(f"{a:5s} {g:18s} n={m.sum():5d} own {o[0]:+.3f} [{o[1]:+.3f},{o[2]:+.3f}] | action {t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}] | language {l[0]:+.3f} [{l[1]:+.3f},{l[2]:+.3f}]")
json.dump(rows, open("two_channel_ood.json", "w"), indent=1)
