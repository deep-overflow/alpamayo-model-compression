"""Draw the recovery-training and probe-validation sets, bucket-balanced by geometry.

Official-train pool: coc_generated/train/coc_train.parquet (full-model CoC rollouts over
a 2-13 s t0 grid, per-sample minADE attached). Filters: calib_100 clips out, any clip in
the OOD manifest out (its val half is the probe), degenerate CoC out (eval_lib
thresholds), minADE <= --max-ade (do not distill reasoning from a rollout whose
trajectory was bad). One sample per clip. The GT future trajectory for bucketing comes
from the egomotion label zips alone -- no video decode.

Probe-validation official half: drawn from the eval cache (official val at t0=5.1 s),
excluding every val_500(=indist_500) clip so the report set is never touched by
checkpoint selection. The OOD half of the probe is outputs/eval_sets/ood_val.parquet
as-is, so this script only draws the official 238.

Buckets: {turn_left, turn_right, decel_stop, accel, cruise} -- eval_lib.bucket with the
net turn signed (y is left in the ego frame, so positive net turn = left). Draws are
balanced across buckets, scarcest first, with unfilled quota redistributed.

Writes outputs/recovery_sets/{train_official_<n>.parquet, val_official_<n>.parquet,
bucket_pool_train.parquet, config.json, summary.txt, plots/*.png}.

Usage:
  python experiments/recovery/make_train_set.py [--n-train 1200] [--n-val 238] \
      [--max-ade 3.0] [--workers 16]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))
sys.path.insert(0, str(REPO / "experiments" / "evaluation"))

import eval_lib as el
import sample_cache as sc

AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
COC_TRAIN = Path("/mnt/nvme1n1/ad_vla/data/coc_generated/train/coc_train.parquet")
BUCKETS = ("turn_left", "turn_right", "decel_stop", "accel", "cruise")
SEED = 20260819

BG, INK, MUTED = "#FAF9F5", "#29261B", "#6B6555"
C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def bucket5(xy):
    """eval_lib.bucket with the turn split by sign. xy (T, 2) in the ego frame at t0."""
    vel = np.diff(xy, axis=0) / el.DT  # (T-1, 2)
    speed = np.linalg.norm(vel, axis=1)  # (T-1,)
    moving = speed > 0.5
    if moving.sum() >= 2:
        head = np.arctan2(vel[moving, 1], vel[moving, 0])
        signed = float(np.rad2deg(np.angle(np.exp(1j * (head[-1] - head[0])))))
    else:
        signed = 0.0
    k = max(len(speed) // 8, 1)
    v0, v_end = float(speed[:k].mean()), float(speed[-k:].mean())
    if v_end < 0.5 or v_end < 0.5 * v0:
        b = "decel_stop"
    elif abs(signed) >= 30.0:
        b = "turn_left" if signed > 0 else "turn_right"
    elif v_end > 1.5 * max(v0, 0.5):
        b = "accel"
    else:
        b = "cruise"
    return b, signed, v0, v_end


_AVDI = None


def _avdi():
    global _AVDI
    if _AVDI is None:
        import physical_ai_av
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=AV)
    return _AVDI


def ego_future_xy(egomotion, t0_us):
    """Future 6.4 s xy in the ego frame at t0, from the egomotion interpolator. (64, 2)"""
    import scipy.spatial.transform as spt
    fut_ts = t0_us + np.arange(1, 65, dtype=np.int64) * 100_000
    ref = egomotion(np.array([t0_us], dtype=np.int64))
    fut = egomotion(fut_ts)
    rot_inv = spt.Rotation.from_quat(ref.pose.rotation.as_quat()[0]).inv()
    local = rot_inv.apply(fut.pose.translation - ref.pose.translation[0])  # (64, 3)
    return local[:, :2]


def bucket_clip(task):
    """Bucket every candidate t0 of one clip from a single egomotion read."""
    clip_id, t0s = task
    try:
        ego = _avdi().get_clip_feature(clip_id, "egomotion", maybe_stream=True)
    except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
        return [{"clip_id": clip_id, "t0_us": int(t), "error": str(e)[:120]} for t in t0s]
    rows = []
    for t in t0s:
        try:
            b, signed, v0, v_end = bucket5(ego_future_xy(ego, int(t)))
            rows.append({"clip_id": clip_id, "t0_us": int(t), "bucket": b,
                         "turn_deg": signed, "v0": v0, "v_end": v_end})
        except Exception as e:  # noqa: BLE001
            rows.append({"clip_id": clip_id, "t0_us": int(t), "error": str(e)[:120]})
    return rows


def bucket_pool_official(df, workers, out_path):
    """Bucket the filtered official pool via egomotion labels; cached across reruns."""
    if out_path.exists():
        pool = pd.read_parquet(out_path)
        have = set(zip(pool.clip_id, pool.t0_us))
        df = df[[(c, t) not in have for c, t in zip(df.clip_id, df.t0_us)]]
        if not len(df):
            return pool
    else:
        pool = pd.DataFrame()
    tasks = [(c, g.t0_us.tolist()) for c, g in df.groupby("clip_id")]
    print(f"bucketing {len(df)} samples over {len(tasks)} clips ...", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(bucket_clip, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            if i % 200 == 0 or i == len(tasks):
                el_ = time.time() - t0
                print(f"[{i}/{len(tasks)}] elapsed {el_ / 60:.1f}m "
                      f"eta {el_ / i * (len(tasks) - i) / 60:.1f}m", flush=True)
    pool = pd.concat([pool, pd.DataFrame(rows)], ignore_index=True)
    pool.to_parquet(out_path)
    return pool


def _bucket_eval_stem(stem):
    """Bucket one eval-cache sample, reading ego_future straight off the npz."""
    clip_id, t0 = stem.split("__t0_")
    z = np.load(sc.path_for("eval", clip_id, int(t0)))
    b, signed, v0, v_end = bucket5(z["ego_future_xyz"][0, 0, :, :2])
    return {"clip_id": clip_id, "t0_us": int(t0), "bucket": b,
            "turn_deg": signed, "v0": v0, "v_end": v_end}


def bucket_pool_eval_cache(exclude, workers):
    """Bucket every eval-cache clip not excluded (top-level fn: pool workers pickle it)."""
    stems = [s for s in sc.index("eval") if s.split("__t0_")[0] not in exclude]
    print(f"bucketing {len(stems)} eval-cache clips ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_bucket_eval_stem, stems, chunksize=256))
    return pd.DataFrame(rows)


def balanced_draw(pool, total, rng):
    """Scarcity-first balanced draw, at most one sample per clip, quota redistributed."""
    order = sorted(BUCKETS, key=lambda b: pool[pool.bucket == b].clip_id.nunique())
    base, rem = divmod(total, len(BUCKETS))
    target = {b: base + (1 if i < rem else 0) for i, b in enumerate(order)}
    used, picks = set(), []

    def draw_from(b, n):
        cand = pool[(pool.bucket == b) & ~pool.clip_id.isin(used)]
        clips = np.asarray(cand.clip_id.unique(), dtype=object)
        rng.shuffle(clips)
        for c in clips[:n]:
            rows = cand[cand.clip_id == c]
            picks.append(rows.iloc[rng.integers(len(rows))])
            used.add(c)
        return min(n, len(clips))

    for b in order:
        got = draw_from(b, target[b])
        if got < target[b]:
            print(f"bucket {b}: only {got}/{target[b]} clips available", flush=True)
    while len(picks) < total:
        spare = [(pool[(pool.bucket == b) & ~pool.clip_id.isin(used)].clip_id.nunique(), b)
                 for b in BUCKETS]
        spare = [(n, b) for n, b in spare if n > 0]
        if not spare:
            break
        _, b = max(spare)
        draw_from(b, 1)
    return pd.DataFrame(picks).reset_index(drop=True)


def dist_plot(dfs, labels, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.8 / len(dfs)
    for i, (df, lab) in enumerate(zip(dfs, labels)):
        counts = [int((df.bucket == b).sum()) for b in BUCKETS]
        ax.bar(np.arange(len(BUCKETS)) + i * w, counts, width=w,
               color=[C1, C2, C4][i % 3], label=lab)
    ax.set_xticks(np.arange(len(BUCKETS)) + w * (len(dfs) - 1) / 2)
    ax.set_xticklabels(BUCKETS)
    ax.set_ylabel("samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_plot(rows_xy, path):
    """A few GT trajectories per bucket -- the eyeball check that left is left."""
    fig, axes = plt.subplots(1, len(BUCKETS), figsize=(3.0 * len(BUCKETS), 3.2))
    for ax, b in zip(axes, BUCKETS):
        for xy in rows_xy.get(b, []):
            ax.plot(xy[:, 1], xy[:, 0], lw=1.2, color=C1, alpha=0.7)  # x fwd = up, y left
        ax.set_title(b)
        ax.invert_xaxis()  # y (left) grows to the left of the plot
        ax.set_aspect("equal", adjustable="datalim")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.axvline(0, color=MUTED, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=238)
    ap.add_argument("--max-ade", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = REPO / "outputs" / "recovery_sets"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- official-train pool: filter, bucket, draw ----
    df = pd.read_parquet(COC_TRAIN)
    n0 = len(df)
    calib = set(pd.read_parquet(REPO / "outputs" / "eval_sets" / "calib_100.parquet").clip_id)
    ood = set(pd.read_parquet(REPO / "outputs" / "eval_sets" / "ood.parquet").clip_id)
    df = df[~df.clip_id.isin(calib) & ~df.clip_id.isin(ood)]
    n_clip_filter = len(df)
    deg = df.coc.map(lambda t: el.coc_degenerate(t)["degenerate"])
    df = df[~deg]
    n_deg = len(df)
    df = df[df.minADE <= args.max_ade]
    n_ade = len(df)
    print(f"official pool: {n0} -> clip filters {n_clip_filter} -> coc ok {n_deg} "
          f"-> minADE<={args.max_ade} {n_ade} ({df.clip_id.nunique()} clips)", flush=True)

    pool = bucket_pool_official(df[["clip_id", "t0_us"]], args.workers,
                                out / "bucket_pool_train.parquet")
    errs = pool[pool.get("error").notna()] if "error" in pool else pd.DataFrame()
    pool = pool[pool.bucket.notna()] if "error" in pool else pool
    pool = pool.merge(df[["clip_id", "t0_us", "chunk", "coc", "minADE"]],
                      on=["clip_id", "t0_us"], how="inner")
    train = balanced_draw(pool, args.n_train, rng)
    train.to_parquet(out / f"train_official_{len(train)}.parquet")

    # ---- probe-validation official half ----
    val500 = set(pd.read_parquet(REPO / "outputs" / "eval_sets" / "val_500.parquet").clip_id)
    vpool = bucket_pool_eval_cache(val500 | ood, args.workers)
    val = balanced_draw(vpool, args.n_val, rng)
    val.to_parquet(out / f"val_official_{len(val)}.parquet")

    # ---- plots + records ----
    dist_plot([pool, train], ["pool", "selected"], out / "plots" / "train_buckets.png")
    dist_plot([vpool, val], ["pool", "selected"], out / "plots" / "val_buckets.png")
    ex_xy = {}
    for b in BUCKETS:
        sel = train[train.bucket == b].head(5)
        xys = []
        for r in sel.itertuples():
            try:
                ego = _avdi().get_clip_feature(r.clip_id, "egomotion", maybe_stream=True)
                xys.append(ego_future_xy(ego, int(r.t0_us)))
            except Exception as e:  # noqa: BLE001  plot-only
                print(f"example plot skip {r.clip_id}: {e}", flush=True)
        ex_xy[b] = xys
    example_plot(ex_xy, out / "plots" / "train_examples.png")

    cfg = {"seed": SEED, "n_train": args.n_train, "n_val": args.n_val,
           "max_ade": args.max_ade, "coc_train": str(COC_TRAIN),
           "filters": {"rows": n0, "after_clip_filters": n_clip_filter,
                       "after_coc_ok": n_deg, "after_ade": n_ade},
           "excluded": {"calib_100": len(calib), "ood_manifest_clips": len(ood),
                        "val_500": len(val500)},
           "bucket_errors": len(errs)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    lines = [f"recovery sets (seed {SEED})",
             f"train official: {len(train)} samples, {train.clip_id.nunique()} clips",
             "  " + "  ".join(f"{b}={int((train.bucket == b).sum())}" for b in BUCKETS),
             f"val official probe: {len(val)} samples (eval cache minus val_500/ood)",
             "  " + "  ".join(f"{b}={int((val.bucket == b).sum())}" for b in BUCKETS),
             (f"pool: {len(pool)} samples ({pool.clip_id.nunique()} clips), "
              f"bucket errors {len(errs)}")]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved ->", out)


if __name__ == "__main__":
    main()
