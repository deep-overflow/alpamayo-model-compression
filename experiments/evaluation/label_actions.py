"""Action labels for calibration-draw candidates, and the raw-action check on cached clips.

Two modes, both CPU-only (plans/2026-09-11_action-stratified-calib.md, stage 1):

  --candidates N   draw N random official-train clips that are outside every existing
                   calibration manifest, the OOD pool and every evaluation set, and label
                   each with `make_train_set.bucket5` at CALIB_T0 from the egomotion label
                   zips alone (no video decode). This is the pool `make_calib_strat.py`
                   stratifies. The label is a property of (clip, t0), which is why t0 is
                   pinned here and checked again after the cache is built: the `tr` sets
                   lost their bucket balance exactly by re-picking the window.

  --raw-from-cache for each set=manifest:cache, recompute bucket5 from the cached
                   `ego_future_xyz` (gate G0-c: must equal the egomotion label) and build
                   the raw flow-matching target `traj_to_action(...)` -> (64, 2) normalized
                   unicycle (accel, curvature) with the released `action_space_cfg`, so the
                   strata can be checked in the space `I_traj` is actually fitted on
                   (gate G0-d). `UnicycleAccelCurvatureActionSpace` is a plain nn.Module and
                   all four inputs are in the npz, so no model is loaded.

Usage:
  python experiments/evaluation/label_actions.py --candidates 3000 --seed 20260911
  python experiments/evaluation/label_actions.py --raw-from-cache \
      --sets se_a=calib_se100_a:calib_strat rd_a=calib_rd100_a:calib_rd_a ...
"""

import argparse
import configparser
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# token in $HOME, blobs on /mnt/nvme1n1 -- set both or neither works (CLAUDE.md)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", "/mnt/nvme1n1/ad_vla/cache/hub")
if "HF_TOKEN" not in os.environ:
    _cp = configparser.ConfigParser()
    _cp.read(os.path.expanduser("~/.cache/huggingface/stored_tokens"))
    if "full_right" in _cp:
        os.environ["HF_TOKEN"] = _cp["full_right"]["hf_token"]

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "experiments" / "recovery"))
sys.path.insert(0, str(REPO / "experiments" / "head_analysis"))

from make_eval_sets import derive
from make_train_set import bucket5, bucket_clip

AV = Path("/mnt/nvme1n1/ad_vla/data/physicalai_av")
PRE = AV / "pre_processed"
ES = REPO / "outputs" / "eval_sets"
CALIB_T0 = 5_100_000
ORDER = ["cruise", "decel_stop", "accel", "turn_left", "turn_right"]
EVAL_SETS = ("test_500", "val_500", "indist_500", "ood_val", "ood")
MODEL = "nvidia/Alpamayo-1.5-10B"


def held_out():
    """Every clip that is already a calibration, evaluation or OOD clip."""
    ids = set(pd.read_parquet(AV / "reasoning" / "ood_reasoning.parquet").index.astype(str))
    src = {"ood_reasoning": len(ids)}
    for p in sorted(glob.glob(str(ES / "calib_*.parquet"))) + \
            [str(ES / f"{s}.parquet") for s in EVAL_SETS]:
        if not Path(p).exists():
            continue
        d = pd.read_parquet(p)
        col = "clip_id" if "clip_id" in d.columns else d.columns[0]
        s = set(d[col].astype(str))
        src[Path(p).stem] = len(s)
        ids |= s
    return ids, src


def candidates(args, out):
    ci = pd.read_parquet(AV / "clip_index.parquet")
    dc = pd.read_parquet(AV / "metadata" / "data_collection.parquet")
    full = derive(ci[ci.split == "train"], dc)
    full.index = full.index.astype(str)
    prior, src = held_out()
    pool = full[~full.index.isin(prior)]
    print(f"official train {len(full):,}  -held out {len(full) - len(pool):,}  "
          f"-> eligible {len(pool):,}", flush=True)
    rng = np.random.default_rng(args.seed)
    pick = pool.iloc[rng.choice(len(pool), args.candidates, replace=False)]

    t = time.time()
    rows = []
    todo = [(cid, [CALIB_T0]) for cid in pick.index]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(bucket_clip, task) for task in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t
                print(f"  [{i}/{len(todo)}] {el / 60:.1f}m eta {el / i * (len(todo) - i) / 60:.1f}m",
                      flush=True)
    got = pd.DataFrame(rows)
    err = got[got["error"].notna()] if "error" in got.columns else got.iloc[:0]
    ok = got[got["bucket"].notna()] if "bucket" in got.columns else got.iloc[:0]
    print(f"labelled {len(ok)}  failed {len(err)}  {(time.time() - t) / 60:.1f} min", flush=True)

    attrs = ["chunk", "country", "platform_class", "radar_config", "month", "hour_of_day",
             "time_of_day", "season"]
    lab = ok.merge(pick[attrs].reset_index(names="clip_id"), on="clip_id", how="left")
    lab = lab[["clip_id", "t0_us", "bucket", "turn_deg", "v0", "v_end"] + attrs]
    lab.to_parquet(out / "labels.parquet", index=False)
    (out / "labels_errors.json").write_text(err.to_json(orient="records", indent=1))
    comp = lab["bucket"].value_counts(normalize=True).reindex(ORDER).fillna(0)
    (out / "config_labels.json").write_text(json.dumps({
        "candidates": args.candidates, "seed": args.seed, "t0_us": CALIB_T0,
        "pool": {"official_train": len(full), "eligible": len(pool)},
        "held_out_sources": src, "labelled": len(lab), "failed": len(err),
        "composition": {k: float(v) for k, v in comp.items()},
        "minutes": (time.time() - t) / 60}, indent=2))
    lines = [(f"{len(lab)} labelled candidates at t0={CALIB_T0} us (of {args.candidates} drawn "
              f"from {len(pool):,} eligible official-train clips), {len(err)} failed"),
             "composition: " + "  ".join(f"{k} {v * 100:.1f}%" for k, v in comp.items()),
             "counts:      " + "  ".join(f"{k} {int(n)}" for k, n in
                                         lab["bucket"].value_counts().reindex(ORDER).items())]
    (out / "summary_labels.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def action_space():
    """The released action space, from the cached config -- no weights are touched."""
    from alpamayo1_5.action_space import UnicycleAccelCurvatureActionSpace
    from huggingface_hub import constants
    snaps = Path(constants.HF_HUB_CACHE) / f"models--{MODEL.replace('/', '--')}" / "snapshots"
    cfgs = sorted(snaps.glob("*/config.json"))
    cfg = json.loads(cfgs[-1].read_text())["action_space_cfg"]
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return UnicycleAccelCurvatureActionSpace(**cfg), cfg


def raw_one(space, cfg, p):
    import torch
    z = np.load(p, allow_pickle=True)
    fx = z["ego_future_xyz"].reshape(-1, 3)  # (64, 3)
    b, signed, v0, v_end = bucket5(fx[:, :2])
    with torch.no_grad():
        act = space.traj_to_action(
            torch.as_tensor(z["ego_history_xyz"].reshape(1, -1, 3), dtype=torch.float32),
            torch.as_tensor(z["ego_history_rot"].reshape(1, -1, 3, 3), dtype=torch.float32),
            torch.as_tensor(fx.reshape(1, 64, 3), dtype=torch.float32),
            torch.as_tensor(z["ego_future_rot"].reshape(1, 64, 3, 3), dtype=torch.float32),
        )[0].numpy()  # (64, 2) normalized (accel, curvature)
    a = act[:, 0] * cfg["accel_std"] + cfg["accel_mean"]          # m/s^2
    k = act[:, 1] * cfg["curvature_std"] + cfg["curvature_mean"]  # 1/m
    return {"bucket_cache": b, "turn_deg_cache": signed, "v0": v0, "v_end": v_end,
            "a_mean_abs": float(np.abs(a).mean()), "a_max_decel": float(-a.min()),
            "a_max_accel": float(a.max()), "k_mean_abs": float(np.abs(k).mean()),
            "k_max_abs": float(np.abs(k).max()),
            "rms_a_norm": float(np.sqrt((act[:, 0] ** 2).mean())),
            "rms_k_norm": float(np.sqrt((act[:, 1] ** 2).mean()))}


def raw_from_cache(args, out):
    space, cfg = action_space()
    labels = None
    if (out / "labels.parquet").exists():
        labels = pd.read_parquet(out / "labels.parquet").set_index("clip_id")["bucket"]
    rows = []
    for spec in args.sets:
        name, rest = spec.split("=")
        man, cache = rest.split(":")
        m = pd.read_parquet(ES / f"{man}.parquet")
        t0s = m["t0_us"].astype("int64") if "t0_us" in m.columns else \
            pd.Series(CALIB_T0, index=m.index)
        miss = 0
        for cid, t0 in zip(m["clip_id"].astype(str), t0s):
            p = PRE / cache / "samples" / f"{cid}__t0_{int(t0)}.npz"
            if not p.exists():
                miss += 1
                continue
            r = {"set": name, "clip_id": cid, "t0_us": int(t0), **raw_one(space, cfg, p)}
            if "bucket" in m.columns:
                r["bucket_label"] = str(m.loc[m["clip_id"] == cid, "bucket"].iloc[0])
            elif labels is not None and cid in labels.index:
                r["bucket_label"] = str(labels[cid])
            rows.append(r)
        print(f"{name}: {len(m) - miss}/{len(m)} cached", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "raw_actions.parquet", index=False)

    # G0-c: the label the draw used must be the label the cache carries
    lab = df[df["bucket_label"].notna()] if "bucket_label" in df.columns else df.iloc[:0]
    mism = lab[lab["bucket_label"] != lab["bucket_cache"]]
    # G0-d: strata must separate in the raw action space
    med = df.groupby("bucket_cache")[["k_mean_abs", "a_max_decel", "rms_a_norm",
                                      "rms_k_norm", "v0"]].median().reindex(ORDER)
    turn = med.loc[["turn_left", "turn_right"], "k_mean_abs"].min()
    g0 = {
        "c_label_vs_cache": {"checked": len(lab), "mismatch": len(mism),
                             "pass": bool(len(lab) > 0 and len(mism) == 0)},
        "d_strata_separate": {
            "k_mean_abs_turn_min_vs_cruise": [float(turn), float(med.loc["cruise", "k_mean_abs"])],
            "a_max_decel_stop_vs_cruise": [float(med.loc["decel_stop", "a_max_decel"]),
                                           float(med.loc["cruise", "a_max_decel"])],
            "rms_a_norm_accel_stop_vs_cruise": [float(med.loc["accel", "rms_a_norm"]),
                                                float(med.loc["decel_stop", "rms_a_norm"]),
                                                float(med.loc["cruise", "rms_a_norm"])],
            "pass": bool(turn > med.loc["cruise", "k_mean_abs"]
                         and med.loc["decel_stop", "a_max_decel"] > med.loc["cruise", "a_max_decel"]
                         and med.loc["accel", "rms_a_norm"] > med.loc["cruise", "rms_a_norm"]
                         and med.loc["decel_stop", "rms_a_norm"] > med.loc["cruise", "rms_a_norm"])},
        "action_space_cfg": cfg,
        "per_set_composition": {s: g["bucket_cache"].value_counts(normalize=True)
                                .reindex(ORDER).fillna(0).round(3).to_dict()
                                for s, g in df.groupby("set")},
    }
    (out / "g0_raw.json").write_text(json.dumps(g0, indent=2))
    pd.set_option("display.width", 200)
    print("\n== raw-action medians per bucket (all sets pooled) ==")
    print(med.round(4).to_string())
    print(f"\nG0-c label==cache: {len(lab) - len(mism)}/{len(lab)}  -> "
          f"{'PASS' if g0['c_label_vs_cache']['pass'] else 'FAIL'}")
    if len(mism):
        print(mism[["set", "clip_id", "bucket_label", "bucket_cache"]].to_string())
    print(f"G0-d strata separate in raw action space -> "
          f"{'PASS' if g0['d_strata_separate']['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--raw-from-cache", action="store_true")
    ap.add_argument("--sets", nargs="*", default=[], help="name=manifest:cache")
    ap.add_argument("--out", default="outputs/strat_calib")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.candidates:
        candidates(args, out)
    if args.raw_from_cache:
        raw_from_cache(args, out)


if __name__ == "__main__":
    main()
