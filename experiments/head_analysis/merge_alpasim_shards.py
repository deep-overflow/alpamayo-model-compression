"""Stitch sharded closed-loop runs back into one run directory.

`launch_alpasim_shards.sh` splits one config's scenes over several GPUs, so a config
ends up as N run directories instead of one. `analyze_alpasim.py` reads exactly two
things from a run directory -- `aggregate/results-summary.json` and
`rollouts/<scene>/<rollout_id>/rollout.asl` -- so merging is concatenating the
`rollouts` array and linking the per-rollout directories.

The scoring fields (`score`, `passed`, `score_criteria`) are taken verbatim from each
shard's own summary; nothing is recomputed here. A shard that never finished has no
`aggregate/`, and its rollouts are deliberately NOT salvaged: reconstructing `score`
from `metrics.parquet` would mean re-implementing alpasim's scoring rule and risking a
silent disagreement with the shards that were scored properly.

Rollout dirs are hardlinked file by file (same filesystem, so free) rather than
symlinked, so the merged run stays readable if a shard directory is later removed.

Usage:
  python experiments/head_analysis/merge_alpasim_shards.py \
      --runs-root /home/cvlab21/project/chan/alpasim-runs \
      --shards m2601_baseline_sh0 m2601_baseline_sh1 m2601_baseline_sh2 m2601_baseline_sh3 \
      --out m2601_merged_baseline
"""

import argparse
import json
import shutil
from pathlib import Path


def link_tree(src, dst):
    """Hardlink every file under src into dst, creating directories as needed."""
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            continue
        try:
            q.hardlink_to(p)
        except OSError:          # cross-device or permission -- fall back to a copy
            shutil.copy2(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="merged run dir name under --runs-root")
    ap.add_argument("--expect-scenes", type=int, default=None,
                    help="fail if the merged run does not have exactly this many scenes")
    args = ap.parse_args()

    out_dir = args.runs_root / args.out
    (out_dir / "aggregate").mkdir(parents=True, exist_ok=True)

    merged, seen, base = [], {}, None
    for name in args.shards:
        run = args.runs_root / name
        summ = run / "aggregate" / "results-summary.json"
        if not summ.exists():
            raise SystemExit(f"{name}: no aggregate/results-summary.json -- the shard did "
                             "not finish, so its rollouts have no score. Re-run it.")
        d = json.loads(summ.read_text())
        if base is None:
            base = {k: v for k, v in d.items() if k not in ("rollouts", "metrics_results")}
        elif d.get("score_criteria") != base.get("score_criteria"):
            raise SystemExit(f"{name}: score_criteria differs from {args.shards[0]}; "
                             "these shards were not scored the same way")
        for r in d["rollouts"]:
            key = (r["clipgt_id"], r["rollout_id"])
            if key in seen:
                raise SystemExit(f"duplicate rollout {key} in {name} and {seen[key]}; "
                                 "the shards overlap")
            seen[key] = name
            merged.append(r)
        link_tree(run / "rollouts", out_dir / "rollouts")
        print(f"{name}: {len(d['rollouts'])} rollouts", flush=True)

    scenes = {r["clipgt_id"] for r in merged}
    base["rollouts"] = merged
    # metrics_results is alpasim's own cross-scene aggregate; it cannot be concatenated
    # meaningfully, and analyze_alpasim recomputes everything it needs from `rollouts`
    base["metrics_results"] = []
    base["merged_from"] = list(args.shards)
    (out_dir / "aggregate" / "results-summary.json").write_text(json.dumps(base, indent=2))

    print(f"\nmerged {len(args.shards)} shards -> {out_dir}")
    print(f"  {len(merged)} rollouts over {len(scenes)} scenes")
    if args.expect_scenes is not None and len(scenes) != args.expect_scenes:
        raise SystemExit(f"expected {args.expect_scenes} scenes, got {len(scenes)}")


if __name__ == "__main__":
    main()
