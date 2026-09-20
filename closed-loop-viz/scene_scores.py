"""Per-scene score for one arm, as `<score>\\t<scene_id>` lines.

The score is the scene's mean over its rollouts -- the quantity every table in this repo
calls the scene score -- not the score of whichever rollout a video happens to show. The
renderer draws the WORSE rollout by default, so those two differ on a split scene, and the
filename should carry the one the analysis uses.

Written to three decimals so that sorting the filenames sorts by score.
"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path

RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="run name (resolved under alpasim-runs) or an absolute path")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    run = Path(args.run) if Path(args.run).is_absolute() else RUNS / args.run
    d = json.loads((run / "aggregate/results-summary.json").read_text())
    by = collections.defaultdict(list)
    for r in d["rollouts"]:
        by[r["clipgt_id"]].append(float(r["score"]))

    lines = [f"{st.mean(v):.3f}\t{k}" for k, v in sorted(by.items())]
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"{len(lines)}개 -> {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
