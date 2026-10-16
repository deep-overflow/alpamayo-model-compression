"""Splice run_l0_select.py's layer-0 choice into its arm: outputs/slim_swap_<name>/slim_meta.json.

plans/2026-10-14_l0-functional-redundancy.md, step 3. The arm's kept set is copied unchanged
except layer 0's Q heads, which become the 19 the greedy P2 search kept. Both drop 13 heads, so
the parameter count is the arm's own. Refuses an incomplete search.

Usage:
  python experiments/evaluation/make_l0_select_swaps.py --arm dual_seed_n1 --sel l0sel_n1 \
      --name n1_L0p2
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="slim dir name without the slim_ prefix")
    ap.add_argument("--sel", required=True, help="run_l0_select.py exp id")
    ap.add_argument("--name", required=True, help="writes outputs/slim_swap_<name>")
    args = ap.parse_args()
    out = REPO / "outputs"
    sel = json.loads((out / args.sel / "metrics.json").read_text())
    assert sel["complete"] and sel["arm"] == args.arm, (sel["complete"], sel["arm"])
    drop = sel["selected_drop"]
    assert len(drop) == 13, drop
    m = json.loads((out / f"slim_{args.arm}" / "slim_meta.json").read_text())
    before = sorted(set(range(32)) - set(m["vlm"][0]["q"]))
    m["vlm"][0]["q"] = sorted(set(range(32)) - set(drop))
    m["l0_select"] = {"source_arm": args.arm, "selector": args.sel, "as_built_drop": before,
                      "selected_drop": drop}
    dd = out / f"slim_swap_{args.name}"
    dd.mkdir(exist_ok=True)
    (dd / "slim_meta.json").write_text(json.dumps(m))
    print(f"{args.arm}: layer 0 drops {before} -> {drop}; changed "
          f"{sorted(set(before) ^ set(drop))} -> {dd}")


if __name__ == "__main__":
    main()
