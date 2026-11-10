"""Scene lists for the u50 closed loop on Origin150 x 2 (iclr27-rebuttal/plans/2026-11-09_u50-closedloop-origin150.md).

The user split Origin150 75 / 75 between the two card types (11-10): 75 scenes on Blackwell 0-3, 75 on Ada 4-7.
Every u50 arm uses the same lists, so all arms share one scene -> card map. Rule, from the patched-harness dense run
hofix_merged_baseline (4 shards on Ada 4-7):
  - scenes sorted by scene_id; even positions stay on the Ada card dense used (shard 4 + k for hofix shard k),
    odd positions move to the Blackwell card with the same index (shard k)
  - shard i runs on GPU i of the list "0 1 2 3 4 5 6 7" (launch_alpasim_shards.sh `shard` column)
Smoke list: per shard the first scene whose two dense rollouts both passed (no abort, collision or offroad).

  .venv/bin/python iclr27-rebuttal/code/make_u50_scene_lists.py
writes outputs/iclr27-rebuttal/u50_origin150/{scenes_main.csv, scenes_smoke8.csv, scene_lists.txt}
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path("/home/cvlab21/project/chan/alpamayo-model-compression")
RUNS = Path("/home/cvlab21/project/chan/alpasim-runs")
OUT = REPO / "outputs" / "iclr27-rebuttal" / "u50_origin150"


def main():
    hof = pd.read_csv(RUNS / "shard_suites_hofix_baseline.csv")  # test_suite_id, scene_id, uuid
    hof["hofix_shard"] = hof.test_suite_id.str.extract(r"_sh(\d)$")[0].astype(int)
    assert len(hof) == 150 and hof.scene_id.is_unique and set(hof.hofix_shard) == {0, 1, 2, 3}
    # hofix's shards are themselves round-robin over the sorted ids, so alternate WITHIN each shard (a global
    # even/odd split would send whole shards to one card type). Shards of 38/38/37/37 give 76 Ada scenes; the
    # last Ada scene of the last odd-sized shard moves to Blackwell, so the split is exactly 75 / 75.
    hof = hof.sort_values(["hofix_shard", "scene_id"]).reset_index(drop=True)
    rank = hof.groupby("hofix_shard").cumcount()
    ada = rank % 2 == 0
    odd_shards = [k for k, n in hof.hofix_shard.value_counts().sort_index().items() if n % 2]
    for k in odd_shards[::-1][:int(ada.sum()) - 75]:
        ada[hof.index[(hof.hofix_shard == k) & ada][-1]] = False
    hof["shard"] = (hof.hofix_shard + 4).where(ada, hof.hofix_shard)
    hof["card_type"] = pd.Series("ada", index=hof.index).where(ada, "blackwell")
    hof["same_card_as_dense"] = ada
    assert (hof.card_type == "ada").sum() == 75 and (hof.card_type == "blackwell").sum() == 75

    summ = json.loads((RUNS / "hofix_merged_baseline" / "aggregate" / "results-summary.json").read_text())
    ok = pd.DataFrame([{"scene_id": r["clipgt_id"], "ok": r.get("status") == "pass" and r.get("score") is not None}
                       for r in summ["rollouts"]]).groupby("scene_id").ok.all()
    smoke = (hof[hof.scene_id.map(ok).fillna(False).astype(bool)]
             .sort_values(["shard", "scene_id"]).groupby("shard").head(1))
    assert sorted(smoke.shard) == list(range(8)), smoke.shard.tolist()

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["scene_id", "uuid", "shard", "card_type", "same_card_as_dense", "hofix_shard"]
    hof[cols].to_csv(OUT / "scenes_main.csv", index=False)
    smoke[cols].to_csv(OUT / "scenes_smoke8.csv", index=False)
    lines = ["u50 Origin150 scene lists (75 Blackwell 0-3 / 75 Ada 4-7)",
             "scenes per shard (GPU): " + ", ".join(f"{s}:{n}" for s, n in hof.shard.value_counts().sort_index().items()),
             f"smoke scenes: {', '.join(smoke.scene_id)}"]
    (OUT / "scene_lists.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
