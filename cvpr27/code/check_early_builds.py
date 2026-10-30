"""G0 for step 2 of cvpr27/plans/2026-10-29_early-mlp-cut.md: are the maskfile builds what the plan says?

  (a) slim_early_b0 == slim_dual_u40_l0keep_v2 unit for unit (VLM and expert kept indices, KV-only
      layers) and removes 2,657,447,936 params, so the stored l0keep rows (swap_ship_L0fullM_*,
      the same selection) are a valid B0 for the open-loop pairing
  (b) slim_early_a10 / a05 differ from B0 only in the MLP of layers 1-4, keep 1229 / 614 channels
      there, nested in B0's
  (c) their removed params equal masks_config.json's

Exits 1 on any failure; writes outputs/<exp-id>/g0_builds.json.

Usage:
  .venv/bin/python cvpr27/code/check_early_builds.py [--exp-id cvpr27/early_mlp]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAYERS = (1, 2, 3, 4)
ARMS = {"a10": ("e10", 1229), "a05": ("e05", 614)}


def meta(name):
    return json.loads((REPO / "outputs" / name / "slim_meta.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="cvpr27/early_mlp")
    ap.add_argument("--ref", default="slim_dual_u40_l0keep_v2")
    args = ap.parse_args()
    mcfg = json.loads((REPO / "outputs" / args.exp_id / "masks_config.json").read_text())["configs"]
    ref, b0 = meta(args.ref), meta("cvpr27/slim_early_b0")
    checks = {}

    checks["a_vlm_identical"] = all(sorted(r["q"]) == sorted(b["q"]) and sorted(r["mlp"]) == sorted(b["mlp"])
                                    for r, b in zip(ref["vlm"], b0["vlm"]))
    checks["a_expert_identical"] = all(sorted(r["q"]) == sorted(b["q"]) and sorted(r["mlp"]) == sorted(b["mlp"])
                                       for r, b in zip(ref["expert"], b0["expert"]))
    checks["a_kvonly_identical"] = ref["kvonly_layers"] == b0["kvonly_layers"]
    checks["a_removed"] = b0["params"]["removed"] == ref["params"]["removed"] == 2_657_447_936

    for arm, (cfg, k) in ARMS.items():
        a = meta(f"cvpr27/slim_early_{arm}")
        same_elsewhere = all(
            sorted(x["q"]) == sorted(y["q"]) and (li in LAYERS or sorted(x["mlp"]) == sorted(y["mlp"]))
            for li, (x, y) in enumerate(zip(a["vlm"], b0["vlm"])))
        checks[f"b_{arm}_only_l1_4_mlp_differ"] = same_elsewhere and a["expert"] == b0["expert"]
        checks[f"b_{arm}_count_nested"] = all(
            len(a["vlm"][li]["mlp"]) == k and set(a["vlm"][li]["mlp"]) <= set(b0["vlm"][li]["mlp"])
            for li in LAYERS)
        checks[f"c_{arm}_removed"] = a["params"]["removed"] == mcfg[cfg]["removed"]

    ok = all(checks.values())
    (REPO / "outputs" / args.exp_id / "g0_builds.json").write_text(json.dumps(
        {"ref": args.ref, "checks": checks, "pass": ok}, indent=2))
    for name, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {name}")
    print("G0", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
