#!/bin/bash
# Closed-loop evaluation of the Tyr-the-Pruner arms on the same 150 scenes x 2 rollouts
# every shipped arm used, then merge + analyze against the stored baseline and dual runs.
#
# This is a thin, opinionated wrapper around launch_alpasim_shards.sh: it adds the
# preflight checks that make the comparison fair, and nothing else. Every simulation
# setting (suite, scene order, rollouts, topology, ASL image skip, aggregation policy)
# comes from that script unchanged, so the Tyr runs are produced by the same code path
# as m2601_merged_{baseline,slim_dual_u40_v2,...}.
#
# What the preflight enforces (each is a way the comparison could silently break):
#   1. slim_state.pt present. The alpasim driver's load_slim does an unconditional
#      torch.load(slim_state.pt) + strict load, so a recipe-only dir does not fall back
#      to the base weights -- it raises. For Tyr this file *is* the OSSCAR
#      reconstruction: without it the arm would not be Tyr at all (cf. the selection-only
#      mix-up recorded in plans/2026-08-20_tyr-baseline.md).
#   2. Matched budget. slim_meta.json must report exactly -2,657,452,032 removed
#      parameters, the same budget as dual / traj / coc / j / jtraj.
#   3. Driver dir is a hardlink of outputs/<cfg>/, same filesystem, verified by inode.
#      A cross-filesystem copy would pin a second 16 GB of a disk that is at 98%.
#   4. Identical scenes. The first 150 scene_ids of public_2601 must equal the scene set
#      actually present in the stored comparison runs; the shards are then generated from
#      the suite CSV (never scenes.scene_ids, which resolves to 26.04 renders).
#
# Thread setting -- read this before changing it. The shipped runs are NOT uniform:
# baseline, traj, coc, j, w4, w8, znorm and recover ran the driver with
# OMP_NUM_THREADS=8, but slim_dual_u40_v2 (both shards) and slim_jtraj_u40_v2 shard 1
# ran with it unset. Thread count changes CPU reduction order, so this is a real (if
# small) inconsistency that predates this script. The default here matches the baseline
# and the majority of arms; set DRIVER_OMP_THREADS= (empty) to match dual instead.
#
# Usage:
#   bash experiments/head_analysis/launch_alpasim_tyr.sh                 # full run
#   MODE=preflight bash .../launch_alpasim_tyr.sh                        # checks only
#   MODE=prepare   bash .../launch_alpasim_tyr.sh                        # + driver hardlink
#   GPUS="6 7" CONFIGS="slim_tyr_u40_r slim_tyr_uniform_u40_recon" bash .../launch_alpasim_tyr.sh
#
# Cost: ~11.6 min/scene/driver. 150 scenes over 2 cards is ~14.5 h per config.
set -u

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
ALPASIM=/home/cvlab21/project/chan/alpasim
RUNS=/home/cvlab21/project/chan/alpasim-runs
DRIVERS=/mnt/nvme1n1/ad_vla/data/alpasim/drivers
SUITE=${SUITE-public_2601}
PREFIX=${PREFIX-m2601_}
N_SCENES=${N_SCENES-150}
N_ROLLOUTS=${N_ROLLOUTS-2}
GPUS=${GPUS-"6 7"}
CONFIGS=${CONFIGS-"slim_tyr_u40_r"}
MODE=${MODE-run}
EXPECTED_REMOVED=2657452032
# the arms whose stored runs this comparison is against; also the scene-set reference
REF_RUNS=${REF_RUNS-"m2601_merged_baseline m2601_merged_slim_dual_u40_v2"}
export DRIVER_OMP_THREADS=${DRIVER_OMP_THREADS-8}

read -r -a CFG_ARR <<<"$CONFIGS"
read -r -a GPU_ARR <<<"$GPUS"
n_shards=${#GPU_ARR[@]}

echo "[tyr-cl] configs: ${CFG_ARR[*]}"
echo "[tyr-cl] $N_SCENES scenes x $N_ROLLOUTS rollouts, suite $SUITE, GPUs ${GPU_ARR[*]} ($n_shards shards)"
echo "[tyr-cl] driver OMP_NUM_THREADS='${DRIVER_OMP_THREADS}' (shipped dual ran with this UNSET)"
echo "[tyr-cl] mode: $MODE"

# ---------------------------------------------------------------- preflight
fail=0
for cfg in "${CFG_ARR[@]}"; do
  src=$REPO/outputs/$cfg
  if [ ! -f "$src/slim_meta.json" ]; then echo "[FAIL] $cfg: no slim_meta.json"; fail=1; continue; fi
  if [ ! -f "$src/slim_state.pt" ]; then
    echo "[FAIL] $cfg: no slim_state.pt -- the driver requires it and for Tyr it carries"
    echo "       the OSSCAR reconstruction; rebuild with make_slim (without --no-state)."
    fail=1; continue
  fi
  removed=$("$REPO/.venv/bin/python" -c "
import json,sys;print(json.load(open('$src/slim_meta.json'))['params']['removed'])")
  if [ "$removed" != "$EXPECTED_REMOVED" ]; then
    echo "[FAIL] $cfg: removed $removed != $EXPECTED_REMOVED (budget not matched)"; fail=1
  else
    echo "[ok]   $cfg: state present, removed $removed"
  fi
done

# scene set identical to the stored comparison runs
"$REPO/.venv/bin/python" - "$SUITE" "$N_SCENES" "$RUNS" $REF_RUNS <<'PY' || fail=1
import sys
from pathlib import Path
import pandas as pd
suite, n, runs = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
su = pd.read_csv("/home/cvlab21/project/chan/alpasim/data/scenes/sim_suites.csv")
want = set(su[su.test_suite_id == suite].sort_values("scene_id").head(n).scene_id)
if len(want) != n:
    print(f"[FAIL] suite {suite} yielded {len(want)} scenes, expected {n}"); sys.exit(1)
ok = True
for ref in sys.argv[4:]:
    d = runs / ref / "rollouts"
    if not d.is_dir():
        print(f"[warn] reference run {ref} not found, skipping scene check"); continue
    have = {p.name for p in d.iterdir() if p.is_dir()}
    if have != want:
        print(f"[FAIL] scene set differs from {ref}: "
              f"{len(want - have)} missing, {len(have - want)} extra"); ok = False
    else:
        print(f"[ok]   scene set identical to {ref} ({len(have)} scenes)")
sys.exit(0 if ok else 1)
PY

[ "$fail" -ne 0 ] && { echo "[tyr-cl] preflight FAILED"; exit 1; }
[ "$MODE" = preflight ] && { echo "[tyr-cl] preflight only, stopping"; exit 0; }

# ------------------------------------------------- driver dirs (hardlinks, same fs)
for cfg in "${CFG_ARR[@]}"; do
  src=$REPO/outputs/$cfg
  dst=$DRIVERS/$cfg
  if [ -d "$dst" ]; then echo "[tyr-cl] driver dir exists: $dst"; else
    src_dev=$(stat -c %d "$src"); dst_dev=$(stat -c %d "$DRIVERS")
    if [ "$src_dev" != "$dst_dev" ]; then
      echo "[FAIL] $src and $DRIVERS are on different filesystems; a copy would pin"
      echo "       another 16 GB. Move outputs or drivers so they share a mount."; exit 1
    fi
    cp -al "$src" "$dst" || exit 1
    echo "[tyr-cl] hardlinked $src -> $dst"
  fi
  a=$(stat -c %i "$src/slim_state.pt"); b=$(stat -c %i "$dst/slim_state.pt")
  [ "$a" = "$b" ] || { echo "[FAIL] $dst/slim_state.pt is a copy, not a hardlink"; exit 1; }
  echo "[ok]   $cfg driver dir shares inodes with outputs/"
done

[ "$MODE" = prepare ] && { echo "[tyr-cl] prepared, not launching"; exit 0; }

# ---------------------------------------------------------------- run + merge + analyze
for cfg in "${CFG_ARR[@]}"; do
  echo "[tyr-cl] === $cfg: launching $n_shards shards"
  SUITE="$SUITE" PREFIX="$PREFIX" DRIVER_OMP_THREADS="$DRIVER_OMP_THREADS" \
    bash "$REPO/experiments/head_analysis/launch_alpasim_shards.sh" \
      "$cfg" "$N_SCENES" "$N_ROLLOUTS" "$GPUS" || { echo "[tyr-cl] $cfg shards failed"; exit 1; }

  shards=()
  for ((i = 0; i < n_shards; i++)); do shards+=("${PREFIX}${cfg}_sh${i}"); done
  "$REPO/.venv/bin/python" "$REPO/experiments/head_analysis/merge_alpasim_shards.py" \
    --runs-root "$RUNS" --shards "${shards[@]}" \
    --out "${PREFIX}merged_${cfg}" --expect-scenes "$N_SCENES" \
    || { echo "[tyr-cl] $cfg merge failed"; exit 1; }
  echo "[tyr-cl] merged -> $RUNS/${PREFIX}merged_${cfg}"
done

# analyze_alpasim needs alpasim's venv (alpasim_utils / pandas for the ASL logs)
ANALYZE_CFGS=(baseline slim_dual_u40_v2 "${CFG_ARR[@]}")
echo "[tyr-cl] analyzing: ${ANALYZE_CFGS[*]}"
( cd "$ALPASIM" && uv run python \
    "$REPO/experiments/head_analysis/analyze_alpasim.py" \
    --runs-root "$RUNS" --prefix "${PREFIX}merged_" \
    --configs "${ANALYZE_CFGS[@]}" \
    --out "$REPO/outputs/alpasim_tyr_2601" )
echo "[tyr-cl] done -> $REPO/outputs/alpasim_tyr_2601"
