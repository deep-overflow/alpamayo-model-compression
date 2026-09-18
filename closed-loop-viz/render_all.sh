#!/bin/bash
# Render every scene of the 150-scene matrix as a six-arm top-down replay.
#
# CPU only -- CUDA_VISIBLE_DEVICES is emptied per worker, so this cannot touch the cards
# the closed-loop runs are holding. The real shared resource here is cores: the box has 64
# and a running alpasim config keeps four driver stacks on them, so WORKERS stays well
# under what the machine could take. Unset OMP thread limits are what made a previous run
# put 256 spinning threads on 64 cores; matplotlib does not thread, but the worker count
# is the same kind of knob and is deliberately conservative.
#
# ~90 s per video at six panels, so 150 scenes is ~3.7 h serial and ~37 min at 6 workers.
# Re-running skips scenes whose mp4 already exists, so an interrupted batch resumes.

set -uo pipefail

REPO=/home/cvlab21/project/chan/alpamayo-model-compression
WT=$REPO/.claude/worktrees/closed-loop-viz
ALPASIM=/home/cvlab21/project/chan/alpasim
RUNS=/home/cvlab21/project/chan/alpasim-runs
OUT=${OUT:-$REPO/closed-loop-viz/video/matrix150}
WORKERS=${WORKERS:-6}
LOG=${LOG:-/home/cvlab21/project/chan/.claude/jobs/285e9a74/tmp/render_all.log}

ARMS=(
  "dual=slim_dual_u40_v2"
  "coc=slim_coc_u40_v2"
  "traj=slim_traj_u40_v2"
  # The real LLM-Pruner baseline, not a stand-in. Its 150-scene run lives in soowon's
  # runs_root rather than under our m2601_merged_ prefix, which is why an earlier batch
  # used spg_s1_uni_nr (0.733) instead: a search of our prefix simply does not find it.
  # Verified same 150 scenes, same record schema, GT path agreeing with ours at 0.0000 m.
  "llm-pruner=/mnt/nvme1n1/ad_vla/outputs/soowon/alpasim-analysis/runs_root/lp_r50"
  "tyr=slim_tyr_u40_r"
  "wanda=slim_wanda_u40_v2"
)

mkdir -p "$OUT"
: > "$LOG"

# the scene list comes from a run that has all 150, and every arm is checked to cover it
mapfile -t SCENES < <("$REPO/.venv/bin/python" - "$RUNS" "${ARMS[@]}" <<'PY'
import json, sys
from pathlib import Path
runs = Path(sys.argv[1])
sets = []
for spec in sys.argv[2:]:
    cfg = spec.split("=", 1)[1]
    # An absolute path is the run directory itself. Blindly prefixing it produced
    # `m2601_merged_/mnt/nvme1n1/...` and the batch died before rendering anything --
    # the same mistake render_arm_all.sh had, in a second place.
    run = Path(cfg) if Path(cfg).is_absolute() else runs / f"m2601_merged_{cfg}"
    d = json.loads((run / "aggregate/results-summary.json").read_text())
    sets.append({r["clipgt_id"] for r in d["rollouts"]})
common = sorted(set.intersection(*sets))
union = sorted(set.union(*sets))
if len(common) != len(union):
    sys.exit(f"FATAL: arm 간 씬 집합이 다릅니다 (공통 {len(common)}, 합집합 {len(union)})")
print("\n".join(common))
PY
)
[ "${#SCENES[@]}" -gt 0 ] || { echo "씬 목록을 못 만들었습니다"; exit 1; }
echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] 씬 ${#SCENES[@]}개, arm ${#ARMS[@]}개, 워커 $WORKERS" | tee -a "$LOG"

render_one() {
  local scene="$1"
  local out="$OUT/${scene}.mp4"
  if [ -s "$out" ]; then
    echo "skip  $scene (이미 있음)" >> "$LOG"
    return 0
  fi
  # bash cannot export an array, so the worker rebuilds it from the exported string;
  # the labels and configs contain no spaces, which is what makes this safe
  local args=()
  local a
  for a in $ARMS_STR; do args+=(--arm "$a"); done
  cd "$ALPASIM" || return 1
  if CUDA_VISIBLE_DEVICES="" uv run python "$WT/closed-loop-viz/render_replay.py" \
       --scene "$scene" "${args[@]}" --out "$out" >> "$LOG.$scene" 2>&1; then
    echo "ok    $scene  $(du -h "$out" | cut -f1)" >> "$LOG"
    rm -f "$LOG.$scene"
  else
    echo "FAIL  $scene -- $LOG.$scene 참조" >> "$LOG"
  fi
}
export -f render_one
export OUT ALPASIM WT LOG
export ARMS_STR="${ARMS[*]}"

printf '%s\n' "${SCENES[@]}" | xargs -P "$WORKERS" -I{} bash -c 'render_one "$@"' _ {}

ok=$(grep -c '^ok' "$LOG")
sk=$(grep -c '^skip' "$LOG")
bad=$(grep -c '^FAIL' "$LOG")
echo "[$(date -u -d '+9 hours' '+%m-%d %H:%M KST')] 완료: ok $ok / skip $sk / FAIL $bad" | tee -a "$LOG"
echo "-> $OUT"
